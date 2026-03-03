from __future__ import annotations

"""Clinically constrained counterfactual generation (custom, zero-order).

This module is designed to work with this repo's *existing* inference artifacts:

- A trained estimator saved in `models/*.joblib` (e.g., `models/rf_baseline.joblib`).
- A fitted canonical preprocessor saved in `models/preprocessor.pkl`.

It does NOT retrain any model and does NOT modify preprocessing/training code.
It preserves the canonical feature order strictly:

    ['hemo', 'sc', 'al', 'htn', 'age', 'dm']

The search is heuristic hill-climbing (gradient-free / zero-order).
As with most heuristic searches, it can converge to a local optimum; the
implementation includes simple random restarts to mitigate this but does not
guarantee a globally optimal counterfactual.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
 
import joblib
import numpy as np
import pandas as pd

from src.canonical import CANONICAL_FEATURES, assert_canonical_schema
from src.config import PREPROCESSOR_PATH, PREPROC_DIR


def _repo_root() -> Path:
    # repo root is the parent of src/
    return Path(__file__).resolve().parents[1]


def _resolve_repo_relative_path(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = _repo_root() / path
    return path


# -----------------------------
# Canonical clinical constraints
# -----------------------------

# Immutable features (cannot change)
IMMUTABLE_FEATURES: set[str] = {'age'}

# Binary features (must be 0 or 1)
BINARY_FEATURES: set[str] = {'htn', 'dm'}

# Physiological bounds (simple, for clipping)
PHYSIO_BOUNDS: dict[str, tuple[float, float]] = {
    'hemo': (3.0, 18.0),
    'sc': (0.4, 15.0),
    'al': (0.0, 5.0),
    'age': (1.0, 100.0),
}

# Directional constraints for a "healthier" counterfactual
# sc must not increase relative to the original patient.
DIRECTIONAL_NONINCREASE: set[str] = {'sc'}


def _as_single_row_dataframe(patient_input: Any) -> pd.DataFrame:
    """Coerce input into a 1-row DataFrame.

    Supported forms:
    - dict-like
    - pandas Series
    - pandas DataFrame (first row used if multiple rows)
    - numpy array / list-like of length == len(CANONICAL_FEATURES)
    """

    if isinstance(patient_input, pd.DataFrame):
        if patient_input.empty:
            raise ValueError('patient_input DataFrame is empty')
        return patient_input.iloc[[0]].copy()

    if isinstance(patient_input, pd.Series):
        return patient_input.to_frame().T

    if isinstance(patient_input, dict):
        return pd.DataFrame([patient_input])

    if isinstance(patient_input, (list, tuple, np.ndarray)):
        arr = np.asarray(patient_input, dtype=float).reshape(-1)
        if arr.size != len(CANONICAL_FEATURES):
            raise ValueError(
                'Array-like patient_input must have length '
                f"{len(CANONICAL_FEATURES)} (got {arr.size})."
            )
        return pd.DataFrame([arr], columns=CANONICAL_FEATURES)

    raise TypeError(
        'Unsupported patient_input type. Use dict, Series, DataFrame, or array-like of length 6.'
    )


def load_model(model_path: str | Path) -> Any:
    """Load a trained model (inference-only)."""

    path = _resolve_repo_relative_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f'Model not found: {path}')
    return joblib.load(path)


def load_preprocessor(preprocessor_path: str | Path = PREPROCESSOR_PATH) -> Any:
    """Load the fitted canonical preprocessor used in this repo."""

    path = _resolve_repo_relative_path(preprocessor_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Canonical preprocessor not found: {path}. '
            'Run src/cleaning.py once to generate it.'
        )
    return joblib.load(path)


def prepare_patient_canonical(
    patient_input: Any,
    *,
    preprocessor: Any | None = None,
    preprocess: bool = True,
) -> pd.DataFrame:
    """Return a 1-row canonical DataFrame in strict feature order.

    If preprocess=True (default), this applies the repo's fitted canonical
    preprocessor (`models/preprocessor.pkl`) for consistent coercion + imputation.

    This does not change the preprocessing pipeline; it reuses the existing one.
    """

    raw_df = _as_single_row_dataframe(patient_input)

    # If the input has canonical features (possibly among many), select them.
    missing = [c for c in CANONICAL_FEATURES if c not in raw_df.columns]
    if missing:
        raise ValueError(
            'Missing required canonical columns. '
            f"Expected at least: {CANONICAL_FEATURES}. Missing: {missing}"
        )
    raw_df = raw_df[CANONICAL_FEATURES].copy()

    if preprocess:
        pre = preprocessor if preprocessor is not None else load_preprocessor()
        X = pre.transform(raw_df)
    else:
        # Light coercion only; assumes already preprocessed.
        X = raw_df.copy()
        for col in ['hemo', 'sc', 'al', 'age']:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        for col in ['htn', 'dm']:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X[CANONICAL_FEATURES].copy()
    assert_canonical_schema(X)
    if X.isna().any().any():
        raise ValueError(
            'Canonical patient row contains NaNs after preparation. '
            'If providing raw input, keep preprocess=True so imputation runs.'
        )
    return X


def predict_proba_cf(model: Any, X_canonical: pd.DataFrame) -> np.ndarray:
    """Prediction wrapper returning full probability vector."""

    assert_canonical_schema(X_canonical)
    proba = model.predict_proba(X_canonical)
    return np.asarray(proba, dtype=float)


def _model_classes(model: Any) -> np.ndarray | None:
    classes = getattr(model, 'classes_', None)
    if classes is None:
        return None
    arr = np.asarray(classes)
    if arr.ndim != 1 or arr.size < 2:
        return None
    return arr


def _resolve_target_from_prediction(
    *,
    model: Any,
    X_canonical: pd.DataFrame,
    target_class: int | None,
) -> tuple[int, int]:
    """Resolve (target_label, target_index) robustly.

    - If target_class is None: set target to the *opposite* of the model's current predicted label.
    - If target_class is provided: treat it as the desired *label*.

    Returns:
        target_label: the class label to aim for (as returned by model.predict).
        target_index: the probability-vector index corresponding to target_label.
    """

    assert_canonical_schema(X_canonical)
    classes = _model_classes(model)

    # Determine predicted label
    try:
        pred_label_raw = model.predict(X_canonical)
        pred_label = int(np.asarray(pred_label_raw).reshape(-1)[0])
    except Exception:
        # Fall back to argmax proba if predict is unavailable.
        proba = predict_proba_cf(model, X_canonical)[0]
        if classes is not None and len(classes) == len(proba):
            pred_label = int(classes[int(np.argmax(proba))])
        else:
            pred_label = int(np.argmax(proba))

    # Choose target label
    if target_class is None:
        if classes is not None and int(classes.size) == 2:
            if pred_label == int(classes[0]):
                target_label = int(classes[1])
            else:
                target_label = int(classes[0])
        else:
            # Binary fallback: assume labels {0,1}
            target_label = 1 - int(pred_label)
    else:
        target_label = int(target_class)

    # Map label -> proba index
    if classes is not None:
        idxs = np.where(classes.astype(int) == int(target_label))[0]
        if idxs.size == 0:
            raise ValueError(
                f"target_class={target_label} not in model.classes_={classes.tolist()}. "
                "Pass target_class as one of the model's labels, or leave it None for auto-target."
            )
        target_index = int(idxs[0])
    else:
        # Assume label matches proba column index.
        target_index = int(target_label)

    return int(target_label), int(target_index)


def apply_constraints(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    *,
    feature_order: list[str] | None = None,
) -> np.ndarray:
    """Project a candidate into the clinically feasible space.

    - age immutable
    - htn, dm binary
    - sc cannot increase relative to original
    - clip physiologic bounds
    """

    feats = feature_order or CANONICAL_FEATURES
    out = np.asarray(x_cf, dtype=float).copy()

    # Bound clipping
    for i, f in enumerate(feats):
        if f in PHYSIO_BOUNDS:
            lo, hi = PHYSIO_BOUNDS[f]
            out[i] = float(np.clip(out[i], lo, hi))

    # Immutable
    for i, f in enumerate(feats):
        if f in IMMUTABLE_FEATURES:
            out[i] = float(x_original[i])

    # Binary
    for i, f in enumerate(feats):
        if f in BINARY_FEATURES:
            out[i] = 1.0 if out[i] >= 0.5 else 0.0

    # Directional: non-increase
    for i, f in enumerate(feats):
        if f in DIRECTIONAL_NONINCREASE:
            out[i] = float(min(out[i], x_original[i]))

    return out


def clinical_violation_penalty(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    *,
    feature_order: list[str] | None = None,
    big_penalty: float = 1_000.0,
) -> float:
    """Return a large penalty if any constraint is violated.

    Even though `apply_constraints` should prevent violations, this penalty makes
    the optimization robust to any accidental numerical issues or user misuse.
    """

    feats = feature_order or CANONICAL_FEATURES
    penalty = 0.0
    x_cf = np.asarray(x_cf, dtype=float)

    # Bounds
    for i, f in enumerate(feats):
        if f in PHYSIO_BOUNDS:
            lo, hi = PHYSIO_BOUNDS[f]
            if not (lo <= float(x_cf[i]) <= hi):
                penalty += big_penalty

    # Immutable
    for i, f in enumerate(feats):
        if f in IMMUTABLE_FEATURES and float(x_cf[i]) != float(x_original[i]):
            penalty += big_penalty

    # Binary
    for i, f in enumerate(feats):
        if f in BINARY_FEATURES and float(x_cf[i]) not in (0.0, 1.0):
            penalty += big_penalty

    # Directional
    for i, f in enumerate(feats):
        if f in DIRECTIONAL_NONINCREASE and float(x_cf[i]) > float(x_original[i]) + 1e-12:
            penalty += big_penalty

    return float(penalty)


def _feature_ranges(feature_order: list[str] | None = None) -> np.ndarray:
    feats = feature_order or CANONICAL_FEATURES
    ranges = []
    for f in feats:
        if f in PHYSIO_BOUNDS:
            lo, hi = PHYSIO_BOUNDS[f]
            ranges.append(float(hi - lo))
        else:
            ranges.append(1.0)
    return np.asarray(ranges, dtype=float)


def _load_training_feature_scales(
    *,
    feature_order: list[str] | None = None,
    preproc_dir: str | Path = PREPROC_DIR,
) -> np.ndarray:
    """Load per-feature std dev from the canonical preprocessed TRAIN split.

    This repo intentionally does not StandardScale features. Using z-score style
    normalization for proximity avoids domination by large-range variables.

    Falls back to feature ranges (bounded) if the training file is missing.
    """

    feats = feature_order or CANONICAL_FEATURES
    path = _resolve_repo_relative_path(Path(preproc_dir) / 'X_train_preproc.csv')
    try:
        Xtr = pd.read_csv(path)
        Xtr = Xtr[feats].copy()
        assert_canonical_schema(Xtr)
        scales = Xtr.std(axis=0, ddof=0).to_numpy(dtype=float)
        scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
        return np.asarray(scales, dtype=float)
    except Exception:
        # Conservative fallback: use physiologic ranges.
        return np.maximum(_feature_ranges(feats), 1e-12)


def prediction_loss(proba_vec: np.ndarray, target_index: int, eps: float = 1e-9) -> float:
    p = float(np.clip(proba_vec[int(target_index)], eps, 1.0))
    return float(-np.log(p))


def proximity_loss(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    *,
    feature_order: list[str] | None = None,
    mode: str = 'zscore',
    feature_scales: np.ndarray | None = None,
) -> float:
    feats = feature_order or CANONICAL_FEATURES
    diff = np.abs(np.asarray(x_cf, dtype=float) - np.asarray(x_original, dtype=float))

    # Proximity normalization choice:
    # - 'zscore': divide by per-feature std learned from train split (default)
    # - 'range' : divide by physiologic bounds range (legacy)
    # - 'none'  : raw L1 in feature units
    if mode == 'zscore':
        scales = feature_scales
        if scales is None:
            scales = _load_training_feature_scales(feature_order=feats)
        scales = np.asarray(scales, dtype=float).reshape(-1)
        if scales.size != len(feats):
            raise ValueError('feature_scales must match number of features')
        diff = diff / np.maximum(scales, 1e-12)
    elif mode == 'range':
        diff = diff / np.maximum(_feature_ranges(feats), 1e-12)
    elif mode == 'none':
        pass
    else:
        raise ValueError("mode must be one of {'zscore','range','none'}")

    # Age is immutable so contributes 0; keep it in for simplicity.
    return float(np.sum(diff))


def diversity_loss(
    x_cf: np.ndarray,
    accepted: Iterable[np.ndarray],
    *,
    feature_order: list[str] | None = None,
    normalize: bool = True,
) -> float:
    accepted_list = [np.asarray(a, dtype=float) for a in accepted]
    if not accepted_list:
        return 0.0
    feats = feature_order or CANONICAL_FEATURES
    ranges = _feature_ranges(feats)
    dists = []
    for a in accepted_list:
        d = np.abs(np.asarray(x_cf, dtype=float) - a)
        if normalize:
            d = d / np.maximum(ranges, 1e-12)
        dists.append(float(np.sum(d)))
    return float(np.mean(dists))


def dice_similarity_continuous(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Continuous Dice similarity for probability vectors."""

    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    inter = float(np.sum(np.minimum(a, b)))
    denom = float(np.sum(a) + np.sum(b) + eps)
    return float(2.0 * inter / denom)


def robustness_score(
    model: Any,
    x_cf: np.ndarray,
    *,
    n_noise: int = 25,
    noise_frac: float = 0.01,
    rng: np.random.Generator | None = None,
    feature_order: list[str] | None = None,
) -> float:
    """Robustness as mean Dice similarity under small Gaussian noise."""

    feats = feature_order or CANONICAL_FEATURES
    rng = rng or np.random.default_rng(0)

    base_df = pd.DataFrame([x_cf], columns=feats)
    assert_canonical_schema(base_df)
    base_pred = predict_proba_cf(model, base_df)[0]

    ranges = _feature_ranges(feats)
    sims: list[float] = []
    for _ in range(int(n_noise)):
        noise = rng.normal(loc=0.0, scale=ranges * float(noise_frac), size=len(feats))

        # Do not add Gaussian noise to binary/immutable features.
        for i, f in enumerate(feats):
            if f in BINARY_FEATURES or f in IMMUTABLE_FEATURES:
                noise[i] = 0.0

        x_noisy = apply_constraints(x_cf + noise, x_cf, feature_order=feats)
        noisy_df = pd.DataFrame([x_noisy], columns=feats)
        pred = predict_proba_cf(model, noisy_df)[0]
        sims.append(dice_similarity_continuous(base_pred, pred))

    return float(np.mean(sims)) if sims else 0.0


@dataclass
class LossWeights:
    lambda_prox: float = 0.5
    lambda_div: float = 0.3
    lambda_rob: float = 0.2
    lambda_clin: float = 2.0


def total_loss(
    *,
    model: Any,
    x_cf: np.ndarray,
    x_original: np.ndarray,
    target_index: int,
    accepted: Iterable[np.ndarray],
    weights: LossWeights,
    rng: np.random.Generator,
    proximity_mode: str = 'zscore',
    feature_scales: np.ndarray | None = None,
    feature_order: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    feats = feature_order or CANONICAL_FEATURES

    X = pd.DataFrame([x_cf], columns=feats)
    proba_vec = predict_proba_cf(model, X)[0]

    l_pred = prediction_loss(proba_vec, target_index)
    l_prox = proximity_loss(
        x_cf,
        x_original,
        feature_order=feats,
        mode=proximity_mode,
        feature_scales=feature_scales,
    )
    l_div = diversity_loss(x_cf, accepted, feature_order=feats, normalize=True)

    # Robustness is expensive (many extra predict_proba calls).
    # If the user sets lambda_rob=0 for debugging / live demos, skip it entirely.
    if float(getattr(weights, 'lambda_rob', 0.0)) == 0.0:
        rob = 1.0
        l_rob = 0.0
    else:
        rob = robustness_score(model, x_cf, rng=rng, feature_order=feats)
        l_rob = 1.0 - float(np.clip(rob, 0.0, 1.0))
    l_clin = clinical_violation_penalty(x_cf, x_original, feature_order=feats)

    # Spec: L = L_pred + λp*L_prox - λd*L_div + λr*L_rob + λc*L_clin
    L = (
        l_pred
        + weights.lambda_prox * l_prox
        - weights.lambda_div * l_div
        + weights.lambda_rob * l_rob
        + weights.lambda_clin * l_clin
    )
    parts = {
        'L_total': float(L),
        'L_pred': float(l_pred),
        'L_prox': float(l_prox),
        'L_div': float(l_div),
        'L_rob': float(l_rob),
        'L_clin': float(l_clin),
        'robustness': float(rob),
        'p_target': float(proba_vec[int(target_index)]),
    }
    return float(L), parts


def _neighbor_candidates(
    x_current: np.ndarray,
    x_original: np.ndarray,
    *,
    rng: np.random.Generator,
    n_neighbors: int = 40,
    step_frac: float = 0.05,
    feature_order: list[str] | None = None,
) -> list[np.ndarray]:
    """Generate neighbor candidates via small random perturbations."""

    feats = feature_order or CANONICAL_FEATURES
    ranges = _feature_ranges(feats)

    mutable_idx = [i for i, f in enumerate(feats) if f not in IMMUTABLE_FEATURES]
    if not mutable_idx:
        return []

    out: list[np.ndarray] = []
    for _ in range(int(n_neighbors)):
        i = int(rng.choice(mutable_idx))
        f = feats[i]

        cand = np.asarray(x_current, dtype=float).copy()

        if f in BINARY_FEATURES:
            cand[i] = 1.0 - float(cand[i] >= 0.5)
        else:
            step = float(ranges[i] * step_frac)
            if f in DIRECTIONAL_NONINCREASE:
                # Bias toward decrease
                delta = -abs(float(rng.normal(loc=0.0, scale=step)))
            else:
                delta = float(rng.normal(loc=0.0, scale=step))
            cand[i] = float(cand[i] + delta)

            # Rare global resample within bounds to escape local minima
            if rng.random() < 0.05 and f in PHYSIO_BOUNDS:
                lo, hi = PHYSIO_BOUNDS[f]
                cand[i] = float(rng.uniform(lo, hi))

        cand = apply_constraints(cand, x_original, feature_order=feats)
        out.append(cand)

    return out


def hill_climb_counterfactual(
    *,
    model: Any,
    x_original: np.ndarray,
    target_index: int,
    accepted: Iterable[np.ndarray] = (),
    weights: LossWeights | None = None,
    max_iter: int = 300,
    target_prob: float = 0.90,
    rng: np.random.Generator | None = None,
    n_neighbors: int = 40,
    proximity_mode: str = 'zscore',
    feature_scales: np.ndarray | None = None,
    feature_order: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Zero-order hill-climbing search for one counterfactual."""

    feats = feature_order or CANONICAL_FEATURES
    rng = rng or np.random.default_rng(0)
    weights = weights or LossWeights()

    # Start near original with small noise (keeps constraints intact)
    ranges = _feature_ranges(feats)
    noise = rng.normal(loc=0.0, scale=ranges * 0.02, size=len(feats))
    for i, f in enumerate(feats):
        if f in BINARY_FEATURES or f in IMMUTABLE_FEATURES:
            noise[i] = 0.0
    x_best = apply_constraints(x_original + noise, x_original, feature_order=feats)

    best_L, best_parts = total_loss(
        model=model,
        x_cf=x_best,
        x_original=x_original,
        target_index=target_index,
        accepted=accepted,
        weights=weights,
        rng=rng,
        proximity_mode=proximity_mode,
        feature_scales=feature_scales,
        feature_order=feats,
    )

    stall = 0
    for _ in range(int(max_iter)):
        # Early stop if we already achieved the target confidently
        if best_parts['p_target'] >= float(target_prob):
            break

        neighbors = _neighbor_candidates(
            x_best,
            x_original,
            rng=rng,
            n_neighbors=n_neighbors,
            step_frac=0.05,
            feature_order=feats,
        )
        if not neighbors:
            break

        improved = False

        # Major speed-up: evaluate neighbor probabilities in one batch call.
        # (Calling RandomForest.predict_proba per-candidate is very expensive.)
        Xn = pd.DataFrame(neighbors, columns=feats)
        probas = predict_proba_cf(model, Xn)

        for cand, proba_vec in zip(neighbors, probas):
            l_pred = prediction_loss(proba_vec, target_index)
            l_prox = proximity_loss(
                cand,
                x_original,
                feature_order=feats,
                mode=proximity_mode,
                feature_scales=feature_scales,
            )
            l_div = diversity_loss(cand, accepted, feature_order=feats, normalize=True)
            l_clin = clinical_violation_penalty(cand, x_original, feature_order=feats)

            if float(getattr(weights, 'lambda_rob', 0.0)) == 0.0:
                rob = 1.0
                l_rob = 0.0
            else:
                rob = robustness_score(model, cand, rng=rng, feature_order=feats)
                l_rob = 1.0 - float(np.clip(rob, 0.0, 1.0))

            L = (
                l_pred
                + weights.lambda_prox * l_prox
                - weights.lambda_div * l_div
                + weights.lambda_rob * l_rob
                + weights.lambda_clin * l_clin
            )

            if float(L) < best_L:
                best_L = float(L)
                x_best = np.asarray(cand, dtype=float)
                best_parts = {
                    'L_total': float(L),
                    'L_pred': float(l_pred),
                    'L_prox': float(l_prox),
                    'L_div': float(l_div),
                    'L_rob': float(l_rob),
                    'L_clin': float(l_clin),
                    'robustness': float(rob),
                    'p_target': float(proba_vec[int(target_index)]),
                }
                improved = True

        if improved:
            stall = 0
        else:
            stall += 1
            # Simple restart if stuck for too long
            if stall >= 25:
                stall = 0
                noise = rng.normal(loc=0.0, scale=ranges * 0.05, size=len(feats))
                for i, f in enumerate(feats):
                    if f in BINARY_FEATURES or f in IMMUTABLE_FEATURES:
                        noise[i] = 0.0
                x_best = apply_constraints(x_original + noise, x_original, feature_order=feats)
                best_L, best_parts = total_loss(
                    model=model,
                    x_cf=x_best,
                    x_original=x_original,
                    target_index=target_index,
                    accepted=accepted,
                    weights=weights,
                    rng=rng,
                    proximity_mode=proximity_mode,
                    feature_scales=feature_scales,
                    feature_order=feats,
                )

    return x_best, best_parts


def _pareto_mask(
    metrics_df: pd.DataFrame,
    *,
    objectives: list[tuple[str, str]],
) -> np.ndarray:
    """Compute Pareto-optimal mask.

    objectives: list of (column, direction) where direction is 'min' or 'max'.
    """

    if metrics_df.empty:
        return np.zeros((0,), dtype=bool)

    cols = [c for c, _ in objectives]
    vals = metrics_df[cols].to_numpy(dtype=float)
    dirs = [d for _, d in objectives]

    # Convert all objectives to minimization.
    for j, d in enumerate(dirs):
        if d == 'max':
            vals[:, j] = -vals[:, j]
        elif d != 'min':
            raise ValueError("Objective direction must be 'min' or 'max'")

    n = vals.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        vi = vals[i]
        # Any point j dominates i if it is <= in all dims and < in at least one.
        for j in range(n):
            if i == j or not keep[i]:
                continue
            vj = vals[j]
            if np.all(vj <= vi) and np.any(vj < vi):
                keep[i] = False
                break
    return keep


def _sparsity_count(
    x_cf: np.ndarray,
    x_original: np.ndarray,
    *,
    feature_order: list[str] | None = None,
    tol: float = 1e-6,
) -> int:
    feats = feature_order or CANONICAL_FEATURES
    count = 0
    for i, f in enumerate(feats):
        if f in IMMUTABLE_FEATURES:
            continue
        if f in BINARY_FEATURES:
            if int(round(float(x_cf[i]))) != int(round(float(x_original[i]))):
                count += 1
        else:
            if abs(float(x_cf[i]) - float(x_original[i])) > float(tol):
                count += 1
    return int(count)


def generate_k_counterfactuals(
    *,
    model: Any,
    patient_canonical: pd.DataFrame,
    target_label: int,
    target_index: int,
    k: int = 3,
    max_iter: int = 300,
    target_prob: float = 0.90,
    weights: LossWeights | None = None,
    seed: int = 42,
    proximity_mode: str = 'zscore',
    feature_scales: np.ndarray | None = None,
    selection: str = 'pareto',
    pool_size: int | None = None,
    max_attempts: int | None = None,
    n_neighbors: int = 40,
    feature_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate counterfactuals and return Pareto-optimal subset by default.

    Notes:
    - The optimizer is single-objective (weighted loss), but the returned set can
      be filtered to a Pareto front over multiple metrics.
    - selection='pareto' returns up to k non-dominated CFs (by default metrics).
    - selection='rank' reproduces legacy behavior (rank by p_target then proximity).
    """

    feats = feature_order or CANONICAL_FEATURES
    assert_canonical_schema(patient_canonical)

    x_original = patient_canonical.iloc[0].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    weights = weights or LossWeights()

    accepted: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    # Build a pool of valid CFs, then select (Pareto or rank).
    # NOTE: attempts can be expensive; keep defaults reasonable for demos.
    desired_pool = int(pool_size) if pool_size is not None else max(int(k) * 4, int(k))
    max_attempts_eff = int(max_attempts) if max_attempts is not None else max(desired_pool * 3, int(k) * 6)

    for attempt in range(int(max_attempts_eff)):
        local_rng = np.random.default_rng(int(seed) * 10_000 + int(attempt))

        x_cf, parts = hill_climb_counterfactual(
            model=model,
            x_original=x_original,
            target_index=int(target_index),
            accepted=accepted,
            weights=weights,
            max_iter=max_iter,
            target_prob=target_prob,
            rng=local_rng,
            n_neighbors=int(n_neighbors),
            proximity_mode=proximity_mode,
            feature_scales=feature_scales,
            feature_order=feats,
        )

        # Keep only if it truly achieves the target class (label match)
        proba = predict_proba_cf(model, pd.DataFrame([x_cf], columns=feats))[0]
        try:
            pred_label = int(model.predict(pd.DataFrame([x_cf], columns=feats)).reshape(-1)[0])
        except Exception:
            classes = _model_classes(model)
            if classes is not None and int(classes.size) == int(len(proba)):
                pred_label = int(classes[int(np.argmax(proba))])
            else:
                pred_label = int(np.argmax(proba))
        valid = int(pred_label) == int(target_label)
        if not valid:
            continue

        accepted.append(x_cf)
        rows.append(
            {
                'validity': bool(valid),
                'proximity': float(
                    proximity_loss(
                        x_cf,
                        x_original,
                        feature_order=feats,
                        mode=proximity_mode,
                        feature_scales=feature_scales,
                    )
                ),
                'sparsity': int(_sparsity_count(x_cf, x_original, feature_order=feats)),
                'p_target': float(proba[int(target_index)]),
                'robustness': float(parts.get('robustness', 0.0)),
            }
        )

        if len(accepted) >= int(desired_pool):
            break

    if not accepted:
        # Return empty results but keep schema stable.
        return (
            pd.DataFrame(columns=feats),
            pd.DataFrame(columns=['validity', 'proximity', 'sparsity', 'p_target', 'robustness']),
        )

    cfs_df = pd.DataFrame(accepted, columns=feats)
    metrics_df = pd.DataFrame(rows)

    # Selection
    if selection not in ('pareto', 'rank'):
        raise ValueError("selection must be one of {'pareto','rank'}")

    if selection == 'pareto':
        objectives = [('proximity', 'min'), ('sparsity', 'min'), ('p_target', 'max'), ('robustness', 'max')]
        mask = _pareto_mask(metrics_df, objectives=objectives)
        metrics_df = metrics_df.assign(pareto_optimal=mask.astype(bool))
        pareto_idx = metrics_df.index[metrics_df['pareto_optimal']].to_list()
        if pareto_idx:
            # If more than k, return the most confident + closest among the Pareto front.
            sub = metrics_df.loc[pareto_idx].sort_values(by=['p_target', 'proximity'], ascending=[False, True])
            keep_idx = sub.index.to_list()[: int(k)]
            cfs_df = cfs_df.loc[keep_idx].reset_index(drop=True)
            metrics_df = metrics_df.loc[keep_idx].reset_index(drop=True)
        else:
            # Should not happen (mask empty while metrics_df non-empty), but be defensive.
            cfs_df = cfs_df.head(int(k)).reset_index(drop=True)
            metrics_df = metrics_df.head(int(k)).reset_index(drop=True)
    else:
        # Legacy rank selection
        order = metrics_df.sort_values(by=['p_target', 'proximity'], ascending=[False, True]).index
        cfs_df = cfs_df.loc[order]
        metrics_df = metrics_df.loc[order]
        cfs_df = cfs_df.head(int(k)).reset_index(drop=True)
        metrics_df = metrics_df.head(int(k)).reset_index(drop=True)

    return cfs_df, metrics_df


def _safe_import_shap() -> Any | None:
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


def explain_original_vs_counterfactual(
    *,
    model: Any,
    x_original: pd.DataFrame,
    x_cf_best: pd.DataFrame,
    target_label: int,
    target_index: int,
    stability_runs: int = 0,
    stability_k: int = 5,
    stability_noise_frac: float = 0.01,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """SHAP explanation for original and best counterfactual (TreeExplainer)."""

    assert_canonical_schema(x_original)
    assert_canonical_schema(x_cf_best)

    shap = _safe_import_shap()
    orig_proba = predict_proba_cf(model, x_original)[0]
    cf_proba = predict_proba_cf(model, x_cf_best)[0]

    out: dict[str, Any] = {
        'target_class': int(target_label),
        'target_index': int(target_index),
        'original_proba': orig_proba.tolist(),
        'counterfactual_proba': cf_proba.tolist(),
        'shap_available': bool(shap is not None),
    }

    if shap is None:
        out['note'] = 'Install shap to enable SHAP explanations (pip install shap).'
        return out

    try:
        explainer = shap.TreeExplainer(model)
        shap_orig = explainer.shap_values(x_original)
        shap_cf = explainer.shap_values(x_cf_best)
    except Exception as e:
        out['note'] = f'Could not compute TreeExplainer SHAP: {e}'
        return out

    # Normalize to (n_samples, n_features) for the selected class
    def _select_class(shap_values: Any) -> np.ndarray:
        if isinstance(shap_values, list):
            # Binary classification often returns [class0, class1]
            idx = int(target_index) if len(shap_values) > int(target_index) else -1
            arr = np.asarray(shap_values[idx])
        else:
            arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_classes, n_samples, n_features)
            arr = arr[int(target_index)]
        return np.asarray(arr, dtype=float)

    s0 = _select_class(shap_orig)[0]
    s1 = _select_class(shap_cf)[0]
    delta = s1 - s0

    # Simple explanation consistency: overlap of top-|SHAP| features.
    k_cons = min(5, len(CANONICAL_FEATURES))
    top0 = set(np.argsort(-np.abs(s0))[:k_cons].tolist())
    top1 = set(np.argsort(-np.abs(s1))[:k_cons].tolist())
    union = top0 | top1
    inter = top0 & top1
    out['shap_topk_overlap'] = {
        'k': int(k_cons),
        'jaccard': float(len(inter) / max(1, len(union))),
        'features_original': [CANONICAL_FEATURES[i] for i in sorted(top0)],
        'features_counterfactual': [CANONICAL_FEATURES[i] for i in sorted(top1)],
    }

    out['shap_original'] = {f: float(v) for f, v in zip(CANONICAL_FEATURES, s0)}
    out['shap_counterfactual'] = {f: float(v) for f, v in zip(CANONICAL_FEATURES, s1)}
    out['shap_delta'] = {f: float(v) for f, v in zip(CANONICAL_FEATURES, delta)}

    top = sorted(zip(CANONICAL_FEATURES, np.abs(delta)), key=lambda x: x[1], reverse=True)
    out['top_shap_delta_features'] = [f for f, _ in top[:5]]

    # Optional: input-perturbation stability (not the same as multi-seed re-training stability).
    if int(stability_runs) > 1:
        try:
            from src.explanation_metrics import (  # noqa: WPS433
                explanation_stability_score,
                cross_model_consistency_jaccard,
            )

            rng = rng or np.random.default_rng(0)

            def _stack_runs(base_row: pd.DataFrame) -> np.ndarray:
                feats = CANONICAL_FEATURES
                x0 = base_row.iloc[0].to_numpy(dtype=float)
                ranges = _feature_ranges(feats)
                runs: list[np.ndarray] = []
                for r in range(int(stability_runs)):
                    noise = rng.normal(loc=0.0, scale=ranges * float(stability_noise_frac), size=len(feats))
                    for i, f in enumerate(feats):
                        if f in BINARY_FEATURES or f in IMMUTABLE_FEATURES:
                            noise[i] = 0.0
                    x_noisy = apply_constraints(x0 + noise, x0, feature_order=feats)
                    df = pd.DataFrame([x_noisy], columns=feats)
                    sv = explainer.shap_values(df)
                    runs.append(_select_class(sv)[0])
                return np.stack(runs, axis=0).reshape(int(stability_runs), 1, len(feats))

            runs_orig = _stack_runs(x_original)
            runs_cf = _stack_runs(x_cf_best)

            out['shap_input_stability'] = {
                'note': (
                    'Computed on SHAP values over small input perturbations; '
                    'this measures explanation sensitivity to feature noise, '
                    'not variability across re-training seeds.'
                ),
                'runs': int(stability_runs),
                'k': int(stability_k),
                'ess_original': float(explanation_stability_score(runs_orig)),
                'ess_counterfactual': float(explanation_stability_score(runs_cf)),
                'topk_jaccard_original': float(cross_model_consistency_jaccard(runs_orig, k=int(stability_k))),
                'topk_jaccard_counterfactual': float(cross_model_consistency_jaccard(runs_cf, k=int(stability_k))),
            }
        except Exception as e:
            out['shap_input_stability'] = {
                'note': f'Could not compute SHAP input-perturbation stability: {e}',
                'runs': int(stability_runs),
            }

    return out


def _clinical_interpretation_text(
    x_original: pd.Series,
    x_cf: pd.Series,
    *,
    decimals: int = 2,
) -> str:
    """Generate a simple human-readable interpretation of changes."""

    lines: list[str] = []

    def _fmt(v: float) -> str:
        return f'{float(v):.{int(decimals)}f}'

    # Numeric
    for f in ['sc', 'hemo', 'al']:
        if f not in x_original.index:
            continue
        a = float(x_original[f])
        b = float(x_cf[f])
        if abs(a - b) < 1e-9:
            continue
        direction = 'decreased' if b < a else 'increased'
        label = {
            'sc': 'serum creatinine (sc)',
            'hemo': 'hemoglobin (hemo)',
            'al': 'albumin (al)',
        }[f]
        lines.append(f"{label} {direction} from {_fmt(a)} to {_fmt(b)}")

    # Binary
    for f in ['htn', 'dm']:
        if f not in x_original.index:
            continue
        a = int(round(float(x_original[f])))
        b = int(round(float(x_cf[f])))
        if a == b:
            continue
        label = {'htn': 'hypertension (htn)', 'dm': 'diabetes (dm)'}[f]
        lines.append(f"{label} changed from {a} to {b}")

    if not lines:
        return 'No actionable feature changes were found (counterfactual identical to original under constraints).'

    return 'Clinical interpretation (feature changes):\n- ' + '\n- '.join(lines)


def generate_counterfactual(
    patient_input: Any,
    *,
    model_path: str | Path = 'models/rf_augmented_42_v1.joblib',
    preprocessor_path: str | Path = PREPROCESSOR_PATH,
    target_class: int | None = None,
    k: int = 3,
    preprocess_input: bool = True,
    max_iter: int = 300,
    target_prob: float = 0.90,
    seed: int = 42,
    weights: LossWeights | None = None,
    proximity_mode: str = 'zscore',
    selection: str = 'pareto',
    explanation_stability_runs: int = 10,
    compute_explanation: bool = True,
    pool_size: int | None = None,
    max_attempts: int | None = None,
    n_neighbors: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Main entrypoint.

    Notes:
    - If `target_class` is None (default), the target is set automatically to the
      *opposite* of the model's current predicted label for this patient.
      This prevents silent mismatches if label conventions differ.

    Returns:
    - best_counterfactuals_df: canonical features for each CF (k rows)
    - metrics_df: validity/proximity/sparsity/probability/robustness per CF
    - explanation: dict (SHAP + text + key deltas)
    - comparison_table: original vs best CF (2 rows)
    """

    model = load_model(model_path)
    preproc = load_preprocessor(preprocessor_path)

    patient_canon = prepare_patient_canonical(
        patient_input,
        preprocessor=preproc,
        preprocess=preprocess_input,
    )

    # Auto-resolve target based on current prediction (binary flip)
    resolved_target_label, target_index = _resolve_target_from_prediction(
        model=model,
        X_canonical=patient_canon,
        target_class=target_class,
    )

    # Proximity scales (train-set std) for standardized proximity
    feature_scales = None
    if proximity_mode == 'zscore':
        feature_scales = _load_training_feature_scales(feature_order=CANONICAL_FEATURES)

    cfs_df, metrics_df = generate_k_counterfactuals(
        model=model,
        patient_canonical=patient_canon,
        target_label=int(resolved_target_label),
        target_index=int(target_index),
        k=int(k),
        max_iter=int(max_iter),
        target_prob=float(target_prob),
        weights=weights,
        seed=int(seed),
        proximity_mode=proximity_mode,
        feature_scales=feature_scales,
        selection=selection,
        pool_size=pool_size,
        max_attempts=max_attempts,
        n_neighbors=int(n_neighbors),
    )

    # Choose the best CF (highest p_target, then lowest proximity)
    explanation: dict[str, Any]
    comparison: pd.DataFrame
    if metrics_df.empty:
        explanation = {
            'note': 'No valid counterfactual found under constraints with the given search budget.',
            'original_proba': predict_proba_cf(model, patient_canon)[0].tolist(),
            'target_class': int(resolved_target_label),
            'auto_target': bool(target_class is None),
            'target_index': int(target_index),
        }
        comparison = pd.concat(
            [
                patient_canon.assign(row='original'),
            ],
            ignore_index=True,
        )
        return cfs_df, metrics_df, explanation, comparison

    best_idx = (
        metrics_df.sort_values(by=['p_target', 'proximity'], ascending=[False, True])
        .index
        .to_list()[0]
    )
    best_cf = cfs_df.loc[[best_idx]].copy()
    best_cf = best_cf[CANONICAL_FEATURES].copy()
    assert_canonical_schema(best_cf)

    # Explanation
    x0 = patient_canon.iloc[0]
    x1 = best_cf.iloc[0]
    changed = {
        f: {'original': float(x0[f]), 'counterfactual': float(x1[f])}
        for f in CANONICAL_FEATURES
        if f not in IMMUTABLE_FEATURES and float(x0[f]) != float(x1[f])
    }
    if bool(compute_explanation):
        explanation = explain_original_vs_counterfactual(
            model=model,
            x_original=patient_canon,
            x_cf_best=best_cf,
            target_label=int(resolved_target_label),
            target_index=int(target_index),
            stability_runs=int(explanation_stability_runs),
            rng=np.random.default_rng(int(seed)),
        )
        explanation['auto_target'] = bool(target_class is None)
    else:
        explanation = {
            'note': 'explanation disabled (compute_explanation=False)',
            'original_proba': predict_proba_cf(model, patient_canon)[0].tolist(),
            'counterfactual_proba': predict_proba_cf(model, best_cf)[0].tolist(),
            'target_class': int(resolved_target_label),
            'target_index': int(target_index),
            'auto_target': bool(target_class is None),
        }
    explanation['top_features_changed'] = list(changed.keys())
    explanation['feature_changes'] = changed
    explanation['clinical_text'] = _clinical_interpretation_text(x0, x1)

    # Comparison table
    comparison = pd.concat(
        [
            patient_canon.assign(row='original'),
            best_cf.assign(row='best_counterfactual'),
        ],
        ignore_index=True,
    )
    cols = ['row'] + CANONICAL_FEATURES
    comparison = comparison[cols]

    return cfs_df, metrics_df, explanation, comparison

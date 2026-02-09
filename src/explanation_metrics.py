from __future__ import annotations

"""Reusable explanation stability/consistency metrics.

This module is intentionally **model-training independent**: it operates purely on
pre-computed explanation tensors (e.g., SHAP) plus optional `pred_fn`/`mask_fn`
callbacks for agreement-style metrics.

All public metrics return values clipped to [0, 1] for comparability.

Notation
--------
- R: number of repeated runs of the same model
- M: number of different models
- n: number of samples
- d: number of features

Expected explanation tensor shapes
---------------------------------
- Runs:   (R, n, d)
- Models: (M, n, d)

The high-level `compute_all_metrics` assumes `x` corresponds to one of the rows
in the explanation tensors; by default `sample_index=0` is used.

Only NumPy + SciPy are used (no pandas).
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata


ArrayLike = np.ndarray


def _as_3d_stack(name: str, arrs: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Normalize a list/array of SHAP arrays into a float64 ndarray of shape (T, n, d)."""
    if isinstance(arrs, np.ndarray):
        x = np.asarray(arrs)
        if x.ndim != 3:
            raise ValueError(f"{name} must have shape (T, n_samples, d); got ndarray with shape {x.shape}.")
        out = x
    else:
        if len(arrs) == 0:
            raise ValueError(f"{name} must be a non-empty list of arrays.")
        mats: list[np.ndarray] = []
        for idx, a in enumerate(arrs):
            aa = np.asarray(a)
            if aa.ndim == 1:
                aa = aa.reshape(1, -1)
            if aa.ndim != 2:
                raise ValueError(
                    f"{name}[{idx}] must have shape (n_samples, d) or (d,); got {aa.shape}."
                )
            mats.append(aa)
        try:
            out = np.stack(mats, axis=0)
        except Exception as e:  # pragma: no cover
            shapes = [m.shape for m in mats]
            raise ValueError(f"{name} could not be stacked; shapes={shapes}") from e

    out = np.asarray(out, dtype=np.float64)
    if not np.isfinite(out).all():
        bad = np.argwhere(~np.isfinite(out))
        raise ValueError(f"{name} contains non-finite values; first bad index: {bad[0].tolist() if bad.size else 'unknown'}")
    return out


def _validate_feature_names(feature_names: Sequence[str], d: int) -> list[str]:
    names = list(feature_names)
    if len(names) != d:
        raise ValueError(f"feature_names must have length d={d}; got {len(names)}.")
    if any(not isinstance(s, str) or not s for s in names):
        raise ValueError("feature_names must be a sequence of non-empty strings.")
    return names


def _validate_k(k: int, d: int) -> int:
    if not isinstance(k, int):
        raise TypeError(f"k must be int; got {type(k)}")
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k > d:
        raise ValueError(f"k must be <= d={d}; got k={k}")
    return k


def _validate_dmax(D_max: float) -> float:
    D = float(D_max)
    if not np.isfinite(D) or D <= 0:
        raise ValueError(f"D_max must be a positive finite float; got {D_max!r}.")
    return D


def _clip01(x: np.ndarray | float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _topk_indices_by_abs(phi: np.ndarray, k: int) -> np.ndarray:
    """Return top-k feature indices by descending |phi|.

    Parameters
    ----------
    phi:
        Array of shape (..., d)
    k:
        Number of top features.

    Returns
    -------
    np.ndarray
        Integer indices of shape (..., k).
    """
    if phi.ndim < 1:
        raise ValueError("phi must have at least 1 dimension")
    d = phi.shape[-1]
    k = _validate_k(k, d)

    scores = np.abs(phi)

    # Use argpartition for O(d) selection, then sort within the top-k.
    part = np.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    topk_scores = np.take_along_axis(scores, part, axis=-1)
    order = np.argsort(-topk_scores, axis=-1)
    return np.take_along_axis(part, order, axis=-1)


def explanation_stability_score(shap_runs: Sequence[np.ndarray] | np.ndarray) -> float:
    """Compute Explanation Stability Score (ESS) across consecutive runs.

    ESS is computed per-sample using Spearman-rank distance of feature importances,
    using the provided closed-form formula:

    ESS = 1 − [6 * Σ (r_i − r'_i)^2] / [d(d^2 − 1)]

    where ranks are computed on |SHAP| (higher |SHAP| => better rank).

    Parameters
    ----------
    shap_runs:
        List of SHAP arrays or a stacked array of shape (R, n, d).

    Returns
    -------
    float
        Mean ESS over samples and consecutive run pairs, clipped to [0, 1].

    Raises
    ------
    ValueError
        If shapes are invalid or R < 2.
    """
    runs = _as_3d_stack("shap_runs", shap_runs)
    R, n, d = runs.shape
    if R < 2:
        raise ValueError(f"ESS requires at least 2 runs; got R={R}.")
    if d < 2:
        return 1.0

    # Rank features by descending |SHAP|: rank 1 = most important.
    ranks = rankdata(-np.abs(runs), axis=-1, method="average")

    diff = ranks[1:, :, :] - ranks[:-1, :, :]
    sum_sq = np.sum(diff * diff, axis=-1)  # (R-1, n)

    denom = float(d * (d * d - 1))
    ess = 1.0 - (6.0 * sum_sq) / denom
    return _clip01(np.mean(ess))


def attributive_stability_index(shap_runs: Sequence[np.ndarray] | np.ndarray, D_max: float) -> float:
    """Compute Attributive Stability Index (ASI) across consecutive runs.

    ASI = 1 − (|| |phi| − |phi'| ||₁ / D_max)

    Parameters
    ----------
    shap_runs:
        List of SHAP arrays or stacked array of shape (R, n, d).
    D_max:
        Positive normalization constant representing the maximum possible L1
        distance across features.

    Returns
    -------
    float
        Mean ASI over samples and consecutive run pairs, clipped to [0, 1].

    Raises
    ------
    ValueError
        If shapes are invalid or R < 2.
    """
    runs = _as_3d_stack("shap_runs", shap_runs)
    R, n, d = runs.shape
    if R < 2:
        raise ValueError(f"ASI requires at least 2 runs; got R={R}.")
    D = _validate_dmax(D_max)

    abs_runs = np.abs(runs)
    l1 = np.sum(np.abs(abs_runs[1:, :, :] - abs_runs[:-1, :, :]), axis=-1)  # (R-1, n)
    asi = 1.0 - (l1 / D)
    return _clip01(np.mean(asi))


def cross_model_consistency_jaccard(
    shap_models: Sequence[np.ndarray] | np.ndarray,
    k: int = 3,
) -> float:
    """Compute Cross-Model Consistency (Jaccard), averaged over model pairs.

    CMC_J = (2 / M(M−1)) * Σ_{i<j} |TopK_i ∩ TopK_j| / |TopK_i ∪ TopK_j|

    TopK is computed per-sample by descending |SHAP|.

    Parameters
    ----------
    shap_models:
        List of SHAP arrays or stacked array of shape (M, n, d).
    k:
        Number of top features.

    Returns
    -------
    float
        Mean CMC_J over samples and model pairs, clipped to [0, 1].

    Raises
    ------
    ValueError
        If M < 2 or shapes are invalid.
    """
    models = _as_3d_stack("shap_models", shap_models)
    M, n, d = models.shape
    if M < 2:
        raise ValueError(f"CMC_J requires at least 2 models; got M={M}.")
    k = _validate_k(k, d)

    topk = _topk_indices_by_abs(models, k=k)  # (M, n, k)

    # Boolean top-k mask to compute intersections/unions vectorized over samples.
    mask = np.zeros((M, n, d), dtype=bool)
    sample_idx = np.arange(n)[:, None]
    for m in range(M):
        mask[m, sample_idx, topk[m]] = True

    total = 0.0
    count = 0
    for i in range(M):
        for j in range(i + 1, M):
            inter = np.sum(mask[i] & mask[j], axis=-1)  # (n,)
            union = np.sum(mask[i] | mask[j], axis=-1)  # (n,)
            # union should be > 0, but be defensive.
            jac = np.where(union > 0, inter / union, 0.0)
            total += float(np.mean(jac))
            count += 1

    return _clip01(total / count if count else 0.0)


def cross_model_consistency_attribution(
    shap_models: Sequence[np.ndarray] | np.ndarray,
    D_max: float,
) -> float:
    """Compute Cross-Model Consistency (Attribution), averaged over model pairs.

    CMC_A = 1 − (2 / M(M−1)) * Σ_{i<j} ||φ_i − φ_j||₁ / D_max

    Parameters
    ----------
    shap_models:
        List of SHAP arrays or stacked array of shape (M, n, d).
    D_max:
        Positive normalization constant.

    Returns
    -------
    float
        Mean CMC_A over samples and model pairs, clipped to [0, 1].

    Raises
    ------
    ValueError
        If M < 2 or shapes are invalid.
    """
    models = _as_3d_stack("shap_models", shap_models)
    M, n, d = models.shape
    if M < 2:
        raise ValueError(f"CMC_A requires at least 2 models; got M={M}.")
    D = _validate_dmax(D_max)

    total = 0.0
    count = 0
    for i in range(M):
        for j in range(i + 1, M):
            l1 = np.sum(np.abs(models[i] - models[j]), axis=-1)  # (n,)
            total += float(np.mean(l1 / D))
            count += 1

    cmc_a = 1.0 - (total / count if count else 0.0)
    return _clip01(cmc_a)


def model_explanation_agreement_index(
    *,
    phi_x: np.ndarray,
    feature_names: Sequence[str],
    pred_fn: Callable[[Any], Any],
    mask_fn: Callable[[Any, Sequence[str]], Any],
    x: Any,
    k: int = 3,
    eps: float = 1e-12,
) -> float:
    """Compute Model–Explanation Agreement Index (MEAI) for a single instance.

    MEAI = (f(x) − f(x_masked)) / f(x)

    The top-k features are selected by descending |phi_x|.

    Notes
    -----
    - `pred_fn` is expected to return a probability-like scalar for x.
    - For numerical safety when f(x) ≈ 0, we return 0.0 if f(x) <= eps.
    - The result is clipped to [0, 1].

    Parameters
    ----------
    phi_x:
        Explanation vector for the instance x, shape (d,).
    feature_names:
        Feature names in canonical order, length d.
    pred_fn:
        Callable returning model prediction probability for x.
    mask_fn:
        Callable that masks the provided features in x.
    x:
        Single input instance (in the same canonical feature order).
    k:
        Number of top features to mask.
    eps:
        Safety constant for division.

    Returns
    -------
    float
        MEAI clipped to [0, 1].
    """
    phi = np.asarray(phi_x, dtype=np.float64)
    if phi.ndim != 1:
        raise ValueError(f"phi_x must have shape (d,); got {phi.shape}.")
    d = phi.shape[0]
    names = _validate_feature_names(feature_names, d)
    k = _validate_k(k, d)

    def _to_scalar(y: Any) -> float:
        yy = np.asarray(y)
        if yy.size == 0:
            raise ValueError("pred_fn returned an empty array")
        val = float(yy.reshape(-1)[0])
        if not np.isfinite(val):
            raise ValueError(f"pred_fn returned a non-finite value: {val}")
        return val

    fx = _to_scalar(pred_fn(x))
    if fx <= eps:
        return 0.0

    topk_idx = _topk_indices_by_abs(phi, k=k)
    feats = [names[int(i)] for i in np.asarray(topk_idx).reshape(-1)]

    x_masked = mask_fn(x, feats)
    fx_masked = _to_scalar(pred_fn(x_masked))

    meai = (fx - fx_masked) / max(fx, eps)
    return _clip01(meai)


def counterfactual_feasibility_score(
    P: float,
    L: float,
    A: float,
    S: float,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    tol: float = 1e-9,
) -> float:
    """Compute Counterfactual Feasibility Score (CFS).

    CFS = αP + βL + γA + δS, with α + β + γ + δ = 1.

    This function validates that all components and weights are finite and in [0, 1],
    and that weights sum to 1 (within `tol`).

    Parameters
    ----------
    P, L, A, S:
        Component scores, expected in [0, 1].
    alpha, beta, gamma, delta:
        Non-negative weights summing to 1.
    tol:
        Tolerance for weight normalization.

    Returns
    -------
    float
        CFS clipped to [0, 1].
    """
    comps = np.asarray([P, L, A, S], dtype=np.float64)
    w = np.asarray([alpha, beta, gamma, delta], dtype=np.float64)

    if not np.isfinite(comps).all():
        raise ValueError("CFS components must be finite.")
    if not np.isfinite(w).all():
        raise ValueError("CFS weights must be finite.")

    if np.any(comps < 0.0) or np.any(comps > 1.0):
        raise ValueError(f"CFS components must be in [0,1]; got {comps.tolist()}")
    if np.any(w < 0.0):
        raise ValueError(f"CFS weights must be non-negative; got {w.tolist()}")

    s = float(np.sum(w))
    if not np.isfinite(s) or abs(s - 1.0) > tol:
        raise ValueError(f"CFS weight normalization failed: alpha+beta+gamma+delta must be 1 (±{tol}); got {s}.")

    return _clip01(float(np.dot(w, comps)))


def compute_all_metrics(
    *,
    shap_runs: Sequence[np.ndarray] | np.ndarray,
    shap_models: Sequence[np.ndarray] | np.ndarray,
    feature_names: Sequence[str],
    pred_fn: Callable[[Any], Any],
    mask_fn: Callable[[Any, Sequence[str]], Any],
    x: Any,
    k: int = 3,
    D_max: float,
    cfs_components: tuple[float, float, float, float],
    cfs_weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    sample_index: int = 0,
) -> dict[str, float]:
    """Compute ESS, ASI, CMC_J, CMC_A, MEAI, and CFS.

    Parameters
    ----------
    shap_runs:
        SHAP values from repeated runs of the *same* model. Shape (R, n, d).
    shap_models:
        SHAP values from different models. Shape (M, n, d).
    feature_names:
        Names of the d features (canonical order).
    pred_fn, mask_fn, x:
        Callbacks and instance used for MEAI.
    k:
        Top-k features for CMC_J and MEAI.
    D_max:
        Normalization constant for ASI and CMC_A (and any other L1-normalized metric).
    cfs_components:
        Tuple (P, L, A, S) each in [0, 1].
    cfs_weights:
        Tuple (alpha, beta, gamma, delta) non-negative, summing to 1.
    sample_index:
        Which row in the explanation tensors corresponds to `x`.

    Returns
    -------
    dict[str, float]
        {"ESS","ASI","CMC_J","CMC_A","MEAI","CFS"} with floats in [0, 1].

    Raises
    ------
    ValueError
        If shapes are inconsistent or required inputs are invalid.
    """
    runs = _as_3d_stack("shap_runs", shap_runs)
    models = _as_3d_stack("shap_models", shap_models)

    if runs.shape[-1] != models.shape[-1]:
        raise ValueError(f"shap_runs and shap_models must agree on d; got {runs.shape[-1]} vs {models.shape[-1]}")

    R, n_r, d = runs.shape
    M, n_m, d2 = models.shape
    if d != d2:
        raise AssertionError("internal shape mismatch")

    if n_r != n_m:
        raise ValueError(f"shap_runs and shap_models must agree on n_samples; got {n_r} vs {n_m}")

    names = _validate_feature_names(feature_names, d)
    k = _validate_k(k, d)
    D = _validate_dmax(D_max)

    if not isinstance(sample_index, int):
        raise TypeError("sample_index must be int")
    if sample_index < 0 or sample_index >= n_r:
        raise ValueError(f"sample_index out of range for n_samples={n_r}; got {sample_index}")

    ess = explanation_stability_score(runs)
    asi = attributive_stability_index(runs, D_max=D)
    cmc_j = cross_model_consistency_jaccard(models, k=k)
    cmc_a = cross_model_consistency_attribution(models, D_max=D)

    # MEAI: use mean explanation across models for the selected sample.
    phi_x = np.mean(models[:, sample_index, :], axis=0)
    meai = model_explanation_agreement_index(
        phi_x=phi_x,
        feature_names=names,
        pred_fn=pred_fn,
        mask_fn=mask_fn,
        x=x,
        k=k,
    )

    P, L, A, S = cfs_components
    alpha, beta, gamma, delta = cfs_weights
    cfs = counterfactual_feasibility_score(
        P,
        L,
        A,
        S,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        delta=float(delta),
    )

    return {
        "ESS": float(ess),
        "ASI": float(asi),
        "CMC_J": float(cmc_j),
        "CMC_A": float(cmc_a),
        "MEAI": float(meai),
        "CFS": float(cfs),
    }


def _as_2d_weights(name: str, lime_weight_runs: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Normalize LIME weight runs into shape (R, d) float64 array."""
    x = np.asarray(lime_weight_runs, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (R, d); got {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError(f"{name} requires at least 2 runs; got R={x.shape[0]}.")
    if not np.isfinite(x).all():
        bad = np.argwhere(~np.isfinite(x))
        raise ValueError(
            f"{name} contains non-finite values; first bad index: {bad[0].tolist() if bad.size else 'unknown'}"
        )
    return x


def lime_feature_selection_stability(lime_weight_runs: Sequence[np.ndarray] | np.ndarray, *, k: int = 3) -> float:
    """LIME Feature Selection Stability (LFSS), equivalent to Top-K overlap.

    Measures whether LIME selects the same important features across runs.

    LFSS = (1/(R-1)) * Σ_{i=1..R-1} |TopK_i ∩ TopK_{i+1}| / |TopK_i ∪ TopK_{i+1}|

    TopK is computed by descending |w| per run.

    Parameters
    ----------
    lime_weight_runs:
        Weight matrix of shape (R, d), where each row is one LIME run and columns
        correspond to a fixed feature space.
    k:
        Size of Top-K set.

    Returns
    -------
    float
        Mean consecutive-run Jaccard similarity, clipped to [0, 1].
    """
    W = _as_2d_weights("lime_weight_runs", lime_weight_runs)
    R, d = W.shape
    if d < 1:
        return 0.0
    k = _validate_k(int(k), d)

    topk = _topk_indices_by_abs(W, k=k)  # (R, k)

    vals: list[float] = []
    for i in range(R - 1):
        a = set(int(x) for x in np.asarray(topk[i]).reshape(-1))
        b = set(int(x) for x in np.asarray(topk[i + 1]).reshape(-1))
        union = len(a | b)
        jac = (len(a & b) / union) if union else 0.0
        vals.append(float(jac))
    return _clip01(float(np.mean(vals)) if vals else 0.0)


def _pearson_corr(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("Pearson correlation requires equal-length vectors")
    if a.size < 2:
        return 1.0
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom <= eps:
        # If both are (near) constant, treat as perfectly stable iff identical.
        return 1.0 if float(np.max(np.abs(a - b))) <= 0.0 else 0.0
    return float(np.sum(a0 * b0) / denom)


def lime_rank_stability(lime_weight_runs: Sequence[np.ndarray] | np.ndarray) -> float:
    """LIME Rank Stability (LRS): Spearman-based stability over consecutive runs.

    Computes the mean Spearman rank correlation between consecutive runs, where ranks
    are computed on |w| (higher |w| = more important).

    Parameters
    ----------
    lime_weight_runs:
        Weight matrix of shape (R, d).

    Returns
    -------
    float
        Mean consecutive-run Spearman correlation mapped to [0, 1] via clipping.
        (Negative correlations are clipped to 0.)
    """
    W = _as_2d_weights("lime_weight_runs", lime_weight_runs)
    R, d = W.shape
    if d < 2:
        return 1.0

    ranks = rankdata(-np.abs(W), axis=1, method="average")  # (R, d)
    vals: list[float] = []
    for i in range(R - 1):
        rho = _pearson_corr(ranks[i], ranks[i + 1])
        vals.append(float(rho))
    return _clip01(float(np.mean(vals)) if vals else 0.0)


def lime_sign_consistency(lime_weight_runs: Sequence[np.ndarray] | np.ndarray, *, k: int = 3) -> float:
    """LIME Sign Consistency (LSC) over consecutive runs.

    Measures whether the *direction* (positive/negative) of LIME weights is stable
    for the Top-K features of each run.

    For consecutive runs i and i+1:
        LSC_i = (1/K) * Σ_{j in TopK_i} I[sign(w_{i,j}) == sign(w_{i+1,j})]

    Parameters
    ----------
    lime_weight_runs:
        Weight matrix of shape (R, d).
    k:
        Number of top features (by |w|) from run i to check.

    Returns
    -------
    float
        Mean LSC over consecutive run pairs, clipped to [0, 1].
    """
    W = _as_2d_weights("lime_weight_runs", lime_weight_runs)
    R, d = W.shape
    if d < 1:
        return 0.0
    k = _validate_k(int(k), d)

    topk = _topk_indices_by_abs(W, k=k)  # (R, k)
    sgn = np.sign(W)

    vals: list[float] = []
    for i in range(R - 1):
        idx = np.asarray(topk[i], dtype=int).reshape(-1)
        agree = np.mean((sgn[i, idx] == sgn[i + 1, idx]).astype(np.float64))
        vals.append(float(agree))
    return _clip01(float(np.mean(vals)) if vals else 0.0)


def compute_lime_stability_metrics(lime_weight_runs: Sequence[np.ndarray] | np.ndarray, *, k: int = 3) -> dict[str, float]:
    """Convenience wrapper to compute LFSS, LRS, and LSC for LIME runs."""
    return {
        "LFSS": lime_feature_selection_stability(lime_weight_runs, k=k),
        "LRS": lime_rank_stability(lime_weight_runs),
        "LSC": lime_sign_consistency(lime_weight_runs, k=k),
    }

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from typing import Literal, cast

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Allow running as: python src/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREPROC_DIR, RESULTS_DIR, SPLIT_DIR  # noqa: E402
from src.cleaning import apply_clinical_bounds, model_based_impute  # noqa: E402
from src.canonical import (  # noqa: E402
    CANONICAL_FEATURES,
    CanonicalPreprocessor,
    assert_canonical_schema,
    forbid_onehot_residuals,
)
from src.utils import ensure_dir  # noqa: E402


def _synth_fold_gcopula_canonical(
    X_train_canon: pd.DataFrame,
    y_train: np.ndarray,
    multiplier: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate fold-local synthetic samples in canonical feature space.

    This avoids leakage versus using a synthetic dataset generated from the full
    training split (which would indirectly encode validation-fold information).
    """

    # Local import to avoid heavy deps during baseline runs.
    from src.synthesizer import GaussianCopulaGenerator, postprocess_synth  # noqa: WPS433

    assert_canonical_schema(X_train_canon)
    forbid_onehot_residuals(list(X_train_canon.columns))

    n_total = int(len(X_train_canon) * int(multiplier))
    if n_total <= 0:
        raise ValueError('multiplier must yield at least 1 synthetic sample')

    rng = np.random.default_rng(int(seed))

    X0 = X_train_canon[y_train == 0].reset_index(drop=True)
    X1 = X_train_canon[y_train == 1].reset_index(drop=True)
    if X0.empty or X1.empty:
        raise ValueError('Cannot synthesize per-class: one class has zero samples in this fold.')

    p1 = float(len(X1) / len(y_train))
    n1 = int(round(n_total * p1))
    n0 = int(n_total - n1)
    if n0 <= 0 or n1 <= 0:
        # Keep both classes represented
        n0 = max(1, n0)
        n1 = max(1, n_total - n0)

    cop0 = GaussianCopulaGenerator()
    cop0.fit(X0)
    Xs0 = cop0.sample(n0, seed=int(seed))

    cop1 = GaussianCopulaGenerator()
    cop1.fit(X1)
    Xs1 = cop1.sample(n1, seed=int(seed) + 1)

    Xs = pd.concat([Xs0, Xs1], ignore_index=True)
    ys = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    perm = rng.permutation(len(Xs))
    Xs = Xs.iloc[perm].reset_index(drop=True)
    ys = ys[perm]

    # Postprocess for bounds + binary flags, then re-apply canonical preprocessor to enforce dtypes/order
    Xs = postprocess_synth(Xs, X_train_canon)
    pre = CanonicalPreprocessor().fit(X_train_canon)
    Xs = pre.transform(Xs)

    assert_canonical_schema(Xs)
    forbid_onehot_residuals(list(Xs.columns))
    return Xs, ys


def _row_hashes(df: pd.DataFrame) -> set[int]:
    """Fast per-row hashes for overlap detection."""
    # hash_pandas_object returns uint64; convert to Python int for set ops.
    h = pd.util.hash_pandas_object(df, index=False)
    return set(int(x) for x in h.to_numpy())


def _assert_no_train_test_overlap(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    inter = _row_hashes(X_train) & _row_hashes(X_test)
    if inter:
        raise ValueError(f'Data leakage: found {len(inter)} duplicate rows between train and test.')


def _drop_leakage_columns(X: pd.DataFrame) -> pd.DataFrame:
    leak_cols = [c for c in ['target', 'classification'] if c in X.columns]
    return X.drop(columns=leak_cols) if leak_cols else X


def _load_split_preproc() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    preproc_dir = Path(PREPROC_DIR)
    X_train = pd.read_csv(preproc_dir / 'X_train_preproc.csv')
    X_test = pd.read_csv(preproc_dir / 'X_test_preproc.csv')

    split_dir = Path('data/processed/splits')
    y_train = pd.read_csv(split_dir / 'y_train.csv')['target'].to_numpy()
    y_test = pd.read_csv(split_dir / 'y_test.csv')['target'].to_numpy()

    forbid_onehot_residuals(list(X_train.columns))
    forbid_onehot_residuals(list(X_test.columns))
    assert_canonical_schema(X_train)
    assert_canonical_schema(X_test)
    return X_train, X_test, y_train, y_test


def _load_split_raw_train() -> tuple[pd.DataFrame, np.ndarray]:
    split_dir = Path(SPLIT_DIR)
    X_train = pd.read_csv(split_dir / 'X_train_raw.csv')
    y_train = pd.read_csv(split_dir / 'y_train.csv')['target'].to_numpy()
    return X_train, y_train


def _cv_evaluate_baseline_raw(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
    *,
    variant: str = 'baseline',
    synth_backend: str = 'gcopula',
) -> pd.DataFrame:
    """Strict CV in canonical feature space with fold-local preprocessing (no leakage).

    For augmented variants (1x/3x), this performs fold-local synthesis using
    the specified backend so that the validation fold is not used to generate
    synthetic samples.
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _fold_scores(estimator_factory, model_name: str) -> dict[str, object]:
        fold_metrics: list[dict[str, float]] = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
            X_tr = X_raw.iloc[train_idx].copy()
            y_tr = y[train_idx]
            X_va = X_raw.iloc[val_idx].copy()
            y_va = y[val_idx]

            # Fold-local regression imputation learned from fold-train only (paper-aligned)
            X_tr_learn = X_tr.copy()
            X_tr_imp = model_based_impute(X_tr_learn, X_tr)
            X_va_imp = model_based_impute(X_tr_learn, X_va)

            # Bounds after imputation
            X_tr_imp = apply_clinical_bounds(X_tr_imp)
            X_va_imp = apply_clinical_bounds(X_va_imp)

            # Fold-local canonical preprocessing (fit on fold-train only)
            preproc = CanonicalPreprocessor().fit(X_tr_imp)
            X_tr_c = preproc.transform(X_tr_imp)
            X_va_c = preproc.transform(X_va_imp)

            # Hard safety checks required by spec
            assert_canonical_schema(X_tr_c)
            assert_canonical_schema(X_va_c)

            # Optional: fold-local augmentation in canonical space
            if variant in ('1x', '3x'):
                mult = 1 if variant == '1x' else 3
                if synth_backend != 'gcopula':
                    raise ValueError(
                        "Only 'gcopula' fold-local synthesis is supported for CV. "
                        f"Got synth_backend={synth_backend!r}."
                    )
                fold_seed = int(random_state) * 10_000 + int(fold_idx)
                Xs, ys = _synth_fold_gcopula_canonical(X_tr_c, y_tr, multiplier=mult, seed=fold_seed)
                X_tr_fit = pd.concat([X_tr_c, Xs], ignore_index=True)
                y_tr_fit = np.concatenate([y_tr, ys])
                assert_canonical_schema(X_tr_fit)
            else:
                X_tr_fit = X_tr_c
                y_tr_fit = y_tr

            base = estimator_factory()
            base.fit(X_tr_fit, y_tr_fit)
            y_prob = base.predict_proba(X_va_c)[:, 1]
            fold_metrics.append(_metrics(y_va, y_prob))

        df = pd.DataFrame(fold_metrics)
        out: dict[str, object] = {'model': model_name, 'cv_folds': float(n_splits)}
        for c in ['roc_auc', 'precision', 'recall', 'f1']:
            out[f'{c}_mean'] = float(df[c].mean())
            out[f'{c}_std'] = float(df[c].std(ddof=0))
        return out

    def _lr():
        return LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=0)

    def _rf():
        return RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)

    rows = [
        _fold_scores(_lr, 'lr'),
        _fold_scores(_rf, 'rf'),
    ]

    try:
        from xgboost import XGBClassifier

        def _xgb():
            return XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=0,
                eval_metric='logloss',
            )

        rows.append(_fold_scores(_xgb, 'xgb'))
    except Exception:
        pass

    return pd.DataFrame(rows)


def _load_synth(multiplier: int) -> pd.DataFrame:
    candidates = [
        Path('data/synthetic') / f'X_synth_{multiplier}x_preproc.csv',
        Path('data/synthetic') / f'X_synth_{multiplier}x_gcopula_preproc.csv',
    ]
    for path in candidates:
        if path.exists():
            X = pd.read_csv(path)
            forbid_onehot_residuals(list(X.columns))
            assert_canonical_schema(X)
            return X
    raise FileNotFoundError(
        f"Missing synthetic preprocessed file for {multiplier}x. Tried: "
        + ', '.join(str(p) for p in candidates)
        + f". Run src/synthesizer.py --multiplier {multiplier} first."
    )


def _load_y_synth(multiplier: int) -> np.ndarray:
    candidates = [
        Path('data/synthetic') / f'y_synth_{multiplier}x.csv',
        Path('data/synthetic') / f'y_synth_{multiplier}x_gcopula.csv',
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)['target'].to_numpy()
    raise FileNotFoundError(
        f"Missing synthetic y file for {multiplier}x. Tried: " + ', '.join(str(p) for p in candidates)
    )


def _load_synth_with_backend(multiplier: int, backend: str) -> tuple[pd.DataFrame, np.ndarray]:
    # Backward/forward compatible loader. Canonical schema is enforced.
    x_candidates = [
        Path('data/synthetic') / f'X_synth_{multiplier}x_{backend}_preproc.csv',
        Path('data/synthetic') / f'X_synth_{multiplier}x_gcopula_preproc.csv',
        Path('data/synthetic') / f'X_synth_{multiplier}x_preproc.csv',
    ]
    y_candidates = [
        Path('data/synthetic') / f'y_synth_{multiplier}x_{backend}.csv',
        Path('data/synthetic') / f'y_synth_{multiplier}x_gcopula.csv',
        Path('data/synthetic') / f'y_synth_{multiplier}x.csv',
    ]

    X_path = next((p for p in x_candidates if p.exists()), None)
    y_path = next((p for p in y_candidates if p.exists()), None)
    if X_path is None or y_path is None:
        raise FileNotFoundError(
            f"Missing synthetic files for {multiplier}x backend={backend}. Tried X: "
            + ', '.join(str(p) for p in x_candidates)
            + ' | y: '
            + ', '.join(str(p) for p in y_candidates)
        )

    X = pd.read_csv(X_path)
    forbid_onehot_residuals(list(X.columns))
    assert_canonical_schema(X)
    y = pd.read_csv(y_path)['target'].to_numpy()
    return X, y


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Train LR/RF/XGB on baseline or augmented preprocessed data.')
    parser.add_argument('--variant', type=str, default='baseline', choices=['baseline', '1x', '3x'])
    parser.add_argument('--synth-backend', type=str, default='gcopula', choices=['gcopula', 'native', 'sdv_gcopula', 'sdv_ctgan'])
    parser.add_argument('--cv-folds', type=int, default=5)
    args = parser.parse_args()

    if args.variant != 'baseline':
        print('WARNING: Synthetic data used ONLY for robustness testing, not explanation attribution.')

    # Strict CV (all variants). Augmented variants use fold-local synthesis to prevent leakage.
    if args.cv_folds and args.cv_folds >= 2:
        X_raw_train, y_raw_train = _load_split_raw_train()
        cv_df = _cv_evaluate_baseline_raw(
            X_raw_train,
            y_raw_train,
            n_splits=args.cv_folds,
            random_state=0,
            variant=args.variant,
            synth_backend=args.synth_backend,
        )
        cv_df.insert(1, 'variant', args.variant)
        cv_df.insert(2, 'synth_backend', args.synth_backend)
        out_dir = ensure_dir(RESULTS_DIR)
        cv_suffix = args.variant if args.variant == 'baseline' else f"{args.variant}_{args.synth_backend}"
        out_path = out_dir / f'cv_metrics_{cv_suffix}.csv'
        cv_df.to_csv(out_path, index=False)
        print('Saved CV metrics:', out_path)
        for _, r in cv_df.iterrows():
            print(
                f"CV {int(args.cv_folds)}-fold {r['model'].upper()} ({cv_suffix}): "
                f"AUC {r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}, "
                f"F1 {r['f1_mean']:.4f}±{r['f1_std']:.4f}, "
                f"Precision {r['precision_mean']:.4f}±{r['precision_std']:.4f}, "
                f"Recall {r['recall_mean']:.4f}±{r['recall_std']:.4f}"
            )

    X_train, X_test, y_train, y_test = _load_split_preproc()

    # Defensive leakage guards
    X_train = _drop_leakage_columns(X_train)
    X_test = _drop_leakage_columns(X_test)
    _assert_no_train_test_overlap(X_train, X_test)

    if args.variant == 'baseline':
        X_train_use = X_train
        y_train_use = y_train
    else:
        mult = 1 if args.variant == '1x' else 3
        X_synth, y_synth = _load_synth_with_backend(mult, args.synth_backend)
        X_train_use = pd.concat([X_train, X_synth], ignore_index=True)
        y_train_use = np.concatenate([y_train, y_synth])

    # Required safety check: strict schema and strict order before training
    forbid_onehot_residuals(list(X_train_use.columns))
    forbid_onehot_residuals(list(X_test.columns))
    assert_canonical_schema(X_train_use)
    assert_canonical_schema(X_test)

    # Explicit feature order enforcement (defensive)
    X_train_use = X_train_use[CANONICAL_FEATURES].copy()
    X_test = X_test[CANONICAL_FEATURES].copy()
    assert_canonical_schema(X_train_use)
    assert_canonical_schema(X_test)

    results = []

    # Logistic Regression
    lr = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=0)
    lr.fit(X_train_use, y_train_use)
    y_prob = lr.predict_proba(X_test)[:, 1]
    results.append({'model': 'lr', 'variant': args.variant, **_metrics(y_test, y_prob)})
    ensure_dir('models')
    joblib.dump(lr, Path('models') / f'lr_{args.variant}.joblib')

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)
    rf.fit(X_train_use, y_train_use)
    y_prob = rf.predict_proba(X_test)[:, 1]
    results.append({'model': 'rf', 'variant': args.variant, **_metrics(y_test, y_prob)})
    joblib.dump(rf, Path('models') / f'rf_{args.variant}.joblib')

    # XGBoost (optional)
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=0,
            eval_metric='logloss',
        )
        xgb.fit(X_train_use, y_train_use)
        y_prob = xgb.predict_proba(X_test)[:, 1]
        results.append({'model': 'xgb', 'variant': args.variant, **_metrics(y_test, y_prob)})
        joblib.dump(xgb, Path('models') / f'xgb_{args.variant}.joblib')
    except Exception:
        pass

    out_dir = ensure_dir(RESULTS_DIR)
    suffix = args.variant if args.variant == 'baseline' else f"{args.variant}_{args.synth_backend}"
    out_path = out_dir / f'metrics_{suffix}.csv'
    pd.DataFrame(results).to_csv(out_path, index=False)
    print('Saved metrics:', out_path)


if __name__ == '__main__':
    main()

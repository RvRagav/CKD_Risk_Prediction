from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from typing import Literal, cast

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow running as: python src/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CAT_COLS, CONT_COLS, ORD_COLS, PREPROC_DIR, RESULTS_DIR, SPLIT_DIR  # noqa: E402
from src.cleaning import apply_clinical_bounds, model_based_impute  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


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
    return X_train, X_test, y_train, y_test


def _load_split_raw_train() -> tuple[pd.DataFrame, np.ndarray]:
    split_dir = Path(SPLIT_DIR)
    X_train = pd.read_csv(split_dir / 'X_train_raw.csv')
    y_train = pd.read_csv(split_dir / 'y_train.csv')['target'].to_numpy()
    return X_train, y_train


def _build_preprocessor_from_X(X: pd.DataFrame) -> ColumnTransformer:
    cont_cols = [c for c in CONT_COLS if c in X.columns]
    ord_cols = [c for c in ORD_COLS if c in X.columns]
    cat_cols = [c for c in CAT_COLS if c in X.columns]

    num_cols = cont_cols + ord_cols
    cont_pipe = Pipeline([
        ('scaler', StandardScaler()),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe),
    ])

    return ColumnTransformer(
        transformers=[
            ('num', cont_pipe, num_cols),
            ('cat', cat_pipe, cat_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def _cv_evaluate_baseline_raw(
    X_raw: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Strict CV on raw features with fold-local imputation+preprocessing (no leakage)."""

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _fold_scores(estimator_factory, model_name: str) -> dict[str, object]:
        fold_metrics: list[dict[str, float]] = []
        for train_idx, val_idx in skf.split(X_raw, y):
            X_tr = X_raw.iloc[train_idx].copy()
            y_tr = y[train_idx]
            X_va = X_raw.iloc[val_idx].copy()
            y_va = y[val_idx]

            # Fold-local imputation learned from X_tr only (no leakage)
            X_tr_learn = X_tr.copy()
            X_tr = model_based_impute(X_tr_learn, X_tr)
            X_va = model_based_impute(X_tr_learn, X_va)
            X_tr = apply_clinical_bounds(X_tr)
            X_va = apply_clinical_bounds(X_va)

            preprocessor = _build_preprocessor_from_X(X_tr)
            base = estimator_factory()

            pipe = Pipeline([
                ('preprocess', preprocessor),
                ('model', base),
            ])

            pipe.fit(X_tr, y_tr)
            y_prob = pipe.predict_proba(X_va)[:, 1]
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
                use_label_encoder=False,
            )

        rows.append(_fold_scores(_xgb, 'xgb'))
    except Exception:
        pass

    return pd.DataFrame(rows)


def _load_synth(multiplier: int) -> pd.DataFrame:
    path = Path('data/synthetic') / f'X_synth_{multiplier}x_preproc.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/synthesizer.py --multiplier {multiplier} first.")
    return pd.read_csv(path)


def _load_y_synth(multiplier: int) -> np.ndarray:
    path = Path('data/synthetic') / f'y_synth_{multiplier}x.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/synthesizer.py --multiplier {multiplier} first.")
    return pd.read_csv(path)['target'].to_numpy()


def _load_synth_with_backend(multiplier: int, backend: str) -> tuple[pd.DataFrame, np.ndarray]:
    X_path = Path('data/synthetic') / f'X_synth_{multiplier}x_{backend}_preproc.csv'
    y_path = Path('data/synthetic') / f'y_synth_{multiplier}x_{backend}.csv'
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {X_path} or {y_path}. Run: python src/synthesizer.py --backend {backend} --multiplier {multiplier}"
        )
    return pd.read_csv(X_path), pd.read_csv(y_path)['target'].to_numpy()


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
    parser.add_argument('--synth-backend', type=str, default='sdv_gcopula', choices=['native', 'sdv_gcopula', 'sdv_ctgan'])
    parser.add_argument('--cv-folds', type=int, default=5)
    args = parser.parse_args()

    if args.variant != 'baseline':
        print('WARNING: Synthetic data used ONLY for robustness testing, not explanation attribution.')

    # Strict CV (baseline only) to avoid overly optimistic single split performance.
    if args.variant == 'baseline' and args.cv_folds and args.cv_folds >= 2:
        X_raw_train, y_raw_train = _load_split_raw_train()
        cv_df = _cv_evaluate_baseline_raw(X_raw_train, y_raw_train, n_splits=args.cv_folds)
        cv_df.insert(1, 'variant', args.variant)
        out_dir = ensure_dir(RESULTS_DIR)
        out_path = out_dir / 'cv_metrics_baseline.csv'
        cv_df.to_csv(out_path, index=False)
        print('Saved CV metrics:', out_path)
        # Print mean ± std compactly
        for _, r in cv_df.iterrows():
            print(
                f"CV {int(args.cv_folds)}-fold {r['model'].upper()}: "
                f"AUC {r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f}, "
                f"F1 {r['f1_mean']:.4f}±{r['f1_std']:.4f}, "
                f"Precision {r['precision_mean']:.4f}±{r['precision_std']:.4f}, "
                f"Recall {r['recall_mean']:.4f}±{r['recall_std']:.4f}"
            )
    elif args.variant != 'baseline':
        print('Note: Strict CV is computed for baseline only (to prevent leakage with synthetic augmentation).')

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
            use_label_encoder=False,
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

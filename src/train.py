from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Allow running as: python src/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREPROC_DIR, RESULTS_DIR  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def _load_split_preproc() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    preproc_dir = Path(PREPROC_DIR)
    X_train = pd.read_csv(preproc_dir / 'X_train_preproc.csv')
    X_test = pd.read_csv(preproc_dir / 'X_test_preproc.csv')

    split_dir = Path('data/processed/splits')
    y_train = pd.read_csv(split_dir / 'y_train.csv')['target'].to_numpy()
    y_test = pd.read_csv(split_dir / 'y_test.csv')['target'].to_numpy()
    return X_train, X_test, y_train, y_test


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
        'brier': float(brier_score_loss(y_true, y_prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Train LR/RF/XGB on baseline or augmented preprocessed data.')
    parser.add_argument('--variant', type=str, default='baseline', choices=['baseline', '1x', '3x'])
    parser.add_argument('--synth-backend', type=str, default='sdv_gcopula', choices=['native', 'sdv_gcopula', 'sdv_ctgan'])
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = _load_split_preproc()

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

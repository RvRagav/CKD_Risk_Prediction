from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running as: python src/explain.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREPROC_DIR, RESULTS_DIR  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute SHAP values for a trained model on X_test_preproc.')
    parser.add_argument('--model', type=str, required=True, choices=['lr', 'rf', 'xgb'])
    parser.add_argument('--variant', type=str, default='baseline', choices=['baseline', '1x', '3x'])
    args = parser.parse_args()

    X_test = pd.read_csv(Path(PREPROC_DIR) / 'X_test_preproc.csv')

    model_path = Path('models') / f'{args.model}_{args.variant}.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run src/train.py first.")

    model = joblib.load(model_path)

    try:
        import shap
    except ImportError as e:
        raise SystemExit('Install shap to run explain.py (pip install shap).') from e

    if args.model in ('rf', 'xgb'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # LogisticRegression
        try:
            explainer = shap.LinearExplainer(model, X_test, feature_perturbation='interventional')
            shap_values = explainer.shap_values(X_test)
        except Exception:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
            shap_values = explainer.shap_values(X_test)

    # Normalize output to a 2D array for binary classification.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        # (n_classes, n_samples, n_features)
        shap_arr = shap_arr[1]

    out_dir = ensure_dir(RESULTS_DIR)
    out_path = out_dir / f'shap_{args.model}_{args.variant}.npy'
    np.save(out_path, shap_arr)
    print('Saved SHAP:', out_path)


if __name__ == '__main__':
    main()

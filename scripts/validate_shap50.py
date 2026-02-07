import joblib
import numpy as np
import pandas as pd
import shap
import xgboost


def main() -> None:
    print(f"shap={shap.__version__}")
    print(f"xgboost={xgboost.__version__}")

    X = pd.read_csv("data/processed/preprocessed/X_test_preproc.csv")
    model = joblib.load("models/xgb.joblib")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[:20, :])

    arr = np.asarray(shap_values)
    print(f"shap_values ndarray shape: {arr.shape}")


if __name__ == "__main__":
    main()

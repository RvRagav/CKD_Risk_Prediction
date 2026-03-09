"""Re-save all .joblib model files with the current sklearn version.

Run once to eliminate InconsistentVersionWarning after a sklearn upgrade.
Usage:
    python scripts/resave_models.py
"""
import pathlib
import warnings

import joblib

warnings.filterwarnings("ignore")

models_dir = pathlib.Path(__file__).parent.parent / "models"
for p in sorted(models_dir.glob("*.joblib")):
    obj = joblib.load(p)
    joblib.dump(obj, p)
    print(f"re-saved: {p.name}")

print("Done — all models re-saved with current sklearn version.")

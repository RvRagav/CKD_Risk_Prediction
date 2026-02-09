# CKD Risk Prediction — Module Summary

This document summarizes the project implementation in three modules:

1. **Data preparation** (cleaning + preprocessing + synthetic data generation)
2. **Model training and evaluation**
3. **Model explanation and explanation evaluation**

The code is implemented primarily in the Python modules under [src/](src/) and orchestrated via notebooks under [notebooks/](notebooks/).

---

## 0) Shared design decisions (apply to all modules)

### Canonical feature space (6 features)
The project deliberately uses a fixed, “paper-aligned” canonical feature space:

- Features: `hemo`, `sc`, `al`, `htn`, `age`, `dm`
- Enforced in: [src/canonical.py](src/canonical.py)

Why this matters:

- All training, augmentation, and explanation steps operate on the same strict schema.
- The code actively *rejects* accidental one-hot artifacts (for example `dm_0`, `htn_1`) to keep the feature space stable.

### Reproducibility + leakage safety
Several guardrails are used throughout the pipeline:

- Train/test split is stratified and fixed via constants in [src/config.py](src/config.py).
- Regression-based imputation is trained on *train only* and then applied to other splits.
- Cross-validation for augmented variants uses **fold-local synthesis**, so the validation fold is not used to generate synthetic samples.

### Key folders and artifacts
- Raw dataset: [dataset/](dataset/)
- Processed artifacts: [data/processed/](data/processed/)
- Synthetic artifacts: [data/synthetic/](data/synthetic/)
- Trained models: [models/](models/)
- Metrics, plots, reports: [results/](results/)

---

## 1) Data preparation (preprocessing + synthetic data generation)

### 1.1 Cleaning + train/test splitting
**Primary implementation:** [src/cleaning.py](src/cleaning.py)

What it does:

1. Loads the raw dataset configured in [src/config.py](src/config.py) (`RAW_DATA_PATH`).
2. Tidies string columns (trims whitespace/tabs, converts `?` to missing).
3. Coerces numeric columns to numeric types.
4. Normalizes categorical encodings into numeric flags for key columns (yes/no, normal/abnormal, present/notpresent).
5. Applies basic physiologic clipping bounds for a subset of numeric columns (helps avoid unstable tails).
6. Maps the target label from `classification` → `target` (binary), with a safe mapping that avoids substring mistakes (e.g., `ckd` inside `notckd`).
7. Creates a stratified train/test split and writes split artifacts.
8. Produces a *regression-based imputed* version of the raw splits (trained on train only), then constructs the **canonical preprocessed** matrices.

Key outputs:

- Cleaned dataset:
  - [data/processed/kidney_disease_cleaned.csv](data/processed/kidney_disease_cleaned.csv)
- Raw splits:
  - [data/processed/splits/X_train_raw.csv](data/processed/splits/X_train_raw.csv)
  - [data/processed/splits/X_test_raw.csv](data/processed/splits/X_test_raw.csv)
  - [data/processed/splits/y_train.csv](data/processed/splits/y_train.csv)
  - [data/processed/splits/y_test.csv](data/processed/splits/y_test.csv)
- Imputed raw splits (used as the basis for canonical features):
  - [data/processed/splits/X_train_imputed_raw.csv](data/processed/splits/X_train_imputed_raw.csv)
  - [data/processed/splits/X_test_imputed_raw.csv](data/processed/splits/X_test_imputed_raw.csv)
- Canonical preprocessed matrices (6 columns, strict order):
  - [data/processed/preprocessed/X_train_preproc.csv](data/processed/preprocessed/X_train_preproc.csv)
  - [data/processed/preprocessed/X_test_preproc.csv](data/processed/preprocessed/X_test_preproc.csv)
  - [data/processed/preprocessed/feature_names.json](data/processed/preprocessed/feature_names.json)
- Saved canonical preprocessor:
  - [models/preprocessor.pkl](models/preprocessor.pkl)

How to run (CLI):

- `python src/cleaning.py`

Important notes:

- The canonical preprocessor is implemented in [src/canonical.py](src/canonical.py) as `CanonicalPreprocessor`.
- The regression-based imputation in [src/cleaning.py](src/cleaning.py) trains per-feature regressors on the training data only (to reduce leakage risk).

### 1.2 Synthetic data generation (Gaussian Copula)
**Primary implementation:** [src/synthesizer.py](src/synthesizer.py)

What it does:

1. Loads the imputed training split (expects [data/processed/splits/X_train_imputed_raw.csv](data/processed/splits/X_train_imputed_raw.csv)).
2. Restricts to canonical features only.
3. Fits a **class-conditional Gaussian copula** (one copula for class 0, one for class 1), then samples synthetic records.
4. Post-processes synthetic samples:
   - clips to configured clinical bounds
   - forces binary flags (`htn`, `dm`) into {0,1}
   - snaps ordinal-like features (where applicable) to observed supports
5. Re-applies the saved `CanonicalPreprocessor` to guarantee canonical schema and dtypes.
6. Writes synthetic features/labels to disk.
7. Computes quality control (QC) metrics and writes QC reports.

Key outputs:

- Synthetic preprocessed features (examples already in repo):
  - [data/synthetic/X_synth_1x_gcopula_preproc.csv](data/synthetic/X_synth_1x_gcopula_preproc.csv)
  - [data/synthetic/X_synth_3x_gcopula_preproc.csv](data/synthetic/X_synth_3x_gcopula_preproc.csv)
- Synthetic labels:
  - [data/synthetic/y_synth_1x_gcopula.csv](data/synthetic/y_synth_1x_gcopula.csv)
  - [data/synthetic/y_synth_3x_gcopula.csv](data/synthetic/y_synth_3x_gcopula.csv)
- QC summary and KS reports (naming includes multiplier and seed):
  - Example patterns: [results/](results/) → `qc_report_*.csv`, `ks_*.csv`

How to run (CLI):

- `python src/synthesizer.py --multiplier 1 --seed 42`
- `python src/synthesizer.py --multiplier 3 --seed 42`

Important notes:

- The synthesizer operates in the canonical space; this is intentional so that augmentation matches the model’s training features exactly.
- QC metrics implemented include KS similarity per feature, correlation-difference summary, and a “real vs synthetic” discriminator AUC.

---

## 2) Model training and evaluation

### 2.1 Training variants
**Primary implementation:** [src/train.py](src/train.py)

Supported variants:

- `baseline`: train only on real canonical preprocessed training data.
- `1x`: augment real training set with +1× synthetic samples.
- `3x`: augment real training set with +3× synthetic samples.

Supported models:

- Logistic Regression (always)
- Random Forest (always)
- XGBoost (optional; only if `xgboost` is installed)

### 2.2 Evaluation strategy
This module uses two complementary evaluation modes:

1. **Strict cross-validation (CV) on the raw training split**
   - Implemented in `_cv_evaluate_baseline_raw` in [src/train.py](src/train.py)
   - For each fold:
     - regression-based imputation is trained on the fold-train portion only
     - canonical preprocessing is fit on fold-train only
     - for augmented variants, **fold-local synthesis** is generated only from fold-train canonical features
   - Writes CV summaries to [results/](results/) as `cv_metrics_*.csv`

2. **Holdout test evaluation**
   - Loads canonical [data/processed/preprocessed/X_train_preproc.csv](data/processed/preprocessed/X_train_preproc.csv) and [data/processed/preprocessed/X_test_preproc.csv](data/processed/preprocessed/X_test_preproc.csv)
   - Optionally concatenates synthetic data into the training set for augmented variants
   - Trains models and evaluates on the test split
   - Saves metrics to [results/](results/) as `metrics_*.csv`
   - Saves trained models to [models/](models/)

Key outputs:

- Models (examples already present):
  - [models/lr_baseline.joblib](models/lr_baseline.joblib)
  - [models/rf_baseline.joblib](models/rf_baseline.joblib)
  - [models/xgb_baseline.joblib](models/xgb_baseline.joblib)
  - Augmented equivalents: `*_1x.joblib`, `*_3x.joblib`
- Metrics:
  - CV metrics: [results/cv_metrics_baseline.csv](results/cv_metrics_baseline.csv) and related files
  - Holdout metrics: [results/metrics_baseline.csv](results/metrics_baseline.csv) and related files

How to run (CLI):

- Baseline:
  - `python src/train.py --variant baseline --cv-folds 5`
- Augmented (uses fold-local synthesis for CV):
  - `python src/train.py --variant 1x --synth-backend gcopula --cv-folds 5`
  - `python src/train.py --variant 3x --synth-backend gcopula --cv-folds 5`

Important notes:

- There is an explicit defensive check for train/test overlap (row hash intersection) before final training.
- For CV, augmented variants do not rely on pre-generated synthetic CSVs; they synthesize per-fold to avoid leakage.
- For the final holdout training, augmented variants can load synthetic files from [data/synthetic/](data/synthetic/).

### 2.3 Visualization of results
**Primary implementation:** [src/visualize.py](src/visualize.py)

What it does:

- Discovers and aggregates `metrics_*.csv` in [results/](results/)
- Generates plots into [results/visualizations/](results/visualizations/)
  - model comparison plot
  - performance summary (heatmap + bar charts)
  - synthetic quality summary (QC + KS)
  - feature distribution comparisons (real vs synthetic)

How to run (CLI):

- `python src/visualize.py --output-dir results/visualizations`

---

## 3) Model explanation and evaluation

This module focuses on **post-hoc explanations**.

- **SHAP** is implemented as a runnable script via [src/explain.py](src/explain.py).
- **LIME** is used in the explanation notebooks (not currently exposed as a standalone `src/` CLI script).

### 3.1 Generating explanations (SHAP)
**Primary implementation:** [src/explain.py](src/explain.py)

What it does:

1. Loads the canonical test matrix [data/processed/preprocessed/X_test_preproc.csv](data/processed/preprocessed/X_test_preproc.csv).
2. Loads a trained model from [models/](models/) based on `--model` and `--variant`.
3. Computes SHAP values:
   - tree models (`rf`, `xgb`) use `shap.TreeExplainer`
   - logistic regression (`lr`) tries `shap.LinearExplainer`, with a fallback to `shap.KernelExplainer`
4. Normalizes SHAP outputs to a 2D array shape `(n_samples, n_features)` for binary classification.
5. Saves the SHAP array to [results/](results/) as `shap_{model}_{variant}.npy`.

Key outputs:

- SHAP arrays:
  - Example: [results/](results/) → `shap_rf_baseline.npy` (and similar naming)

How to run (CLI):

- `python src/explain.py --model rf --variant baseline`
- `python src/explain.py --model xgb --variant 3x`

Environment compatibility note:

- Your [README.md](README.md) and [scripts/validate_shap50.py](scripts/validate_shap50.py) indicate special attention to SHAP compatibility with newer XGBoost.
- If SHAP/XGBoost versions mismatch, the quick diagnostic script can be used:
  - `python scripts/validate_shap50.py`

### 3.1b Generating explanations (LIME, notebook-based)
**Primary implementation:** explanation notebooks (not a `src/` CLI entrypoint).

Where LIME is used:

- [notebooks/CKD_Model_Explanations.ipynb](notebooks/CKD_Model_Explanations.ipynb)
- [notebooks/CKD_Explanation_Refactored.ipynb](notebooks/CKD_Explanation_Refactored.ipynb)

How it fits this project:

- LIME is applied as a local, model-agnostic explainer (via `LimeTabularExplainer`).
- It perturbs **raw/canonical clinical features** and relies on a prediction wrapper so that model prediction uses the same canonical preprocessing as training.

How to run:

- Open either notebook above and run the LIME sections/cells after the models are trained and the processed splits exist under [data/processed/](data/processed/).

### 3.2 Explanation evaluation: stability metrics
**Primary implementation:** [src/stability.py](src/stability.py)

What it measures:

Given multiple SHAP runs saved to disk (for example, different seeds or repeated runs), it computes per-sample stability metrics:

- Mean pairwise Spearman correlation of absolute SHAP values
- Mean pairwise Jaccard similarity of Top-$k$ features by absolute SHAP value

Key outputs:

- Stability table (CSV): default is `results/stability.csv`

How to run (CLI):

- `python src/stability.py --glob "results/shap_rf_baseline_seed*.npy" --k 5 --out results/explanation_stability_metrics.csv`

Notes:

- This script assumes you have multiple SHAP result files matching the `--glob` pattern.
- The repo already contains [results/explanation_stability_metrics.csv](results/explanation_stability_metrics.csv), which suggests stability has been computed at least once.

---

## Suggested execution order (end-to-end)

1. Cleaning + preprocessing:
   - `python src/cleaning.py`
2. Synthetic data generation (optional; needed for holdout augmented training, not needed for CV):
   - `python src/synthesizer.py --multiplier 1 --seed 42`
   - `python src/synthesizer.py --multiplier 3 --seed 42`
3. Training + evaluation:
   - `python src/train.py --variant baseline --cv-folds 5`
   - `python src/train.py --variant 1x --synth-backend gcopula --cv-folds 5`
   - `python src/train.py --variant 3x --synth-backend gcopula --cv-folds 5`
4. Explanations:
   - `python src/explain.py --model rf --variant baseline`
  - For LIME (notebook-based): run [notebooks/CKD_Model_Explanations.ipynb](notebooks/CKD_Model_Explanations.ipynb) (or [notebooks/CKD_Explanation_Refactored.ipynb](notebooks/CKD_Explanation_Refactored.ipynb))
5. Stability (only if you have multiple SHAP runs):
   - `python src/stability.py --glob "results/shap_rf_baseline_seed*.npy" --k 5`
6. Plots:
   - `python src/visualize.py`

---

## Where the notebooks fit

The notebooks in [notebooks/](notebooks/) are the “start here” experience and largely orchestrate these same steps:

- [notebooks/CKD_Risk_Prediction.ipynb](notebooks/CKD_Risk_Prediction.ipynb)
- [notebooks/CKD_Training.ipynb](notebooks/CKD_Training.ipynb)
- [notebooks/CKD_Comparative_Analysis.ipynb](notebooks/CKD_Comparative_Analysis.ipynb)
- Explanation-focused notebooks such as [notebooks/CKD_Model_Explanations.ipynb](notebooks/CKD_Model_Explanations.ipynb)
- Additional explanation notebook: [notebooks/CKD_Explanation_Refactored.ipynb](notebooks/CKD_Explanation_Refactored.ipynb)

If you want, I can also add a short “entrypoints cheat sheet” section mapping each notebook cell block to the underlying [src/](src/) functions/scripts, but I kept this document scoped to the three requested modules.

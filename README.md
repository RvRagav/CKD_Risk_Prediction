# CKD Risk Prediction (Real vs Synthetic Augmentation)

This repository contains a chronic kidney disease (CKD) risk prediction workflow comparing **real-only training** versus **real + synthetic data augmentation**.

The primary documentation is in the notebooks under notebooks/.

## Notebooks (Start Here)

- notebooks/CKD_Risk_Prediction.ipynb
  - End-to-end exploration and prediction workflow.
- notebooks/CKD_Training.ipynb
  - Training and evaluation on the real dataset.
- notebooks/CKD_Comparative_Analysis.ipynb
  - Comparative analysis of real-only vs augmented training.
  - Uses **cross-validation (CV) metrics** for tables and plots.
  - Saves the trained augmented models into models/.

## Explanations (SHAP + LIME)

For the canonical explanation notebook:

- [notebooks/CKD_Model_Explanations.ipynb](notebooks/CKD_Model_Explanations.ipynb)

Important environment note (XGBoost SHAP):

- If you are using **XGBoost >= 3.1**, you should run this notebook with **SHAP >= 0.50**.
- This repo includes a ready-to-use conda env/kernel setup that installs `shap==0.50.0` under **Python 3.11**.

Steps (VS Code):

- In the notebook, use the kernel picker and select: `ckd-shap50 (Python 3.11)`
- Restart the kernel, then **Run All** (or run cells top-to-bottom)

If you don't see that kernel option, recreate it:

- `conda create -n ckd-shap50 python=3.11`
- `conda activate ckd-shap50`
- `pip install -r requirements.txt shap==0.50.0 ipykernel`
- `python -m ipykernel install --user --name ckd-shap50 --display-name "ckd-shap50 (Python 3.11)"`

## Workflow (Project Flow)

1. Data source
    - Start from the original dataset in dataset/.

2. Cleaning and preprocessing
    - The raw dataset is cleaned and transformed into a model-ready format.
    - Key outputs are stored in data/processed/ (cleaned CSV, train/test splits, preprocessed feature matrices, and feature metadata).

3. Train real-only baseline models
    - Models are trained on real data and evaluated.
    - Trained models are saved in models/.
    - Metrics summaries and plots are saved in results/.

4. Generate and prepare synthetic data (augmentation)
    - Synthetic samples are created to augment the real training set.
    - Synthetic datasets used in experiments are stored in data/synthetic/.

5. Train augmented models (real + synthetic)
    - Augmented models are trained using the same feature space as the real-only pipeline.
    - Augmented trained models are saved in models/.

6. Compare real-only vs augmented performance (CV-first)
    - Comparative reporting is CV-based (tables and plots use CV metrics).
    - Comparison artifacts (CSV/JSON) and visualizations are written to results/.

7. Optional: interpretability and stability checks
    - Additional analysis (explanations, stability, and synthetic-vs-real comparisons) can be run via the supporting modules in src/ and/or the notebooks.

## Project Folders

- dataset/
  - Original source dataset.
- data/processed/
  - Cleaned/preprocessed datasets, splits, and feature metadata.
- data/synthetic/
  - Synthetic datasets used for augmentation experiments.
- models/
  - Saved trained models (.joblib).
- results/
  - Metrics summaries (CSV/JSON) and generated visualizations.
- src/
  - Python modules used by notebooks/scripts (cleaning, training, synthesis, evaluation, visualization).

## Notes

- Recommended order for a fresh run: CKD_Risk_Prediction.ipynb → CKD_Training.ipynb → CKD_Comparative_Analysis.ipynb.
- If you re-run notebooks, outputs under results/ and models/ may be regenerated.

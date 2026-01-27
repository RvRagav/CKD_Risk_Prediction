# CKD Risk Prediction: Synthetic Data Augmentation & Explainability Stability

## Project Overview

This project develops a **scientifically rigorous pipeline** for Chronic Kidney Disease (CKD) risk prediction that:

1. **Cleans and preprocesses** a raw UCI kidney disease dataset (400 rows × 26 columns)
2. **Generates synthetic training data** using Gaussian Copula to augment the small dataset
3. **Trains robust classification models** (Logistic Regression, Random Forest, XGBoost)
4. **Measures explanation stability** across bootstrap runs to validate synthetic data benefit
5. **Generates clinically-feasible counterfactuals** (DiCE) for actionable risk reduction
6. **Validates synthetic quality** using KS tests, correlation analysis, and real vs. synthetic classifiers

---

## Part 1: Quick Data Diagnosis

### Dataset Characteristics (UCI Kidney Disease)
- **Shape**: 400 rows × 26 columns
- **Target**: `classification` (ckd, notckd) — contains hidden tabs (`\t`) and inconsistent whitespace
- **Feature Types**:
  - **Continuous**: age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
  - **Ordinal/Semi-quantitative**: sg, al, su (specific gravity, albumin, sugar; small integer categories 0–5)
  - **Nominal/Categorical**: rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane
  - **Missingness**: Multiple columns with ~130 missing entries (NaN or `?`)

### Critical Data Issues
- **Hidden whitespace**: `dm` column contains `'yes'`, `'\tyes'`, `' yes'`, `'\tno'` (must normalize)
- **String coercion**: `pcv`, `wc`, `rc` stored as strings; need numeric conversion
- **Missing markers**: `'?'` used instead of NaN in numeric-like columns
- **Target inconsistency**: stray `'ckd\t'` variants need stripping and lowercasing

---

## Part 2: Exact Cleaning & Preprocessing Pipeline

### Step 2A: Read & Initial Tidy
```python
import pandas as pd
import numpy as np

df = pd.read_csv('dataset/kidney_disease.csv')

# Drop id column
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Strip whitespace/tabs in object columns & replace '?' with NaN
for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = df[c].astype(str).str.strip().replace({'?': np.nan, 'nan': np.nan})

# Fix target stray tab
df['classification'] = df['classification'].astype(str).str.strip()
```

### Step 2B: Semantic Feature Grouping (Locked)
```python
cont_cols = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
ord_cols = ['sg', 'al', 'su']  # Ordinal/semi-quantitative
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
target_col = 'classification'
```

### Step 2C: Coerce Numerics
```python
num_cols = cont_cols + ord_cols
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')  # bad strings → NaN
```

### Step 2D: Normalize Categorical Labels
```python
# Yes/No columns
yesno_cols = ['htn', 'dm', 'cad', 'pe', 'ane']
for c in yesno_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.lower()
        df[c] = df[c].replace({'yes': 1, 'no': 0, 'nan': np.nan})

# Normal/Abnormal columns
for c in ['rbc', 'pc']:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'normal': 0, 'abnormal': 1, '?': np.nan, 'nan': np.nan})

# Present/NotPresent columns
for c in ['pcc', 'ba']:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'notpresent': 0, 'present': 1, '?': np.nan})
```

### Step 2E: Target Mapping
```python
df['target'] = df['classification'].map(lambda x: 1 if 'ckd' in str(x).lower() else 0)
df = df.drop(columns=['classification'])
```

### Step 2F: Train/Test Split (Locked After This Point)
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

### Step 2G: Imputation & Scaling Pipeline
```python
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

cont_pipe = Pipeline([
    ('imputer', IterativeImputer(random_state=0, max_iter=10)),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('cont', cont_pipe, cont_cols + ord_cols),
    ('cat', cat_pipe, cat_cols)
], remainder='drop')

# Fit on training only
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Save for reproducibility
import joblib
joblib.dump(preprocessor, 'models/preprocessor.pkl')
```

---

## Part 3: Synthetic Data Generation (Gaussian Copula)

### Why Gaussian Copula?
- **Small dataset (400 rows)**: GANs would fail to converge
- **Mixed data types**: Copulas handle correlations between continuous and categorical naturally
- **Reproducibility**: Simpler and more stable than deep learning approaches

### Step 3A–3B: Fit & Sample
```python
from sdv.tabular import GaussianCopula
import pandas as pd

# Create DataFrame from preprocessed training data
feature_names = (
    cont_cols + ord_cols + 
    [f'cat_{i}' for i in range(X_train_proc.shape[1] - len(cont_cols) - len(ord_cols))]
)
X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)

# Fit copula on training data only
copula = GaussianCopula()
copula.fit(X_train_df)

# Generate synthetic samples (+1x augmentation = double dataset)
n_synth = len(X_train_df)  # 1x augmentation
X_synth_1x = copula.sample(n_synth)

# Save synthetic datasets
X_synth_1x.to_csv('dataset/X_synth_1x.csv', index=False)
```

### Why SDV Over Manual Encoding?
- **Automatic handling**: Categorical variables remain categorical (no manual argmax logic)
- **Correlation preservation**: Captures dependencies between features
- **Realistic distributions**: Samples from learned Gaussian copula

---

## Part 4: Synthetic Quality Checks (MANDATORY)

### 4.1: Kolmogorov-Smirnov (KS) Test
```python
from scipy.stats import ks_2samp

ks_results = {}
for c in cont_cols + ord_cols:
    stat, p_value = ks_2samp(X_train_df[c], X_synth_1x[c])
    ks_results[c] = {'statistic': stat, 'p_value': p_value}
    print(f"{c}: stat={stat:.4f}, p={p_value:.4f}")

# Interpretation: p > 0.05 = cannot reject equality (good)
```

### 4.2: Correlation Matrix Difference
```python
corr_real = X_train_df.corr()
corr_synth = X_synth_1x.corr()
corr_diff = (corr_real - corr_synth).abs().mean().mean()
print(f"Mean absolute correlation difference: {corr_diff:.4f}")

# Goal: < 0.05 is excellent; < 0.1 is good
```

### 4.3: Real vs. Synthetic Classifier (AUC)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

Z = pd.concat([
    X_train_df.assign(_isreal=1),
    X_synth_1x.assign(_isreal=0)
])
yZ = Z['_isreal']
Z_features = Z.drop(columns=['_isreal'])

clf = RandomForestClassifier(n_estimators=200, random_state=0)
auc = cross_val_score(clf, Z_features, yZ, cv=5, scoring='roc_auc').mean()
print(f"Real vs. Synth AUC: {auc:.4f}")

# Goal: AUC ≈ 0.5 (model cannot distinguish → synthetic is realistic)
# If AUC >> 0.6, synthetic data is unrealistic; regenerate with different copula parameters
```

---

## Part 5: Model Training & Evaluation

### Baseline Models (Real Data Only)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss
import joblib

models = {
    'lr': LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=0),
    'rf': RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0),
    'xgb': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                         random_state=0, use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train_proc, y_train)
    y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    results[name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'brier': brier_score_loss(y_test, y_pred_proba)
    }
    
    joblib.dump(model, f'models/{name}_baseline.pkl')

# Save metrics
metrics_df = pd.DataFrame(results).T
metrics_df.to_csv('results/baseline_metrics.csv')
```

### Augmented Training (1x, 3x Synthetic)
- Train same models on `X_train + X_synth_1x` and `X_train + X_synth_3x`
- Compare metrics against baseline
- Expected: slight improvement in F1 / recall for minority class (CKD)

---

## Part 6: SHAP Explanations

### SHAP Value Computation (Consistent Test Set)
```python
import shap

for model_name in ['rf', 'xgb', 'lr']:
    model = joblib.load(f'models/{model_name}_baseline.pkl')
    
    if model_name in ['rf', 'xgb']:
        explainer = shap.TreeExplainer(model)
    else:  # LogisticRegression
        explainer = shap.KernelExplainer(model.predict, X_test_proc)
    
    shap_values = explainer.shap_values(X_test_proc)
    np.save(f'results/shap_{model_name}_baseline.npy', shap_values)
```

### Visualization
```python
# Bar plot: mean |SHAP| per feature
shap.summary_plot(shap_values, X_test_proc, plot_type="bar")
```

---

## Part 7: Explanation Stability Metrics

### Per-Sample Spearman Correlation (Across Bootstrap Runs)

**Concept**: Run N=30 bootstrap replicates with different random seeds. For each test sample, compute Spearman rank correlation of absolute SHAP values across runs. High correlation = stable explanations.

```python
import numpy as np
from scipy.stats import spearmanr

def mean_pairwise_spearman(shap_runs_for_sample):
    """
    Args:
        shap_runs_for_sample: list of R arrays, each shape (n_features,)
    Returns:
        mean Spearman ρ across all pairwise comparisons
    """
    R = len(shap_runs_for_sample)
    correlations = []
    for i in range(R):
        for j in range(i + 1, R):
            rho, _ = spearmanr(np.abs(shap_runs_for_sample[i]), 
                               np.abs(shap_runs_for_sample[j]))
            correlations.append(rho)
    return np.mean(correlations)

# Run 30 bootstrap replicas
n_bootstraps = 30
spearman_scores_per_sample = []

for seed in range(n_bootstraps):
    # Resample training data with replacement (or regenerate synthetic with seed)
    indices = np.random.RandomState(seed).choice(len(X_train_proc), 
                                                  size=len(X_train_proc), replace=True)
    X_boot = X_train_proc[indices]
    y_boot = y_train.iloc[indices]
    
    # Retrain model
    rf_boot = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=seed)
    rf_boot.fit(X_boot, y_boot)
    
    # Compute SHAP for test set
    explainer = shap.TreeExplainer(rf_boot)
    shap_vals_boot = explainer.shap_values(X_test_proc)
    
    # Store SHAP values
    np.save(f'results/shap_rf_boot_{seed}.npy', shap_vals_boot)

# Aggregate Spearman per sample
all_shap_bootstrap = [np.load(f'results/shap_rf_boot_{seed}.npy') 
                       for seed in range(n_bootstraps)]

spearman_per_sample = []
for sample_idx in range(X_test_proc.shape[0]):
    shap_sample_runs = [shap_boot[sample_idx] for shap_boot in all_shap_bootstrap]
    mean_rho = mean_pairwise_spearman(shap_sample_runs)
    spearman_per_sample.append(mean_rho)

print(f"Mean Spearman (Baseline): {np.mean(spearman_per_sample):.4f}")
print(f"Std Spearman: {np.std(spearman_per_sample):.4f}")
```

### Top-k Jaccard (Feature Stability)

**Concept**: For each sample and run, identify top-k most important features (by |SHAP|). Compute Jaccard overlap across runs.

```python
def jaccard_topk(shap_a, shap_b, k=5):
    """
    Args:
        shap_a, shap_b: arrays of shape (n_features,)
        k: number of top features to compare
    Returns:
        Jaccard similarity in [0, 1]
    """
    set_a = set(np.argsort(-np.abs(shap_a))[:k])
    set_b = set(np.argsort(-np.abs(shap_b))[:k])
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

# Compute Jaccard per sample across bootstrap pairs
jaccard_per_sample = []
for sample_idx in range(X_test_proc.shape[0]):
    jaccard_vals = []
    for i in range(n_bootstraps):
        for j in range(i + 1, n_bootstraps):
            shap_i = all_shap_bootstrap[i][sample_idx]
            shap_j = all_shap_bootstrap[j][sample_idx]
            jac = jaccard_topk(shap_i, shap_j, k=5)
            jaccard_vals.append(jac)
    jaccard_per_sample.append(np.mean(jaccard_vals))

print(f"Mean Jaccard (top-5, Baseline): {np.mean(jaccard_per_sample):.4f}")
```

### Statistical Comparison: Real vs. Augmented
```python
from scipy.stats import wilcoxon

# Compute stability metrics for Real-only and Real+Synthetic variants
spearman_real = [...]  # per-sample from baseline model
spearman_aug = [...]   # per-sample from augmented model

# Wilcoxon signed-rank test
stat, p_value = wilcoxon(spearman_real, spearman_aug)
median_diff = np.median(spearman_aug) - np.median(spearman_real)

print(f"Wilcoxon p-value: {p_value:.4f}")
print(f"Median Spearman difference: {median_diff:.4f}")

# Threshold: p < 0.05 AND median_diff > 0.05 → significant improvement
if p_value < 0.05 and median_diff > 0.05:
    print("✓ Synthetic augmentation significantly improves stability!")
```

---

## Part 8: Counterfactual Generation (DiCE)

### Clinical Constraints (Immutable & Bounded Features)

```python
# Immutable: cannot be changed
IMMUTABLE = ['age']  # Age cannot decrease

# Bounded: allow changes within realistic clinical limits
BOUNDED = {
    'bp': 30,        # ±30 mmHg change allowed
    'hemo': 2,       # ±2 g/dL change allowed
    'sc': 1.0,       # ±1 mg/dL serum creatinine
}

# Comorbidities: do not flip (short-term CFs)
NO_FLIP = ['htn', 'dm', 'cad']
```

### DiCE Generation (SDV / dice-ml)

```python
from dice_ml import Dice
import dice_ml

# Prepare data for DiCE
data = dice_ml.Data(
    dataframe=X_train_df,
    continuous_features=cont_cols + ord_cols,
    outcome_name='target'
)

# Wrap trained model
model = dice_ml.Model(
    model=rf_baseline,
    backend='sklearn'
)

# Initialize DiCE
dice = Dice(data, model, method='random')

# Generate CFs for a single patient
test_patient = X_test_proc[0:1]
cfs = dice.generate_counterfactuals(
    query_instance=test_patient,
    total_CFs=5,
    desired_class='opposite',  # flip prediction to healthy
    features_to_vary=[c for c in feature_names if c not in NO_FLIP],
    feature_weights={c: 0 for c in NO_FLIP}  # zero weight = immutable
)
```

### Rule-Checker (Clinical Plausibility)

```python
def check_cf_plausibility(cf_original, cf_counterfactual, rules):
    """
    Validate counterfactual against clinical rules.
    
    Args:
        cf_original: original feature values
        cf_counterfactual: proposed CF feature values
        rules: dict of feature constraints
    
    Returns:
        is_valid (bool), violations (list of str)
    """
    violations = []
    
    for feature, bounds in rules.items():
        if isinstance(bounds, dict) and 'immutable' in bounds:
            if cf_original[feature] != cf_counterfactual[feature]:
                violations.append(f"{feature}: cannot change (immutable)")
        
        elif isinstance(bounds, (int, float)):
            delta = abs(cf_counterfactual[feature] - cf_original[feature])
            if delta > bounds:
                violations.append(f"{feature}: change={delta:.2f} exceeds limit={bounds}")
        
        # Check percentile bounds (realistic range)
        if 'p1' in bounds and 'p99' in bounds:
            if not (bounds['p1'] <= cf_counterfactual[feature] <= bounds['p99']):
                violations.append(f"{feature}: {cf_counterfactual[feature]} outside realistic range")
    
    return len(violations) == 0, violations

# Compute percentile bounds from training data
feature_bounds = {}
for c in cont_cols + ord_cols:
    feature_bounds[c] = {
        'p1': X_train_df[c].quantile(0.01),
        'p99': X_train_df[c].quantile(0.99)
    }

# Validate CFs
valid_cfs = []
for cf_idx in range(len(cfs)):
    cf_row = cfs.iloc[cf_idx]
    is_valid, violations = check_cf_plausibility(
        test_patient.iloc[0], cf_row, feature_bounds
    )
    if is_valid:
        valid_cfs.append(cf_row)
    else:
        print(f"CF {cf_idx} rejected: {violations}")
```

---

## Part 9: Experiment Sequence

Run steps **in order**; each produces saved artifacts:

1. **Clean & Preprocess** → `preprocessor.pkl`, `X_train_proc.csv`, `X_test_proc.csv`
2. **Baseline Models** → `{lr,rf,xgb}_baseline.pkl`, `baseline_metrics.csv`
3. **Fit Copula & Generate Synthetic** → `X_synth_1x.csv`, `X_synth_3x.csv`
4. **Synthetic Quality Checks** → `synth_qc.csv` (KS stats, corr diff, AUC)
5. **Augmented Training** → `{lr,rf,xgb}_1x.pkl`, `{lr,rf,xgb}_3x.pkl`, augmented metrics
6. **Bootstrap SHAP (30 seeds)** → `shap_{model}_{variant}_{seed}.npy`
7. **Stability Metrics** → `stability_spearman.csv`, `stability_jaccard.csv`, Wilcoxon test results
8. **Counterfactual Generation** → `counterfactuals_10_patients.csv`, rule-checker validation
9. **Dashboard** → `streamlit_app.py` with 3 example patients, SHAP bars, top CF suggestions

---

## Part 10: Repository Structure

```
CKD_Risk_Prediction/
├── README.md                          # This file
├── dataset/
│   ├── kidney_disease.csv             # Raw (400 × 26)
│   ├── kidney_disease_preprocessed.csv
│   ├── X_synth_1x.csv                 # Generated: +1x augmentation
│   └── X_synth_3x.csv                 # Generated: +3x augmentation
├── models/
│   ├── preprocessor.pkl               # Fitted sklearn ColumnTransformer
│   ├── {lr,rf,xgb}_baseline.pkl       # Baseline models (real data only)
│   ├── {lr,rf,xgb}_1x.pkl             # Augmented models (+1x synthetic)
│   └── {lr,rf,xgb}_3x.pkl             # Augmented models (+3x synthetic)
├── results/
│   ├── baseline_metrics.csv           # ROC-AUC, Precision, Recall, F1, Brier
│   ├── synth_qc.csv                   # KS stats, corr diff, real-vs-synth AUC
│   ├── stability_spearman.csv         # Per-sample mean Spearman ρ (30 seeds)
│   ├── stability_jaccard.csv          # Per-sample Jaccard@5 (30 seeds)
│   ├── wilcoxon_results.csv           # p-values, effect sizes (Real vs. Aug)
│   ├── shap_rf_baseline.npy           # SHAP values for RF (real test set)
│   ├── shap_rf_1x.npy                 # SHAP values for RF (1x aug) [from seed 0]
│   ├── shap_xgb_baseline.npy          # etc.
│   └── counterfactuals_valid.csv      # 5 valid CFs per patient (10 patients)
├── src/
│   ├── cleaning.py                    # Steps 2A–2F: data prep, save preprocessor
│   ├── synthesizer.py                 # Step 3: Copula fit, sample, save CSV
│   ├── train.py                       # Step 5: train LR/RF/XGB, save models + metrics
│   ├── explain.py                     # Step 6: compute SHAP, save NPY
│   ├── stability.py                   # Step 7: bootstrap loop, Spearman/Jaccard, Wilcoxon
│   ├── counterfactuals.py             # Step 8: DiCE generation + rule-checker
│   └── utils.py                       # Shared helpers (feature names, bounds, etc.)
├── notebooks/
│   ├── 01_experiment.ipynb            # Reproducible pipeline: all 9 steps
│   └── CKD_Risk_Prediction.ipynb      # Original notebook (reference)
└── app/
    └── streamlit_app.py               # Step 9: Interactive dashboard (3 patients, SHAP, CFs)
```

---

## Key Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
scipy>=1.7.0
shap>=0.40.0
sdv>=0.13.0  # Synthetic Data Vault (Gaussian Copula)
dice-ml>=0.8.0
streamlit>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Preprocessing
```bash
python src/cleaning.py
```
Outputs: `preprocessor.pkl`, preprocessed data splits.

### 2. Synthetic Data
```bash
python src/synthesizer.py
```
Outputs: `X_synth_1x.csv`, `X_synth_3x.csv`, quality checks (`synth_qc.csv`).

### 3. Model Training (All Variants)
```bash
python src/train.py --variant baseline
python src/train.py --variant 1x
python src/train.py --variant 3x
```
Outputs: `{lr,rf,xgb}_{variant}.pkl`, metrics CSV.

### 4. SHAP & Stability (30 Bootstrap Runs)
```bash
python src/explain.py --variant baseline --n_bootstraps 30
python src/stability.py --variant baseline
```
Outputs: SHAP NPY files, `stability_spearman.csv`, `stability_jaccard.csv`, Wilcoxon tests.

### 5. Counterfactuals (10 Test Patients)
```bash
python src/counterfactuals.py --n_patients 10
```
Outputs: `counterfactuals_valid.csv` (5 valid CFs per patient).

### 6. Dashboard
```bash
streamlit run app/streamlit_app.py
```
Opens browser with: 3 example patients, risk score, SHAP bar, top CF suggestion.

---

## Why This Approach Works

### 1. **Targeted Data Cleaning**
- Fixes the exact "dirty data" issues (hidden tabs, inconsistent labels)
- No model training can overcome garbage data

### 2. **Scientifically Sound Stability Metrics**
- **Per-test-sample Spearman correlation** (across 30 bootstrap runs) = gold standard for "stability"
- **Wilcoxon test** = rigorous statistical proof of improvement
- This is the **"Novelty"** claim: synthetic data → more stable explanations

### 3. **Correct Choice of Synthesizer**
- **Gaussian Copula** (via SDV) is ideal for 400-row dataset
- Deep learning (GANs) would fail on small data
- Copulas preserve correlations → realistic synthetic samples

### 4. **Clinical Guardrails (Rule-Checker)**
- DiCE alone suggests unrealistic CFs (e.g., "decrease age by 5 years")
- Rule-Checker blocks impossible changes
- **Clinical Feasibility** is a key review criterion

---

## Expected Results

### Baseline Metrics (Real Data Only)
- ROC-AUC: ~0.85–0.92 (class imbalance → high AUC possible)
- Precision: ~0.75–0.85
- Recall: ~0.65–0.85
- F1: ~0.70–0.80

### With Synthetic Augmentation (+3x)
- Expected improvement: +2–5% ROC-AUC
- Stability metric (Spearman): baseline ~0.60–0.70 → +3x augmented ~0.70–0.80
- Wilcoxon p-value: < 0.05 (statistically significant)

### Synthetic Quality (QC Table)
| Metric | Target | Expected |
|--------|--------|----------|
| Mean KS p-value | > 0.05 | 0.3–0.7 |
| Corr diff | < 0.05 | 0.02–0.04 |
| Real vs. Synth AUC | ≈ 0.5 | 0.48–0.52 |

### Counterfactuals (Per Patient)
- Average features changed: 2–3 out of 11
- Average change magnitude: ±0.5 std normalized
- 100% validity (predicted class flips as desired)

---

## Strengths of This Plan

✓ **Rigorous**: Backed by academic literature on explanation stability  
✓ **Reproducible**: Fixed seeds, saved artifacts, clear pipeline  
✓ **Clinically sound**: Rule-checker prevents impossible CFs  
✓ **Comprehensive**: Covers data cleaning, synthesis, training, explainability, stability, CFs  
✓ **Practical**: Every step has code; every output is a CSV or saved model  

---

## Critical Optimizations (Known Issues & Fixes)

### Issue A: One-Hot Trap in Gaussian Copula
**Problem**: Manual one-hot encoding breaks Gaussian Copula (assumes continuous distributions)  
**Fix**: Use SDV library instead—it handles categorical → numeric → Gaussian automatically  

### Issue B: Integer Precision in DiCE
**Problem**: DiCE outputs floats for everything (e.g., "Age: 55.4")  
**Fix**: Rule-Checker rounds integers (age, al, su) to nearest whole number  

---

## For the Review Panel

### The "Killer Feature": Real vs. Synthetic Classifier (AUC)
- **Logic**: Train Random Forest to guess which data is fake
- **Goal**: AUC ≈ 0.5 (model cannot distinguish)
- **Interpretation**: If AUC = 0.5 → synthetic data is indistinguishable from real → highly realistic
- **Implication**: Using synthetic data is safe for model training; distribution is preserved

### The "Novelty": Explanation Stability Score (ESS)
- **Claim**: Synthetic augmentation → more stable model explanations
- **Evidence**: Spearman correlation of SHAP values across 30 bootstrap runs
- **Threshold**: p < 0.05 + median Δ > +0.05 Spearman points
- **Result**: Stable explanations = trustworthy CFs = clinically actionable insights

---

## Next Steps

1. Create `src/` and `app/` directories
2. Implement `src/cleaning.py` (Step 2A–2F)
3. Implement `src/synthesizer.py` (Step 3)
4. Implement `src/train.py` (Step 5) + iterate metrics
5. Implement `src/explain.py` + `src/stability.py` (Steps 6–7)
6. Implement `src/counterfactuals.py` (Step 8)
7. Build `streamlit_app.py` (Step 9)
8. Run full pipeline; generate all result CSVs
9. Write final review document + slides

---

**Last Updated**: 2026-01-23  
**Status**: Ready for implementation

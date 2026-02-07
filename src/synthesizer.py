from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Allow running as: python src/synthesizer.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLINICAL_BOUNDS,
    HIGH_RISK_NUMERIC,
    ORD_COLS,
    PREPROCESSOR_PATH,
    SPLIT_DIR,
    SYNTH_DIR,
    TARGET_COL,
    PREPROC_DIR,
)
from src.utils import ensure_dir
from src.canonical import (
    CANONICAL_FEATURES,
    CanonicalPreprocessor,
    assert_canonical_schema,
    forbid_onehot_residuals,
)

# -------------------------------------------------------------------------
#  1. Native Gaussian Copula Implementation
# -------------------------------------------------------------------------

class GaussianCopulaGenerator:
    """
    Native implementation of Gaussian Copula Synthesis.
    
    1. Models marginal distributions using Empirical CDF (quantiles).
    2. Models correlations using the covariance of the Gaussian-transformed variables.
    """
    def __init__(self):
        self.cov_matrix = None
        self.marginals = {}
        self.columns = None
        self.categorical_mappings = {}  # Store label encodings for categorical columns
        self.imputers = {} # Store fill values to ensure reproducibility

    def fit(self, df: pd.DataFrame) -> None:
        self.columns = df.columns
        df_encoded = df.copy()

        # --- STEP 1: SAFETY IMPUTATION (CRITICAL FIX) ---
        # We must fill NaNs because cleaning.py only fixed numerics.
        # We cannot drop rows, or we lose the minority class.
        for col in df_encoded.columns:
            if df_encoded[col].isna().any():
                if pd.api.types.is_numeric_dtype(df_encoded[col]):
                    fill_val = df_encoded[col].median()
                else:
                    # Fill categorical NaNs with Mode (most frequent)
                    mode_series = df_encoded[col].mode()
                    fill_val = mode_series[0] if not mode_series.empty else 'Missing'
                
                df_encoded[col] = df_encoded[col].fillna(fill_val)
                self.imputers[col] = fill_val

        # --- STEP 2: ENCODE CATEGORICALS (kept for completeness) ---
        # Canonical synthesis uses numeric/binary columns, so this usually no-ops.
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                unique_vals = np.sort(df_encoded[col].unique())
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.categorical_mappings[col] = {idx: val for val, idx in mapping.items()}
                df_encoded[col] = df_encoded[col].map(mapping)

        # Convert to dense matrix (No dropna needed now!)
        X = df_encoded.to_numpy(dtype=float)
        n_obs, n_features = X.shape

        # --- STEP 3: TRANSFORM TO GAUSSIAN ---
        Z = np.zeros_like(X)
        epsilon = 1e-6  # Stability for norm.ppf

        for i in range(n_features):
            col_data = X[:, i]
            self.marginals[i] = np.sort(col_data)
            
            # Rank -> Uniform (0, 1)
            ranks = pd.Series(col_data).rank(method='average').to_numpy()
            U = np.clip((ranks - 0.5) / n_obs, epsilon, 1 - epsilon)
            
            # Uniform -> Standard Normal
            Z[:, i] = norm.ppf(U)
            
            # Safety clamp for extreme outliers
            Z[:, i] = np.clip(Z[:, i], -5.0, 5.0)

        # --- STEP 4: COVARIANCE MATRIX ---
        self.cov_matrix = np.cov(Z, rowvar=False)
        
        # Regularization (ensure positive semi-definite)
        min_eig = np.min(np.real(np.linalg.eigvals(self.cov_matrix)))
        if min_eig < 0:
            self.cov_matrix -= 10 * min_eig * np.eye(n_features)

    def sample(self, n_samples: int, seed: int | None = None) -> pd.DataFrame:
        if self.columns is None:
            raise ValueError("Model has not been fitted yet.")
        if self.cov_matrix is None:
            raise ValueError("Model has not been fitted yet.")
        
        rng = np.random.default_rng(seed)
        n_features = len(self.columns)

        # 1. Sample Multivariate Normal
        mean = np.zeros(n_features)
        cov = np.real(self.cov_matrix).astype(np.float64)
        
        # 'check_valid=warn' handles minor numerical instability
        Z_synth = rng.multivariate_normal(mean, cov, size=n_samples, check_valid='warn')

        # 2. Transform back to Uniform
        U_synth = norm.cdf(Z_synth)

        # 3. Transform Uniform back to Original Marginals
        X_synth = np.zeros_like(U_synth)
        
        for i in range(n_features):
            valid_support = self.marginals[i]
            # Use interpolation to map Uniform back to the empirical distribution
            X_synth[:, i] = np.interp(U_synth[:, i], np.linspace(0, 1, len(valid_support)), valid_support)

        df_synth = pd.DataFrame(X_synth, columns=self.columns)
        
        # 4. Decode Categoricals
        for col in self.columns:
            if col in self.categorical_mappings:
                # Round to nearest integer code
                codes = np.round(df_synth[col]).astype(int)
                
                # Clip to valid range of codes
                max_code = max(self.categorical_mappings[col].keys())
                codes = np.clip(codes, 0, max_code)
                
                # Map back to strings
                df_synth[col] = [self.categorical_mappings[col][c] for c in codes]
        
        return df_synth
# -------------------------------------------------------------------------
#  2. Post-Processing Utilities
# -------------------------------------------------------------------------

def _snap_to_observed(col: pd.Series, ref: pd.Series) -> pd.Series:
    """Snap values to the nearest observed value (for ordinal/discrete)."""
    x = pd.to_numeric(col, errors='coerce').to_numpy()
    obs = np.sort(pd.to_numeric(ref, errors='coerce').dropna().unique())
    if obs.size == 0: 
        return pd.Series(x)

    idx = np.searchsorted(obs, x)
    idx = np.clip(idx, 0, obs.size - 1)
    
    left = obs[np.clip(idx - 1, 0, obs.size - 1)]
    right = obs[idx]
    
    # Choose nearest neighbor
    snapped = np.where(np.abs(x - left) <= np.abs(x - right), left, right)
    return pd.Series(snapped)

def postprocess_synth(X_synth: pd.DataFrame, X_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce clinical logic and data types on the raw synthetic data.
    """
    out = X_synth.copy()

    # 1. Enforce specific clinical bounds (Physiological limits)
    for col, (lo, hi) in CLINICAL_BOUNDS.items():
        if col in out.columns:
            out[col] = out[col].clip(lo, hi)

    # 1b. Force binary clinical flags to {0,1}
    for col in ['htn', 'dm']:
        if col in out.columns:
            x = pd.to_numeric(out[col], errors='coerce')
            # If generator produced continuous values, threshold at 0.5
            x = (x >= 0.5).astype(int)
            out[col] = x

    # 2. Snap ordinal columns (e.g., stages 1,2,3) to integers
    for col in ORD_COLS:
        if col in out.columns and col in X_ref.columns:
            out[col] = _snap_to_observed(out[col], X_ref[col])

    # 3. Round High Risk Numerics if they are essentially integers in reality (optional)
    #    For now, we just ensure they stay within observed quantiles to avoid outliers
    for col in HIGH_RISK_NUMERIC:
        if col in out.columns and col in X_ref.columns:
            # Clip to observed min/max to prevent explosion
            lo = X_ref[col].min()
            hi = X_ref[col].max()
            out[col] = out[col].clip(lo, hi)

    return out

# -------------------------------------------------------------------------
#  3. Data Loading & Preprocessing Wrappers
# -------------------------------------------------------------------------

def _load_imputed_train() -> pd.DataFrame:
    """
    CRITICAL CHANGE: Loads 'X_train_imputed_raw.csv' instead of 'X_train_raw.csv'.
    We must use the dataset where missing values have been filled by Linear Regression.
    """
    split_dir = Path(SPLIT_DIR)
    path = split_dir / 'X_train_imputed_raw.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/cleaning.py first.")
    print(f"Loading imputed training data from: {path}")
    return pd.read_csv(path)

def _load_y_train() -> np.ndarray:
    split_dir = Path(SPLIT_DIR)
    path = split_dir / 'y_train.csv'
    return pd.read_csv(path)[TARGET_COL].to_numpy()

def _load_canonical_preprocessor() -> CanonicalPreprocessor:
    preproc = joblib.load(PREPROCESSOR_PATH)
    if not isinstance(preproc, CanonicalPreprocessor):
        raise TypeError(
            'Saved preprocessor is not CanonicalPreprocessor. '
            'Re-run: python src/cleaning.py to regenerate canonical artifacts.'
        )
    return preproc

def _load_preprocessed_train() -> pd.DataFrame:
    """Used only for QC comparison."""
    path = Path(PREPROC_DIR) / 'X_train_preproc.csv'
    return pd.read_csv(path)

# -------------------------------------------------------------------------
#  4. Main Synthesis Logic
# -------------------------------------------------------------------------

def generate_by_class(X: pd.DataFrame, y: np.ndarray, n_total: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Splits data by class, fits a Copula per class, samples, and recombines.
    This preserves the distinct correlations often found in disease vs healthy groups.
    """
    rng = np.random.default_rng(seed)
    
    # Separate classes
    X0 = X[y == 0].reset_index(drop=True)
    X1 = X[y == 1].reset_index(drop=True)
    
    # Calculate counts based on original ratio
    p1 = len(X1) / len(y)
    n1 = int(round(n_total * p1))
    n0 = n_total - n1
    
    print(f"Training Class 0 Copula ({len(X0)} real samples) -> Generating {n0} synthetic...")
    copula0 = GaussianCopulaGenerator()
    copula0.fit(X0)
    X_synth0 = copula0.sample(n0, seed=seed)
    
    print(f"Training Class 1 Copula ({len(X1)} real samples) -> Generating {n1} synthetic...")
    copula1 = GaussianCopulaGenerator()
    copula1.fit(X1)
    # Use different seed for second batch
    X_synth1 = copula1.sample(n1, seed=seed + 1 if seed else None)
    
    # Combine
    X_synth = pd.concat([X_synth0, X_synth1], ignore_index=True)
    y_synth = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    
    # Shuffle
    perm = rng.permutation(len(X_synth))
    return X_synth.iloc[perm].reset_index(drop=True), y_synth[perm]

def main():
    parser = argparse.ArgumentParser(description='Gaussian Copula Synthesizer (Native)')
    parser.add_argument('--multiplier', type=int, default=1, help='Size multiplier relative to train set')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backend', type=str, default='gcopula', choices=['gcopula'])
    args = parser.parse_args()

    # 1. Load Data (raw-imputed), then STRICTLY filter to canonical feature set
    X_train_full = _load_imputed_train()
    X_train = X_train_full[CANONICAL_FEATURES].copy()
    y_train = _load_y_train()
    
    n_synth = len(X_train) * args.multiplier
    
    print(f"\n--- Starting Synthesis (Gaussian Copula) ---\nTarget size: {n_synth} samples")

    # 2. Synthesize in canonical feature space ONLY.
    X_synth_raw, y_synth = generate_by_class(X_train, y_train, n_synth, args.seed)

    # 3. Post-Process (Bounds & Binary Flags)
    X_synth_raw = postprocess_synth(X_synth_raw, X_train)

    # 4. Canonical preprocessing (same fitted preprocessor as real training)
    preproc = _load_canonical_preprocessor()
    X_synth_preproc = preproc.transform(X_synth_raw)

    forbid_onehot_residuals(list(X_synth_preproc.columns))
    assert_canonical_schema(X_synth_preproc)

    # 5. Save
    out_dir = ensure_dir(SYNTH_DIR)
    
    # Save Preprocessed Synthetic (for model training)
    out_path_preproc = out_dir / f'X_synth_{args.multiplier}x_gcopula_preproc.csv'
    X_synth_preproc.to_csv(out_path_preproc, index=False)
    
    # Save Labels
    out_path_y = out_dir / f'y_synth_{args.multiplier}x_gcopula.csv'
    pd.DataFrame({TARGET_COL: y_synth}).to_csv(out_path_y, index=False)

    print(f"Saved Synthetic Data: {out_path_preproc}")
    print(f"Saved Synthetic Labels: {out_path_y}")

    # -------------------------------------------------------------------------
    #  QC (Quality Control)
    # -------------------------------------------------------------------------
    print("\n--- Running QC Metrics ---")
    
    # Load real preprocessed data for comparison
    X_real_preproc = _load_preprocessed_train()
    forbid_onehot_residuals(list(X_real_preproc.columns))
    assert_canonical_schema(X_real_preproc)
    
    common_cols = CANONICAL_FEATURES.copy()
    
    # Metric 1: KS Test (Distribution similarity)
    # Lower statistic = distributions are more similar
    ks_results = []
    for col in common_cols:
        stat, pval = ks_2samp(X_real_preproc[col], X_synth_preproc[col])
        ks_results.append({'feature': col, 'ks_stat': stat, 'ks_p': pval})
    
    ks_df = pd.DataFrame(ks_results)
    avg_ks = ks_df['ks_stat'].mean()
    print(f"Average KS Statistic: {avg_ks:.4f} (Lower is better)")
    
    # Metric 2: Correlation Structure Preservation
    corr_real = X_real_preproc[common_cols].corr()
    corr_synth = X_synth_preproc[common_cols].corr()
    diff_matrix = (corr_real - corr_synth).abs()
    mean_corr_diff = diff_matrix.mean().mean()
    print(f"Mean Absolute Correlation Difference: {mean_corr_diff:.4f} (Lower is better)")

    # Metric 3: Discriminator AUC (Can a classifier tell real from fake?)
    Z = pd.concat([
        X_real_preproc[common_cols].assign(is_real=1),
        X_synth_preproc[common_cols].assign(is_real=0)
    ], ignore_index=True)
    
    # Shuffle
    Z = Z.sample(frac=1, random_state=42).reset_index(drop=True)
    y_disc = Z['is_real']
    X_disc = Z.drop(columns=['is_real'])
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    auc_scores = cross_val_score(clf, X_disc, y_disc, cv=3, scoring='roc_auc')
    print(f"Discriminator AUC: {auc_scores.mean():.4f} (0.5 is ideal, 1.0 is bad)")

    # Save QC Report with consistent naming
    results_dir = ensure_dir('results')
    tag = f'{args.multiplier}x_gcopula_seed{args.seed}'
    
    qc_path = results_dir / f'qc_report_{tag}.csv'
    pd.DataFrame([{
        'avg_ks': avg_ks,
        'mean_corr_diff': mean_corr_diff,
        'discriminator_auc': auc_scores.mean()
    }]).to_csv(qc_path, index=False)
    print(f"Saved QC Report: {qc_path}")
    
    # Save KS Test Results (per feature)
    ks_path = results_dir / f'ks_{tag}.csv'
    ks_df.to_csv(ks_path, index=False)
    print(f"Saved KS Results: {ks_path}")
    
    # Clean up old files with different naming patterns
    old_patterns = ['*_sdv_gcopula_*', 'qc_report_[0-9]x.csv']
    for pattern in old_patterns:
        for old_file in results_dir.glob(pattern):
            if old_file.exists() and old_file.stem != f'qc_report_{args.multiplier}x_gcopula_seed{args.seed}' and old_file.stem != f'ks_{args.multiplier}x_gcopula_seed{args.seed}':
                print(f"Removing old file: {old_file.name}")
                old_file.unlink()

if __name__ == '__main__':
    main()
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Allow running as: python src/synthesizer.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREPROC_DIR, SYNTH_DIR  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def _load_preprocessed_train() -> pd.DataFrame:
    preproc_dir = Path(PREPROC_DIR)
    path = preproc_dir / 'X_train_preproc.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/cleaning.py first.")
    return pd.read_csv(path)


def _load_y_train() -> np.ndarray:
    split_dir = Path('data/processed/splits')
    path = split_dir / 'y_train.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/cleaning.py first.")
    return pd.read_csv(path)['target'].to_numpy()


def _gaussian_copula_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Gaussian copula sampler using empirical marginals.

    Works without sdv/copulas; relies only on numpy/scipy.
    """
    rng = np.random.default_rng(seed)
    X = df.to_numpy(dtype=float)
    n_obs, n_features = X.shape

    # Rank -> uniform -> normal scores (Gaussianize)
    ranks = np.apply_along_axis(lambda col: pd.Series(col).rank(method='average').to_numpy(), 0, X)
    U = (ranks - 0.5) / n_obs
    Z = norm.ppf(U)

    # Correlation in Gaussian space (regularize for stability)
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2
    eps = 1e-6
    corr.flat[:: n_features + 1] += eps

    # Sample multivariate normal
    L = np.linalg.cholesky(corr)
    Zs = rng.standard_normal(size=(n, n_features)) @ L.T
    Us = norm.cdf(Zs)

    # Inverse empirical CDF via quantiles
    out = np.empty_like(Us)
    q_grid = np.linspace(0.0, 1.0, n_obs)
    for j in range(n_features):
        x_sorted = np.sort(X[:, j])
        out[:, j] = np.interp(Us[:, j], q_grid, x_sorted)

    return pd.DataFrame(out, columns=df.columns)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _sdv_sample(df: pd.DataFrame, n: int, seed: int, kind: str, epochs: int) -> pd.DataFrame:
    """Sample using SDV single-table synthesizers.

    kind: 'gcopula' | 'ctgan'
    """
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SDV is not installed or not supported in this Python. Install with: pip install 'sdv>=1.10'"
        ) from e

    _set_seed(seed)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    if kind == 'ctgan':
        synth = CTGANSynthesizer(metadata, epochs=epochs, verbose=False)
    elif kind == 'gcopula':
        synth = GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError(f'Unknown SDV synthesizer kind: {kind}')

    # Best-effort deterministic behavior (SDV versions differ)
    try:
        setter = getattr(synth, 'set_random_state', None)
        if callable(setter):
            setter(seed)
    except Exception:
        pass

    synth.fit(df)
    sample_df = synth.sample(n)

    # Ensure same column order
    return sample_df[df.columns]


def _class_conditional_sample(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    n_total: int,
    seed: int,
    backend: str,
    ctgan_epochs: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError('Both classes must be present to do class-conditional synthesis.')

    p1 = len(idx1) / len(y_train)
    n1 = int(round(n_total * p1))
    n0 = n_total - n1

    X0 = X_train_df.iloc[idx0]
    X1 = X_train_df.iloc[idx1]

    seed0 = int(rng.integers(0, 2**31 - 1))
    seed1 = int(rng.integers(0, 2**31 - 1))

    if backend == 'native':
        Xs0 = _gaussian_copula_sample(X0, n=n0, seed=seed0)
        Xs1 = _gaussian_copula_sample(X1, n=n1, seed=seed1)
    elif backend == 'sdv_gcopula':
        Xs0 = _sdv_sample(X0, n=n0, seed=seed0, kind='gcopula', epochs=ctgan_epochs)
        Xs1 = _sdv_sample(X1, n=n1, seed=seed1, kind='gcopula', epochs=ctgan_epochs)
    elif backend == 'sdv_ctgan':
        Xs0 = _sdv_sample(X0, n=n0, seed=seed0, kind='ctgan', epochs=ctgan_epochs)
        Xs1 = _sdv_sample(X1, n=n1, seed=seed1, kind='ctgan', epochs=ctgan_epochs)
    else:
        raise ValueError(f'Unknown backend: {backend}')

    Xs = pd.concat([Xs0, Xs1], ignore_index=True)
    ys = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    # Shuffle
    perm = rng.permutation(len(Xs))
    return Xs.iloc[perm].reset_index(drop=True), ys[perm]


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit Gaussian copula on X_train_preproc and sample synthetic rows.')
    parser.add_argument('--multiplier', type=int, default=1, help='1 for +1x, 3 for +3x')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--backend', type=str, default='sdv_gcopula', choices=['native', 'sdv_gcopula', 'sdv_ctgan'])
    parser.add_argument('--ctgan-epochs', type=int, default=150, help='Epochs used only for sdv_ctgan')
    args = parser.parse_args()

    X_train_df = _load_preprocessed_train()
    y_train = _load_y_train()
    n_synth = len(X_train_df) * int(args.multiplier)

    X_synth, y_synth = _class_conditional_sample(
        X_train_df,
        y_train,
        n_total=n_synth,
        seed=args.seed,
        backend=args.backend,
        ctgan_epochs=int(args.ctgan_epochs),
    )
    backend_used = args.backend

    synth_dir = ensure_dir(SYNTH_DIR)
    out_path = synth_dir / f'X_synth_{args.multiplier}x_{backend_used}_preproc.csv'
    X_synth.to_csv(out_path, index=False)
    y_path = synth_dir / f'y_synth_{args.multiplier}x_{backend_used}.csv'
    pd.DataFrame({'target': y_synth}).to_csv(y_path, index=False)
    print('Saved synthetic:', out_path)
    print('Saved synthetic labels:', y_path)
    print('Backend:', backend_used)

    # --- Mandatory QC ---
    qc_dir = ensure_dir('results')

    # (1) KS tests per feature
    from scipy.stats import ks_2samp

    ks_rows = []
    for col in X_train_df.columns:
        res = ks_2samp(X_train_df[col].to_numpy(), X_synth[col].to_numpy())
        stat = getattr(res, 'statistic', res[0])
        pval = getattr(res, 'pvalue', res[1])

        stat_f = float(np.asarray(stat).reshape(-1)[0].item())
        pval_f = float(np.asarray(pval).reshape(-1)[0].item())
        ks_rows.append({'feature': col, 'ks_stat': stat_f, 'ks_p': pval_f})

    ks_df = pd.DataFrame(ks_rows)
 
    # (2) Correlation matrix difference
    corr_diff = (X_train_df.corr(numeric_only=True) - X_synth.corr(numeric_only=True)).abs().mean().mean()

    # (3) Real vs synth classifier AUC
    Z = pd.concat([X_train_df.assign(_isreal=1), X_synth.assign(_isreal=0)], ignore_index=True)
    yZ = Z['_isreal'].to_numpy(dtype=int)
    Z = Z.drop(columns=['_isreal'])

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    auc = float(cross_val_score(clf, Z, yZ, cv=5, scoring='roc_auc').mean())

    summary = {
        'multiplier': int(args.multiplier),
        'seed': int(args.seed),
        'backend': backend_used,
        'mean_abs_corr_diff': float(corr_diff),
        'real_vs_synth_auc': auc,
        'mean_ks_p': float(ks_df['ks_p'].mean()),
        'median_ks_p': float(ks_df['ks_p'].median()),
    }

    ks_out = qc_dir / f'ks_{args.multiplier}x_{backend_used}_seed{args.seed}.csv'
    ks_df.to_csv(ks_out, index=False)

    summary_out = qc_dir / f'synth_qc_{args.multiplier}x_{backend_used}_seed{args.seed}.csv'
    pd.DataFrame([summary]).to_csv(summary_out, index=False)

    print('Saved KS table:', ks_out)
    print('Saved QC summary:', summary_out)


if __name__ == '__main__':
    main()

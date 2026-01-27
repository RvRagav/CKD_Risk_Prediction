from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

# Allow running as: python src/stability.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir  # noqa: E402


def mean_pairwise_spearman(shap_runs_for_sample: list[np.ndarray]) -> float:
    R = len(shap_runs_for_sample)
    vals: list[float] = []
    for i in range(R):
        for j in range(i + 1, R):
            rho, _ = spearmanr(np.abs(shap_runs_for_sample[i]), np.abs(shap_runs_for_sample[j]))
            vals.append(float(np.asarray(rho).reshape(-1)[0].item()))
    return float(np.mean(vals)) if vals else float('nan')


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    set_a = set(np.argsort(-np.abs(a))[:k])
    set_b = set(np.argsort(-np.abs(b))[:k])
    denom = len(set_a | set_b)
    return float(len(set_a & set_b) / denom) if denom else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute stability metrics from saved SHAP runs.')
    parser.add_argument('--glob', type=str, required=True, help='Glob like results/shap_rf_baseline_seed*.npy')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--out', type=str, default='results/stability.csv')
    args = parser.parse_args()

    paths = sorted(Path('.').glob(args.glob))
    if not paths:
        raise FileNotFoundError('No SHAP files matched --glob')

    norm_runs = []
    for p in paths:
        r = np.load(p, allow_pickle=True)
        # If saved object-array scalar containing a list/other object
        if isinstance(r, np.ndarray) and r.dtype == object and r.shape == ():
            r = r.item()
        # If list of class arrays
        if isinstance(r, list):
            r = r[1] if len(r) > 1 else r[0]
        r = np.asarray(r)
        if r.ndim == 3:
            r = r[1]
        norm_runs.append(r)

    n_samples = norm_runs[0].shape[0]
    n_features = norm_runs[0].shape[1]

    spearman_list = []
    jaccard_list = []

    for i in range(n_samples):
        shap_runs_for_sample = [r[i, :] for r in norm_runs]
        spearman_list.append(mean_pairwise_spearman(shap_runs_for_sample))

        jac_vals = []
        for a in range(len(shap_runs_for_sample)):
            for b in range(a + 1, len(shap_runs_for_sample)):
                jac_vals.append(jaccard_topk(shap_runs_for_sample[a], shap_runs_for_sample[b], k=args.k))
        jaccard_list.append(float(np.mean(jac_vals)) if jac_vals else float('nan'))

    out_df = pd.DataFrame({
        'sample_idx': np.arange(n_samples),
        'mean_pairwise_spearman': spearman_list,
        f'mean_pairwise_jaccard_top{args.k}': jaccard_list,
    })

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    out_df.to_csv(out_path, index=False)
    print('Saved stability:', out_path)

    print('Summary Spearman mean:', float(np.nanmean(out_df['mean_pairwise_spearman'])))
    print('Summary Jaccard mean:', float(np.nanmean(out_df[f'mean_pairwise_jaccard_top{args.k}'])))


if __name__ == '__main__':
    main()

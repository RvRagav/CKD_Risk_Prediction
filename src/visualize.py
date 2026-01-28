from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Allow running as: python src/visualize.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PREPROC_DIR, RESULTS_DIR, SPLIT_DIR  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

# Set style
try:
    sns.set_theme(style='whitegrid', palette='colorblind')
except AttributeError:
    # Fallback for older seaborn versions
    sns.set_style('whitegrid')
    sns.set_palette('colorblind')

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f'Skipping {path.name}: failed to read CSV ({e})')
        return None


def _discover_metrics(results_dir: Path) -> pd.DataFrame:
    """Load and combine all metrics_*.csv files in results_dir."""
    frames: list[pd.DataFrame] = []
    for p in sorted(results_dir.glob('metrics_*.csv')):
        df = _safe_read_csv(p)
        if df is None or df.empty:
            continue
        df = df.copy()
        df['run'] = p.stem.replace('metrics_', '', 1)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    for col in ['roc_auc', 'precision', 'recall', 'f1']:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _load_test_split() -> tuple[pd.DataFrame, np.ndarray] | None:
    X_path = Path(PREPROC_DIR) / 'X_test_preproc.csv'
    y_path = Path(SPLIT_DIR) / 'y_test.csv'
    if not X_path.exists() or not y_path.exists():
        return None
    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path)['target'].to_numpy()
    return X_test, y_test


def _parse_model_variant_from_filename(name: str) -> tuple[str, str] | None:
    if not name.endswith('.joblib'):
        return None
    stem = name[:-7]
    parts = stem.split('_')
    if len(parts) < 2:
        return None
    model = parts[0]
    variant = '_'.join(parts[1:])
    return model, variant


def plot_model_comparison(results_dir: Path, output_dir: Path) -> None:
    """Compare model performance across all available runs (metrics_*.csv)."""

    metrics = _discover_metrics(results_dir)
    if metrics.empty:
        print('Skipping model comparison: no metrics_*.csv found')
        return

    run_order = list(dict.fromkeys(metrics['run'].tolist()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison (All Runs)', fontsize=16, fontweight='bold', y=0.995)

    plot_metrics = ['roc_auc', 'f1', 'precision', 'recall']
    titles = ['ROC-AUC', 'F1', 'Precision', 'Recall']

    for ax, metric, title in zip(axes.flat, plot_metrics, titles):
        df = metrics[['model', 'run', metric]].dropna()
        if df.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=df, x='model', y=metric, hue='run', hue_order=run_order, ax=ax)
        ax.set_ylabel(title)
        ax.set_xlabel('Model')
        ax.set_title(title, fontsize=12, pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def plot_synthetic_quality(results_dir: Path, output_dir: Path, synth_tag: str | None = None) -> None:
    """Visualize synthetic data quality metrics (best-effort)."""

    qc_files = sorted(results_dir.glob('synth_qc_*.csv'))
    ks_files = sorted(results_dir.glob('ks_*.csv'))
    if not qc_files or not ks_files:
        print('Skipping synthetic quality: missing synth_qc_*.csv or ks_*.csv')
        return

    def _pick(files: list[Path]) -> Path:
        if synth_tag:
            tagged = [p for p in files if synth_tag in p.stem]
            if tagged:
                return tagged[0]
        for preferred in ['3x_sdv_gcopula', '1x_sdv_gcopula', '3x', '1x']:
            cand = [p for p in files if preferred in p.stem]
            if cand:
                return cand[0]
        return files[0]

    qc_path = _pick(qc_files)
    ks_path = _pick(ks_files)

    qc = _safe_read_csv(qc_path)
    ks = _safe_read_csv(ks_path)
    if qc is None or ks is None or qc.empty or ks.empty:
        print('Skipping synthetic quality: selected QC/KS file empty or unreadable')
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Synthetic Data Quality Assessment ({qc_path.stem})',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Quality metrics summary
    ax1 = axes[0, 0]
    def _get0(col: str) -> float:
        return float(qc[col].iloc[0]) if col in qc.columns else float('nan')

    metrics_data = {
        'Mean Abs\nCorr Diff': _get0('mean_abs_corr_diff'),
        'Real vs Synth\nAUC': _get0('real_vs_synth_auc'),
        'Mean KS\np-value': _get0('mean_ks_p'),
        'Median KS\np-value': _get0('median_ks_p'),
    }
    
    bars = ax1.barh(list(metrics_data.keys()), list(metrics_data.values()), 
                    color=['#2ecc71', '#e74c3c', '#3498db', '#9b59b6'], alpha=0.7)
    ax1.set_xlabel('Value', fontsize=11, fontweight='bold')
    ax1.set_title('Quality Metrics Summary', fontsize=12, pad=10)
    ax1.set_xlim((0, 1))
    ax1.grid(axis='x', alpha=0.3)
    
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 2. KS test p-values distribution
    ax2 = axes[0, 1]
    ks_sorted = ks.sort_values('ks_p', ascending=False)
    colors = ['green' if p > 0.05 else 'orange' if p > 0.01 else 'red' 
              for p in ks_sorted['ks_p']]
    
    ax2.barh(range(len(ks_sorted)), ks_sorted['ks_p'], color=colors, alpha=0.6)
    ax2.set_xlabel('KS Test p-value', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Feature', fontsize=11, fontweight='bold')
    ax2.set_title('KS Test Results by Feature', fontsize=12, pad=10)
    ax2.axvline(0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax2.set_yticks(range(len(ks_sorted)))
    ax2.set_yticklabels(ks_sorted['feature'], fontsize=7)
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. KS statistics histogram
    ax3 = axes[1, 0]
    ax3.hist(ks['ks_stat'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('KS Statistic', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of KS Statistics', fontsize=12, pad=10)
    ax3.axvline(ks['ks_stat'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {ks["ks_stat"].mean():.3f}')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. P-value significance count
    ax4 = axes[1, 1]
    sig_counts = {
        'p > 0.05\n(Good)': (ks['ks_p'] > 0.05).sum(),
        '0.01 < p ≤ 0.05\n(Moderate)': ((ks['ks_p'] > 0.01) & (ks['ks_p'] <= 0.05)).sum(),
        'p ≤ 0.01\n(Different)': (ks['ks_p'] <= 0.01).sum()
    }
    
    wedges, texts, autotexts = ax4.pie(
        list(sig_counts.values()), 
        labels=list(sig_counts.keys()),
        colors=['#2ecc71', '#f39c12', '#e74c3c'],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0, 0)
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax4.set_title('Feature Distribution Similarity', fontsize=12, pad=10)
    
    plt.tight_layout()
    tag = synth_tag or qc_path.stem.replace('synth_qc_', '', 1)
    output_path = output_dir / f'synthetic_quality_{tag}.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def plot_feature_distributions(data_dir: Path, results_dir: Path, output_dir: Path) -> None:
    """Compare real vs synthetic feature distributions."""

    X_train_path = Path(PREPROC_DIR) / 'X_train_preproc.csv'
    if not X_train_path.exists():
        print('Skipping feature distributions: missing X_train_preproc.csv')
        return

    synth_files = sorted((data_dir / 'synthetic').glob('X_synth_*_preproc.csv'))
    ks_files = sorted(results_dir.glob('ks_*.csv'))
    if not synth_files or not ks_files:
        print('Skipping feature distributions: missing X_synth_*_preproc.csv or ks_*.csv')
        return

    preferred = ['3x_sdv_gcopula', '1x_sdv_gcopula', '3x', '1x']
    X_synth_path = None
    ks_path = None
    for tag in preferred:
        if X_synth_path is None:
            for p in synth_files:
                if tag in p.stem:
                    X_synth_path = p
                    break
        if ks_path is None:
            for p in ks_files:
                if tag in p.stem:
                    ks_path = p
                    break
        if X_synth_path is not None and ks_path is not None:
            break
    X_synth_path = X_synth_path or synth_files[0]
    ks_path = ks_path or ks_files[0]

    X_train = pd.read_csv(X_train_path)
    X_synth = pd.read_csv(X_synth_path)
    ks = pd.read_csv(ks_path)
    
    # Select top 6 features by KS p-value (best matches) and bottom 6 (worst matches)
    ks_sorted = ks.sort_values('ks_p', ascending=False)
    top_features = ks_sorted.head(6)['feature'].tolist()
    bottom_features = ks_sorted.tail(6)['feature'].tolist()
    
    # Plot best matches
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Distributions: Best Matches (Real vs Synthetic)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for ax, feat in zip(axes.flat, top_features):
        if feat in X_train.columns and feat in X_synth.columns:
            ax.hist(X_train[feat], bins=30, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(X_synth[feat], bins=30, alpha=0.5, label='Synthetic', color='orange', density=True)
            
            # Get KS stats
            ks_stat = ks[ks['feature'] == feat]['ks_stat'].values[0]
            ks_p = ks[ks['feature'] == feat]['ks_p'].values[0]
            
            ax.set_title(f'{feat}\nKS p={ks_p:.4f}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'feature_distributions_best_{X_synth_path.stem}.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()
    
    # Plot worst matches
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Distributions: Challenging Features (Real vs Synthetic)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for ax, feat in zip(axes.flat, bottom_features):
        if feat in X_train.columns and feat in X_synth.columns:
            ax.hist(X_train[feat], bins=30, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(X_synth[feat], bins=30, alpha=0.5, label='Synthetic', color='orange', density=True)
            
            ks_stat = ks[ks['feature'] == feat]['ks_stat'].values[0]
            ks_p = ks[ks['feature'] == feat]['ks_p'].values[0]
            
            ax.set_title(f'{feat}\nKS p={ks_p:.4f}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'feature_distributions_challenging_{X_synth_path.stem}.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def plot_performance_summary(results_dir: Path, output_dir: Path) -> None:
    """Create a comprehensive summary visualization."""

    metrics = _discover_metrics(results_dir)
    if metrics.empty:
        print('Skipping performance summary: no metrics_*.csv found')
        return
    
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, top=0.95, bottom=0.05, left=0.05, right=0.98)
    
    fig.suptitle('CKD Risk Prediction: Complete Performance Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Main heatmap
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    combined = metrics.copy()
    combined['label'] = combined['model'].astype(str).str.upper() + '\n' + combined['run'].astype(str)

    metrics_for_heatmap = ['roc_auc', 'precision', 'recall', 'f1', 'brier', 'ece']
    heatmap_df = combined.pivot_table(index='label', values=metrics_for_heatmap, aggfunc=lambda s: s.iloc[0])
    heatmap_df = heatmap_df.rename(
        columns={
            'roc_auc': 'ROC-AUC',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'brier': 'Brier',
            'ece': 'ECE',
        }
    )

    sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax_main, cbar_kws={'label': 'Score'})
    ax_main.set_title('Performance + Calibration Heatmap', fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('')
    ax_main.set_ylabel('Model + Run', fontsize=11, fontweight='bold')
    
    # ROC-AUC trend for up to 3 runs
    ax1 = fig.add_subplot(gs[0, 2])

    runs = list(dict.fromkeys(combined['run'].tolist()))[:3]
    models = sorted(combined['model'].unique().tolist())
    x_pos = np.arange(len(models))

    for run in runs:
        yv = [
            float(combined[(combined['model'] == m) & (combined['run'] == run)]['roc_auc'].iloc[0])
            if not combined[(combined['model'] == m) & (combined['run'] == run)].empty else np.nan
            for m in models
        ]
        ax1.plot(x_pos, yv, marker='o', linewidth=2, label=run)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.set_ylabel('ROC-AUC', fontweight='bold')
    ax1.set_title('ROC-AUC', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # F1 trend
    ax2 = fig.add_subplot(gs[1, 2])

    for run in runs:
        yv = [
            float(combined[(combined['model'] == m) & (combined['run'] == run)]['f1'].iloc[0])
            if not combined[(combined['model'] == m) & (combined['run'] == run)].empty else np.nan
            for m in models
        ]
        ax2.plot(x_pos, yv, marker='o', linewidth=2, label=run)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Performance delta between the first two runs (if available)
    ax3 = fig.add_subplot(gs[2, :])

    if len(runs) >= 2:
        base_run, other_run = runs[0], runs[1]
        metrics_to_compare = ['roc_auc', 'precision', 'recall', 'f1']
        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics_to_compare):
            base_vals = np.array([
                float(combined[(combined['model'] == m) & (combined['run'] == base_run)][metric].iloc[0])
                if not combined[(combined['model'] == m) & (combined['run'] == base_run)].empty else np.nan
                for m in models
            ])
            other_vals = np.array([
                float(combined[(combined['model'] == m) & (combined['run'] == other_run)][metric].iloc[0])
                if not combined[(combined['model'] == m) & (combined['run'] == other_run)].empty else np.nan
                for m in models
            ])
            delta = (other_vals - base_vals) * 100
            ax3.bar(x + i * width, delta, width, label=metric.upper(), alpha=0.7)

        ax3.set_title(f'Performance Delta (%): {other_run} vs {base_run}', fontsize=12, fontweight='bold')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels([m.upper() for m in models])
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.legend(loc='upper right', ncol=4)
        ax3.grid(axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Need at least 2 runs for delta plot', ha='center', va='center')
        ax3.set_axis_off()
    
    if len(runs) >= 2:
        ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Performance Change (%)', fontsize=11, fontweight='bold')
    
    output_path = output_dir / 'performance_summary.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate visualizations for CKD prediction results.')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--output-dir', type=str, default='results/visualizations')
    parser.add_argument('--synth-tag', type=str, default=None, help='Substring to select which synth run to visualize')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    
    print('=' * 60)
    print('CKD Risk Prediction - Results Visualization')
    print('=' * 60)
    
    print('\n1. Generating model comparison plots...')
    plot_model_comparison(results_dir, output_dir)
    
    print('\n2. Generating synthetic quality assessment...')
    plot_synthetic_quality(results_dir, output_dir, synth_tag=args.synth_tag)
    
    print('\n3. Generating feature distribution comparisons...')
    plot_feature_distributions(data_dir, results_dir, output_dir)
    
    print('\n4. Generating performance summary...')
    plot_performance_summary(results_dir, output_dir)
    
    print('\n' + '=' * 60)
    print(f'✓ All visualizations saved to: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()

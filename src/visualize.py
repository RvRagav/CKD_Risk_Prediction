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

    # Filter to only feat6 for final results
    metrics = metrics[metrics['run'] == 'feat6'].copy()
    
    if metrics.empty:
        print('Skipping model comparison: no feat6 metrics found')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.98)

    plot_metrics = ['roc_auc', 'f1', 'precision', 'recall']
    titles = ['ROC-AUC', 'F1 Score', 'Precision', 'Recall']

    for ax, metric, title in zip(axes.flat, plot_metrics, titles):
        df = metrics[['model', metric]].dropna()
        if df.empty:
            ax.set_visible(False)
            continue
        
        # Create bar plot
        bars = ax.bar(df['model'], df[metric], alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        # Set y-axis to show differences clearly
        y_min = max(0.75, df[metric].min() - 0.05)
        y_max = min(1.0, df[metric].max() + 0.05)
        ax.set_ylim([y_min, y_max])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Uppercase model names
        ax.set_xticks(range(len(df['model'])))
        ax.set_xticklabels([m.upper() for m in df['model']])

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_synthetic_quality(results_dir: Path, output_dir: Path, synth_tag: str | None = None) -> None:
    """Visualize synthetic data quality metrics (best-effort)."""

    qc_files = sorted(list(results_dir.glob('synth_qc_*.csv')) + list(results_dir.glob('qc_report_*.csv')))
    ks_files = sorted(results_dir.glob('ks_*.csv'))
    if not qc_files:
        print('Skipping synthetic quality: missing synth_qc_*.csv or qc_report_*.csv')
        return

    def _pick(files: list[Path]) -> Path:
        if synth_tag:
            tagged = [p for p in files if synth_tag in p.stem]
            if tagged:
                return tagged[0]
        for preferred in ['3x_gcopula', '3x_sdv_gcopula', '1x_sdv_gcopula', '3x', '1x']:
            cand = [p for p in files if preferred in p.stem]
            if cand:
                return cand[0]
        return files[0]

    qc_path = _pick(qc_files)
    ks_path = _pick(ks_files) if ks_files else None

    qc = _safe_read_csv(qc_path)
    ks = _safe_read_csv(ks_path) if ks_path else None
    if qc is None or ks is None or qc.empty or ks.empty:
        print('Skipping synthetic quality: selected QC/KS file empty or unreadable')
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'Synthetic Data Quality Assessment ({qc_path.stem})',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 1. Quality metrics summary
    ax1 = axes[0, 0]
    def _get0(col: str) -> float:
        return float(qc[col].iloc[0]) if col in qc.columns else float('nan')

    metrics_data = {
        'Mean Abs\nCorr Diff': _get0('mean_abs_corr_diff'),
        'Real vs Synth\nAUC': _get0('real_vs_synth_auc'),
        'Avg KS\nStatistic': _get0('avg_ks'),
    }
    
    # Add KS p-value metrics only if available
    if 'mean_ks_p' in qc.columns:
        metrics_data['Mean KS\np-value'] = _get0('mean_ks_p')
    if 'median_ks_p' in qc.columns:
        metrics_data['Median KS\np-value'] = _get0('median_ks_p')
    
    bars = ax1.barh(list(metrics_data.keys()), list(metrics_data.values()), 
                    color=['#2ecc71', '#e74c3c', '#3498db', '#9b59b6'], alpha=0.7)
    ax1.set_xlabel('Value', fontsize=10, fontweight='bold')
    ax1.set_title('Quality Metrics Summary', fontsize=11, pad=10)
    ax1.set_xlim((0, 1.15))  # Extended to accommodate text labels
    ax1.grid(axis='x', alpha=0.3)
    
    # Adjust y-axis label font size
    ax1.tick_params(axis='y', labelsize=9)
    
    for bar in bars:
        width = bar.get_width()
        if not np.isnan(width):
            # Position text inside bar if width > 0.5, else outside
            if width > 0.5:
                ax1.text(width - 0.02, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', ha='right', va='center', fontsize=9, 
                        fontweight='bold', color='white')
            else:
                ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 2-4. KS test plots (only if KS data is available)
    if ks is not None and not ks.empty and 'ks_p' in ks.columns:
        # 2. KS test p-values distribution
        ax2 = axes[0, 1]
        ks_sorted = ks.sort_values('ks_p', ascending=False)
        colors = ['green' if p > 0.05 else 'orange' if p > 0.01 else 'red' 
                  for p in ks_sorted['ks_p']]
        
        ax2.barh(range(len(ks_sorted)), ks_sorted['ks_p'], color=colors, alpha=0.6)
        ax2.set_xlabel('KS Test p-value', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Feature', fontsize=10, fontweight='bold')
        ax2.set_title('KS Test Results by Feature', fontsize=11, pad=10)
        ax2.axvline(0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
        ax2.set_yticks(range(len(ks_sorted)))
        ax2.set_yticklabels(ks_sorted['feature'], fontsize=6)
        ax2.legend(fontsize=8, loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. KS statistics histogram
        ax3 = axes[1, 0]
        ax3.hist(ks['ks_stat'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('KS Statistic', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax3.set_title('Distribution of KS Statistics', fontsize=11, pad=10)
        ax3.axvline(ks['ks_stat'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {ks["ks_stat"].mean():.3f}')
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='both', labelsize=9)
        
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
            explode=(0.05, 0, 0),
            textprops={'fontsize': 9}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax4.set_title('Feature Distribution Similarity', fontsize=11, pad=10)
    else:
        # Hide KS-specific plots if data not available
        for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
            ax.text(0.5, 0.5, 'KS data not available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    tag = synth_tag or qc_path.stem.replace('synth_qc_', '', 1).replace('qc_report_', '')
    output_path = output_dir / f'synthetic_quality_{tag}.png'
    plt.savefig(output_path, bbox_inches='tight')
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
    if not synth_files:
        print('Skipping feature distributions: missing X_synth_*_preproc.csv')
        return

    preferred = ['3x_gcopula', '3x_sdv_gcopula', '1x_sdv_gcopula', '3x', '1x']
    X_synth_path = None
    ks_path = None
    for tag in preferred:
        if X_synth_path is None:
            for p in synth_files:
                if tag in p.stem:
                    X_synth_path = p
                    break
        if ks_path is None and ks_files:
            for p in ks_files:
                if tag in p.stem:
                    ks_path = p
                    break
        if X_synth_path is not None and (ks_path is not None or not ks_files):
            break
    X_synth_path = X_synth_path or synth_files[0]
    ks_path = ks_path or (ks_files[0] if ks_files else None)

    X_train = pd.read_csv(X_train_path)
    X_synth = pd.read_csv(X_synth_path)
    
    # Get common features between train and synth
    common_features = [col for col in X_train.columns if col in X_synth.columns]
    
    if ks_path and ks_path.exists():
        ks = pd.read_csv(ks_path)
        # Select top 6 features by KS p-value (best matches) and bottom 6 (worst matches)
        ks_sorted = ks.sort_values('ks_p', ascending=False)
        top_features = ks_sorted.head(6)['feature'].tolist()
        bottom_features = ks_sorted.tail(6)['feature'].tolist()
    else:
        # Without KS data, just pick first 6 and last 6 features
        ks = None
        top_features = common_features[:6]
        bottom_features = common_features[-6:]
    
    # Plot best matches
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Distributions: Best Matches (Real vs Synthetic)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    for ax, feat in zip(axes.flat, top_features):
        if feat in X_train.columns and feat in X_synth.columns:
            ax.hist(X_train[feat], bins=30, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(X_synth[feat], bins=30, alpha=0.5, label='Synthetic', color='orange', density=True)
            
            # Get KS stats if available
            if ks is not None and not ks.empty:
                ks_row = ks[ks['feature'] == feat]
                if not ks_row.empty and 'ks_p' in ks_row.columns:
                    ks_p = ks_row['ks_p'].values[0]
                    ax.set_title(f'{feat}\nKS p={ks_p:.4f}', fontsize=10, fontweight='bold')
                else:
                    ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
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
                 fontsize=14, fontweight='bold', y=0.98)
    
    for ax, feat in zip(axes.flat, bottom_features):
        if feat in X_train.columns and feat in X_synth.columns:
            ax.hist(X_train[feat], bins=30, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(X_synth[feat], bins=30, alpha=0.5, label='Synthetic', color='orange', density=True)
            
            # Get KS stats if available
            if ks is not None and not ks.empty:
                ks_row = ks[ks['feature'] == feat]
                if not ks_row.empty and 'ks_p' in ks_row.columns:
                    ks_p = ks_row['ks_p'].values[0]
                    ax.set_title(f'{feat}\nKS p={ks_p:.4f}', fontsize=10, fontweight='bold')
                else:
                    ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'{feat}', fontsize=10, fontweight='bold')
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
    
    # Filter to only feat6 for final results
    metrics = metrics[metrics['run'] == 'feat6'].copy()
    
    if metrics.empty:
        print('Skipping performance summary: no feat6 metrics found')
        return
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, top=0.92, bottom=0.08, left=0.08, right=0.95)
    
    fig.suptitle('CKD Risk Prediction: Model Performance', 
                 fontsize=18, fontweight='bold', y=0.97)
    
    # Main heatmap (top-left, spanning 2 columns)
    ax_main = fig.add_subplot(gs[0, :])
    
    combined = metrics.copy()
    combined['model'] = combined['model'].str.upper()

    metrics_for_heatmap = ['roc_auc', 'precision', 'recall', 'f1']
    heatmap_df = combined.set_index('model')[metrics_for_heatmap]
    heatmap_df = heatmap_df.rename(
        columns={
            'roc_auc': 'ROC-AUC',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1-Score',
        }
    )

    sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax_main, 
                cbar_kws={'label': 'Score'}, vmin=0.75, vmax=1.0)
    ax_main.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    # ROC-AUC comparison (bottom-left)
    ax1 = fig.add_subplot(gs[1, 0])
    models = sorted(combined['model'].unique().tolist())
    auc_vals = [combined[combined['model'] == m]['roc_auc'].iloc[0] for m in models]
    
    bars = ax1.bar(models, auc_vals, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylim((max(0.75, min(auc_vals) - 0.05), min(1.0, max(auc_vals) + 0.05)))
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('ROC-AUC', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_title('ROC-AUC Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # F1 comparison (bottom-right)
    ax2 = fig.add_subplot(gs[1, 1])
    f1_vals = [combined[combined['model'] == m]['f1'].iloc[0] for m in models]
    
    bars = ax2.bar(models, f1_vals, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylim((max(0.75, min(f1_vals) - 0.05), min(1.0, max(f1_vals) + 0.05)))
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax2.set_title('F1 Score Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    output_path = output_dir / 'performance_summary.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
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
    
    print('\n2. Generating performance summary...')
    plot_performance_summary(results_dir, output_dir)
    
    print('\n3. Generating synthetic quality assessment...')
    plot_synthetic_quality(results_dir, output_dir, synth_tag=args.synth_tag)
    
    print('\n4. Generating feature distribution comparisons...')
    plot_feature_distributions(data_dir, results_dir, output_dir)
    
    print('\n' + '=' * 60)
    print(f'✓ All visualizations saved to: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Allow running as: python src/visualize.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def plot_model_comparison(results_dir: Path, output_dir: Path) -> None:
    """Compare model performance: baseline vs synthetic augmentation."""
    
    # Load metrics
    baseline_path = results_dir / 'metrics_baseline.csv'
    synth_3x_path = results_dir / 'metrics_3x_sdv_gcopula.csv'
    
    if not baseline_path.exists() or not synth_3x_path.exists():
        print('Skipping model comparison: missing metrics files')
        return
    
    baseline = pd.read_csv(baseline_path)
    synth_3x = pd.read_csv(synth_3x_path)
    
    # Combine data
    baseline['dataset'] = 'Baseline'
    synth_3x['dataset'] = '3x Synthetic'
    combined = pd.concat([baseline, synth_3x], ignore_index=True)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison: Baseline vs 3x Synthetic Augmentation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['roc_auc', 'f1', 'precision', 'recall']
    titles = ['ROC-AUC Score', 'F1 Score', 'Precision', 'Recall']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        # Group by model and dataset
        plot_data = combined.pivot(index='model', columns='dataset', values=metric)
        
        x = np.arange(len(plot_data.index))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, plot_data['Baseline'], width, 
                       label='Baseline', alpha=0.8, color='#1f77b4')
        bars2 = ax.bar(x + width/2, plot_data['3x Synthetic'], width, 
                       label='3x Synthetic', alpha=0.8, color='#ff7f0e')
        
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in plot_data.index])
        ax.legend(loc='lower right')
        ax.set_ylim((0.94, 1.005))
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'model_comparison.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def plot_synthetic_quality(results_dir: Path, output_dir: Path) -> None:
    """Visualize synthetic data quality metrics."""
    
    # Load quality metrics
    qc_path = results_dir / 'synth_qc_3x_sdv_gcopula_seed0.csv'
    ks_path = results_dir / 'ks_3x_sdv_gcopula_seed0.csv'
    
    if not qc_path.exists() or not ks_path.exists():
        print('Skipping synthetic quality: missing QC files')
        return
    
    qc = pd.read_csv(qc_path)
    ks = pd.read_csv(ks_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Synthetic Data Quality Assessment (3x Gaussian Copula)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Quality metrics summary
    ax1 = axes[0, 0]
    metrics_data = {
        'Mean Abs\nCorr Diff': qc['mean_abs_corr_diff'].values[0],
        'Real vs Synth\nClassifier AUC': qc['real_vs_synth_auc'].values[0],
        'Mean KS\np-value': qc['mean_ks_p'].values[0],
        'Median KS\np-value': qc['median_ks_p'].values[0]
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
    output_path = output_dir / 'synthetic_quality.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def plot_brier_score_comparison(results_dir: Path, output_dir: Path) -> None:
    """Compare Brier scores (calibration metric)."""
    
    baseline_path = results_dir / 'metrics_baseline.csv'
    synth_3x_path = results_dir / 'metrics_3x_sdv_gcopula.csv'
    
    if not baseline_path.exists() or not synth_3x_path.exists():
        print('Skipping Brier score comparison: missing metrics files')
        return
    
    baseline = pd.read_csv(baseline_path)
    synth_3x = pd.read_csv(synth_3x_path)
    
    baseline['dataset'] = 'Baseline'
    synth_3x['dataset'] = '3x Synthetic'
    combined = pd.concat([baseline, synth_3x], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot and plot
    plot_data = combined.pivot(index='model', columns='dataset', values='brier')
    
    x = np.arange(len(plot_data.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, plot_data['Baseline'], width, 
                   label='Baseline', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, plot_data['3x Synthetic'], width, 
                   label='3x Synthetic', alpha=0.8, color='#ff7f0e')
    
    ax.set_ylabel('Brier Score (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Calibration: Brier Score Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in plot_data.index])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / 'brier_score_comparison.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def plot_feature_distributions(data_dir: Path, results_dir: Path, output_dir: Path) -> None:
    """Compare real vs synthetic feature distributions."""
    
    # Load preprocessed data
    X_train_path = data_dir / 'processed' / 'preprocessed' / 'X_train_preproc.csv'
    X_synth_path = data_dir / 'synthetic' / 'X_synth_3x_sdv_gcopula_preproc.csv'
    ks_path = results_dir / 'ks_3x_sdv_gcopula_seed0.csv'
    
    if not all([X_train_path.exists(), X_synth_path.exists(), ks_path.exists()]):
        print('Skipping feature distributions: missing data files')
        return
    
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
    output_path = output_dir / 'feature_distributions_best.png'
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
    output_path = output_dir / 'feature_distributions_challenging.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def plot_performance_summary(results_dir: Path, output_dir: Path) -> None:
    """Create a comprehensive summary visualization."""
    
    baseline_path = results_dir / 'metrics_baseline.csv'
    synth_3x_path = results_dir / 'metrics_3x_sdv_gcopula.csv'
    
    if not baseline_path.exists() or not synth_3x_path.exists():
        print('Skipping performance summary: missing metrics files')
        return
    
    baseline = pd.read_csv(baseline_path)
    synth_3x = pd.read_csv(synth_3x_path)
    
    fig = plt.figure(figsize=(16, 10), constrained_layout=False)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, top=0.95, bottom=0.05, left=0.05, right=0.98)
    
    fig.suptitle('CKD Risk Prediction: Complete Performance Summary', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Main heatmap
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    
    baseline['dataset'] = 'Baseline'
    synth_3x['dataset'] = '3x Synth'
    combined = pd.concat([baseline, synth_3x], ignore_index=True)
    
    # Create pivot table for heatmap
    metrics_for_heatmap = ['roc_auc', 'precision', 'recall', 'f1', 'brier']
    heatmap_data = []
    
    for model in ['lr', 'rf', 'xgb']:
        for dataset in ['Baseline', '3x Synth']:
            row_data = combined[(combined['model'] == model) & (combined['dataset'] == dataset)]
            if len(row_data) > 0:
                values = [row_data[m].values[0] for m in metrics_for_heatmap]
                heatmap_data.append(values)
    
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=[f'{m.upper()}\n{d}' for m in ['lr', 'rf', 'xgb'] for d in ['Baseline', '3x Synth']],
        columns=['ROC-AUC', 'Precision', 'Recall', 'F1', 'Brier']
    )
    
    sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', 
                center=0.97, vmin=0.94, vmax=1.0, ax=ax_main, cbar_kws={'label': 'Score'})
    ax_main.set_title('Performance Metrics Heatmap', fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('')
    ax_main.set_ylabel('Model + Dataset', fontsize=11, fontweight='bold')
    
    # ROC-AUC comparison
    ax1 = fig.add_subplot(gs[0, 2])
    models = baseline['model'].values
    x_pos = np.arange(len(models))
    
    ax1.plot(x_pos, baseline['roc_auc'].to_numpy(), 'o-', label='Baseline', linewidth=2, markersize=8)
    ax1.plot(x_pos, synth_3x['roc_auc'].to_numpy(), 's-', label='3x Synth', linewidth=2, markersize=8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.set_ylabel('ROC-AUC', fontweight='bold')
    ax1.set_title('ROC-AUC', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim((0.99, 1.002))
    ax1.grid(alpha=0.3)
    
    # F1 Score comparison
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.plot(x_pos, baseline['f1'].to_numpy(), 'o-', label='Baseline', linewidth=2, markersize=8)
    ax2.plot(x_pos, synth_3x['f1'].to_numpy(), 's-', label='3x Synth', linewidth=2, markersize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_ylim((0.975, 0.992))
    ax2.grid(alpha=0.3)
    
    # Performance delta
    ax3 = fig.add_subplot(gs[2, :])
    
    metrics_to_compare = ['roc_auc', 'precision', 'recall', 'f1']
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_compare):
        delta = (synth_3x[metric].to_numpy() - baseline[metric].to_numpy()) * 100
        color = ['green' if d >= 0 else 'red' for d in delta]
        ax3.bar(x + i*width, delta, width, label=metric.upper(), alpha=0.7)
    
    ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Performance Change (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Performance Delta: Synthetic vs Baseline', fontsize=12, fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([m.upper() for m in models])
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.legend(loc='upper right', ncol=4)
    ax3.grid(axis='y', alpha=0.3)
    
    output_path = output_dir / 'performance_summary.png'
    plt.savefig(output_path)
    print(f'Saved: {output_path}')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate visualizations for CKD prediction results.')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='results/visualizations')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    
    print('=' * 60)
    print('CKD Risk Prediction - Results Visualization')
    print('=' * 60)
    
    print('\n1. Generating model comparison plots...')
    plot_model_comparison(results_dir, output_dir)
    
    print('\n2. Generating synthetic quality assessment...')
    plot_synthetic_quality(results_dir, output_dir)
    
    print('\n3. Generating Brier score comparison...')
    plot_brier_score_comparison(results_dir, output_dir)
    
    print('\n4. Generating feature distribution comparisons...')
    plot_feature_distributions(data_dir, results_dir, output_dir)
    
    print('\n5. Generating performance summary...')
    plot_performance_summary(results_dir, output_dir)
    
    print('\n' + '=' * 60)
    print(f'✓ All visualizations saved to: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()

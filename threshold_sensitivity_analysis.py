#!/usr/bin/env python3
"""
Threshold Sensitivity Analysis for Dominating Set Evaluation
- ROC curve (TPR vs FPR)
- PR curve (Precision vs Recall)
- Performance change analysis by threshold
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_threshold_results(exp_dir):
    """Load threshold-specific results from experiment result directory (all binaries)"""
    results = []
    
    # threshold 디렉토리들을 찾기
    exp_path = Path(exp_dir)
    threshold_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('threshold_')])
    
    # Binary list
    binaries = ['bool', 'direvent', 'gmp', 'libcrypto', 'libssl']
    
    for threshold_dir in threshold_dirs:
        threshold = int(threshold_dir.name.split('_')[1])
        
        # Collect results from all binaries
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_tps = []
        all_fps = []
        all_fns = []
        all_dom_sizes = []
        
        for binary in binaries:
            binary_dir = threshold_dir / binary
            if binary_dir.exists():
                metrics_file = binary_dir / 'dominating_set_metrics.csv'
                if metrics_file.exists():
                    try:
                        df = pd.read_csv(metrics_file)
                        if len(df) > 0:
                            # Calculate average for each binary
                            avg_precision = df['Precision'].mean()
                            avg_recall = df['Recall'].mean()
                            avg_f1 = df['F1-score'].mean()
                            total_tp = df['TP'].sum()
                            total_fp = df['FP'].sum()
                            total_fn = df['FN'].sum()
                            avg_dom_size = df['Dominating Set Size'].mean()
                            
                            all_precisions.append(avg_precision)
                            all_recalls.append(avg_recall)
                            all_f1s.append(avg_f1)
                            all_tps.append(total_tp)
                            all_fps.append(total_fp)
                            all_fns.append(total_fn)
                            all_dom_sizes.append(avg_dom_size)
                    except Exception as e:
                        print(f"Error reading {metrics_file}: {e}")
        
        # Calculate precision from total TP/FP (fair evaluation)
        if all_precisions:  # If at least one binary has data
            total_tp = np.sum(all_tps)
            total_fp = np.sum(all_fps)
            total_fn = np.sum(all_fns)
            
            # Calculate precision from total TP/FP (not average of averages)
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'dominating_set_size': np.mean(all_dom_sizes),
                'num_binaries': len(all_precisions)
            })
    
    return pd.DataFrame(results)

def calculate_roc_metrics(df):
    """Calculate TPR, FPR for ROC curve"""
    # TPR = Recall = TP / (TP + FN)
    # FPR = FP / (FP + TN) - approximated as FP / (FP + TP) here
    df['tpr'] = df['recall']  # TPR = Recall
    df['fpr'] = df['fp'] / (df['fp'] + df['tp'])  # FPR 근사
    return df

def plot_roc_curve(df, save_path=None, exp_dir=None):
    """Plot ROC curve"""
    # Sort data by threshold
    df_sorted = df.sort_values('threshold').reset_index(drop=True)
    
    plt.figure(figsize=(10, 8))
    
    # ROC curve
    plt.subplot(2, 2, 1)
    plt.plot(df_sorted['fpr'], df_sorted['tpr'], 'bo-', linewidth=2, markersize=8, label='Dominating Set')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Threshold Sensitivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Threshold별 TPR, FPR
    plt.subplot(2, 2, 2)
    plt.plot(df_sorted['threshold'], df_sorted['tpr'], 'bo-', label='TPR (Recall)', linewidth=2, markersize=6)
    plt.plot(df_sorted['threshold'], df_sorted['fpr'], 'ro-', label='FPR', linewidth=2, markersize=6)
    plt.xlabel('TLSH Threshold')
    plt.ylabel('Rate')
    plt.title('TPR vs FPR by Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall curve
    plt.subplot(2, 2, 3)
    plt.plot(df_sorted['recall'], df_sorted['precision'], 'go-', linewidth=2, markersize=8, label='Dominating Set')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Threshold별 Precision, Recall, F1-Score
    plt.subplot(2, 2, 4)
    plt.plot(df_sorted['threshold'], df_sorted['precision'], 'go-', label='Precision', linewidth=2, markersize=6)
    plt.plot(df_sorted['threshold'], df_sorted['recall'], 'bo-', label='Recall', linewidth=2, markersize=6)
    plt.plot(df_sorted['threshold'], df_sorted['f1'], 'mo-', label='F1-Score', linewidth=2, markersize=6)
    plt.xlabel('TLSH Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, F1-Score by Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        if exp_dir:
            # Save within exp_dir
            save_path = os.path.join(exp_dir, save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC/PR curves saved to: {save_path}")
    
    plt.show()

def plot_detailed_analysis(df, save_path=None, exp_dir=None):
    """Detailed threshold analysis graphs"""
    # Sort data by threshold
    df_sorted = df.sort_values('threshold').reset_index(drop=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. F1-score vs Threshold
    axes[0, 0].plot(df_sorted['threshold'], df_sorted['f1'], 'mo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('TLSH Threshold')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('F1-Score vs TLSH Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Dominating Set Size vs Threshold
    axes[0, 1].plot(df_sorted['threshold'], df_sorted['dominating_set_size'], 'co-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('TLSH Threshold')
    axes[0, 1].set_ylabel('Average Dominating Set Size')
    axes[0, 1].set_title('Dominating Set Size vs TLSH Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. TP, FP vs Threshold
    axes[0, 2].plot(df_sorted['threshold'], df_sorted['tp'], 'go-', label='True Positives', linewidth=2, markersize=6)
    axes[0, 2].plot(df_sorted['threshold'], df_sorted['fp'], 'ro-', label='False Positives', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('TLSH Threshold')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('TP vs FP by Threshold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision vs Recall (scatter)
    scatter = axes[1, 0].scatter(df_sorted['recall'], df_sorted['precision'], c=df_sorted['threshold'], 
                                cmap='viridis', s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Recall (colored by threshold)')
    plt.colorbar(scatter, ax=axes[1, 0], label='TLSH Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Performance metrics comparison
    axes[1, 1].plot(df_sorted['threshold'], df_sorted['precision'], 'go-', label='Precision', linewidth=2, markersize=6)
    axes[1, 1].plot(df_sorted['threshold'], df_sorted['recall'], 'bo-', label='Recall', linewidth=2, markersize=6)
    axes[1, 1].plot(df_sorted['threshold'], df_sorted['f1'], 'mo-', label='F1-Score', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('TLSH Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics vs TLSH Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Threshold sensitivity heatmap
    metrics_df = df_sorted[['threshold', 'precision', 'recall', 'f1']].set_index('threshold')
    sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=axes[1, 2], cbar_kws={'label': 'Score'})
    axes[1, 2].set_title('Threshold Sensitivity Heatmap')
    axes[1, 2].set_xlabel('TLSH Threshold')
    axes[1, 2].set_ylabel('Metrics')
    
    plt.tight_layout()
    
    if save_path:
        if exp_dir:
            # Save within exp_dir
            save_path = os.path.join(exp_dir, save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed analysis saved to: {save_path}")
    
    plt.show()

def print_analysis_summary(df):
    """Print analysis result summary (all binaries)"""
    print("=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS SUMMARY (ALL BINARIES)")
    print("=" * 80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   - Number of thresholds tested: {len(df)}")
    print(f"   - Threshold range: {df['threshold'].min()} - {df['threshold'].max()}")
    if 'num_binaries' in df.columns:
        print(f"   - Average number of binaries per threshold: {df['num_binaries'].mean():.1f}")
    
    print(f"\n🎯 Best Performance:")
    best_f1_idx = df['f1'].idxmax()
    best_precision_idx = df['precision'].idxmax()
    best_recall_idx = df['recall'].idxmax()
    
    print(f"   - Best F1-Score: {df.loc[best_f1_idx, 'f1']:.4f} at threshold {df.loc[best_f1_idx, 'threshold']}")
    print(f"   - Best Precision: {df.loc[best_precision_idx, 'precision']:.4f} at threshold {df.loc[best_precision_idx, 'threshold']}")
    print(f"   - Best Recall: {df.loc[best_recall_idx, 'recall']:.4f} at threshold {df.loc[best_recall_idx, 'threshold']}")
    
    print(f"\n📈 Performance Trends:")
    print(f"   - Precision range: {df['precision'].min():.4f} - {df['precision'].max():.4f}")
    print(f"   - Recall range: {df['recall'].min():.4f} - {df['recall'].max():.4f}")
    print(f"   - F1-Score range: {df['f1'].min():.4f} - {df['f1'].max():.4f}")
    print(f"   - Dominating set size range: {df['dominating_set_size'].min():.2f} - {df['dominating_set_size'].max():.2f}")
    
    print(f"\n🔍 Key Insights:")
    print(f"   - Analysis includes all 5 binaries: bool, direvent, gmp, libcrypto, libssl")
    print(f"   - Metrics are averaged across all available binaries for each threshold")
    print(f"   - TP, FP, FN are summed across all binaries")
    print(f"   - Dominating set size is averaged across all binaries")
    
    print(f"\n📋 Detailed Results:")
    if 'num_binaries' in df.columns:
        print(df[['threshold', 'precision', 'recall', 'f1', 'dominating_set_size', 'num_binaries']].to_string(index=False))
    else:
        print(df[['threshold', 'precision', 'recall', 'f1', 'dominating_set_size']].to_string(index=False))

def main():
    """Main analysis function"""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python threshold_sensitivity_analysis.py <exp_dir>")
        print("Example: python threshold_sensitivity_analysis.py /path/to/dominating_results/exp_20250924_174242")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    
    # Check if directory exists
    if not os.path.exists(exp_dir):
        print(f"❌ Directory not found: {exp_dir}")
        sys.exit(1)
    
    print(f"🔍 Loading threshold sensitivity results from all binaries in: {exp_dir}")
    df = load_threshold_results(exp_dir)
    
    if df.empty:
        print("❌ No results found!")
        return
    
    print(f"✅ Loaded {len(df)} threshold results (averaged across all binaries)")
    
    # Calculate ROC metrics
    df = calculate_roc_metrics(df)
    
    # Print result summary
    print_analysis_summary(df)
    
    # Generate graphs
    print("\n📊 Generating ROC/PR curves...")
    plot_roc_curve(df, save_path="threshold_roc_pr_curves.png", exp_dir=exp_dir)
    
    print("\n📊 Generating detailed analysis...")
    plot_detailed_analysis(df, save_path="threshold_detailed_analysis.png", exp_dir=exp_dir)
    
    # Save results to CSV (sorted by threshold)
    output_file = os.path.join(exp_dir, "threshold_sensitivity_results.csv")
    df_sorted = df.sort_values('threshold').reset_index(drop=True)
    df_sorted.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
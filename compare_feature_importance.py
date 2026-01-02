#!/usr/bin/env python3
"""
Compare feature importance across different feature extraction approaches.

This script compares feature importance from:
- Original features (56 features)
- Features V2 (46 features)
- Features V3 (39 features)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_feature_importance(base_dir: str, iteration_ids: list, model_type: str = 'level1'):
    """
    Load and aggregate feature importance across iterations.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for stability analysis
    iteration_ids : list
        List of iteration IDs to process
    model_type : str
        Model type ('level1' or 'raw')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and their mean importance across iterations
    """
    all_importance = []
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(base_dir, f'iteration_{iter_id:03d}')
        importance_file = os.path.join(iter_dir, f'{model_type}_model', 'feature_importance.csv')
        
        if os.path.exists(importance_file):
            df = pd.read_csv(importance_file)
            df['iteration'] = iter_id
            all_importance.append(df)
    
    if not all_importance:
        return pd.DataFrame()
    
    # Combine all iterations
    combined = pd.concat(all_importance, ignore_index=True)
    
    # Calculate mean importance per feature
    aggregated = combined.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
    aggregated = aggregated.sort_values('mean', ascending=False)
    aggregated.columns = ['feature', 'mean_importance', 'std_importance', 'n_iterations']
    
    return aggregated


def create_comparison_plots(orig_importance, v2_importance, v3_importance,
                           output_path: str, model_type: str = 'level1'):
    """
    Create comparison plots for feature importance.
    
    Parameters:
    -----------
    orig_importance : pd.DataFrame
        Feature importance for original features
    v2_importance : pd.DataFrame
        Feature importance for features_v2
    v3_importance : pd.DataFrame
        Feature importance for features_v3
    output_path : str
        Path to save the plot
    model_type : str
        Model type for title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Feature Importance Comparison: {model_type.upper()} Model', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Top 20 features for each version
    ax1 = axes[0, 0]
    
    # Get top 20 from each
    top_orig = orig_importance.head(20) if len(orig_importance) > 0 else pd.DataFrame()
    top_v2 = v2_importance.head(20) if len(v2_importance) > 0 else pd.DataFrame()
    top_v3 = v3_importance.head(20) if len(v3_importance) > 0 else pd.DataFrame()
    
    x = np.arange(min(20, max(len(top_orig), len(top_v2), len(top_v3))))
    width = 0.25
    
    if len(top_orig) > 0:
        ax1.bar(x - width, top_orig['mean_importance'].head(len(x)), width, 
               label='Original (56)', alpha=0.8, color='#2E86AB')
    if len(top_v2) > 0:
        ax1.bar(x, top_v2['mean_importance'].head(len(x)), width,
               label='V2 (46)', alpha=0.8, color='#A23B72')
    if len(top_v3) > 0:
        ax1.bar(x + width, top_v3['mean_importance'].head(len(x)), width,
               label='V3 (39)', alpha=0.8, color='#F18F01')
    
    ax1.set_xlabel('Rank', fontsize=11)
    ax1.set_ylabel('Mean Importance', fontsize=11)
    ax1.set_title('Top 20 Features Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Feature importance distribution
    ax2 = axes[0, 1]
    
    if len(orig_importance) > 0:
        ax2.hist(orig_importance['mean_importance'], bins=20, alpha=0.5, 
                label='Original', color='#2E86AB', edgecolor='black')
    if len(v2_importance) > 0:
        ax2.hist(v2_importance['mean_importance'], bins=20, alpha=0.5,
                label='V2', color='#A23B72', edgecolor='black')
    if len(v3_importance) > 0:
        ax2.hist(v3_importance['mean_importance'], bins=20, alpha=0.5,
                label='V3', color='#F18F01', edgecolor='black')
    
    ax2.set_xlabel('Feature Importance', fontsize=11)
    ax2.set_ylabel('Number of Features', fontsize=11)
    ax2.set_title('Importance Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cumulative importance
    ax3 = axes[1, 0]
    
    if len(orig_importance) > 0:
        orig_cumsum = np.cumsum(orig_importance['mean_importance'].sort_values(ascending=False))
        ax3.plot(range(len(orig_cumsum)), orig_cumsum / orig_cumsum.max() * 100,
                label='Original', linewidth=2, color='#2E86AB')
    if len(v2_importance) > 0:
        v2_cumsum = np.cumsum(v2_importance['mean_importance'].sort_values(ascending=False))
        ax3.plot(range(len(v2_cumsum)), v2_cumsum / v2_cumsum.max() * 100,
                label='V2', linewidth=2, color='#A23B72')
    if len(v3_importance) > 0:
        v3_cumsum = np.cumsum(v3_importance['mean_importance'].sort_values(ascending=False))
        ax3.plot(range(len(v3_cumsum)), v3_cumsum / v3_cumsum.max() * 100,
                label='V3', linewidth=2, color='#F18F01')
    
    ax3.set_xlabel('Number of Features', fontsize=11)
    ax3.set_ylabel('Cumulative Importance (%)', fontsize=11)
    ax3.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top 10 features side-by-side
    ax4 = axes[1, 1]
    
    # Get top 10 from each
    top10_orig = orig_importance.head(10) if len(orig_importance) > 0 else pd.DataFrame()
    top10_v2 = v2_importance.head(10) if len(v2_importance) > 0 else pd.DataFrame()
    top10_v3 = v3_importance.head(10) if len(v3_importance) > 0 else pd.DataFrame()
    
    # Create comparison for top 10
    all_top_features = set()
    if len(top10_orig) > 0:
        all_top_features.update(top10_orig['feature'].tolist())
    if len(top10_v2) > 0:
        all_top_features.update(top10_v2['feature'].tolist())
    if len(top10_v3) > 0:
        all_top_features.update(top10_v3['feature'].tolist())
    
    comparison_data = []
    for feat in sorted(all_top_features)[:10]:
        orig_val = top10_orig[top10_orig['feature'] == feat]['mean_importance'].values[0] if len(top10_orig) > 0 and feat in top10_orig['feature'].values else 0
        v2_val = top10_v2[top10_v2['feature'] == feat]['mean_importance'].values[0] if len(top10_v2) > 0 and feat in top10_v2['feature'].values else 0
        v3_val = top10_v3[top10_v3['feature'] == feat]['mean_importance'].values[0] if len(top10_v3) > 0 and feat in top10_v3['feature'].values else 0
        comparison_data.append({'feature': feat, 'Original': orig_val, 'V2': v2_val, 'V3': v3_val})
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        x_pos = np.arange(len(comp_df))
        width = 0.25
        
        ax4.bar(x_pos - width, comp_df['Original'], width, label='Original', alpha=0.8, color='#2E86AB')
        ax4.bar(x_pos, comp_df['V2'], width, label='V2', alpha=0.8, color='#A23B72')
        ax4.bar(x_pos + width, comp_df['V3'], width, label='V3', alpha=0.8, color='#F18F01')
        
        ax4.set_xlabel('Feature', fontsize=11)
        ax4.set_ylabel('Importance', fontsize=11)
        ax4.set_title('Top 10 Features Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(comp_df['feature'], rotation=45, ha='right', fontsize=9)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved to: {output_path}")


def generate_comparison_report(orig_importance, v2_importance, v3_importance,
                              output_file: str, model_type: str = 'level1'):
    """
    Generate a markdown comparison report.
    
    Parameters:
    -----------
    orig_importance : pd.DataFrame
        Feature importance for original features
    v2_importance : pd.DataFrame
        Feature importance for features_v2
    v3_importance : pd.DataFrame
        Feature importance for features_v3
    output_file : str
        Path to save the report
    model_type : str
        Model type for title
    """
    report_lines = [
        f"# Feature Importance Comparison: {model_type.upper()} Model",
        "",
        "## Overview",
        "",
        "This report compares feature importance across three feature extraction approaches:",
        "",
        "1. **Original Features** (56 features): Baseline feature set",
        "2. **Features V2** (46 features): Removed redundant features",
        "3. **Features V3** (39 features): Improved calculations + new features",
        "",
        "## Top Features Comparison",
        "",
        "### Top 20 Features",
        "",
        "| Rank | Original | V2 | V3 |",
        "|------|----------|----|----|"
    ]
    
    # Get top 20 from each
    top20_orig = orig_importance.head(20) if len(orig_importance) > 0 else pd.DataFrame()
    top20_v2 = v2_importance.head(20) if len(v2_importance) > 0 else pd.DataFrame()
    top20_v3 = v3_importance.head(20) if len(v3_importance) > 0 else pd.DataFrame()
    
    # Create comparison table
    all_features = set()
    if len(top20_orig) > 0:
        all_features.update(top20_orig['feature'].tolist())
    if len(top20_v2) > 0:
        all_features.update(top20_v2['feature'].tolist())
    if len(top20_v3) > 0:
        all_features.update(top20_v3['feature'].tolist())
    
    # Get top features by average rank
    feature_ranks = {}
    for feat in all_features:
        orig_rank = top20_orig[top20_orig['feature'] == feat].index[0] + 1 if len(top20_orig) > 0 and feat in top20_orig['feature'].values else None
        v2_rank = top20_v2[top20_v2['feature'] == feat].index[0] + 1 if len(top20_v2) > 0 and feat in top20_v2['feature'].values else None
        v3_rank = top20_v3[top20_v3['feature'] == feat].index[0] + 1 if len(top20_v3) > 0 and feat in top20_v3['feature'].values else None
        
        ranks = [r for r in [orig_rank, v2_rank, v3_rank] if r is not None]
        avg_rank = np.mean(ranks) if ranks else 999
        
        feature_ranks[feat] = {
            'avg_rank': avg_rank,
            'orig_rank': orig_rank,
            'v2_rank': v2_rank,
            'v3_rank': v3_rank,
            'orig_importance': top20_orig[top20_orig['feature'] == feat]['mean_importance'].values[0] if len(top20_orig) > 0 and feat in top20_orig['feature'].values else None,
            'v2_importance': top20_v2[top20_v2['feature'] == feat]['mean_importance'].values[0] if len(top20_v2) > 0 and feat in top20_v2['feature'].values else None,
            'v3_importance': top20_v3[top20_v3['feature'] == feat]['mean_importance'].values[0] if len(top20_v3) > 0 and feat in top20_v3['feature'].values else None,
        }
    
    # Sort by average rank
    sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1]['avg_rank'])[:20]
    
    for rank, (feat, data) in enumerate(sorted_features, 1):
        orig_str = f"{data['orig_rank']} ({data['orig_importance']:.4f})" if data['orig_rank'] is not None else "-"
        v2_str = f"{data['v2_rank']} ({data['v2_importance']:.4f})" if data['v2_rank'] is not None else "-"
        v3_str = f"{data['v3_rank']} ({data['v3_importance']:.4f})" if data['v3_rank'] is not None else "-"
        
        report_lines.append(f"| {rank} | {feat} | {orig_str} | {v2_str} | {v3_str} |")
    
    report_lines.extend([
        "",
        "## Statistics",
        "",
        "### Feature Count",
        f"- Original: {len(orig_importance)} features",
        f"- V2: {len(v2_importance)} features",
        f"- V3: {len(v3_importance)} features",
        "",
        "### Importance Statistics",
        ""
    ])
    
    # Add statistics for each version
    for name, df in [("Original", orig_importance), ("V2", v2_importance), ("V3", v3_importance)]:
        if len(df) > 0:
            report_lines.extend([
                f"#### {name}",
                f"- Mean importance: {df['mean_importance'].mean():.6f}",
                f"- Std importance: {df['mean_importance'].std():.6f}",
                f"- Max importance: {df['mean_importance'].max():.6f}",
                f"- Min importance: {df['mean_importance'].min():.6f}",
                ""
            ])
    
    # New features in V3
    report_lines.extend([
        "## New Features in V3",
        "",
        "The following features are new in V3:",
        ""
    ])
    
    if len(v3_importance) > 0:
        v3_features = set(v3_importance['feature'].tolist())
        if len(v2_importance) > 0:
            v2_features = set(v2_importance['feature'].tolist())
            new_in_v3 = v3_features - v2_features
            
            for feat in sorted(new_in_v3):
                importance = v3_importance[v3_importance['feature'] == feat]['mean_importance'].values[0]
                rank = (v3_importance['feature'] == feat).idxmax() + 1 if (v3_importance['feature'] == feat).any() else None
                report_lines.append(f"- **{feat}**: Rank {rank}, Importance {importance:.6f}")
    
    report_lines.extend([
        "",
        "## Removed Features",
        "",
        "### Removed in V2",
        ""
    ])
    
    if len(orig_importance) > 0 and len(v2_importance) > 0:
        orig_features = set(orig_importance['feature'].tolist())
        v2_features = set(v2_importance['feature'].tolist())
        removed_in_v2 = orig_features - v2_features
        
        for feat in sorted(removed_in_v2):
            importance = orig_importance[orig_importance['feature'] == feat]['mean_importance'].values[0]
            rank = (orig_importance['feature'] == feat).idxmax() + 1 if (orig_importance['feature'] == feat).any() else None
            report_lines.append(f"- **{feat}**: Rank {rank}, Importance {importance:.6f}")
    
    report_lines.extend([
        "",
        "### Removed in V3 (from V2)",
        ""
    ])
    
    if len(v2_importance) > 0 and len(v3_importance) > 0:
        v2_features = set(v2_importance['feature'].tolist())
        v3_features = set(v3_importance['feature'].tolist())
        removed_in_v3 = v2_features - v3_features
        
        for feat in sorted(removed_in_v3):
            importance = v2_importance[v2_importance['feature'] == feat]['mean_importance'].values[0]
            rank = (v2_importance['feature'] == feat).idxmax() + 1 if (v2_importance['feature'] == feat).any() else None
            report_lines.append(f"- **{feat}**: Rank {rank}, Importance {importance:.6f}")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Report saved to: {output_file}")


def main():
    """Generate feature importance comparison."""
    print("=" * 80)
    print("FEATURE IMPORTANCE COMPARISON")
    print("=" * 80)
    print()
    
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    # Load feature importance for each version
    print("Loading feature importance...")
    
    print("  Original features...")
    orig_level1 = load_feature_importance('stability_analysis_v3', iteration_ids, 'level1')
    orig_raw = load_feature_importance('stability_analysis_v3', iteration_ids, 'raw')
    print(f"    Level1: {len(orig_level1)} features, {orig_level1['n_iterations'].sum() if len(orig_level1) > 0 else 0} total iterations")
    print(f"    Raw: {len(orig_raw)} features, {orig_raw['n_iterations'].sum() if len(orig_raw) > 0 else 0} total iterations")
    
    print("  Features V2...")
    v2_level1 = load_feature_importance('stability_analysis_v3_features_v2', iteration_ids, 'level1')
    v2_raw = load_feature_importance('stability_analysis_v3_features_v2', iteration_ids, 'raw')
    print(f"    Level1: {len(v2_level1)} features, {v2_level1['n_iterations'].sum() if len(v2_level1) > 0 else 0} total iterations")
    print(f"    Raw: {len(v2_raw)} features, {v2_raw['n_iterations'].sum() if len(v2_raw) > 0 else 0} total iterations")
    
    print("  Features V3...")
    v3_level1 = load_feature_importance('stability_analysis_v3_features_v3', iteration_ids, 'level1')
    v3_raw = load_feature_importance('stability_analysis_v3_features_v3', iteration_ids, 'raw')
    print(f"    Level1: {len(v3_level1)} features, {v3_level1['n_iterations'].sum() if len(v3_level1) > 0 else 0} total iterations")
    print(f"    Raw: {len(v3_raw)} features, {v3_raw['n_iterations'].sum() if len(v3_raw) > 0 else 0} total iterations")
    print()
    
    # Create output directory
    output_dir = 'feature_importance_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comparisons for both model types
    for model_type in ['level1', 'raw']:
        print(f"Generating comparison for {model_type} model...")
        
        if model_type == 'level1':
            orig = orig_level1
            v2 = v2_level1
            v3 = v3_level1
        else:
            orig = orig_raw
            v2 = v2_raw
            v3 = v3_raw
        
        # Create plots
        plot_path = os.path.join(output_dir, f'feature_importance_comparison_{model_type}.png')
        create_comparison_plots(orig, v2, v3, plot_path, model_type)
        
        # Generate report
        report_path = os.path.join(output_dir, f'feature_importance_comparison_{model_type}.md')
        generate_comparison_report(orig, v2, v3, report_path, model_type)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Compare feature importance across feature extraction approaches including V4.

This script compares feature importance from:
- Original features (56 features)
- Features V2 (46 features)
- Features V3 (39 features)
- Features V4 (~40-42 features)
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


def generate_comparison_report(orig_importance, v2_importance, v3_importance, v4_importance,
                              output_file: str, model_type: str = 'level1'):
    """Generate a markdown comparison report including V4."""
    report_lines = [
        f"# Feature Importance Comparison: {model_type.upper()} Model (Including V4)",
        "",
        "## Overview",
        "",
        "This report compares feature importance across four feature extraction approaches:",
        "",
        "1. **Original Features** (56 features): Baseline feature set",
        "2. **Features V2** (46 features): Removed redundant features",
        "3. **Features V3** (39 features): Improved calculations + new features",
        "4. **Features V4** (~40-42 features): Phase 1 & 2 improvements",
        "",
        "## Top Features Comparison",
        "",
        "### Top 20 Features",
        "",
        "| Rank | Feature | Original | V2 | V3 | V4 |",
        "|------|---------|----------|----|----|----|"
    ]
    
    # Get top 20 from each
    top20_orig = orig_importance.head(20) if len(orig_importance) > 0 else pd.DataFrame()
    top20_v2 = v2_importance.head(20) if len(v2_importance) > 0 else pd.DataFrame()
    top20_v3 = v3_importance.head(20) if len(v3_importance) > 0 else pd.DataFrame()
    top20_v4 = v4_importance.head(20) if len(v4_importance) > 0 else pd.DataFrame()
    
    # Create comparison table
    all_features = set()
    if len(top20_orig) > 0:
        all_features.update(top20_orig['feature'].tolist())
    if len(top20_v2) > 0:
        all_features.update(top20_v2['feature'].tolist())
    if len(top20_v3) > 0:
        all_features.update(top20_v3['feature'].tolist())
    if len(top20_v4) > 0:
        all_features.update(top20_v4['feature'].tolist())
    
    # Get feature data for comparison
    feature_data = {}
    for feat in all_features:
        feature_data[feat] = {
            'orig_rank': None,
            'orig_importance': None,
            'v2_rank': None,
            'v2_importance': None,
            'v3_rank': None,
            'v3_importance': None,
            'v4_rank': None,
            'v4_importance': None,
        }
        
        if len(top20_orig) > 0 and feat in top20_orig['feature'].values:
            row = top20_orig[top20_orig['feature'] == feat].iloc[0]
            feature_data[feat]['orig_rank'] = top20_orig[top20_orig['feature'] == feat].index[0] + 1
            feature_data[feat]['orig_importance'] = row['mean_importance']
        
        if len(top20_v2) > 0 and feat in top20_v2['feature'].values:
            row = top20_v2[top20_v2['feature'] == feat].iloc[0]
            feature_data[feat]['v2_rank'] = top20_v2[top20_v2['feature'] == feat].index[0] + 1
            feature_data[feat]['v2_importance'] = row['mean_importance']
        
        if len(top20_v3) > 0 and feat in top20_v3['feature'].values:
            row = top20_v3[top20_v3['feature'] == feat].iloc[0]
            feature_data[feat]['v3_rank'] = top20_v3[top20_v3['feature'] == feat].index[0] + 1
            feature_data[feat]['v3_importance'] = row['mean_importance']
        
        if len(top20_v4) > 0 and feat in top20_v4['feature'].values:
            row = top20_v4[top20_v4['feature'] == feat].iloc[0]
            feature_data[feat]['v4_rank'] = top20_v4[top20_v4['feature'] == feat].index[0] + 1
            feature_data[feat]['v4_importance'] = row['mean_importance']
    
    # Sort by highest importance value (not rank)
    def get_max_importance(data):
        importances = [imp for imp in [data['orig_importance'], data['v2_importance'], 
                                       data['v3_importance'], data['v4_importance']] if imp is not None]
        return max(importances) if importances else 0
    
    sorted_features = sorted(feature_data.items(), key=lambda x: get_max_importance(x[1]), reverse=True)[:20]
    
    for rank, (feat, data) in enumerate(sorted_features, 1):
        orig_str = f"{data['orig_rank']} ({data['orig_importance']:.4f})" if data['orig_rank'] is not None else "-"
        v2_str = f"{data['v2_rank']} ({data['v2_importance']:.4f})" if data['v2_rank'] is not None else "-"
        v3_str = f"{data['v3_rank']} ({data['v3_importance']:.4f})" if data['v3_rank'] is not None else "-"
        v4_str = f"{data['v4_rank']} ({data['v4_importance']:.4f})" if data['v4_rank'] is not None else "-"
        
        report_lines.append(f"| {rank} | {feat} | {orig_str} | {v2_str} | {v3_str} | {v4_str} |")
    
    report_lines.extend([
        "",
        "## Statistics",
        "",
        "### Feature Count",
        f"- Original: {len(orig_importance)} features",
        f"- V2: {len(v2_importance)} features",
        f"- V3: {len(v3_importance)} features",
        f"- V4: {len(v4_importance)} features",
        "",
        "### Importance Statistics",
        ""
    ])
    
    # Add statistics for each version
    for name, df in [("Original", orig_importance), ("V2", v2_importance), ("V3", v3_importance), ("V4", v4_importance)]:
        if len(df) > 0:
            report_lines.extend([
                f"#### {name}",
                f"- Mean importance: {df['mean_importance'].mean():.6f}",
                f"- Std importance: {df['mean_importance'].std():.6f}",
                f"- Max importance: {df['mean_importance'].max():.6f}",
                f"- Min importance: {df['mean_importance'].min():.6f}",
                f"- Top feature: {df.iloc[0]['feature']} ({df.iloc[0]['mean_importance']:.6f})",
                ""
            ])
    
    # New features in V4
    report_lines.extend([
        "## New Features in V4",
        "",
        "The following features are new in V4:",
        ""
    ])
    
    if len(v4_importance) > 0:
        v4_features = set(v4_importance['feature'].tolist())
        if len(v3_importance) > 0:
            v3_features = set(v3_importance['feature'].tolist())
            new_in_v4 = v4_features - v3_features
            
            for feat in sorted(new_in_v4):
                importance = v4_importance[v4_importance['feature'] == feat]['mean_importance'].values[0]
                rank = (v4_importance['feature'] == feat).idxmax() + 1 if (v4_importance['feature'] == feat).any() else None
                report_lines.append(f"- **{feat}**: Rank {rank}, Importance {importance:.6f}")
    
    # Removed features in V4
    report_lines.extend([
        "",
        "## Removed Features in V4 (from V3)",
        ""
    ])
    
    if len(v3_importance) > 0 and len(v4_importance) > 0:
        v3_features = set(v3_importance['feature'].tolist())
        v4_features = set(v4_importance['feature'].tolist())
        removed_in_v4 = v3_features - v4_features
        
        for feat in sorted(removed_in_v4):
            importance = v3_importance[v3_importance['feature'] == feat]['mean_importance'].values[0]
            rank = (v3_importance['feature'] == feat).idxmax() + 1 if (v3_importance['feature'] == feat).any() else None
            report_lines.append(f"- **{feat}**: Rank {rank}, Importance {importance:.6f}")
    
    # Save report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Report saved to: {output_file}")


def main():
    """Generate feature importance comparison including V4."""
    print("=" * 80)
    print("FEATURE IMPORTANCE COMPARISON (Including V4)")
    print("=" * 80)
    print()
    
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    # Load feature importance for each version
    print("Loading feature importance data...")
    
    orig_importance_level1 = load_feature_importance('stability_analysis_v3', iteration_ids, 'level1')
    orig_importance_raw = load_feature_importance('stability_analysis_v3', iteration_ids, 'raw')
    
    v2_importance_level1 = load_feature_importance('stability_analysis_v3_features_v2', iteration_ids, 'level1')
    v2_importance_raw = load_feature_importance('stability_analysis_v3_features_v2', iteration_ids, 'raw')
    
    v3_importance_level1 = load_feature_importance('stability_analysis_v3_features_v3', iteration_ids, 'level1')
    v3_importance_raw = load_feature_importance('stability_analysis_v3_features_v3', iteration_ids, 'raw')
    
    v4_importance_level1 = load_feature_importance('stability_analysis_v3_features_v4', iteration_ids, 'level1')
    v4_importance_raw = load_feature_importance('stability_analysis_v3_features_v4', iteration_ids, 'raw')
    
    print(f"  Original Level1: {len(orig_importance_level1)} features")
    print(f"  Original Raw: {len(orig_importance_raw)} features")
    print(f"  V2 Level1: {len(v2_importance_level1)} features")
    print(f"  V2 Raw: {len(v2_importance_raw)} features")
    print(f"  V3 Level1: {len(v3_importance_level1)} features")
    print(f"  V3 Raw: {len(v3_importance_raw)} features")
    print(f"  V4 Level1: {len(v4_importance_level1)} features")
    print(f"  V4 Raw: {len(v4_importance_raw)} features")
    print()
    
    # Generate reports
    output_dir = 'feature_importance_comparison_v4'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating comparison reports...")
    
    generate_comparison_report(
        orig_importance_level1, v2_importance_level1, v3_importance_level1, v4_importance_level1,
        os.path.join(output_dir, 'feature_importance_comparison_level1.md'),
        'level1'
    )
    
    generate_comparison_report(
        orig_importance_raw, v2_importance_raw, v3_importance_raw, v4_importance_raw,
        os.path.join(output_dir, 'feature_importance_comparison_raw.md'),
        'raw'
    )
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Reports saved to: {output_dir}/")
    print()


if __name__ == '__main__':
    main()


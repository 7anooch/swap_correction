#!/usr/bin/env python3
"""
Generate comprehensive comparison report between features, features_v2, and features_v3.

This script compares all three feature extraction approaches:
- Original features (56 features)
- features_v2 (46 features, removed redundant)
- features_v3 (39 features, improved calculations + new features)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_original_features_data():
    """Load data from original features (stability_analysis_v3)."""
    v3_dir = 'stability_analysis_v3'
    
    # Try to load aggregated metrics if they exist
    level1_file = os.path.join(v3_dir, 'level1_metrics.csv')
    raw_file = os.path.join(v3_dir, 'raw_metrics.csv')
    
    if os.path.exists(level1_file) and os.path.exists(raw_file):
        level1_metrics = pd.read_csv(level1_file)
        raw_metrics = pd.read_csv(raw_file)
    else:
        # Load from individual iterations
        level1_metrics = []
        raw_metrics = []
        iteration_ids = [7, 8, 9, 10, 11, 12]
        
        for iter_id in iteration_ids:
            iter_dir = os.path.join(v3_dir, f'iteration_{iter_id:03d}')
            
            # Level1 model
            level1_eval_file = os.path.join(iter_dir, 'level1_model', 'evaluation_results.json')
            if os.path.exists(level1_eval_file):
                with open(level1_eval_file, 'r') as f:
                    data = json.load(f)
                if 'summary' in data:
                    summary = data['summary']
                    level1_metrics.append({
                        'iteration': iter_id,
                        'eval_mean_f1': summary.get('mean_f1', np.nan),
                        'eval_mean_precision': summary.get('mean_precision', np.nan),
                        'eval_mean_recall': summary.get('mean_recall', np.nan),
                        'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                        'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                        'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                        'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                        'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                    })
            
            # Raw model
            raw_eval_file = os.path.join(iter_dir, 'raw_model', 'evaluation_results.json')
            if os.path.exists(raw_eval_file):
                with open(raw_eval_file, 'r') as f:
                    data = json.load(f)
                if 'summary' in data:
                    summary = data['summary']
                    raw_metrics.append({
                        'iteration': iter_id,
                        'eval_mean_f1': summary.get('mean_f1', np.nan),
                        'eval_mean_precision': summary.get('mean_precision', np.nan),
                        'eval_mean_recall': summary.get('mean_recall', np.nan),
                        'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                        'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                        'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                        'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                        'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                    })
        
        level1_metrics = pd.DataFrame(level1_metrics) if level1_metrics else pd.DataFrame()
        raw_metrics = pd.DataFrame(raw_metrics) if raw_metrics else pd.DataFrame()
    
    return level1_metrics, raw_metrics


def load_features_v2_data():
    """Load data from features_v2."""
    v2_dir = 'stability_analysis_v3_features_v2'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    level1_metrics = []
    raw_metrics = []
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(v2_dir, f'iteration_{iter_id:03d}')
        
        # Level1 model
        level1_eval_file = os.path.join(iter_dir, 'level1_model', 'evaluation_results.json')
        if os.path.exists(level1_eval_file):
            with open(level1_eval_file, 'r') as f:
                data = json.load(f)
            if 'summary' in data:
                summary = data['summary']
                level1_metrics.append({
                    'iteration': iter_id,
                    'eval_mean_f1': summary.get('mean_f1', np.nan),
                    'eval_mean_precision': summary.get('mean_precision', np.nan),
                    'eval_mean_recall': summary.get('mean_recall', np.nan),
                    'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                    'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                    'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                    'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                    'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                })
        
        # Raw model
        raw_eval_file = os.path.join(iter_dir, 'raw_model', 'evaluation_results.json')
        if os.path.exists(raw_eval_file):
            with open(raw_eval_file, 'r') as f:
                data = json.load(f)
            if 'summary' in data:
                summary = data['summary']
                raw_metrics.append({
                    'iteration': iter_id,
                    'eval_mean_f1': summary.get('mean_f1', np.nan),
                    'eval_mean_precision': summary.get('mean_precision', np.nan),
                    'eval_mean_recall': summary.get('mean_recall', np.nan),
                    'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                    'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                    'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                    'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                    'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                })
    
    level1_df = pd.DataFrame(level1_metrics) if level1_metrics else pd.DataFrame()
    raw_df = pd.DataFrame(raw_metrics) if raw_metrics else pd.DataFrame()
    
    return level1_df, raw_df


def load_features_v3_data():
    """Load data from features_v3."""
    v3_dir = 'stability_analysis_v3_features_v3'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    level1_metrics = []
    raw_metrics = []
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(v3_dir, f'iteration_{iter_id:03d}')
        
        # Level1 model
        level1_eval_file = os.path.join(iter_dir, 'level1_model', 'evaluation_results.json')
        if os.path.exists(level1_eval_file):
            with open(level1_eval_file, 'r') as f:
                data = json.load(f)
            if 'summary' in data:
                summary = data['summary']
                level1_metrics.append({
                    'iteration': iter_id,
                    'eval_mean_f1': summary.get('mean_f1', np.nan),
                    'eval_mean_precision': summary.get('mean_precision', np.nan),
                    'eval_mean_recall': summary.get('mean_recall', np.nan),
                    'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                    'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                    'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                    'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                    'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                })
        
        # Raw model
        raw_eval_file = os.path.join(iter_dir, 'raw_model', 'evaluation_results.json')
        if os.path.exists(raw_eval_file):
            with open(raw_eval_file, 'r') as f:
                data = json.load(f)
            if 'summary' in data:
                summary = data['summary']
                raw_metrics.append({
                    'iteration': iter_id,
                    'eval_mean_f1': summary.get('mean_f1', np.nan),
                    'eval_mean_precision': summary.get('mean_precision', np.nan),
                    'eval_mean_recall': summary.get('mean_recall', np.nan),
                    'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                    'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                    'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                    'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                    'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                })
    
    level1_df = pd.DataFrame(level1_metrics) if level1_metrics else pd.DataFrame()
    raw_df = pd.DataFrame(raw_metrics) if raw_metrics else pd.DataFrame()
    
    return level1_df, raw_df


def load_features_v4_data():
    """Load data from features_v4."""
    v4_dir = 'stability_analysis_v3_features_v4'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    level1_metrics = []
    raw_metrics = []
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(v4_dir, f'iteration_{iter_id:03d}')
        
        # Level1 model
        level1_eval_file = os.path.join(iter_dir, 'level1_model', 'evaluation_results.json')
        if os.path.exists(level1_eval_file):
            with open(level1_eval_file, 'r') as f:
                data = json.load(f)
            if 'summary' in data:
                summary = data['summary']
                level1_metrics.append({
                    'iteration': iter_id,
                    'eval_mean_f1': summary.get('mean_f1', np.nan),
                    'eval_mean_precision': summary.get('mean_precision', np.nan),
                    'eval_mean_recall': summary.get('mean_recall', np.nan),
                    'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                    'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                    'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                    'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                    'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                })
        
        # Raw model
        raw_eval_file = os.path.join(iter_dir, 'raw_model', 'evaluation_results.json')
        if os.path.exists(raw_eval_file):
            with open(raw_eval_file, 'r') as f:
                data = json.load(f)
            if 'summary' in data:
                summary = data['summary']
                raw_metrics.append({
                    'iteration': iter_id,
                    'eval_mean_f1': summary.get('mean_f1', np.nan),
                    'eval_mean_precision': summary.get('mean_precision', np.nan),
                    'eval_mean_recall': summary.get('mean_recall', np.nan),
                    'eval_mean_sensitivity': summary.get('mean_sensitivity', np.nan),
                    'eval_mean_specificity': summary.get('mean_specificity', np.nan),
                    'eval_mean_pct_swaps_resolved': summary.get('mean_pct_swaps_resolved', np.nan),
                    'eval_mean_pct_frames_clean_pre': summary.get('mean_pct_frames_clean_pre', np.nan),
                    'eval_mean_pct_frames_clean_post': summary.get('mean_pct_frames_clean_post', np.nan),
                })
    
    level1_df = pd.DataFrame(level1_metrics) if level1_metrics else pd.DataFrame()
    raw_df = pd.DataFrame(raw_metrics) if raw_metrics else pd.DataFrame()
    
    return level1_df, raw_df


def calculate_summary_stats(df, metric_col):
    """Calculate summary statistics for a metric."""
    if df.empty or metric_col not in df.columns:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
    
    values = df[metric_col].dropna()
    if len(values) == 0:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
    
    return {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'max': float(values.max())
    }


def generate_comparison_report():
    """Generate comprehensive comparison report."""
    print("=" * 80)
    print("GENERATING FEATURE COMPARISON REPORT")
    print("=" * 80)
    print()
    
    # Load data from all four feature sets
    print("Loading original features data...")
    orig_level1, orig_raw = load_original_features_data()
    print(f"  Original: {len(orig_level1)} level1, {len(orig_raw)} raw iterations")
    
    print("Loading features_v2 data...")
    v2_level1, v2_raw = load_features_v2_data()
    print(f"  V2: {len(v2_level1)} level1, {len(v2_raw)} raw iterations")
    
    print("Loading features_v3 data...")
    v3_level1, v3_raw = load_features_v3_data()
    print(f"  V3: {len(v3_level1)} level1, {len(v3_raw)} raw iterations")
    
    print("Loading features_v4 data...")
    v4_level1, v4_raw = load_features_v4_data()
    print(f"  V4: {len(v4_level1)} level1, {len(v4_raw)} raw iterations")
    print()
    
    # Generate report
    report_lines = [
        "# Feature Extraction Comparison Report",
        "",
        "## Overview",
        "",
        "This report compares four feature extraction approaches:",
        "",
        "1. **Original Features** (56 features): Baseline feature set",
        "2. **Features V2** (46 features): Removed redundant features (8 positions, 2 velocity magnitudes)",
        "3. **Features V3** (39 features): Improved calculations + new features",
        "   - 3-point derivative for angular velocity",
        "   - Tail path curvature",
        "   - Head/tail curvature ratio",
        "   - Collapsed keypoints binary flag",
        "   - Removed small-window std features and velocity components",
        "4. **Features V4** (~40-42 features): Phase 1 & 2 improvements",
        "   - Removed underperforming v3 features (collapsed_keypoints, raw curvatures)",
        "   - Removed window size 5 features",
        "   - Added acceleration features (head, tail, relative)",
        "   - Added body length normalization to distance features",
        "",
        "## Comparison: Level1 Model",
        "",
        "### Performance Metrics",
        "",
        "| Metric | Original | V2 | V3 | V4 | V2 vs Orig | V3 vs Orig | V4 vs Orig | V3 vs V2 | V4 vs V2 | V4 vs V3 |",
        "|--------|----------|----|----|----|------------|------------|------------|----------|----------|----------|"
    ]
    
    # Metrics to compare
    metrics = [
        ('F1-Score', 'eval_mean_f1'),
        ('Precision', 'eval_mean_precision'),
        ('Recall', 'eval_mean_recall'),
        ('Sensitivity', 'eval_mean_sensitivity'),
        ('Specificity', 'eval_mean_specificity'),
        ('% Swaps Resolved', 'eval_mean_pct_swaps_resolved'),
        ('% Frames Clean Post', 'eval_mean_pct_frames_clean_post'),
    ]
    
    for metric_name, metric_col in metrics:
        orig_stats = calculate_summary_stats(orig_level1, metric_col)
        v2_stats = calculate_summary_stats(v2_level1, metric_col)
        v3_stats = calculate_summary_stats(v3_level1, metric_col)
        v4_stats = calculate_summary_stats(v4_level1, metric_col)
        
        orig_val = f"{orig_stats['mean']:.4f} ± {orig_stats['std']:.4f}" if not np.isnan(orig_stats['mean']) else "N/A"
        v2_val = f"{v2_stats['mean']:.4f} ± {v2_stats['std']:.4f}" if not np.isnan(v2_stats['mean']) else "N/A"
        v3_val = f"{v3_stats['mean']:.4f} ± {v3_stats['std']:.4f}" if not np.isnan(v3_stats['mean']) else "N/A"
        v4_val = f"{v4_stats['mean']:.4f} ± {v4_stats['std']:.4f}" if not np.isnan(v4_stats['mean']) else "N/A"
        
        v2_diff = (v2_stats['mean'] - orig_stats['mean']) if not (np.isnan(v2_stats['mean']) or np.isnan(orig_stats['mean'])) else np.nan
        v3_diff = (v3_stats['mean'] - orig_stats['mean']) if not (np.isnan(v3_stats['mean']) or np.isnan(orig_stats['mean'])) else np.nan
        v4_diff = (v4_stats['mean'] - orig_stats['mean']) if not (np.isnan(v4_stats['mean']) or np.isnan(orig_stats['mean'])) else np.nan
        v3_v2_diff = (v3_stats['mean'] - v2_stats['mean']) if not (np.isnan(v3_stats['mean']) or np.isnan(v2_stats['mean'])) else np.nan
        v4_v2_diff = (v4_stats['mean'] - v2_stats['mean']) if not (np.isnan(v4_stats['mean']) or np.isnan(v2_stats['mean'])) else np.nan
        v4_v3_diff = (v4_stats['mean'] - v3_stats['mean']) if not (np.isnan(v4_stats['mean']) or np.isnan(v3_stats['mean'])) else np.nan
        
        v2_diff_str = f"{v2_diff:+.4f}" if not np.isnan(v2_diff) else "N/A"
        v3_diff_str = f"{v3_diff:+.4f}" if not np.isnan(v3_diff) else "N/A"
        v4_diff_str = f"{v4_diff:+.4f}" if not np.isnan(v4_diff) else "N/A"
        v3_v2_diff_str = f"{v3_v2_diff:+.4f}" if not np.isnan(v3_v2_diff) else "N/A"
        v4_v2_diff_str = f"{v4_v2_diff:+.4f}" if not np.isnan(v4_v2_diff) else "N/A"
        v4_v3_diff_str = f"{v4_v3_diff:+.4f}" if not np.isnan(v4_v3_diff) else "N/A"
        
        report_lines.append(
            f"| {metric_name} | {orig_val} | {v2_val} | {v3_val} | {v4_val} | {v2_diff_str} | {v3_diff_str} | {v4_diff_str} | {v3_v2_diff_str} | {v4_v2_diff_str} | {v4_v3_diff_str} |"
        )
    
    report_lines.extend([
        "",
        "## Comparison: Raw Model",
        "",
        "### Performance Metrics",
        "",
        "| Metric | Original | V2 | V3 | V4 | V2 vs Orig | V3 vs Orig | V4 vs Orig | V3 vs V2 | V4 vs V2 | V4 vs V3 |",
        "|--------|----------|----|----|----|------------|------------|------------|----------|----------|----------|"
    ])
    
    for metric_name, metric_col in metrics:
        orig_stats = calculate_summary_stats(orig_raw, metric_col)
        v2_stats = calculate_summary_stats(v2_raw, metric_col)
        v3_stats = calculate_summary_stats(v3_raw, metric_col)
        v4_stats = calculate_summary_stats(v4_raw, metric_col)
        
        orig_val = f"{orig_stats['mean']:.4f} ± {orig_stats['std']:.4f}" if not np.isnan(orig_stats['mean']) else "N/A"
        v2_val = f"{v2_stats['mean']:.4f} ± {v2_stats['std']:.4f}" if not np.isnan(v2_stats['mean']) else "N/A"
        v3_val = f"{v3_stats['mean']:.4f} ± {v3_stats['std']:.4f}" if not np.isnan(v3_stats['mean']) else "N/A"
        v4_val = f"{v4_stats['mean']:.4f} ± {v4_stats['std']:.4f}" if not np.isnan(v4_stats['mean']) else "N/A"
        
        v2_diff = (v2_stats['mean'] - orig_stats['mean']) if not (np.isnan(v2_stats['mean']) or np.isnan(orig_stats['mean'])) else np.nan
        v3_diff = (v3_stats['mean'] - orig_stats['mean']) if not (np.isnan(v3_stats['mean']) or np.isnan(orig_stats['mean'])) else np.nan
        v4_diff = (v4_stats['mean'] - orig_stats['mean']) if not (np.isnan(v4_stats['mean']) or np.isnan(orig_stats['mean'])) else np.nan
        v3_v2_diff = (v3_stats['mean'] - v2_stats['mean']) if not (np.isnan(v3_stats['mean']) or np.isnan(v2_stats['mean'])) else np.nan
        v4_v2_diff = (v4_stats['mean'] - v2_stats['mean']) if not (np.isnan(v4_stats['mean']) or np.isnan(v2_stats['mean'])) else np.nan
        v4_v3_diff = (v4_stats['mean'] - v3_stats['mean']) if not (np.isnan(v4_stats['mean']) or np.isnan(v3_stats['mean'])) else np.nan
        
        v2_diff_str = f"{v2_diff:+.4f}" if not np.isnan(v2_diff) else "N/A"
        v3_diff_str = f"{v3_diff:+.4f}" if not np.isnan(v3_diff) else "N/A"
        v4_diff_str = f"{v4_diff:+.4f}" if not np.isnan(v4_diff) else "N/A"
        v3_v2_diff_str = f"{v3_v2_diff:+.4f}" if not np.isnan(v3_v2_diff) else "N/A"
        v4_v2_diff_str = f"{v4_v2_diff:+.4f}" if not np.isnan(v4_v2_diff) else "N/A"
        v4_v3_diff_str = f"{v4_v3_diff:+.4f}" if not np.isnan(v4_v3_diff) else "N/A"
        
        report_lines.append(
            f"| {metric_name} | {orig_val} | {v2_val} | {v3_val} | {v4_val} | {v2_diff_str} | {v3_diff_str} | {v4_diff_str} | {v3_v2_diff_str} | {v4_v2_diff_str} | {v4_v3_diff_str} |"
        )
    
    report_lines.extend([
        "",
        "## Summary",
        "",
        "### Key Findings",
        "",
        "1. **Feature Count Reduction**:",
        "   - Original: 56 features",
        "   - V2: 46 features (-10 redundant)",
        "   - V3: 39 features (-17 from original, -7 from V2)",
        "   - V4: ~40-42 features (-3 from V3, +3-6 new)",
        "",
        "2. **Performance Comparison**:",
        "   - See tables above for detailed metrics",
        "   - Positive differences indicate improvement",
        "   - Negative differences indicate regression",
        "",
        "3. **Recommendations**:",
        "   - Use the feature set with best performance for your use case",
        "   - Consider trade-offs between feature count and performance",
        "   - V3 includes improved calculations that may benefit from more data",
        "",
        "## Notes",
        "",
        "- All comparisons use the same data splits (iterations 007-012)",
        "- Metrics are mean ± std across iterations",
        "- Differences are calculated as: New - Original",
        ""
    ])
    
    # Save report
    report_file = 'features_comparison_report.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Comparison report saved to: {report_file}")
    print()


if __name__ == '__main__':
    generate_comparison_report()


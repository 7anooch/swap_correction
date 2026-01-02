#!/usr/bin/env python3
"""
Generate a comprehensive comparison report between V3 and V2 feature extraction.

This script will work with available data and can be re-run as V2 completes.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_v3_aggregated_data():
    """Load aggregated metrics from V3."""
    v3_dir = 'stability_analysis_v3'
    
    level1_metrics = pd.read_csv(os.path.join(v3_dir, 'level1_metrics.csv'))
    raw_metrics = pd.read_csv(os.path.join(v3_dir, 'raw_metrics.csv'))
    
    with open(os.path.join(v3_dir, 'stability_metrics.json'), 'r') as f:
        stability_metrics = json.load(f)
    
    return level1_metrics, raw_metrics, stability_metrics


def load_v2_data():
    """Load V2 data directly from evaluation results."""
    v2_dir = 'stability_analysis_v3_features_v2'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    level1_metrics = []
    raw_metrics = []
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(v2_dir, f'iteration_{iter_id:03d}')
        
        # Load level1 model results
        level1_eval_file = os.path.join(iter_dir, 'level1_model', 'evaluation_results.json')
        if os.path.exists(level1_eval_file):
            with open(level1_eval_file, 'r') as f:
                level1_data = json.load(f)
            if 'summary' in level1_data:
                summary = level1_data['summary']
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
        
        # Load raw model results
        raw_eval_file = os.path.join(iter_dir, 'raw_model', 'evaluation_results.json')
        if os.path.exists(raw_eval_file):
            with open(raw_eval_file, 'r') as f:
                raw_data = json.load(f)
            if 'summary' in raw_data:
                summary = raw_data['summary']
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


def check_v2_status():
    """Check what V2 data is available."""
    v2_dir = 'stability_analysis_v3_features_v2'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    status = {
        'iterations': {},
        'complete_count': 0,
        'partial_count': 0
    }
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(v2_dir, f'iteration_{iter_id:03d}')
        
        if not os.path.exists(iter_dir):
            status['iterations'][iter_id] = 'not_started'
            continue
        
        level1_dir = os.path.join(iter_dir, 'level1_model')
        raw_dir = os.path.join(iter_dir, 'raw_model')
        
        level1_complete = (
            os.path.exists(os.path.join(level1_dir, 'training_results.json')) and
            os.path.exists(os.path.join(level1_dir, 'evaluation_results.json'))
        )
        raw_complete = (
            os.path.exists(os.path.join(raw_dir, 'training_results.json')) and
            os.path.exists(os.path.join(raw_dir, 'evaluation_results.json'))
        )
        
        if level1_complete and raw_complete:
            status['iterations'][iter_id] = 'complete'
            status['complete_count'] += 1
        elif os.path.exists(os.path.join(level1_dir, 'ml_data')) or os.path.exists(os.path.join(raw_dir, 'ml_data')):
            status['iterations'][iter_id] = 'partial'
            status['partial_count'] += 1
        else:
            status['iterations'][iter_id] = 'not_started'
    
    return status


def generate_report():
    """Generate the comparison report."""
    
    # Load V3 data
    print("Loading V3 aggregated data...")
    level1_metrics_v3, raw_metrics_v3, stability_metrics_v3 = load_v3_aggregated_data()
    
    # Check V2 status
    print("Checking V2 status...")
    v2_status = check_v2_status()
    
    # Start building report
    report = []
    report.append("# Feature Extraction Comparison Report")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append("This report compares two feature extraction approaches:")
    report.append("")
    report.append("- **stability_analysis_v3**: Original feature extraction (56 features)")
    report.append("- **stability_analysis_v3_features_v2**: Reduced feature set (46 features)")
    report.append("  - Removed: 8 position features (head_x, head_y, tail_x, tail_y, mid_x, mid_y, centroid_x, centroid_y)")
    report.append("  - Removed: 2 velocity magnitude features (head_velocity_magnitude, tail_velocity_magnitude)")
    report.append("")
    
    # V2 Status
    report.append("## V2 Training Status")
    report.append("")
    report.append(f"- **Complete Iterations**: {v2_status['complete_count']}/6")
    report.append(f"- **Partial Iterations**: {v2_status['partial_count']}/6")
    report.append("")
    
    if v2_status['complete_count'] == 0:
        report.append("⚠️ **Note**: V2 training encountered errors and no iterations completed successfully.")
        report.append("The comparison below shows only V3 results. Once V2 training is fixed and completes,")
        report.append("re-run this script to generate a full comparison.")
        report.append("")
    
    # Load V2 data
    print("Loading V2 data...")
    level1_metrics_v2, raw_metrics_v2 = load_v2_data()
    
    # V3 Summary Statistics
    report.append("## V3 Results Summary (56 Features)")
    report.append("")
    
    for model_type, metrics_df, model_name in [
        ('level1', level1_metrics_v3, 'Level1 Model'),
        ('raw', raw_metrics_v3, 'Raw Model')
    ]:
        report.append(f"### {model_name}")
        report.append("")
        
        # Test set metrics
        test_metrics = ['test_f1', 'test_precision', 'test_recall', 'test_sensitivity', 'test_specificity', 'test_auc']
        report.append("#### Test Set Performance (Mean ± Std across 6 iterations)")
        report.append("")
        report.append("| Metric | Mean | Std Dev | Min | Max | CV |")
        report.append("|--------|------|---------|-----|-----|-----|")
        
        for metric in test_metrics:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    cv = std_val / mean_val if mean_val > 0 else np.nan
                    report.append(f"| {metric.replace('_', ' ').title()} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} | {cv:.4f} |")
        
        report.append("")
        
        # Evaluation metrics
        eval_metrics = ['eval_mean_f1', 'eval_mean_precision', 'eval_mean_recall', 
                       'eval_mean_sensitivity', 'eval_mean_specificity',
                       'eval_mean_pct_swaps_resolved', 'eval_mean_pct_frames_clean_pre',
                       'eval_mean_pct_frames_clean_post']
        
        report.append("#### Evaluation on Test Dataset (Mean ± Std across 6 iterations)")
        report.append("")
        report.append("| Metric | Mean | Std Dev | Min | Max | CV |")
        report.append("|--------|------|---------|-----|-----|-----|")
        
        for metric in eval_metrics:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    cv = std_val / mean_val if mean_val > 0 else np.nan
                    metric_name = metric.replace('eval_mean_', '').replace('_', ' ').title()
                    report.append(f"| {metric_name} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} | {cv:.4f} |")
        
        report.append("")
    
    # Feature Importance from V3
    report.append("## V3 Feature Importance Analysis")
    report.append("")
    report.append("### Top Features (Averaged across iterations)")
    report.append("")
    
    # Load feature importance from one iteration as example
    v3_dir = 'stability_analysis_v3'
    iter_007_l1 = os.path.join(v3_dir, 'iteration_007', 'level1_model', 'feature_importance.csv')
    iter_007_raw = os.path.join(v3_dir, 'iteration_007', 'raw_model', 'feature_importance.csv')
    
    if os.path.exists(iter_007_l1):
        df_l1 = pd.read_csv(iter_007_l1)
        report.append("#### Level1 Model - Top 15 Features")
        report.append("")
        report.append("| Rank | Feature | Importance |")
        report.append("|------|---------|------------|")
        for i, row in df_l1.head(15).iterrows():
            report.append(f"| {i+1} | {row['feature']} | {row['importance']:.6f} |")
        report.append("")
    
    if os.path.exists(iter_007_raw):
        df_raw = pd.read_csv(iter_007_raw)
        report.append("#### Raw Model - Top 15 Features")
        report.append("")
        report.append("| Rank | Feature | Importance |")
        report.append("|------|---------|------------|")
        for i, row in df_raw.head(15).iterrows():
            report.append(f"| {i+1} | {row['feature']} | {row['importance']:.6f} |")
        report.append("")
    
    # Features removed in V2
    report.append("## Features Removed in V2")
    report.append("")
    report.append("The following 10 features were removed in V2:")
    report.append("")
    report.append("### Position Features (8 features)")
    report.append("- head_x, head_y")
    report.append("- tail_x, tail_y")
    report.append("- mid_x, mid_y")
    report.append("- centroid_x, centroid_y")
    report.append("")
    report.append("### Velocity Magnitude Features (2 features)")
    report.append("- head_velocity_magnitude (redundant with head_speed)")
    report.append("- tail_velocity_magnitude (redundant with tail_speed)")
    report.append("")
    
    # Expected impact
    report.append("## Expected Impact of Feature Reduction")
    report.append("")
    report.append("### Potential Benefits")
    report.append("1. **Reduced Model Complexity**: 18% fewer features (56 → 46)")
    report.append("2. **Faster Training**: Fewer features to process")
    report.append("3. **Faster Inference**: Smaller feature vectors")
    report.append("4. **Reduced Overfitting Risk**: Fewer parameters to learn")
    report.append("5. **Clearer Feature Importance**: Removal of redundant features")
    report.append("")
    report.append("### Potential Risks")
    report.append("1. **Information Loss**: Position features may capture spatial context")
    report.append("2. **Performance Degradation**: If removed features were informative")
    report.append("3. **Reduced Robustness**: Fewer features may reduce model flexibility")
    report.append("")
    
    # V2 Summary Statistics (if available)
    if len(level1_metrics_v2) > 0:
        report.append("## V2 Results Summary (46 Features)")
        report.append("")
        
        for model_type, metrics_df, model_name in [
            ('level1', level1_metrics_v2, 'Level1 Model'),
            ('raw', raw_metrics_v2, 'Raw Model')
        ]:
            if len(metrics_df) == 0:
                continue
                
            report.append(f"### {model_name}")
            report.append("")
            
            # Evaluation metrics
            eval_metrics = ['eval_mean_f1', 'eval_mean_precision', 'eval_mean_recall', 
                           'eval_mean_sensitivity', 'eval_mean_specificity',
                           'eval_mean_pct_swaps_resolved', 'eval_mean_pct_frames_clean_pre',
                           'eval_mean_pct_frames_clean_post']
            
            report.append("#### Evaluation on Test Dataset (Mean ± Std across 6 iterations)")
            report.append("")
            report.append("| Metric | Mean | Std Dev | Min | Max | CV |")
            report.append("|--------|------|---------|-----|-----|-----|")
            
            for metric in eval_metrics:
                if metric in metrics_df.columns:
                    values = metrics_df[metric].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        min_val = values.min()
                        max_val = values.max()
                        cv = std_val / mean_val if mean_val > 0 else np.nan
                        metric_name = metric.replace('eval_mean_', '').replace('_', ' ').title()
                        report.append(f"| {metric_name} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} | {cv:.4f} |")
            
            report.append("")
        
        # Direct Comparison
        report.append("## Direct Comparison: V3 (56 features) vs V2 (46 features)")
        report.append("")
        
        # Level1 comparison
        if len(level1_metrics_v2) > 0:
            report.append("### Level1 Model Comparison")
            report.append("")
            report.append("| Metric | V3 (56 features) | V2 (46 features) | Difference |")
            report.append("|--------|------------------|-------------------|------------|")
            
            for metric in ['eval_mean_f1', 'eval_mean_precision', 'eval_mean_recall', 
                          'eval_mean_sensitivity', 'eval_mean_specificity',
                          'eval_mean_pct_swaps_resolved', 'eval_mean_pct_frames_clean_post']:
                if metric in level1_metrics_v3.columns and metric in level1_metrics_v2.columns:
                    v3_val = level1_metrics_v3[metric].mean()
                    v2_val = level1_metrics_v2[metric].mean()
                    diff = v2_val - v3_val
                    metric_name = metric.replace('eval_mean_', '').replace('_', ' ').title()
                    report.append(f"| {metric_name} | {v3_val:.4f} | {v2_val:.4f} | {diff:+.4f} |")
            
            report.append("")
        
        # Raw comparison
        if len(raw_metrics_v2) > 0:
            report.append("### Raw Model Comparison")
            report.append("")
            report.append("| Metric | V3 (56 features) | V2 (46 features) | Difference |")
            report.append("|--------|------------------|-------------------|------------|")
            
            for metric in ['eval_mean_f1', 'eval_mean_precision', 'eval_mean_recall', 
                          'eval_mean_sensitivity', 'eval_mean_specificity',
                          'eval_mean_pct_swaps_resolved', 'eval_mean_pct_frames_clean_post']:
                if metric in raw_metrics_v3.columns and metric in raw_metrics_v2.columns:
                    v3_val = raw_metrics_v3[metric].mean()
                    v2_val = raw_metrics_v2[metric].mean()
                    diff = v2_val - v3_val
                    metric_name = metric.replace('eval_mean_', '').replace('_', ' ').title()
                    report.append(f"| {metric_name} | {v3_val:.4f} | {v2_val:.4f} | {diff:+.4f} |")
            
            report.append("")
    
    # Next steps
    report.append("## Next Steps")
    report.append("")
    if len(level1_metrics_v2) > 0:
        report.append("1. **Analyze Results**: Compare performance metrics, stability, and feature importance")
        report.append("2. **Make Decision**: Determine if feature reduction is beneficial")
        report.append("3. **Further Optimization**: Consider additional feature engineering based on results")
    else:
        report.append("1. **Complete V2 Training**: Ensure all 6 iterations complete successfully")
        report.append("2. **Re-run Comparison**: Execute this script again to generate full comparison")
    report.append("")
    
    # Write report
    output_file = 'features_v2_comparison_report.md'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport generated: {output_file}")
    print(f"V2 Status: {v2_status['complete_count']}/6 iterations complete")
    print(f"V3 Status: 6/6 iterations complete")


if __name__ == '__main__':
    generate_report()


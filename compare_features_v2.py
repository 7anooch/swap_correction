#!/usr/bin/env python3
"""
Compare stability_analysis_v3 (56 features) vs stability_analysis_v3_features_v2 (46 features).

Generates a comprehensive comparison report.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def load_iteration_results(base_dir: str, iteration_ids: List[int]) -> Dict:
    """Load results from all iterations."""
    results = {}
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(base_dir, f'iteration_{iter_id:03d}')
        
        if not os.path.exists(iter_dir):
            continue
        
        iteration_data = {
            'level1': {},
            'raw': {}
        }
        
        for model_type in ['level1', 'raw']:
            model_dir = os.path.join(iter_dir, f'{model_type}_model')
            
            # Load training results
            training_file = os.path.join(model_dir, 'training_results.json')
            if os.path.exists(training_file):
                with open(training_file, 'r') as f:
                    iteration_data[model_type]['training'] = json.load(f)
            
            # Load evaluation results
            eval_file = os.path.join(model_dir, 'evaluation_results.json')
            if os.path.exists(eval_file):
                with open(eval_file, 'r') as f:
                    iteration_data[model_type]['evaluation'] = json.load(f)
            
            # Load feature importance
            importance_file = os.path.join(model_dir, 'feature_importance.csv')
            if os.path.exists(importance_file):
                iteration_data[model_type]['feature_importance'] = pd.read_csv(importance_file)
        
        results[iter_id] = iteration_data
    
    return results


def extract_metrics(results: Dict, model_type: str) -> pd.DataFrame:
    """Extract metrics from results into a DataFrame."""
    metrics_list = []
    
    for iter_id, data in results.items():
        if model_type not in data:
            continue
        
        model_data = data[model_type]
        
        metrics = {
            'iteration': iter_id,
        }
        
        # Training metrics
        if 'training' in model_data:
            training = model_data['training']
            for split in ['train', 'val', 'test']:
                if split in training:
                    split_data = training[split]
                    metrics[f'{split}_f1'] = split_data.get('f1', np.nan)
                    metrics[f'{split}_precision'] = split_data.get('precision', np.nan)
                    metrics[f'{split}_recall'] = split_data.get('recall', np.nan)
                    metrics[f'{split}_sensitivity'] = split_data.get('sensitivity', np.nan)
                    metrics[f'{split}_specificity'] = split_data.get('specificity', np.nan)
                    metrics[f'{split}_auc'] = split_data.get('auc', np.nan)
        
        # Evaluation metrics
        if 'evaluation' in model_data:
            evaluation = model_data['evaluation']
            if 'summary' in evaluation:
                summary = evaluation['summary']
                metrics['eval_mean_f1'] = summary.get('mean_f1', np.nan)
                metrics['eval_mean_precision'] = summary.get('mean_precision', np.nan)
                metrics['eval_mean_recall'] = summary.get('mean_recall', np.nan)
                metrics['eval_mean_sensitivity'] = summary.get('mean_sensitivity', np.nan)
                metrics['eval_mean_specificity'] = summary.get('mean_specificity', np.nan)
                metrics['eval_mean_pct_swaps_resolved'] = summary.get('mean_pct_swaps_resolved', np.nan)
                metrics['eval_mean_pct_frames_clean_pre'] = summary.get('mean_pct_frames_clean_pre', np.nan)
                metrics['eval_mean_pct_frames_clean_post'] = summary.get('mean_pct_frames_clean_post', np.nan)
        
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def calculate_summary_stats(df: pd.DataFrame, prefix: str = '') -> Dict:
    """Calculate summary statistics for a DataFrame."""
    stats = {}
    
    for col in df.columns:
        if col == 'iteration':
            continue
        
        values = df[col].dropna()
        if len(values) > 0:
            stats[col] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'median': float(values.median()),
                'cv': float(values.std() / values.mean()) if values.mean() > 0 else np.nan,
                'n': len(values)
            }
    
    return stats


def compare_feature_importance(results_v3: Dict, results_v2: Dict, model_type: str) -> Dict:
    """Compare feature importance between v3 and v2."""
    comparison = {}
    
    # Collect all feature importance data
    v3_importance = {}
    v2_importance = {}
    
    for iter_id in results_v3.keys():
        if model_type in results_v3[iter_id] and 'feature_importance' in results_v3[iter_id][model_type]:
            df = results_v3[iter_id][model_type]['feature_importance']
            for _, row in df.iterrows():
                feature = row['feature']
                importance = row['importance']
                if feature not in v3_importance:
                    v3_importance[feature] = []
                v3_importance[feature].append(importance)
    
    for iter_id in results_v2.keys():
        if model_type in results_v2[iter_id] and 'feature_importance' in results_v2[iter_id][model_type]:
            df = results_v2[iter_id][model_type]['feature_importance']
            for _, row in df.iterrows():
                feature = row['feature']
                importance = row['importance']
                if feature not in v2_importance:
                    v2_importance[feature] = []
                v2_importance[feature].append(importance)
    
    # Calculate averages
    v3_avg = {f: np.mean(v) for f, v in v3_importance.items()}
    v2_avg = {f: np.mean(v) for f, v in v2_importance.items()}
    
    # Find common features
    common_features = set(v3_avg.keys()) & set(v2_avg.keys())
    v3_only = set(v3_avg.keys()) - set(v2_avg.keys())
    v2_only = set(v2_avg.keys()) - set(v3_avg.keys())
    
    comparison['common_features'] = len(common_features)
    comparison['v3_only_features'] = len(v3_only)
    comparison['v2_only_features'] = len(v2_only)
    comparison['v3_only_feature_list'] = sorted(list(v3_only))
    comparison['v2_only_feature_list'] = sorted(list(v2_only))
    
    # Compare importance for common features
    importance_changes = {}
    for feature in common_features:
        importance_changes[feature] = {
            'v3_mean': v3_avg[feature],
            'v2_mean': v2_avg[feature],
            'change': v2_avg[feature] - v3_avg[feature],
            'change_pct': ((v2_avg[feature] - v3_avg[feature]) / v3_avg[feature] * 100) if v3_avg[feature] > 0 else 0
        }
    
    comparison['importance_changes'] = importance_changes
    
    # Top features in each version
    v3_top = sorted(v3_avg.items(), key=lambda x: x[1], reverse=True)[:10]
    v2_top = sorted(v2_avg.items(), key=lambda x: x[1], reverse=True)[:10]
    
    comparison['v3_top_10'] = v3_top
    comparison['v2_top_10'] = v2_top
    
    return comparison


def generate_comparison_report(results_v3: Dict, results_v2: Dict, output_file: str):
    """Generate comprehensive comparison report."""
    
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
    
    # Extract metrics
    v3_level1_metrics = extract_metrics(results_v3, 'level1')
    v3_raw_metrics = extract_metrics(results_v3, 'raw')
    v2_level1_metrics = extract_metrics(results_v2, 'level1')
    v2_raw_metrics = extract_metrics(results_v2, 'raw')
    
    # Calculate summary statistics
    v3_level1_stats = calculate_summary_stats(v3_level1_metrics)
    v3_raw_stats = calculate_summary_stats(v3_raw_metrics)
    v2_level1_stats = calculate_summary_stats(v2_level1_metrics)
    v2_raw_stats = calculate_summary_stats(v2_raw_metrics)
    
    # Compare metrics
    report.append("## Performance Comparison")
    report.append("")
    
    for model_type, v3_stats, v2_stats, v3_df, v2_df in [
        ('Level1', v3_level1_stats, v2_level1_stats, v3_level1_metrics, v2_level1_metrics),
        ('Raw', v3_raw_stats, v2_raw_stats, v3_raw_metrics, v2_raw_metrics)
    ]:
        report.append(f"### {model_type} Model")
        report.append("")
        
        # Training metrics comparison
        report.append("#### Training Set Performance")
        report.append("")
        report.append("| Metric | V3 (56 features) | V2 (46 features) | Difference |")
        report.append("|--------|-------------------|-----------------|------------|")
        
        for metric in ['train_f1', 'train_precision', 'train_recall', 'train_sensitivity', 'train_specificity', 'train_auc']:
            if metric in v3_stats and metric in v2_stats:
                v3_val = v3_stats[metric]['mean']
                v2_val = v2_stats[metric]['mean']
                diff = v2_val - v3_val
                report.append(f"| {metric.replace('_', ' ').title()} | {v3_val:.4f} ± {v3_stats[metric]['std']:.4f} | {v2_val:.4f} ± {v2_stats[metric]['std']:.4f} | {diff:+.4f} |")
        
        report.append("")
        report.append("#### Validation Set Performance")
        report.append("")
        report.append("| Metric | V3 (56 features) | V2 (46 features) | Difference |")
        report.append("|--------|-------------------|-----------------|------------|")
        
        for metric in ['val_f1', 'val_precision', 'val_recall', 'val_sensitivity', 'val_specificity', 'val_auc']:
            if metric in v3_stats and metric in v2_stats:
                v3_val = v3_stats[metric]['mean']
                v2_val = v2_stats[metric]['mean']
                diff = v2_val - v3_val
                report.append(f"| {metric.replace('_', ' ').title()} | {v3_val:.4f} ± {v3_stats[metric]['std']:.4f} | {v2_val:.4f} ± {v2_stats[metric]['std']:.4f} | {diff:+.4f} |")
        
        report.append("")
        report.append("#### Test Set Performance")
        report.append("")
        report.append("| Metric | V3 (56 features) | V2 (46 features) | Difference |")
        report.append("|--------|-------------------|-----------------|------------|")
        
        for metric in ['test_f1', 'test_precision', 'test_recall', 'test_sensitivity', 'test_specificity', 'test_auc']:
            if metric in v3_stats and metric in v2_stats:
                v3_val = v3_stats[metric]['mean']
                v2_val = v2_stats[metric]['mean']
                diff = v2_val - v3_val
                report.append(f"| {metric.replace('_', ' ').title()} | {v3_val:.4f} ± {v3_stats[metric]['std']:.4f} | {v2_val:.4f} ± {v2_stats[metric]['std']:.4f} | {diff:+.4f} |")
        
        report.append("")
        report.append("#### Evaluation on Test Dataset")
        report.append("")
        report.append("| Metric | V3 (56 features) | V2 (46 features) | Difference |")
        report.append("|--------|-------------------|-----------------|------------|")
        
        for metric in ['eval_mean_f1', 'eval_mean_precision', 'eval_mean_recall', 'eval_mean_sensitivity', 
                       'eval_mean_specificity', 'eval_mean_pct_swaps_resolved', 
                       'eval_mean_pct_frames_clean_pre', 'eval_mean_pct_frames_clean_post']:
            if metric in v3_stats and metric in v2_stats:
                v3_val = v3_stats[metric]['mean']
                v2_val = v2_stats[metric]['mean']
                diff = v2_val - v3_val
                report.append(f"| {metric.replace('eval_mean_', '').replace('_', ' ').title()} | {v3_val:.4f} ± {v3_stats[metric]['std']:.4f} | {v2_val:.4f} ± {v2_stats[metric]['std']:.4f} | {diff:+.4f} |")
        
        report.append("")
    
    # Feature importance comparison
    report.append("## Feature Importance Comparison")
    report.append("")
    
    for model_type in ['level1', 'raw']:
        report.append(f"### {model_type.capitalize()} Model")
        report.append("")
        
        try:
            importance_comp = compare_feature_importance(results_v3, results_v2, model_type)
            
            report.append(f"**Common Features**: {importance_comp['common_features']}")
            report.append(f"**V3 Only Features**: {importance_comp['v3_only_features']}")
            report.append(f"**V2 Only Features**: {importance_comp['v2_only_features']}")
            report.append("")
            
            if importance_comp['v3_only_features'] > 0:
                report.append("**Features Removed in V2**:")
                for feature in importance_comp['v3_only_feature_list']:
                    report.append(f"- {feature}")
                report.append("")
            
            # Top features comparison
            report.append("#### Top 10 Features (V3)")
            report.append("")
            report.append("| Rank | Feature | Mean Importance |")
            report.append("|------|---------|----------------|")
            for i, (feature, importance) in enumerate(importance_comp['v3_top_10'], 1):
                report.append(f"| {i} | {feature} | {importance:.6f} |")
            report.append("")
            
            report.append("#### Top 10 Features (V2)")
            report.append("")
            report.append("| Rank | Feature | Mean Importance |")
            report.append("|------|---------|----------------|")
            for i, (feature, importance) in enumerate(importance_comp['v2_top_10'], 1):
                report.append(f"| {i} | {feature} | {importance:.6f} |")
            report.append("")
            
            # Importance changes for common features
            if importance_comp['importance_changes']:
                report.append("#### Importance Changes for Common Features")
                report.append("")
                report.append("| Feature | V3 Mean | V2 Mean | Change | Change % |")
                report.append("|---------|---------|---------|--------|----------|")
                
                # Sort by absolute change
                sorted_changes = sorted(
                    importance_comp['importance_changes'].items(),
                    key=lambda x: abs(x[1]['change']),
                    reverse=True
                )[:20]  # Top 20 changes
                
                for feature, change_data in sorted_changes:
                    report.append(f"| {feature} | {change_data['v3_mean']:.6f} | {change_data['v2_mean']:.6f} | "
                                f"{change_data['change']:+.6f} | {change_data['change_pct']:+.2f}% |")
                report.append("")
        
        except Exception as e:
            report.append(f"*Error comparing feature importance: {e}*")
            report.append("")
    
    # Stability comparison
    report.append("## Stability Comparison")
    report.append("")
    report.append("### Coefficient of Variation (CV)")
    report.append("")
    report.append("Lower CV indicates more stable performance across iterations.")
    report.append("")
    
    for model_type, v3_stats, v2_stats in [
        ('Level1', v3_level1_stats, v2_level1_stats),
        ('Raw', v3_raw_stats, v2_raw_stats)
    ]:
        report.append(f"#### {model_type} Model")
        report.append("")
        report.append("| Metric | V3 CV | V2 CV |")
        report.append("|--------|-------|-------|")
        
        for metric in ['test_f1', 'test_precision', 'test_recall', 'eval_mean_f1']:
            if metric in v3_stats and metric in v2_stats:
                v3_cv = v3_stats[metric].get('cv', np.nan)
                v2_cv = v2_stats[metric].get('cv', np.nan)
                if not np.isnan(v3_cv) and not np.isnan(v2_cv):
                    report.append(f"| {metric.replace('_', ' ').title()} | {v3_cv:.4f} | {v2_cv:.4f} |")
        report.append("")
    
    # Summary and conclusions
    report.append("## Summary and Conclusions")
    report.append("")
    report.append("### Key Findings")
    report.append("")
    
    # Calculate overall differences
    if 'test_f1' in v3_level1_stats and 'test_f1' in v2_level1_stats:
        l1_f1_diff = v2_level1_stats['test_f1']['mean'] - v3_level1_stats['test_f1']['mean']
        report.append(f"- **Level1 Model Test F1**: {l1_f1_diff:+.4f} change")
    
    if 'test_f1' in v3_raw_stats and 'test_f1' in v2_raw_stats:
        raw_f1_diff = v2_raw_stats['test_f1']['mean'] - v3_raw_stats['test_f1']['mean']
        report.append(f"- **Raw Model Test F1**: {raw_f1_diff:+.4f} change")
    
    report.append("")
    report.append("### Recommendations")
    report.append("")
    report.append("Based on the comparison:")
    report.append("")
    report.append("1. **Feature Reduction Impact**: Evaluate whether removing 10 features (18% reduction) affects model performance")
    report.append("2. **Performance Trade-offs**: Compare accuracy vs. model complexity and training time")
    report.append("3. **Feature Importance**: Analyze how removing redundant features affects feature importance distribution")
    report.append("4. **Stability**: Assess whether reduced feature set improves or degrades stability across iterations")
    report.append("")
    
    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Comparison report saved to: {output_file}")


def main():
    """Main function."""
    v3_dir = 'stability_analysis_v3'
    v2_dir = 'stability_analysis_v3_features_v2'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    output_file = 'features_v2_comparison_report.md'
    
    print("=" * 80)
    print("FEATURE EXTRACTION COMPARISON")
    print("=" * 80)
    print(f"V3 Directory: {v3_dir}")
    print(f"V2 Directory: {v2_dir}")
    print(f"Iterations: {iteration_ids}")
    print()
    
    # Load results
    print("Loading V3 results...")
    results_v3 = load_iteration_results(v3_dir, iteration_ids)
    print(f"  Loaded {len(results_v3)} iterations")
    
    print("Loading V2 results...")
    results_v2 = load_iteration_results(v2_dir, iteration_ids)
    print(f"  Loaded {len(results_v2)} iterations")
    print()
    
    # Check completeness
    v3_complete = sum(1 for r in results_v3.values() 
                     if 'level1' in r and 'training' in r['level1'] and 'evaluation' in r['level1'])
    v2_complete = sum(1 for r in results_v2.values() 
                     if 'level1' in r and 'training' in r['level1'] and 'evaluation' in r['level1'])
    
    print(f"V3 Complete iterations: {v3_complete}/{len(results_v3)}")
    print(f"V2 Complete iterations: {v2_complete}/{len(results_v2)}")
    print()
    
    if v3_complete == 0:
        print("ERROR: No complete V3 results found!")
        return
    
    if v2_complete == 0:
        print("WARNING: No complete V2 results found. Report will be generated with available data.")
        print("         Re-run this script once V2 training completes.")
        print()
    
    # Generate comparison report
    print("Generating comparison report...")
    generate_comparison_report(results_v3, results_v2, output_file)
    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


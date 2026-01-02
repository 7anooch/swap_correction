#!/usr/bin/env python3
"""
Run threshold optimization across all stability analysis iterations and aggregate results.

This script runs threshold optimization for each iteration (007-012) for both
level1 and raw models, then aggregates the results.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import subprocess


def run_threshold_optimization(iteration_id: int, model_type: str, base_dir: str):
    """Run threshold optimization for a single iteration and model type."""
    # Determine feature version from base_dir
    if 'features_v4' in base_dir:
        suffix = '_v4'
    elif 'features_v2' in base_dir:
        suffix = '_v2'
    elif 'features_v3' in base_dir:
        suffix = '_v3'
    else:
        suffix = ''
    
    output_dir = f'threshold_analysis_{model_type}_iter{iteration_id:03d}{suffix}'
    
    cmd = [
        sys.executable,
        'run_threshold_optimization.py',
        '--base-dir', base_dir,
        '--iteration', str(iteration_id),
        '--model-type', model_type,
        '--metric', 'pct_clean_post',
        '--split', 'val',
        '--output-dir', output_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Running threshold optimization: Iteration {iteration_id:03d}, {model_type} model")
    print(f"{'='*80}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Successfully completed iteration {iteration_id:03d}, {model_type} model")
        return True, output_dir
    else:
        print(f"✗ Failed for iteration {iteration_id:03d}, {model_type} model")
        print(f"Error: {result.stderr}")
        return False, output_dir


def load_threshold_results(output_dir: str):
    """Load threshold optimization results from a directory."""
    results_file = os.path.join(output_dir, 'optimal_threshold.json')
    
    if not os.path.exists(results_file):
        return None
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Could not parse {results_file}: {e}")
        return None


def aggregate_results(base_dir: str, iteration_ids: list, model_type: str):
    """Aggregate threshold optimization results across iterations."""
    all_results = []
    
    # Determine feature version from base_dir
    if 'features_v4' in base_dir:
        suffix = '_v4'
    elif 'features_v2' in base_dir:
        suffix = '_v2'
    elif 'features_v3' in base_dir:
        suffix = '_v3'
    else:
        suffix = ''
    
    for iter_id in iteration_ids:
        output_dir = f'threshold_analysis_{model_type}_iter{iter_id:03d}{suffix}'
        results = load_threshold_results(output_dir)
        
        if results is not None:
            results['iteration'] = iter_id
            all_results.append(results)
    
    if not all_results:
        print(f"Warning: No valid results found for {model_type} model")
        return None
    
    if len(all_results) < len(iteration_ids):
        print(f"Warning: Only {len(all_results)}/{len(iteration_ids)} iterations have valid results for {model_type} model")
    
    # Create summary statistics
    thresholds = [r['optimal_threshold'] for r in all_results]
    pct_clean_post = [r['best_metrics']['pct_frames_clean_post'] for r in all_results]
    precision = [r['best_metrics']['precision'] for r in all_results]
    recall = [r['best_metrics']['recall'] for r in all_results]
    f1 = [r['best_metrics']['f1'] for r in all_results]
    pct_swaps_resolved = [r['best_metrics']['pct_swaps_resolved'] for r in all_results]
    
    summary = {
        'model_type': model_type,
        'n_iterations': len(all_results),
        'optimal_threshold': {
            'mean': float(np.mean(thresholds)),
            'std': float(np.std(thresholds)),
            'min': float(np.min(thresholds)),
            'max': float(np.max(thresholds)),
            'median': float(np.median(thresholds))
        },
        'pct_frames_clean_post': {
            'mean': float(np.mean(pct_clean_post)),
            'std': float(np.std(pct_clean_post)),
            'min': float(np.min(pct_clean_post)),
            'max': float(np.max(pct_clean_post)),
            'median': float(np.median(pct_clean_post))
        },
        'precision': {
            'mean': float(np.mean(precision)),
            'std': float(np.std(precision)),
            'min': float(np.min(precision)),
            'max': float(np.max(precision))
        },
        'recall': {
            'mean': float(np.mean(recall)),
            'std': float(np.std(recall)),
            'min': float(np.min(recall)),
            'max': float(np.max(recall))
        },
        'f1': {
            'mean': float(np.mean(f1)),
            'std': float(np.std(f1)),
            'min': float(np.min(f1)),
            'max': float(np.max(f1))
        },
        'pct_swaps_resolved': {
            'mean': float(np.mean(pct_swaps_resolved)),
            'std': float(np.std(pct_swaps_resolved)),
            'min': float(np.min(pct_swaps_resolved)),
            'max': float(np.max(pct_swaps_resolved))
        },
        'per_iteration': all_results
    }
    
    return summary


def generate_aggregated_report(level1_summary: dict, raw_summary: dict, output_file: str, feature_version: str = 'v3'):
    """Generate a markdown report with aggregated results."""
    report_lines = [
        f"# Threshold Optimization: Aggregated Results (Features {feature_version.upper()})",
        "",
        "This report aggregates threshold optimization results across 6 iterations",
        "(007-012) from the stability analysis.",
        "",
        "## Summary Statistics",
        "",
        "### Level1 Model",
        "",
        f"**Optimal Threshold**: {level1_summary['optimal_threshold']['mean']:.4f} ± {level1_summary['optimal_threshold']['std']:.4f}",
        f"- Range: {level1_summary['optimal_threshold']['min']:.4f} - {level1_summary['optimal_threshold']['max']:.4f}",
        f"- Median: {level1_summary['optimal_threshold']['median']:.4f}",
        "",
        f"**% Frames Clean Post**: {level1_summary['pct_frames_clean_post']['mean']:.2f}% ± {level1_summary['pct_frames_clean_post']['std']:.2f}%",
        f"- Range: {level1_summary['pct_frames_clean_post']['min']:.2f}% - {level1_summary['pct_frames_clean_post']['max']:.2f}%",
        "",
        f"**Precision**: {level1_summary['precision']['mean']:.4f} ± {level1_summary['precision']['std']:.4f}",
        f"**Recall**: {level1_summary['recall']['mean']:.4f} ± {level1_summary['recall']['std']:.4f}",
        f"**F1-Score**: {level1_summary['f1']['mean']:.4f} ± {level1_summary['f1']['std']:.4f}",
        f"**% Swaps Resolved**: {level1_summary['pct_swaps_resolved']['mean']:.2f}% ± {level1_summary['pct_swaps_resolved']['std']:.2f}%",
        "",
        "### Raw Model",
        "",
        f"**Optimal Threshold**: {raw_summary['optimal_threshold']['mean']:.4f} ± {raw_summary['optimal_threshold']['std']:.4f}",
        f"- Range: {raw_summary['optimal_threshold']['min']:.4f} - {raw_summary['optimal_threshold']['max']:.4f}",
        f"- Median: {raw_summary['optimal_threshold']['median']:.4f}",
        "",
        f"**% Frames Clean Post**: {raw_summary['pct_frames_clean_post']['mean']:.2f}% ± {raw_summary['pct_frames_clean_post']['std']:.2f}%",
        f"- Range: {raw_summary['pct_frames_clean_post']['min']:.2f}% - {raw_summary['pct_frames_clean_post']['max']:.2f}%",
        "",
        f"**Precision**: {raw_summary['precision']['mean']:.4f} ± {raw_summary['precision']['std']:.4f}",
        f"**Recall**: {raw_summary['recall']['mean']:.4f} ± {raw_summary['recall']['std']:.4f}",
        f"**F1-Score**: {raw_summary['f1']['mean']:.4f} ± {raw_summary['f1']['std']:.4f}",
        f"**% Swaps Resolved**: {raw_summary['pct_swaps_resolved']['mean']:.2f}% ± {raw_summary['pct_swaps_resolved']['std']:.2f}%",
        "",
        "## Per-Iteration Results",
        "",
        "### Level1 Model",
        "",
        "| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |",
        "|-----------|-------------------|--------------|-----------|--------|----|------------------|"
    ]
    
    for result in level1_summary['per_iteration']:
        iter_id = result['iteration']
        threshold = result['optimal_threshold']
        metrics = result['best_metrics']
        report_lines.append(
            f"| {iter_id:03d} | {threshold:.4f} | {metrics['pct_frames_clean_post']:.2f}% | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['pct_swaps_resolved']:.2f}% |"
        )
    
    report_lines.extend([
        "",
        "### Raw Model",
        "",
        "| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |",
        "|-----------|-------------------|--------------|-----------|--------|----|------------------|"
    ])
    
    for result in raw_summary['per_iteration']:
        iter_id = result['iteration']
        threshold = result['optimal_threshold']
        metrics = result['best_metrics']
        report_lines.append(
            f"| {iter_id:03d} | {threshold:.4f} | {metrics['pct_frames_clean_post']:.2f}% | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['pct_swaps_resolved']:.2f}% |"
        )
    
    report_lines.extend([
        "",
        "## Recommendations",
        "",
        f"### Level1 Model",
        f"- **Recommended Threshold**: {level1_summary['optimal_threshold']['median']:.4f} (median across iterations)",
        f"- **Expected Performance**: {level1_summary['pct_frames_clean_post']['mean']:.2f}% ± {level1_summary['pct_frames_clean_post']['std']:.2f}% frames clean post",
        "",
        f"### Raw Model",
        f"- **Recommended Threshold**: {raw_summary['optimal_threshold']['median']:.4f} (median across iterations)",
        f"- **Expected Performance**: {raw_summary['pct_frames_clean_post']['mean']:.2f}% ± {raw_summary['pct_frames_clean_post']['std']:.2f}% frames clean post",
        "",
        "## Usage",
        "",
        "```python",
        "from swap_correction.ml.api import SwapPredictor",
        "",
        "# Level1 model",
        "predictor_level1 = SwapPredictor(model_type='level1')",
        f"predictions = predictor_level1.predict(trial_data, fps=30, threshold={level1_summary['optimal_threshold']['median']:.4f})",
        "",
        "# Raw model",
        "predictor_raw = SwapPredictor(model_type='raw')",
        f"predictions = predictor_raw.predict(trial_data, fps=30, threshold={raw_summary['optimal_threshold']['median']:.4f})",
        "```"
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ Aggregated report saved to: {output_file}")


def main():
    """Run threshold optimization for all iterations and aggregate results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run threshold optimization for all iterations')
    parser.add_argument('--feature-version', type=str, default='v4',
                       choices=['v2', 'v3', 'v4'],
                       help='Feature version to use (v2, v3, or v4)')
    
    args = parser.parse_args()
    
    if args.feature_version == 'v2':
        base_dir = 'stability_analysis_v3_features_v2'
        output_suffix = '_v2'
    elif args.feature_version == 'v3':
        base_dir = 'stability_analysis_v3_features_v3'
        output_suffix = '_v3'
    else:  # v4
        base_dir = 'stability_analysis_v3_features_v4'
        output_suffix = '_v4'
    
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    print("=" * 80)
    print(f"THRESHOLD OPTIMIZATION: ALL ITERATIONS (Features {args.feature_version.upper()})")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Iterations: {iteration_ids}")
    print()
    
    # Run optimization for all iterations
    for model_type in ['level1', 'raw']:
        print(f"\n{'='*80}")
        print(f"PROCESSING {model_type.upper()} MODELS")
        print(f"{'='*80}")
        
        for iter_id in iteration_ids:
            success, output_dir = run_threshold_optimization(iter_id, model_type, base_dir)
            if not success:
                print(f"Warning: Failed for iteration {iter_id:03d}, {model_type} model")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS")
    print("=" * 80)
    
    level1_summary = aggregate_results(base_dir, iteration_ids, 'level1')
    raw_summary = aggregate_results(base_dir, iteration_ids, 'raw')
    
    if level1_summary is None or raw_summary is None:
        print("Error: Could not aggregate results. Some iterations may have failed.")
        return
    
    # Save aggregated summaries
    if 'features_v4' in base_dir:
        agg_dir = 'threshold_analysis_aggregated_v4'
        feature_version = 'v4'
    elif 'features_v2' in base_dir:
        agg_dir = 'threshold_analysis_aggregated_v2'
        feature_version = 'v2'
    else:
        agg_dir = 'threshold_analysis_aggregated'
        feature_version = 'v3'
    
    os.makedirs(agg_dir, exist_ok=True)
    
    with open(f'{agg_dir}/level1_summary.json', 'w') as f:
        json.dump(level1_summary, f, indent=2)
    
    with open(f'{agg_dir}/raw_summary.json', 'w') as f:
        json.dump(raw_summary, f, indent=2)
    
    print("✓ Aggregated summaries saved")
    
    # Generate report
    report_file = f'{agg_dir}/aggregated_threshold_report.md'
    generate_aggregated_report(level1_summary, raw_summary, report_file, feature_version)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nLevel1 Model:")
    print(f"  Optimal Threshold: {level1_summary['optimal_threshold']['mean']:.4f} ± {level1_summary['optimal_threshold']['std']:.4f}")
    print(f"  % Frames Clean Post: {level1_summary['pct_frames_clean_post']['mean']:.2f}% ± {level1_summary['pct_frames_clean_post']['std']:.2f}%")
    
    print(f"\nRaw Model:")
    print(f"  Optimal Threshold: {raw_summary['optimal_threshold']['mean']:.4f} ± {raw_summary['optimal_threshold']['std']:.4f}")
    print(f"  % Frames Clean Post: {raw_summary['pct_frames_clean_post']['mean']:.2f}% ± {raw_summary['pct_frames_clean_post']['std']:.2f}%")
    print()


if __name__ == '__main__':
    main()


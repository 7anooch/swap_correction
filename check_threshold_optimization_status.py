#!/usr/bin/env python3
"""
Check the status of threshold optimization across all iterations.
"""

import os
import json
from pathlib import Path


def check_status():
    """Check status of threshold optimization runs."""
    iteration_ids = [7, 8, 9, 10, 11, 12]
    model_types = ['level1', 'raw']
    feature_versions = ['v2', 'v3']
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION STATUS")
    print("=" * 80)
    print()
    
    total = len(iteration_ids) * len(model_types) * len(feature_versions)
    completed = 0
    failed = 0
    
    results = {}
    
    for feature_version in feature_versions:
        suffix = f'_{feature_version}'
        print(f"\n{'='*80}")
        print(f"FEATURES {feature_version.upper()}")
        print(f"{'='*80}")
        
        results[feature_version] = {}
        
        for model_type in model_types:
            results[feature_version][model_type] = {}
            print(f"\n{model_type.upper()} Model:")
            
            for iter_id in iteration_ids:
                output_dir = f'threshold_analysis_{model_type}_iter{iter_id:03d}{suffix}'
                results_file = os.path.join(output_dir, 'optimal_threshold.json')
            
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        threshold = data['optimal_threshold']
                        pct_clean = data['best_metrics']['pct_frames_clean_post']
                        print(f"  Iteration {iter_id:03d}: ✓ Threshold={threshold:.4f}, % Clean={pct_clean:.2f}%")
                        results[feature_version][model_type][iter_id] = 'completed'
                        completed += 1
                    except Exception as e:
                        print(f"  Iteration {iter_id:03d}: ✗ Error reading results: {e}")
                        results[feature_version][model_type][iter_id] = 'error'
                        failed += 1
                else:
                    print(f"  Iteration {iter_id:03d}: ⏳ Not started or in progress")
                    results[feature_version][model_type][iter_id] = 'pending'
    
    print("\n" + "=" * 80)
    print(f"Overall Progress: {completed}/{total} completed, {failed} failed, {total - completed - failed} pending")
    print("=" * 80)
    
    # Check for aggregated results
    for feature_version in feature_versions:
        if feature_version == 'v2':
            agg_dir = 'threshold_analysis_aggregated_v2'
        else:
            agg_dir = 'threshold_analysis_aggregated'
        
        if os.path.exists(f'{agg_dir}/aggregated_threshold_report.md'):
            print(f"\n✓ {feature_version.upper()} aggregated report available: {agg_dir}/aggregated_threshold_report.md")
    
    return results


if __name__ == '__main__':
    check_status()


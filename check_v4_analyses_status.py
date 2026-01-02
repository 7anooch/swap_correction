#!/usr/bin/env python3
"""
Check the status of threshold optimization and learning curve analyses for features_v4.
"""

import os
import json
from pathlib import Path

def check_threshold_optimization():
    """Check status of threshold optimization."""
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION STATUS (Features V4)")
    print("=" * 80)
    
    iteration_ids = [7, 8, 9, 10, 11, 12]
    model_types = ['level1', 'raw']
    
    completed = []
    in_progress = []
    not_started = []
    
    for model_type in model_types:
        for iter_id in iteration_ids:
            output_dir = f'threshold_analysis_{model_type}_iter{iter_id:03d}_v4'
            results_file = os.path.join(output_dir, 'optimal_threshold.json')
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    threshold = results.get('optimal_threshold', None)
                    pct_clean = results.get('pct_frames_clean_post', None)
                    completed.append({
                        'iteration': iter_id,
                        'model_type': model_type,
                        'optimal_threshold': threshold if threshold is not None else 'N/A',
                        'pct_clean_post': pct_clean if pct_clean is not None else 'N/A'
                    })
                except:
                    in_progress.append({'iteration': iter_id, 'model_type': model_type})
            elif os.path.exists(output_dir):
                in_progress.append({'iteration': iter_id, 'model_type': model_type})
            else:
                not_started.append({'iteration': iter_id, 'model_type': model_type})
    
    print(f"\n✓ Completed: {len(completed)}/{len(iteration_ids) * len(model_types)}")
    if completed:
        print("\nCompleted:")
        for item in completed[:10]:  # Show first 10
            threshold_str = f"{item['optimal_threshold']:.3f}" if isinstance(item['optimal_threshold'], (int, float)) else str(item['optimal_threshold'])
            pct_str = f"{item['pct_clean_post']:.2f}%" if isinstance(item['pct_clean_post'], (int, float)) else str(item['pct_clean_post'])
            print(f"  Iter {item['iteration']:03d} {item['model_type']:6s}: threshold={threshold_str}, clean_post={pct_str}")
        if len(completed) > 10:
            print(f"  ... and {len(completed) - 10} more")
    
    print(f"\n⏳ In Progress: {len(in_progress)}/{len(iteration_ids) * len(model_types)}")
    if in_progress:
        print(f"  Items: {len(in_progress)}")
    
    print(f"\n○ Not Started: {len(not_started)}/{len(iteration_ids) * len(model_types)}")
    
    # Check for aggregated results
    agg_dir = 'threshold_analysis_aggregated_v4'
    if os.path.exists(os.path.join(agg_dir, 'level1_summary.json')):
        print(f"\n✓ Aggregated results available in: {agg_dir}")
    
    print()


def check_learning_curve():
    """Check status of learning curve analysis."""
    print("=" * 80)
    print("LEARNING CURVE ANALYSIS STATUS (Features V4)")
    print("=" * 80)
    
    iteration_ids = [7, 8]
    model_types = ['level1', 'raw']
    
    completed = []
    in_progress = []
    not_started = []
    
    for iter_id in iteration_ids:
        for model_type in model_types:
            output_dir = f'learning_curve_analysis_v4_iter{iter_id:03d}'
            results_file = os.path.join(output_dir, f'{model_type}_results.csv')
            summary_file = os.path.join(output_dir, f'{model_type}_summary_report.md')
            
            if os.path.exists(results_file) and os.path.exists(summary_file):
                completed.append({'iteration': iter_id, 'model_type': model_type})
            elif os.path.exists(output_dir):
                in_progress.append({'iteration': iter_id, 'model_type': model_type})
            else:
                not_started.append({'iteration': iter_id, 'model_type': model_type})
    
    print(f"\n✓ Completed: {len(completed)}/{len(iteration_ids) * len(model_types)}")
    if completed:
        print("\nCompleted:")
        for item in completed:
            print(f"  Iter {item['iteration']:03d} {item['model_type']:6s}")
    
    print(f"\n⏳ In Progress: {len(in_progress)}/{len(iteration_ids) * len(model_types)}")
    if in_progress:
        print("\nIn Progress:")
        for item in in_progress:
            print(f"  Iter {item['iteration']:03d} {item['model_type']:6s}")
    
    print(f"\n○ Not Started: {len(not_started)}/{len(iteration_ids) * len(model_types)}")
    print()


def main():
    """Check status of both analyses."""
    check_threshold_optimization()
    check_learning_curve()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nTo monitor progress:")
    print("  - Threshold optimization: tail -f threshold_optimization_v4.log")
    print("  - Learning curve (iter 7): tail -f learning_curve_v4_iter007.log")
    print("  - Learning curve (iter 8): tail -f learning_curve_v4_iter008.log")
    print()


if __name__ == '__main__':
    main()


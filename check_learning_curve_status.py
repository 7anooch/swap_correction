#!/usr/bin/env python3
"""
Check the status of learning curve analysis.
"""

import os
import pandas as pd


def check_status():
    """Check status of learning curve analysis."""
    output_dir = 'learning_curve_analysis_iter007'
    
    combinations = [
        ('v2', 'level1'),
        ('v2', 'raw'),
        ('v3', 'level1'),
        ('v3', 'raw')
    ]
    
    print("=" * 80)
    print("LEARNING CURVE ANALYSIS STATUS")
    print("=" * 80)
    print()
    
    completed = 0
    in_progress = 0
    not_started = 0
    
    for feature_version, model_type in combinations:
        combo_dir = os.path.join(output_dir, f'{feature_version}_{model_type}')
        results_file = os.path.join(combo_dir, 'learning_curve_results.csv')
        plot_file = os.path.join(combo_dir, 'learning_curves.png')
        
        if os.path.exists(results_file) and os.path.exists(plot_file):
            try:
                df = pd.read_csv(results_file)
                n_points = len(df)
                best_f1 = df['test_f1'].max()
                best_clean = df['test_pct_clean_post'].max()
                print(f"✓ {feature_version.upper()} {model_type.upper()}: "
                      f"Complete ({n_points} points, Best F1={best_f1:.4f}, Best % Clean={best_clean:.2f}%)")
                completed += 1
            except Exception as e:
                print(f"⚠ {feature_version.upper()} {model_type.upper()}: Error reading results: {e}")
                in_progress += 1
        elif os.path.exists(combo_dir):
            print(f"⏳ {feature_version.upper()} {model_type.upper()}: In progress...")
            in_progress += 1
        else:
            print(f"○ {feature_version.upper()} {model_type.upper()}: Not started")
            not_started += 1
    
    print()
    print("=" * 80)
    print(f"Progress: {completed}/4 completed, {in_progress} in progress, {not_started} not started")
    
    if os.path.exists(os.path.join(output_dir, 'summary_report.md')):
        print("\n✓ Summary report available: learning_curve_analysis_iter007/summary_report.md")
    
    if os.path.exists(os.path.join(output_dir, 'comparison_plots.png')):
        print("✓ Comparison plots available: learning_curve_analysis_iter007/comparison_plots.png")


if __name__ == '__main__':
    check_status()


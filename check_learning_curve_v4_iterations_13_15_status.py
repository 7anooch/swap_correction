#!/usr/bin/env python3
"""
Check status of learning curve analysis for iterations 13-15 (features v4).
"""

import os
import json
from pathlib import Path

def check_learning_curve_status():
    """Check status of learning curve analyses."""
    iterations = [13, 14, 15]
    model_types = ['level1', 'raw']
    
    print("=" * 80)
    print("LEARNING CURVE ANALYSIS STATUS: ITERATIONS 13-15 (Features V4)")
    print("=" * 80)
    print()
    
    for iter_num in iterations:
        print(f"Iteration {iter_num:03d}:")
        output_dir = f"learning_curve_analysis_v4_iter{iter_num:03d}"
        
        if not os.path.exists(output_dir):
            print(f"  ⚠ Output directory not found: {output_dir}")
            continue
        
        for model_type in model_types:
            model_dir = os.path.join(output_dir, f"v4_{model_type}")
            
            if not os.path.exists(model_dir):
                print(f"  {model_type}: Not started")
                continue
            
            # Check for results file
            results_file = os.path.join(model_dir, "learning_curve_results.json")
            summary_file = os.path.join(model_dir, "learning_curve_summary.md")
            plot_file = os.path.join(model_dir, "learning_curves.png")
            
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    n_points = len(results.get('results', []))
                    print(f"  {model_type}: ✓ Complete ({n_points} data points)")
                    
                    if os.path.exists(summary_file):
                        print(f"    - Summary report: ✓")
                    if os.path.exists(plot_file):
                        print(f"    - Plot: ✓")
                except:
                    print(f"  {model_type}: ⚠ Results file exists but may be incomplete")
            else:
                print(f"  {model_type}: ⏳ In progress...")
        
        print()
    
    print("=" * 80)
    print("To monitor progress:")
    print("  tail -f learning_curve_v4_iter013.log")
    print("  tail -f learning_curve_v4_iter014.log")
    print("  tail -f learning_curve_v4_iter015.log")
    print("=" * 80)

if __name__ == '__main__':
    check_learning_curve_status()


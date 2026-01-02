#!/usr/bin/env python3
"""
Generate stability report for features_v3 stability analysis.

This script aggregates results from stability_analysis_v3_features_v3
and generates a comprehensive stability report.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from swap_correction.ml.stability.aggregate_results import (
    extract_metrics,
    generate_stability_report
)


def load_features_v3_results(iteration_dirs):
    """Load results from features_v3 iteration directories."""
    all_results = []
    
    for iter_dir in iteration_dirs:
        iter_id_str = os.path.basename(iter_dir).replace('iteration_', '')
        try:
            iter_id = int(iter_id_str)
        except:
            continue
        
        result = {
            'iteration': iter_id,
            'level1_model': {},
            'raw_model': {},
            'iteration_dir': iter_dir  # Needed for extract_metrics to load evaluation files
        }
        
        # Load level1 model results
        level1_training_file = os.path.join(iter_dir, 'level1_model', 'training_results.json')
        level1_eval_file = os.path.join(iter_dir, 'level1_model', 'evaluation_results.json')
        
        if os.path.exists(level1_training_file):
            with open(level1_training_file, 'r') as f:
                training_data = json.load(f)
            result['level1_model']['training'] = training_data
        
        if os.path.exists(level1_eval_file):
            with open(level1_eval_file, 'r') as f:
                eval_data = json.load(f)
            result['level1_model']['evaluation'] = eval_data
        
        # Load raw model results
        raw_training_file = os.path.join(iter_dir, 'raw_model', 'training_results.json')
        raw_eval_file = os.path.join(iter_dir, 'raw_model', 'evaluation_results.json')
        
        if os.path.exists(raw_training_file):
            with open(raw_training_file, 'r') as f:
                training_data = json.load(f)
            result['raw_model']['training'] = training_data
        
        if os.path.exists(raw_eval_file):
            with open(raw_eval_file, 'r') as f:
                eval_data = json.load(f)
            result['raw_model']['evaluation'] = eval_data
        
        # Load trial split info
        split_file = os.path.join(iter_dir, 'trial_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            result['trial_split'] = split_data
        
        all_results.append(result)
    
    return all_results


def main():
    """Generate stability report for features_v3."""
    import json
    
    base_dir = 'stability_analysis_v3_features_v3'
    output_dir = base_dir
    
    print("=" * 80)
    print("GENERATING STABILITY REPORT FOR FEATURES_V3")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Find all iteration directories
    iteration_dirs = []
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(base_dir, f'iteration_{iter_id:03d}')
        if os.path.exists(iter_dir):
            iteration_dirs.append(iter_dir)
            print(f"Found iteration {iter_id:03d}")
        else:
            print(f"Warning: Iteration {iter_id:03d} not found")
    
    if not iteration_dirs:
        print("ERROR: No iteration directories found!")
        return
    
    print(f"\nFound {len(iteration_dirs)} iterations")
    print("Loading results...\n")
    
    # Load all iteration results
    all_results = load_features_v3_results(iteration_dirs)
    
    if not all_results:
        print("ERROR: No results loaded!")
        return
    
    print(f"Loaded {len(all_results)} iteration results")
    
    # Extract metrics for both model types
    print("\nExtracting metrics...")
    level1_metrics = extract_metrics(all_results, model_type='level1')
    raw_metrics = extract_metrics(all_results, model_type='raw')
    
    print(f"Level1 metrics: {len(level1_metrics)} iterations")
    print(f"Raw metrics: {len(raw_metrics)} iterations")
    
    # Generate stability report
    print("\nGenerating stability report...")
    generate_stability_report(all_results, output_dir)
    
    print("\n" + "=" * 80)
    print("STABILITY REPORT GENERATED")
    print("=" * 80)
    print(f"Report saved to: {os.path.join(output_dir, 'stability_report.md')}")
    print()


if __name__ == '__main__':
    main()


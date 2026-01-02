#!/usr/bin/env python3
"""
Check the status of features_v4 testing on stability analysis iterations.
"""

import os
import json
from pathlib import Path

def check_status():
    """Check status of features_v4 testing."""
    base_dir = 'stability_analysis_v3_features_v4'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    print("=" * 80)
    print("FEATURES V4 TESTING STATUS")
    print("=" * 80)
    
    completed = []
    in_progress = []
    not_started = []
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(base_dir, f'iteration_{iter_id:03d}')
        
        if not os.path.exists(iter_dir):
            not_started.append(iter_id)
            continue
        
        # Check if both models are complete
        level1_dir = os.path.join(iter_dir, 'level1_model')
        raw_dir = os.path.join(iter_dir, 'raw_model')
        
        level1_complete = (
            os.path.exists(level1_dir) and
            os.path.exists(os.path.join(level1_dir, 'training_results.json')) and
            os.path.exists(os.path.join(level1_dir, 'evaluation_results.json'))
        )
        
        raw_complete = (
            os.path.exists(raw_dir) and
            os.path.exists(os.path.join(raw_dir, 'training_results.json')) and
            os.path.exists(os.path.join(raw_dir, 'evaluation_results.json'))
        )
        
        if level1_complete and raw_complete:
            # Try to load results
            try:
                with open(os.path.join(level1_dir, 'training_results.json'), 'r') as f:
                    level1_results = json.load(f)
                with open(os.path.join(raw_dir, 'training_results.json'), 'r') as f:
                    raw_results = json.load(f)
                
                level1_f1 = level1_results.get('test', {}).get('f1', 0)
                raw_f1 = raw_results.get('test', {}).get('f1', 0)
                
                completed.append({
                    'iter': iter_id,
                    'level1_f1': level1_f1,
                    'raw_f1': raw_f1
                })
            except:
                in_progress.append(iter_id)
        elif level1_complete or raw_complete or os.path.exists(level1_dir) or os.path.exists(raw_dir):
            in_progress.append(iter_id)
        else:
            not_started.append(iter_id)
    
    print(f"\n✓ Completed: {len(completed)}/{len(iteration_ids)}")
    if completed:
        print("\nCompleted iterations:")
        for item in completed:
            print(f"  Iteration {item['iter']:03d}: Level1 F1={item['level1_f1']:.4f}, Raw F1={item['raw_f1']:.4f}")
    
    print(f"\n⏳ In Progress: {len(in_progress)}/{len(iteration_ids)}")
    if in_progress:
        print(f"  Iterations: {in_progress}")
    
    print(f"\n○ Not Started: {len(not_started)}/{len(iteration_ids)}")
    if not_started:
        print(f"  Iterations: {not_started}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    check_status()


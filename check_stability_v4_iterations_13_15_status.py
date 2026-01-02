#!/usr/bin/env python3
"""
Check the status of stability analysis iterations 13-15 for features_v4.
"""

import os
import json
from pathlib import Path

def check_status():
    """Check status of iterations 13-15."""
    base_dir = 'stability_analysis_v3_features_v4'
    iteration_ids = [13, 14, 15]
    
    print("=" * 80)
    print("STABILITY ANALYSIS STATUS: ITERATIONS 13-15 (Features V4, Sample Size 100)")
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
        # Check what stage they're at
        for iter_id in in_progress:
            iter_dir = os.path.join(base_dir, f'iteration_{iter_id:03d}')
            level1_dir = os.path.join(iter_dir, 'level1_model')
            raw_dir = os.path.join(iter_dir, 'raw_model')
            
            level1_training = os.path.exists(os.path.join(level1_dir, 'training_results.json')) if os.path.exists(level1_dir) else False
            level1_eval = os.path.exists(os.path.join(level1_dir, 'evaluation_results.json')) if os.path.exists(level1_dir) else False
            raw_training = os.path.exists(os.path.join(raw_dir, 'training_results.json')) if os.path.exists(raw_dir) else False
            raw_eval = os.path.exists(os.path.join(raw_dir, 'evaluation_results.json')) if os.path.exists(raw_dir) else False
            
            status_parts = []
            if level1_training:
                status_parts.append("Level1: trained")
            if level1_eval:
                status_parts.append("Level1: evaluated")
            if raw_training:
                status_parts.append("Raw: trained")
            if raw_eval:
                status_parts.append("Raw: evaluated")
            
            if status_parts:
                print(f"    Iter {iter_id:03d}: {', '.join(status_parts)}")
    
    print(f"\n○ Not Started: {len(not_started)}/{len(iteration_ids)}")
    if not_started:
        print(f"  Iterations: {not_started}")
    
    print("\n" + "=" * 80)
    print("To monitor progress:")
    print("  tail -f stability_analysis_v4_iterations_13_15.log")
    print("=" * 80)


if __name__ == '__main__':
    check_status()


#!/usr/bin/env python3
"""
Check the status of features_v3 training/testing.

Reports on completed, in-progress, and not-started iterations.
"""

import os
import json
from pathlib import Path


def check_iteration_status(output_dir: str, iter_id: int) -> str:
    """Check status of a single iteration."""
    iter_dir = os.path.join(output_dir, f'iteration_{iter_id:03d}')
    
    if not os.path.exists(iter_dir):
        return 'not_started'
    
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
        return 'complete'
    elif os.path.exists(os.path.join(level1_dir, 'ml_data')) or os.path.exists(os.path.join(raw_dir, 'ml_data')):
        return 'in_progress'
    else:
        return 'not_started'


def main():
    """Main function."""
    output_dir = 'stability_analysis_v3_features_v3'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    print("=" * 80)
    print("FEATURES_V3 TRAINING STATUS")
    print("=" * 80)
    print(f"Output directory: {output_dir}\n")
    
    statuses = {}
    for iter_id in iteration_ids:
        status = check_iteration_status(output_dir, iter_id)
        statuses[iter_id] = status
    
    complete = [i for i, s in statuses.items() if s == 'complete']
    in_progress = [i for i, s in statuses.items() if s == 'in_progress']
    not_started = [i for i, s in statuses.items() if s == 'not_started']
    
    print(f"Complete: {len(complete)}/{len(iteration_ids)}")
    if complete:
        print(f"  Iterations: {complete}")
    
    print(f"\nIn Progress: {len(in_progress)}/{len(iteration_ids)}")
    if in_progress:
        print(f"  Iterations: {in_progress}")
    
    print(f"\nNot Started: {len(not_started)}/{len(iteration_ids)}")
    if not_started:
        print(f"  Iterations: {not_started}")
    
    # Check log file
    log_file = 'test_features_v3.log'
    if os.path.exists(log_file):
        print(f"\nLog file: {log_file}")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"  Last 5 lines:")
                for line in lines[-5:]:
                    print(f"    {line.rstrip()}")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""Check the status of the features_v2 testing run."""

import os
import json
from pathlib import Path

output_dir = 'stability_analysis_v3_features_v2'
iteration_ids = [7, 8, 9, 10, 11, 12]

print("=" * 80)
print("FEATURES_V2 TESTING STATUS")
print("=" * 80)
print(f"Output directory: {output_dir}")
print()

if not os.path.exists(output_dir):
    print("Output directory does not exist yet. Testing may still be starting...")
    exit(0)

completed = []
in_progress = []
not_started = []

for iter_id in iteration_ids:
    iter_dir = os.path.join(output_dir, f'iteration_{iter_id:03d}')
    
    if not os.path.exists(iter_dir):
        not_started.append(iter_id)
        continue
    
    # Check for both model types
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
        completed.append(iter_id)
    elif os.path.exists(os.path.join(level1_dir, 'ml_data')) or os.path.exists(os.path.join(raw_dir, 'ml_data')):
        in_progress.append(iter_id)
    else:
        not_started.append(iter_id)

print(f"Completed: {len(completed)}/{len(iteration_ids)} iterations")
if completed:
    print(f"  Iterations: {completed}")
print()

print(f"In Progress: {len(in_progress)}/{len(iteration_ids)} iterations")
if in_progress:
    print(f"  Iterations: {in_progress}")
    # Show details for in-progress iterations
    for iter_id in in_progress:
        iter_dir = os.path.join(output_dir, f'iteration_{iter_id:03d}')
        level1_dir = os.path.join(iter_dir, 'level1_model')
        raw_dir = os.path.join(iter_dir, 'raw_model')
        
        print(f"\n  Iteration {iter_id:03d}:")
        if os.path.exists(os.path.join(level1_dir, 'training_results.json')):
            print(f"    level1: Training complete")
        elif os.path.exists(os.path.join(level1_dir, 'ml_data')):
            print(f"    level1: Data prepared")
        else:
            print(f"    level1: Not started")
        
        if os.path.exists(os.path.join(raw_dir, 'training_results.json')):
            print(f"    raw: Training complete")
        elif os.path.exists(os.path.join(raw_dir, 'ml_data')):
            print(f"    raw: Data prepared")
        else:
            print(f"    raw: Not started")
print()

print(f"Not Started: {len(not_started)}/{len(iteration_ids)} iterations")
if not_started:
    print(f"  Iterations: {not_started}")
print()

# Show summary statistics for completed iterations
if completed:
    print("=" * 80)
    print("SUMMARY STATISTICS (Completed Iterations)")
    print("=" * 80)
    
    for iter_id in completed:
        iter_dir = os.path.join(output_dir, f'iteration_{iter_id:03d}')
        
        for model_type in ['level1', 'raw']:
            model_dir = os.path.join(iter_dir, f'{model_type}_model')
            training_file = os.path.join(model_dir, 'training_results.json')
            eval_file = os.path.join(model_dir, 'evaluation_results.json')
            
            if os.path.exists(training_file) and os.path.exists(eval_file):
                with open(training_file) as f:
                    training = json.load(f)
                with open(eval_file) as f:
                    evaluation = json.load(f)
                
                print(f"\nIteration {iter_id:03d} - {model_type} model:")
                print(f"  Training - Test F1: {training.get('test', {}).get('f1', 'N/A'):.4f}")
                if 'summary' in evaluation:
                    summary = evaluation['summary']
                    print(f"  Evaluation - Mean F1: {summary.get('mean_f1', 'N/A'):.4f}")
                    print(f"  Evaluation - Mean Precision: {summary.get('mean_precision', 'N/A'):.4f}")
                    print(f"  Evaluation - Mean Recall: {summary.get('mean_recall', 'N/A'):.4f}")


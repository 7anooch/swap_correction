#!/usr/bin/env python3
"""
Quick script to check the status of the running stability analysis.
"""

import os
import json
from pathlib import Path

results_dir = 'stability_analysis'

if not os.path.exists(results_dir):
    print("Stability analysis directory not found.")
    exit(1)

print("=" * 80)
print("STABILITY ANALYSIS STATUS")
print("=" * 80)

# Count iterations
iterations = sorted([d for d in os.listdir(results_dir) if d.startswith('iteration_')])
print(f"\nTotal iterations: {len(iterations)}/10")

# Check each iteration
for iter_name in iterations:
    iter_dir = os.path.join(results_dir, iter_name)
    results_file = os.path.join(iter_dir, 'iteration_results.json')
    
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        
        status = results.get('status', 'unknown')
        raw_status = results.get('raw_model', {}).get('status', 'unknown')
        level1_status = results.get('level1_model', {}).get('status', 'unknown')
        
        print(f"\n{iter_name}:")
        print(f"  Overall: {status}")
        print(f"  Raw model: {raw_status}")
        print(f"  Level1 model: {level1_status}")
        
        # Check for training results
        raw_model_dir = os.path.join(iter_dir, 'raw_model')
        level1_model_dir = os.path.join(iter_dir, 'level1_model')
        
        if os.path.exists(os.path.join(raw_model_dir, 'training_results.json')):
            print(f"  Raw: Training complete")
        if os.path.exists(os.path.join(level1_model_dir, 'training_results.json')):
            print(f"  Level1: Training complete")
    else:
        print(f"\n{iter_name}: In progress (no results file yet)")

print("\n" + "=" * 80)


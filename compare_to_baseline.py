#!/usr/bin/env python3
"""
Compare new corrected output to existing level1 files to see what changed.
"""

import os
import sys
from swap_correction import tracking_correction as tc
from swap_correction import pivr_loader as loader
from swap_correction import error_analysis
import pandas as pd
import numpy as np

def compare_trial(trial_dir: str):
    """Compare new correction to existing level1."""
    try:
        # Load raw data
        raw_data = loader.load_raw_data(trial_dir)
        fps = loader.get_all_settings(trial_dir)['Framerate']
        
        # Load existing level1
        csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level1.csv')]
        if not csv_files:
            return None
        level1_file = csv_files[0]
        existing_level1 = loader.load_raw_data(trial_dir, level1_file)
        
        # Run NEW correction
        new_level1 = tc.tracking_correction(
            raw_data, fps,
            filterData=False,
            swapCorrection=True,
            validate=False,
            removeErrors=True,
            interp=True,
            debug=False
        )
        
        # Load level2 for comparison
        csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level2.csv')]
        if not csv_files:
            return None
        level2_file = csv_files[0]
        level2_data = loader.load_raw_data(trial_dir, level2_file)
        
        # Compare existing level1 vs level2
        min_len = min(len(existing_level1), len(level2_data))
        existing_swapped = error_analysis.identify_swapped_frames(
            existing_level1.iloc[:min_len],
            level2_data.iloc[:min_len]
        )
        existing_error_rate = len(existing_swapped) / min_len * 100 if min_len > 0 else 0
        
        # Compare new level1 vs level2
        min_len_new = min(len(new_level1), len(level2_data))
        new_swapped = error_analysis.identify_swapped_frames(
            new_level1.iloc[:min_len_new],
            level2_data.iloc[:min_len_new]
        )
        new_error_rate = len(new_swapped) / min_len_new * 100 if min_len_new > 0 else 0
        
        # Check if new and existing level1 differ
        min_len_compare = min(len(new_level1), len(existing_level1))
        diff_frames = error_analysis.identify_swapped_frames(
            new_level1.iloc[:min_len_compare],
            existing_level1.iloc[:min_len_compare]
        )
        changed = len(diff_frames) > 0
        
        return {
            'trial': os.path.basename(trial_dir),
            'existing_error_rate': existing_error_rate,
            'new_error_rate': new_error_rate,
            'improvement': existing_error_rate - new_error_rate,
            'changed': changed,
            'diff_frames': len(diff_frames),
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """Compare all trials."""
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'swap_correction', 'tests', 'test_data'
    )
    
    print("=" * 80)
    print("Comparing New Correction to Existing Level1 (Baseline)")
    print("=" * 80)
    
    trials = [os.path.join(test_data_dir, d) for d in os.listdir(test_data_dir)
              if os.path.isdir(os.path.join(test_data_dir, d))]
    trials.sort()
    
    results = []
    for i, trial_dir in enumerate(trials):
        trial_name = os.path.basename(trial_dir)
        print(f"[{i+1}/{len(trials)}] {trial_name}", end=' ... ', flush=True)
        
        result = compare_trial(trial_dir)
        if result:
            results.append(result)
            if result['changed']:
                print(f"CHANGED: {result['existing_error_rate']:.2f}% -> {result['new_error_rate']:.2f}% "
                      f"(Δ{result['improvement']:+.2f}%)")
            else:
                print(f"No change: {result['existing_error_rate']:.2f}%")
        else:
            print("Failed")
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    changed = df[df['changed'] == True]
    unchanged = df[df['changed'] == False]
    
    print(f"\nTrials that changed: {len(changed)} / {len(results)}")
    print(f"Trials unchanged: {len(unchanged)} / {len(results)}")
    
    if len(changed) > 0:
        print(f"\nChanged trials:")
        for _, row in changed.iterrows():
            print(f"  {row['trial']}: {row['existing_error_rate']:.2f}% -> {row['new_error_rate']:.2f}% "
                  f"(Δ{row['improvement']:+.2f}%)")
        
        improved = changed[changed['improvement'] > 0]
        worsened = changed[changed['improvement'] < 0]
        
        print(f"\nImproved: {len(improved)} trials")
        if len(improved) > 0:
            for _, row in improved.iterrows():
                print(f"  {row['trial']}: {row['improvement']:+.2f}% improvement")
        
        print(f"\nWorsened: {len(worsened)} trials")
        if len(worsened) > 0:
            for _, row in worsened.iterrows():
                print(f"  {row['trial']}: {row['improvement']:.2f}% worse")
    
    print(f"\nAverage existing error rate: {df['existing_error_rate'].mean():.2f}%")
    print(f"Average new error rate: {df['new_error_rate'].mean():.2f}%")
    print(f"Average change: {df['improvement'].mean():.2f}%")
    
    df.to_csv('comparison_results.csv', index=False)
    print(f"\nResults saved to: comparison_results.csv")


if __name__ == '__main__':
    main()


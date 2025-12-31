#!/usr/bin/env python3
"""
Test improved global swap detection on all trials.
Runs swap correction on all trials and compares results to baseline.
"""

import os
import sys
from swap_correction import tracking_correction as tc
from swap_correction import pivr_loader as loader
from swap_correction import error_analysis
import pandas as pd
import numpy as np

def process_trial(trial_dir: str, debug: bool = False):
    """Process a single trial with improved swap correction."""
    try:
        # Load raw data
        raw_data = loader.load_raw_data(trial_dir)
        fps = loader.get_all_settings(trial_dir)['Framerate']
        
        # Run improved correction
        corrected_data = tc.tracking_correction(
            raw_data, fps,
            filterData=False,
            swapCorrection=True,
            validate=False,  # Disable validation for now
            removeErrors=True,
            interp=True,
            debug=debug
        )
        
        # Load level2 for comparison
        csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level2.csv')]
        if not csv_files:
            return None
        
        level2_file = csv_files[0]
        level2_data = loader.load_raw_data(trial_dir, level2_file)
        
        # Calculate error metrics
        min_len = min(len(corrected_data), len(level2_data))
        swapped_frames = error_analysis.identify_swapped_frames(
            corrected_data.iloc[:min_len],
            level2_data.iloc[:min_len]
        )
        error_rate = len(swapped_frames) / min_len * 100 if min_len > 0 else 0
        
        swap_segments = error_analysis.get_swap_segments(
            corrected_data.iloc[:min_len],
            level2_data.iloc[:min_len]
        )
        
        return {
            'trial': os.path.basename(trial_dir),
            'error_rate': error_rate,
            'swapped_frames': len(swapped_frames),
            'num_segments': len(swap_segments),
            'total_frames': min_len,
        }
    except Exception as e:
        print(f"Error processing {trial_dir}: {e}")
        return None


def main():
    """Test on all trials."""
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'swap_correction', 'tests', 'test_data'
    )
    
    print("=" * 80)
    print("Testing Improved Global Swap Detection on All Trials")
    print("=" * 80)
    print(f"Test data directory: {test_data_dir}\n")
    
    # Get all trial directories
    trials = [os.path.join(test_data_dir, d) for d in os.listdir(test_data_dir)
              if os.path.isdir(os.path.join(test_data_dir, d))]
    trials.sort()
    
    print(f"Found {len(trials)} trials\n")
    
    # Process each trial
    results = []
    for i, trial_dir in enumerate(trials):
        trial_name = os.path.basename(trial_dir)
        print(f"[{i+1}/{len(trials)}] Processing: {trial_name}", end=' ... ', flush=True)
        
        result = process_trial(trial_dir, debug=False)
        if result:
            results.append(result)
            print(f"Error rate: {result['error_rate']:.2f}%, Segments: {result['num_segments']}")
        else:
            print("Failed or no level2 data")
    
    if not results:
        print("\nNo results generated!")
        return
    
    # Create summary
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"\nTotal trials analyzed: {len(results)}")
    print(f"Average error rate: {df['error_rate'].mean():.2f}%")
    print(f"Median error rate: {df['error_rate'].median():.2f}%")
    print(f"Min error rate: {df['error_rate'].min():.2f}%")
    print(f"Max error rate: {df['error_rate'].max():.2f}%")
    
    print(f"\nTotal swapped frames: {df['swapped_frames'].sum()}")
    print(f"Average swap segments per trial: {df['num_segments'].mean():.2f}")
    print(f"Median swap segments per trial: {df['num_segments'].median():.0f}")
    
    # Count perfect trials
    perfect = df[df['error_rate'] == 0]
    print(f"\nPerfect trials (0% error): {len(perfect)} / {len(results)} ({100*len(perfect)/len(results):.1f}%)")
    
    # Count problematic trials
    problematic = df[df['error_rate'] > 0]
    print(f"Problematic trials (>0% error): {len(problematic)} / {len(results)} ({100*len(problematic)/len(results):.1f}%)")
    
    if len(problematic) > 0:
        print(f"\nProblematic trials:")
        for _, row in problematic.sort_values('error_rate', ascending=False).iterrows():
            print(f"  {row['trial']}: {row['error_rate']:.2f}% error, {row['num_segments']} segments")
    
    # Compare to baseline (from findings document)
    print("\n" + "=" * 80)
    print("COMPARISON TO BASELINE (from Step 1 analysis)")
    print("=" * 80)
    print("Baseline (before improvements):")
    print("  - Perfect trials: 14 / 25 (56%)")
    print("  - Problematic trials: 11 / 25 (44%)")
    print("  - Average error rate for problematic: ~20-25%")
    print("\nCurrent (with improved global swap detection):")
    print(f"  - Perfect trials: {len(perfect)} / {len(results)} ({100*len(perfect)/len(results):.1f}%)")
    print(f"  - Problematic trials: {len(problematic)} / {len(results)} ({100*len(problematic)/len(results):.1f}%)")
    if len(problematic) > 0:
        print(f"  - Average error rate for problematic: {problematic['error_rate'].mean():.2f}%")
    
    # Save results
    output_file = 'test_results_all_trials.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()


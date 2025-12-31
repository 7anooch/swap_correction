#!/usr/bin/env python3
"""
Test script for collapsed keypoint detection and improved global swap detection.
Tests on the known global swap trial.
"""

import os
import sys
from swap_correction import tracking_correction as tc
from swap_correction import pivr_loader as loader
from swap_correction import error_analysis, metrics
import numpy as np

def test_collapsed_detection(trial_dir: str):
    """Test collapsed keypoint detection on a trial."""
    print(f"\n{'='*80}")
    print(f"Testing collapsed keypoint detection on: {os.path.basename(trial_dir)}")
    print(f"{'='*80}")
    
    # Load raw data
    raw_data = loader.load_raw_data(trial_dir)
    fps = loader.get_all_settings(trial_dir)['Framerate']
    
    # Detect collapsed keypoints
    collapsed = tc.detect_collapsed_keypoints(raw_data, tolerance=0.1, debug=True)
    n_collapsed = np.sum(collapsed)
    n_total = len(collapsed)
    
    print(f"\nCollapsed frames: {n_collapsed} / {n_total} ({100*n_collapsed/n_total:.1f}%)")
    
    if n_collapsed > 0:
        collapsed_frames = np.where(collapsed)[0]
        collapsed_ranges = tc.utils.get_consecutive_ranges(collapsed_frames)
        print(f"Collapsed regions: {len(collapsed_ranges)} segments")
        for i, (start, end) in enumerate(collapsed_ranges[:10]):  # Show first 10
            print(f"  Segment {i+1}: frames {start}-{end} ({end-start+1} frames)")
    
    return collapsed


def test_global_swap_with_collapsed(trial_dir: str, debug: bool = True):
    """Test global swap detection with collapsed keypoint handling."""
    print(f"\n{'='*80}")
    print(f"Testing global swap detection with collapsed keypoint handling")
    print(f"Trial: {os.path.basename(trial_dir)}")
    print(f"{'='*80}")
    
    # Load raw data
    raw_data = loader.load_raw_data(trial_dir)
    fps = loader.get_all_settings(trial_dir)['Framerate']
    
    # Load level2 for comparison - find the actual level2 file
    csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level2.csv')]
    if csv_files:
        level2_file = csv_files[0]
        level2_data = loader.load_raw_data(trial_dir, level2_file)
    else:
        print("Warning: Level2 file not found, skipping comparison")
        level2_data = None
    
    print(f"\nRaw data: {len(raw_data)} frames")
    print(f"Level2 data: {len(level2_data)} frames")
    
    # Test global swap detection directly
    print("\n--- TESTING GLOBAL SWAP DETECTION ---")
    test_data = raw_data.copy()
    test_data = tc.correct_global_swap(test_data, debug=debug)
    
    # Check if swap occurred
    swapped = not np.array_equal(raw_data[['xhead', 'yhead', 'xtail', 'ytail']].values,
                                  test_data[['xhead', 'yhead', 'xtail', 'ytail']].values)
    print(f"\nGlobal swap detected: {swapped}")
    
    # Run full correction
    print("\n--- RUNNING FULL CORRECTION ---")
    corrected_data = tc.tracking_correction(
        raw_data, fps,
        filterData=False,
        swapCorrection=True,
        validate=False,
        removeErrors=True,
        interp=True,
        debug=debug
    )
    
    # Compare with level2
    if level2_data is not None:
        print("\n--- COMPARISON WITH GROUND TRUTH (Level2) ---")
        min_len = min(len(corrected_data), len(level2_data))
        swapped_frames = error_analysis.identify_swapped_frames(
            corrected_data.iloc[:min_len],
            level2_data.iloc[:min_len]
        )
        error_rate = len(swapped_frames) / min_len * 100
        
        print(f"Frames that differ from level2: {len(swapped_frames)}")
        print(f"Error rate: {error_rate:.2f}%")
        
        if len(swapped_frames) > 0:
            swap_segments = error_analysis.get_swap_segments(
                corrected_data.iloc[:min_len],
                level2_data.iloc[:min_len]
            )
            print(f"Swap segments: {len(swap_segments)}")
            if len(swap_segments) > 0:
                segment_lengths = [seg[1] - seg[0] + 1 for seg in swap_segments]
                print(f"Segment lengths: {segment_lengths[:10]}")  # Show first 10
    else:
        error_rate = None
    
    return corrected_data, error_rate


if __name__ == '__main__':
    # Test on the known global swap trial
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'swap_correction', 'tests', 'test_data'
    )
    
    global_swap_trial = os.path.join(test_data_dir, '2024.11.13_00-48-15_Sussex_e2hex')
    
    if os.path.isdir(global_swap_trial):
        # First test collapsed detection
        collapsed = test_collapsed_detection(global_swap_trial)
        
        # Then test global swap with collapsed handling
        corrected, error_rate = test_global_swap_with_collapsed(global_swap_trial, debug=True)
        
        print(f"\n{'='*80}")
        print(f"Test completed. Error rate: {error_rate:.2f}%")
        print(f"{'='*80}\n")
    else:
        print(f"Error: Trial directory not found: {global_swap_trial}")


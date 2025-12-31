#!/usr/bin/env python3
"""
Test script for improved global swap detection.
Tests the implementation on the known global swap trial.
"""

import os
import sys
from swap_correction import tracking_correction as tc
from swap_correction import pivr_loader as loader
from swap_correction import error_analysis, metrics
import numpy as np

def test_global_swap_trial(trial_dir: str, debug: bool = True):
    """Test global swap detection on a specific trial."""
    print(f"\n{'='*80}")
    print(f"Testing global swap detection on: {os.path.basename(trial_dir)}")
    print(f"{'='*80}")
    
    # Load raw data
    raw_data = loader.load_raw_data(trial_dir)
    fps = loader.get_all_settings(trial_dir)['Framerate']
    
    # Load level1 and level2 for comparison
    try:
        level1_data = loader.load_raw_data(trial_dir, '2024.11.13_00-48-15_data_level1.csv')
        level2_data = loader.load_raw_data(trial_dir, '2024.11.13_00-48-15_data_level2.csv')
    except:
        # Try to find level files with trial name
        trial_name = os.path.basename(trial_dir)
        level1_file = f"{trial_name}_data_level1.csv"
        level2_file = f"{trial_name}_data_level2.csv"
        level1_data = loader.load_raw_data(trial_dir, level1_file)
        level2_data = loader.load_raw_data(trial_dir, level2_file)
    
    print(f"\nRaw data: {len(raw_data)} frames")
    print(f"Level1 data: {len(level1_data)} frames")
    print(f"Level2 data: {len(level2_data)} frames")
    
    # Analyze raw data before correction
    print("\n--- BEFORE CORRECTION ---")
    filtered = tc.filter_data(raw_data)
    hspd = metrics.get_speed_from_df(filtered, 'head', fps=fps)
    tspd = metrics.get_speed_from_df(filtered, 'tail', fps=fps)
    cross_sign = metrics.get_ht_cross_sign(filtered)
    valid_signs = cross_sign[~np.isnan(cross_sign)]
    
    print(f"Mean head speed: {np.nanmean(hspd):.3f} mm/s")
    print(f"Mean tail speed: {np.nanmean(tspd):.3f} mm/s")
    print(f"Speed check (tail > head): {np.nanmean(tspd) > np.nanmean(hspd)}")
    
    if len(valid_signs) > 0:
        positive_ratio = np.sum(valid_signs > 0) / len(valid_signs)
        negative_ratio = np.sum(valid_signs < 0) / len(valid_signs)
        print(f"Cross-sign positive ratio: {positive_ratio:.3f}")
        print(f"Cross-sign negative ratio: {negative_ratio:.3f}")
        print(f"Cross-sign check (negative > 0.7): {negative_ratio > 0.7}")
    
    # Test global swap detection directly
    print("\n--- TESTING GLOBAL SWAP DETECTION DIRECTLY ---")
    # Create a copy to test
    test_data = raw_data.copy()
    test_data = tc.correct_global_swap(test_data, debug=True)
    
    # Check if swap occurred
    swapped = not np.array_equal(raw_data[['xhead', 'yhead', 'xtail', 'ytail']].values,
                                  test_data[['xhead', 'yhead', 'xtail', 'ytail']].values)
    print(f"Global swap detected: {swapped}")
    
    # Run correction with debug
    print("\n--- RUNNING FULL CORRECTION ---")
    corrected_data = tc.tracking_correction(
        raw_data, fps,
        filterData=False,
        swapCorrection=True,
        validate=False,  # Disable validation for now
        removeErrors=True,
        interp=True,
        debug=debug
    )
    
    # Analyze corrected data
    print("\n--- AFTER CORRECTION ---")
    filtered_corrected = tc.filter_data(corrected_data)
    hspd_corr = metrics.get_speed_from_df(filtered_corrected, 'head', fps=fps)
    tspd_corr = metrics.get_speed_from_df(filtered_corrected, 'tail', fps=fps)
    cross_sign_corr = metrics.get_ht_cross_sign(filtered_corrected)
    valid_signs_corr = cross_sign_corr[~np.isnan(cross_sign_corr)]
    
    print(f"Mean head speed: {np.nanmean(hspd_corr):.3f} mm/s")
    print(f"Mean tail speed: {np.nanmean(tspd_corr):.3f} mm/s")
    
    if len(valid_signs_corr) > 0:
        positive_ratio_corr = np.sum(valid_signs_corr > 0) / len(valid_signs_corr)
        negative_ratio_corr = np.sum(valid_signs_corr < 0) / len(valid_signs_corr)
        print(f"Cross-sign positive ratio: {positive_ratio_corr:.3f}")
        print(f"Cross-sign negative ratio: {negative_ratio_corr:.3f}")
    
    # Compare with level2 (ground truth)
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
            print(f"Segment lengths: {[seg[1] - seg[0] + 1 for seg in swap_segments[:5]]}")
    
    return corrected_data, error_rate


if __name__ == '__main__':
    # Test on the known global swap trial
    test_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'swap_correction', 'tests', 'test_data'
    )
    
    global_swap_trial = os.path.join(test_data_dir, '2024.11.13_00-48-15_Sussex_e2hex')
    
    if os.path.isdir(global_swap_trial):
        corrected, error_rate = test_global_swap_trial(global_swap_trial, debug=True)
        print(f"\n{'='*80}")
        print(f"Test completed. Error rate: {error_rate:.2f}%")
        print(f"{'='*80}\n")
    else:
        print(f"Error: Trial directory not found: {global_swap_trial}")


#!/usr/bin/env python3
"""
Detailed analysis of specific problematic trials to understand failure modes.
"""

import numpy as np
import pandas as pd
import os
from swap_correction import error_analysis, pivr_loader, metrics, tracking_correction, utils


def analyze_global_swap_trial(trial_dir: str):
    """Detailed analysis of a trial with a global swap."""
    trial_data = error_analysis.load_trial_data(trial_dir)
    
    if trial_data['raw'] is None or trial_data['level1'] is None or trial_data['level2'] is None:
        return None
    
    try:
        settings = pivr_loader.get_all_settings(trial_dir)
        fps = settings['Framerate'] if settings else 30
    except Exception:
        fps = 30
    
    raw_data = trial_data['raw']
    level1 = trial_data['level1']
    level2 = trial_data['level2']
    min_len = min(len(raw_data), len(level1), len(level2))
    
    # Get swap segments
    swap_segments = error_analysis.get_swap_segments(level1, level2)
    
    # Analyze cross-sign
    cross_sign_raw = metrics.get_ht_cross_sign(raw_data.iloc[:min_len])
    cross_sign_level1 = metrics.get_ht_cross_sign(level1.iloc[:min_len])
    cross_sign_level2 = metrics.get_ht_cross_sign(level2.iloc[:min_len])
    
    # Calculate match rates
    raw_vs_level2_match = np.mean(cross_sign_raw == cross_sign_level2)
    level1_vs_level2_match = np.mean(cross_sign_level1 == cross_sign_level2)
    
    # Analyze speeds
    h_raw = metrics.get_speed_from_df(raw_data.iloc[:min_len], 'head', fps=fps)
    t_raw = metrics.get_speed_from_df(raw_data.iloc[:min_len], 'tail', fps=fps)
    h_level1 = metrics.get_speed_from_df(level1.iloc[:min_len], 'head', fps=fps)
    t_level1 = metrics.get_speed_from_df(level1.iloc[:min_len], 'tail', fps=fps)
    h_level2 = metrics.get_speed_from_df(level2.iloc[:min_len], 'head', fps=fps)
    t_level2 = metrics.get_speed_from_df(level2.iloc[:min_len], 'tail', fps=fps)
    
    mean_h_raw = np.nanmean(h_raw)
    mean_t_raw = np.nanmean(t_raw)
    mean_h_level1 = np.nanmean(h_level1)
    mean_t_level1 = np.nanmean(t_level1)
    mean_h_level2 = np.nanmean(h_level2)
    mean_t_level2 = np.nanmean(t_level2)
    
    # Check global swap detection
    current_check = mean_t_raw > mean_h_raw
    should_swap = mean_t_level2 > mean_h_level2
    
    return {
        'trial_name': os.path.basename(trial_dir),
        'num_segments': len(swap_segments),
        'segment_lengths': [seg[1] - seg[0] + 1 for seg in swap_segments],
        'raw_vs_level2_cross_sign_match': raw_vs_level2_match,
        'level1_vs_level2_cross_sign_match': level1_vs_level2_match,
        'mean_head_speed_raw': mean_h_raw,
        'mean_tail_speed_raw': mean_t_raw,
        'mean_head_speed_level1': mean_h_level1,
        'mean_tail_speed_level1': mean_t_level1,
        'mean_head_speed_level2': mean_h_level2,
        'mean_tail_speed_level2': mean_t_level2,
        'current_global_swap_check': current_check,
        'should_swap_globally': should_swap,
        'global_swap_detection_correct': current_check == should_swap,
    }


def analyze_multi_segment_trial(trial_dir: str):
    """Detailed analysis of a trial with multiple segments."""
    trial_data = error_analysis.load_trial_data(trial_dir)
    
    if trial_data['raw'] is None or trial_data['level1'] is None or trial_data['level2'] is None:
        return None
    
    try:
        settings = pivr_loader.get_all_settings(trial_dir)
        fps = settings['Framerate'] if settings else 30
    except Exception:
        fps = 30
    
    raw_data = trial_data['raw']
    level1 = trial_data['level1']
    level2 = trial_data['level2']
    min_len = min(len(raw_data), len(level1), len(level2))
    
    # Get swap segments
    swap_segments = error_analysis.get_swap_segments(level1, level2)
    
    # Analyze each segment
    segment_analyses = []
    for seg in swap_segments:
        start, end = seg
        seg_length = end - start + 1
        
        # Get cross-sign in this segment
        cross_sign1 = metrics.get_ht_cross_sign(level1.iloc[start:end+1])
        cross_sign2 = metrics.get_ht_cross_sign(level2.iloc[start:end+1])
        match_rate = np.mean(cross_sign1 == cross_sign2)
        
        # Get speeds in segment
        h1_seg = metrics.get_speed_from_df(level1.iloc[start:end+1], 'head', fps=fps)
        t1_seg = metrics.get_speed_from_df(level1.iloc[start:end+1], 'tail', fps=fps)
        h2_seg = metrics.get_speed_from_df(level2.iloc[start:end+1], 'head', fps=fps)
        t2_seg = metrics.get_speed_from_df(level2.iloc[start:end+1], 'tail', fps=fps)
        
        mean_h1 = np.nanmean(h1_seg)
        mean_t1 = np.nanmean(t1_seg)
        mean_h2 = np.nanmean(h2_seg)
        mean_t2 = np.nanmean(t2_seg)
        
        segment_analyses.append({
            'start': start,
            'end': end,
            'length': seg_length,
            'cross_sign_match': match_rate,
            'mean_head_speed_level1': mean_h1,
            'mean_tail_speed_level1': mean_t1,
            'mean_head_speed_level2': mean_h2,
            'mean_tail_speed_level2': mean_t2,
        })
    
    return {
        'trial_name': os.path.basename(trial_dir),
        'num_segments': len(swap_segments),
        'segments': segment_analyses,
    }


if __name__ == '__main__':
    # Get the package directory (swap_correction/)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    # test_data is at swap_correction/tests/test_data/
    test_data_dir = os.path.join(package_dir, 'tests', 'test_data')
    
    # Analyze global swap trial
    global_trial = os.path.join(test_data_dir, '2024.11.13_00-48-15_Sussex_e2hex')
    if os.path.isdir(global_trial):
        result = analyze_global_swap_trial(global_trial)
        if result:
            print("GLOBAL SWAP TRIAL ANALYSIS:")
            print(f"Trial: {result['trial_name']}")
            print(f"Segments: {result['num_segments']}, lengths: {result['segment_lengths']}")
            print(f"Raw vs Level2 cross-sign match: {result['raw_vs_level2_cross_sign_match']:.3f}")
            print(f"Level1 vs Level2 cross-sign match: {result['level1_vs_level2_cross_sign_match']:.3f}")
            print(f"Mean speeds (raw): head={result['mean_head_speed_raw']:.3f}, tail={result['mean_tail_speed_raw']:.3f}")
            print(f"Mean speeds (level1): head={result['mean_head_speed_level1']:.3f}, tail={result['mean_tail_speed_level1']:.3f}")
            print(f"Mean speeds (level2): head={result['mean_head_speed_level2']:.3f}, tail={result['mean_tail_speed_level2']:.3f}")
            print(f"Current global swap check: {result['current_global_swap_check']}")
            print(f"Should swap globally: {result['should_swap_globally']}")
            print(f"Global swap detection correct: {result['global_swap_detection_correct']}")
    
    # Analyze multi-segment trial
    multi_trial = os.path.join(test_data_dir, '2024.11.13_00-19-29_Sussex_e2hex')
    if os.path.isdir(multi_trial):
        result = analyze_multi_segment_trial(multi_trial)
        if result:
            print("\n\nMULTI-SEGMENT TRIAL ANALYSIS:")
            print(f"Trial: {result['trial_name']}")
            print(f"Number of segments: {result['num_segments']}")
            print("\nSegment details:")
            for i, seg in enumerate(result['segments'][:5]):  # Show first 5
                print(f"  Segment {i+1}: frames {seg['start']}-{seg['end']} ({seg['length']} frames)")
                print(f"    Cross-sign match: {seg['cross_sign_match']:.3f}")
                print(f"    Level1 speeds: h={seg['mean_head_speed_level1']:.3f}, t={seg['mean_tail_speed_level1']:.3f}")
                print(f"    Level2 speeds: h={seg['mean_head_speed_level2']:.3f}, t={seg['mean_tail_speed_level2']:.3f}")


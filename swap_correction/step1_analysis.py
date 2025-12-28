#!/usr/bin/env python3
"""
Step 1 Analysis: Investigate speed ratio bug and segment-level patterns.

This script performs deep analysis of:
1. Speed ratio calculation issues
2. Segment-level error patterns
3. Comparison of perfect vs problematic trials
4. Why current detection misses entire segments
"""

import numpy as np
import pandas as pd
import os
from swap_correction import error_analysis, pivr_loader, metrics, tracking_correction, utils


def analyze_speed_ratio_bug(trial_dir: str) -> dict:
    """Investigate speed ratio calculation issues."""
    trial_data = error_analysis.load_trial_data(trial_dir)
    
    if trial_data['level1'] is None or trial_data['level2'] is None:
        return {}
    
    try:
        settings = pivr_loader.get_all_settings(trial_dir)
        fps = settings['Framerate'] if settings else 30
    except Exception:
        fps = 30
    
    level1 = trial_data['level1']
    level2 = trial_data['level2']
    min_len = min(len(level1), len(level2))
    
    # Calculate speeds
    h1_speed = metrics.get_speed_from_df(level1.iloc[:min_len], 'head', fps=fps)
    t1_speed = metrics.get_speed_from_df(level1.iloc[:min_len], 'tail', fps=fps)
    h2_speed = metrics.get_speed_from_df(level2.iloc[:min_len], 'head', fps=fps)
    t2_speed = metrics.get_speed_from_df(level2.iloc[:min_len], 'tail', fps=fps)
    
    # Calculate ratios
    h1_ratio = h1_speed / (t1_speed + 1e-6)
    h2_ratio = h2_speed / (t2_speed + 1e-6)
    ratio_error = h2_ratio - h1_ratio
    
    # Find problematic cases
    near_zero_tail = np.sum(t1_speed < 0.01)
    extreme_ratios = np.sum(np.abs(h1_ratio) > 100)
    extreme_errors = np.sum(np.abs(ratio_error) > 1000)
    
    return {
        'near_zero_tail_speed_frames': near_zero_tail,
        'extreme_ratios': extreme_ratios,
        'extreme_errors': extreme_errors,
        'mean_ratio_error': np.nanmean(np.abs(ratio_error)),
        'max_ratio_error': np.nanmax(np.abs(ratio_error)),
        'median_tail_speed': np.nanmedian(t1_speed),
        'min_tail_speed': np.nanmin(t1_speed[t1_speed > 0]),
    }


def analyze_segment_patterns(trial_dir: str) -> dict:
    """Analyze segment-level patterns for a trial."""
    trial_data = error_analysis.load_trial_data(trial_dir)
    
    if trial_data['level1'] is None or trial_data['level2'] is None:
        return {}
    
    level1 = trial_data['level1']
    level2 = trial_data['level2']
    
    # Get swap segments
    swap_segments = error_analysis.get_swap_segments(level1, level2)
    
    if len(swap_segments) == 0:
        return {
            'num_segments': 0,
            'is_perfect': True,
        }
    
    # Analyze segments
    segment_lengths = [seg[1] - seg[0] + 1 for seg in swap_segments]
    segment_starts = [seg[0] for seg in swap_segments]
    segment_ends = [seg[1] for seg in swap_segments]
    
    # Check if global swap (single large segment)
    total_frames = min(len(level1), len(level2))
    is_global = len(swap_segments) == 1 and segment_lengths[0] > total_frames * 0.3
    
    # Analyze cross-sign consistency
    cross_sign1 = metrics.get_ht_cross_sign(level1.iloc[:total_frames])
    cross_sign2 = metrics.get_ht_cross_sign(level2.iloc[:total_frames])
    sign_match_rate = np.mean(cross_sign1 == cross_sign2)
    
    # Analyze speed patterns in swapped segments
    try:
        settings = pivr_loader.get_all_settings(trial_dir)
        fps = settings['Framerate'] if settings else 30
    except Exception:
        fps = 30
    
    h1_speed = metrics.get_speed_from_df(level1.iloc[:total_frames], 'head', fps=fps)
    t1_speed = metrics.get_speed_from_df(level1.iloc[:total_frames], 'tail', fps=fps)
    h2_speed = metrics.get_speed_from_df(level2.iloc[:total_frames], 'head', fps=fps)
    t2_speed = metrics.get_speed_from_df(level2.iloc[:total_frames], 'tail', fps=fps)
    
    # Check if global swap detection would work
    mean_h1 = np.nanmean(h1_speed)
    mean_t1 = np.nanmean(t1_speed)
    mean_h2 = np.nanmean(h2_speed)
    mean_t2 = np.nanmean(t2_speed)
    
    # Current global swap check
    current_check = mean_t1 > mean_h1  # Would current method catch it?
    correct_check = mean_t2 > mean_h2  # What should it be?
    
    return {
        'num_segments': len(swap_segments),
        'is_perfect': False,
        'is_global_swap': is_global,
        'segment_lengths': segment_lengths,
        'max_segment_length': max(segment_lengths),
        'segment_starts': segment_starts,
        'segment_ends': segment_ends,
        'cross_sign_match_rate': sign_match_rate,
        'mean_head_speed_level1': mean_h1,
        'mean_tail_speed_level1': mean_t1,
        'mean_head_speed_level2': mean_h2,
        'mean_tail_speed_level2': mean_t2,
        'current_global_swap_detection': current_check,
        'should_swap_globally': correct_check,
        'global_swap_detection_works': current_check == correct_check,
    }


def analyze_detection_failures(trial_dir: str) -> dict:
    """Analyze why current detection methods fail."""
    trial_data = error_analysis.load_trial_data(trial_dir)
    
    if trial_data['raw'] is None or trial_data['level1'] is None or trial_data['level2'] is None:
        return {}
    
    try:
        settings = pivr_loader.get_all_settings(trial_dir)
        fps = settings['Framerate'] if settings else 30
    except Exception:
        fps = 30
    
    raw_data = trial_data['raw']
    level1 = trial_data['level1']
    level2 = trial_data['level2']
    
    # Ground truth segments
    gt_segments = error_analysis.get_swap_segments(level1, level2)
    gt_frames = error_analysis.identify_swapped_frames(level1, level2)
    gt_set = set(gt_frames)
    
    # Run detection algorithms
    try:
        mdm = tracking_correction.flag_min_delta_mismatches(raw_data, debug=False)
        cosr = tracking_correction.flag_overlap_sign_reversals(raw_data, debug=False)
        comm = tracking_correction.flag_overlap_minimum_mismatches(raw_data, debug=False)
        all_detected = utils.merge(mdm, cosr, comm)
        detected_set = set(all_detected)
        
        # Convert detected frames to segments
        detected_segments = utils.get_consecutive_ranges(list(detected_set))
        
        # Check which ground truth segments are detected
        detected_gt_segments = []
        for gt_seg in gt_segments:
            gt_start, gt_end = gt_seg
            gt_seg_set = set(range(gt_start, gt_end + 1))
            # Check if any detected segment overlaps significantly with GT segment
            overlap_found = False
            for det_seg in detected_segments:
                det_start, det_end = det_seg
                det_seg_set = set(range(det_start, det_end + 1))
                overlap = len(gt_seg_set & det_seg_set)
                if overlap > len(gt_seg_set) * 0.1:  # 10% overlap threshold
                    overlap_found = True
                    break
            detected_gt_segments.append(overlap_found)
        
        segment_detection_rate = np.mean(detected_gt_segments) if len(detected_gt_segments) > 0 else 0
        
    except Exception as e:
        print(f"Error in detection analysis: {e}")
        return {}
    
    return {
        'ground_truth_segments': len(gt_segments),
        'detected_frames': len(all_detected),
        'detected_segments': len(detected_segments),
        'segment_detection_rate': segment_detection_rate,
        'min_delta_mismatches': len(mdm),
        'overlap_sign_reversals': len(cosr),
        'overlap_min_delta_mismatches': len(comm),
    }


def compare_perfect_vs_problematic(test_data_dir: str) -> dict:
    """Compare characteristics of perfect trials vs problematic trials."""
    all_trials = error_analysis.load_all_trials(test_data_dir)
    
    perfect_trials = []
    problematic_trials = []
    
    for trial_name, trial_data in all_trials.items():
        if trial_data['level1'] is None or trial_data['level2'] is None:
            continue
        
        stats = error_analysis.calculate_error_statistics(trial_data['level1'], trial_data['level2'])
        
        trial_info = {
            'name': trial_name,
            'error_rate': stats['error_rate'],
            'num_segments': stats['num_swap_segments'],
        }
        
        if stats['error_rate'] == 0:
            perfect_trials.append(trial_info)
        else:
            problematic_trials.append(trial_info)
    
    return {
        'perfect_trials': perfect_trials,
        'problematic_trials': problematic_trials,
        'num_perfect': len(perfect_trials),
        'num_problematic': len(problematic_trials),
    }


def main():
    """Run Step 1 analysis."""
    # Get the package directory (swap_correction/)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    # test_data is at swap_correction/tests/test_data/
    test_data_dir = os.path.join(package_dir, 'tests', 'test_data')
    
    print("=" * 80)
    print("STEP 1 ANALYSIS: Speed Ratio Bug and Segment-Level Patterns")
    print("=" * 80)
    
    # 1. Analyze speed ratio bug
    print("\n1. ANALYZING SPEED RATIO CALCULATION BUG")
    print("-" * 80)
    
    speed_ratio_analysis = {}
    for trial_name in os.listdir(test_data_dir):
        trial_dir = os.path.join(test_data_dir, trial_name)
        if not os.path.isdir(trial_dir):
            continue
        
        analysis = analyze_speed_ratio_bug(trial_dir)
        if analysis:
            speed_ratio_analysis[trial_name] = analysis
    
    # Summarize speed ratio issues
    if speed_ratio_analysis:
        total_extreme = sum(a.get('extreme_errors', 0) for a in speed_ratio_analysis.values())
        total_near_zero = sum(a.get('near_zero_tail_speed_frames', 0) for a in speed_ratio_analysis.values())
        max_error = max(a.get('max_ratio_error', 0) for a in speed_ratio_analysis.values())
        
        print(f"Total trials analyzed: {len(speed_ratio_analysis)}")
        print(f"Frames with near-zero tail speed: {total_near_zero}")
        print(f"Frames with extreme ratio errors (>1000): {total_extreme}")
        print(f"Maximum ratio error: {max_error:.2f}")
    
    # 2. Analyze segment patterns
    print("\n2. ANALYZING SEGMENT-LEVEL PATTERNS")
    print("-" * 80)
    
    segment_analysis = {}
    global_swap_trials = []
    multi_segment_trials = []
    
    for trial_name in os.listdir(test_data_dir):
        trial_dir = os.path.join(test_data_dir, trial_name)
        if not os.path.isdir(trial_dir):
            continue
        
        analysis = analyze_segment_patterns(trial_dir)
        if analysis:
            segment_analysis[trial_name] = analysis
            if analysis.get('is_global_swap'):
                global_swap_trials.append(trial_name)
            elif analysis.get('num_segments', 0) > 1:
                multi_segment_trials.append(trial_name)
    
    print(f"\nGlobal swap trials (single segment >30% of trial): {len(global_swap_trials)}")
    for trial in global_swap_trials:
        seg_info = segment_analysis[trial]
        print(f"  - {trial}: 1 segment of {seg_info['max_segment_length']} frames")
        print(f"    Cross-sign match: {seg_info['cross_sign_match_rate']:.3f}")
        print(f"    Global swap detection works: {seg_info.get('global_swap_detection_works', False)}")
    
    print(f"\nMulti-segment trials: {len(multi_segment_trials)}")
    for trial in multi_segment_trials[:5]:  # Show first 5
        seg_info = segment_analysis[trial]
        print(f"  - {trial}: {seg_info['num_segments']} segments, max length: {seg_info['max_segment_length']}")
    
    # 3. Compare perfect vs problematic
    print("\n3. COMPARING PERFECT VS PROBLEMATIC TRIALS")
    print("-" * 80)
    
    comparison = compare_perfect_vs_problematic(test_data_dir)
    print(f"Perfect trials: {comparison['num_perfect']}")
    print(f"Problematic trials: {comparison['num_problematic']}")
    
    # 4. Analyze detection failures on problematic trials
    print("\n4. ANALYZING DETECTION FAILURES")
    print("-" * 80)
    
    detection_analysis = {}
    problematic_trial_names = [t['name'] for t in comparison['problematic_trials']]
    for trial_name in problematic_trial_names[:5]:  # Analyze first 5 problematic
        trial_dir = os.path.join(test_data_dir, trial_name)
        analysis = analyze_detection_failures(trial_dir)
        if analysis:
            detection_analysis[trial_name] = analysis
            print(f"\n{trial_name}:")
            print(f"  GT segments: {analysis.get('ground_truth_segments', 0)}")
            print(f"  Detected frames: {analysis.get('detected_frames', 0)}")
            print(f"  Detected segments: {analysis.get('detected_segments', 0)}")
            print(f"  Segment detection rate: {analysis.get('segment_detection_rate', 0):.2%}")
    
    # Save findings
    findings = {
        'speed_ratio_analysis': speed_ratio_analysis,
        'segment_analysis': segment_analysis,
        'detection_analysis': detection_analysis,
        'comparison': comparison,
        'global_swap_trials': global_swap_trials,
        'multi_segment_trials': multi_segment_trials,
    }
    
    return findings


if __name__ == '__main__':
    findings = main()


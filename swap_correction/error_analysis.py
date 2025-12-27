"""
Error analysis module for characterizing swap errors in tracking data.

This module provides functions to:
- Load and compare raw, level1 (auto-corrected), and level2 (manually corrected) data
- Calculate various error metrics
- Identify swap errors and their characteristics
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, List, Optional
from swap_correction import pivr_loader, metrics, tracking_correction, utils


def load_trial_data(trial_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all three data files for a trial (raw, level1, level2).
    
    Parameters:
    -----------
    trial_dir : str
        Directory containing the trial data files
        
    Returns:
    --------
    dict
        Dictionary with keys 'raw', 'level1', 'level2' containing DataFrames.
        Missing files will have None values.
    """
    result = {'raw': None, 'level1': None, 'level2': None}
    
    # Find data files
    files = os.listdir(trial_dir)
    raw_file = None
    level1_file = None
    level2_file = None
    
    for f in files:
        if f.endswith('_data.csv') and not f.endswith('_level1.csv') and not f.endswith('_level2.csv'):
            raw_file = f
        elif f.endswith('_level1.csv'):
            level1_file = f
        elif f.endswith('_level2.csv'):
            level2_file = f
    
    # Load raw data
    if raw_file:
        try:
            result['raw'] = pivr_loader.load_raw_data(trial_dir, raw_file, px2mm=True)
        except Exception as e:
            print(f"Warning: Could not load raw data from {trial_dir}: {e}")
            result['raw'] = None
    
    # Load level1 data
    if level1_file:
        try:
            result['level1'] = pivr_loader.load_raw_data(trial_dir, level1_file, px2mm=True)
        except Exception as e:
            print(f"Warning: Could not load level1 data from {trial_dir}: {e}")
            result['level1'] = None
    
    # Load level2 data (ground truth)
    if level2_file:
        try:
            result['level2'] = pivr_loader.load_raw_data(trial_dir, level2_file, px2mm=True)
        except Exception as e:
            print(f"Warning: Could not load level2 data from {trial_dir}: {e}")
            result['level2'] = None
    
    return result


def load_all_trials(test_data_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load all trials from test_data directory.
    
    Parameters:
    -----------
    test_data_dir : str
        Directory containing trial subdirectories
        
    Returns:
    --------
    dict
        Dictionary mapping trial names to their data dictionaries
    """
    trials = {}
    trial_dirs = utils.get_dirs(test_data_dir)
    
    for trial_dir in trial_dirs:
        trial_name = os.path.basename(trial_dir)
        trial_data = load_trial_data(trial_dir)
        trials[trial_name] = trial_data
    
    return trials


def identify_swapped_frames(level1_data: pd.DataFrame, level2_data: pd.DataFrame) -> np.ndarray:
    """
    Find frames where level1 differs from level2 (indicating swap errors).
    
    A frame is considered swapped if head or tail positions differ significantly
    (more than just rounding differences).
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
        
    Returns:
    --------
    np.ndarray
        Array of frame indices where swaps are detected
    """
    # Align dataframes by index
    min_len = min(len(level1_data), len(level2_data))
    level1 = level1_data.iloc[:min_len].copy()
    level2 = level2_data.iloc[:min_len].copy()
    
    # Check if head positions differ (accounting for NaN values)
    head_diff = np.zeros(min_len, dtype=bool)
    tail_diff = np.zeros(min_len, dtype=bool)
    
    for i in range(min_len):
        # Check head position
        h1_valid = not (pd.isna(level1.iloc[i]['xhead']) or pd.isna(level1.iloc[i]['yhead']))
        h2_valid = not (pd.isna(level2.iloc[i]['xhead']) or pd.isna(level2.iloc[i]['yhead']))
        
        if h1_valid and h2_valid:
            h1_pos = np.array([level1.iloc[i]['xhead'], level1.iloc[i]['yhead']])
            h2_pos = np.array([level2.iloc[i]['xhead'], level2.iloc[i]['yhead']])
            head_dist = np.linalg.norm(h1_pos - h2_pos)
            # Threshold: if positions differ by more than 0.5mm, likely a swap
            head_diff[i] = head_dist > 0.5
        
        # Check tail position
        t1_valid = not (pd.isna(level1.iloc[i]['xtail']) or pd.isna(level1.iloc[i]['ytail']))
        t2_valid = not (pd.isna(level2.iloc[i]['xtail']) or pd.isna(level2.iloc[i]['ytail']))
        
        if t1_valid and t2_valid:
            t1_pos = np.array([level1.iloc[i]['xtail'], level1.iloc[i]['ytail']])
            t2_pos = np.array([level2.iloc[i]['xtail'], level2.iloc[i]['ytail']])
            tail_dist = np.linalg.norm(t1_pos - t2_pos)
            tail_diff[i] = tail_dist > 0.5
    
    # A frame is swapped if head OR tail positions differ significantly
    swapped = head_diff | tail_diff
    
    return np.where(swapped)[0]


def get_swap_segments(level1_data: pd.DataFrame, level2_data: pd.DataFrame) -> np.ndarray:
    """
    Identify continuous swap segments.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
        
    Returns:
    --------
    np.ndarray
        Nx2 array of [start_frame, end_frame] for each swap segment
    """
    swapped_frames = identify_swapped_frames(level1_data, level2_data)
    
    if len(swapped_frames) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Get consecutive ranges
    segments = utils.get_consecutive_ranges(swapped_frames.tolist())
    return np.array(segments)


def calculate_swap_differences(level1_data: pd.DataFrame, level2_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute frame-by-frame differences between level1 and level2.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with difference metrics for each frame
    """
    min_len = min(len(level1_data), len(level2_data))
    level1 = level1_data.iloc[:min_len].copy()
    level2 = level2_data.iloc[:min_len].copy()
    
    differences = pd.DataFrame(index=level1.index[:min_len])
    
    # Head position differences
    h1_x = level1['xhead'].values
    h1_y = level1['yhead'].values
    h2_x = level2['xhead'].values
    h2_y = level2['yhead'].values
    
    head_dx = h2_x - h1_x
    head_dy = h2_y - h1_y
    head_distance = np.sqrt(head_dx**2 + head_dy**2)
    
    # Tail position differences
    t1_x = level1['xtail'].values
    t1_y = level1['ytail'].values
    t2_x = level2['xtail'].values
    t2_y = level2['ytail'].values
    
    tail_dx = t2_x - t1_x
    tail_dy = t2_y - t1_y
    tail_distance = np.sqrt(tail_dx**2 + tail_dy**2)
    
    # Centroid differences
    c1_x = level1['xctr'].values
    c1_y = level1['yctr'].values
    c2_x = level2['xctr'].values
    c2_y = level2['yctr'].values
    
    centroid_dx = c2_x - c1_x
    centroid_dy = c2_y - c1_y
    centroid_distance = np.sqrt(centroid_dx**2 + centroid_dy**2)
    
    # Body length differences
    body_len1 = metrics.get_delta_in_frame(level1, 'head', 'tail')
    body_len2 = metrics.get_delta_in_frame(level2, 'head', 'tail')
    body_len_diff = body_len2 - body_len1
    
    differences['head_dx'] = head_dx
    differences['head_dy'] = head_dy
    differences['head_distance'] = head_distance
    differences['tail_dx'] = tail_dx
    differences['tail_dy'] = tail_dy
    differences['tail_distance'] = tail_distance
    differences['centroid_dx'] = centroid_dx
    differences['centroid_dy'] = centroid_dy
    differences['centroid_distance'] = centroid_distance
    differences['body_length_diff'] = body_len_diff
    
    return differences


def calculate_detection_metrics(raw_data: Optional[pd.DataFrame], 
                                level1_data: pd.DataFrame, 
                                level2_data: pd.DataFrame,
                                fps: int = 30) -> Dict:
    """
    Calculate detection metrics comparing detected swaps to ground truth.
    
    Parameters:
    -----------
    raw_data : pd.DataFrame or None
        Raw data (for running detection algorithms)
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
    fps : int
        Frame rate
        
    Returns:
    --------
    dict
        Dictionary containing detection metrics
    """
    # Ground truth: frames that differ between level1 and level2
    gt_swaps = identify_swapped_frames(level1_data, level2_data)
    gt_set = set(gt_swaps)
    
    metrics_dict = {
        'ground_truth_swaps': len(gt_swaps),
        'ground_truth_segments': len(get_swap_segments(level1_data, level2_data)),
    }
    
    if raw_data is None:
        return metrics_dict
    
    # Run detection algorithms on raw data
    try:
        # Minimum delta mismatches
        mdm = tracking_correction.flag_min_delta_mismatches(raw_data, debug=False)
        mdm_set = set(mdm)
        
        # Overlap sign reversals
        cosr = tracking_correction.flag_overlap_sign_reversals(raw_data, debug=False)
        cosr_set = set(cosr)
        
        # Overlap minimum-delta mismatches
        comm = tracking_correction.flag_overlap_minimum_mismatches(raw_data, debug=False)
        comm_set = set(comm)
        
        # Combined detection
        all_detected = utils.merge(mdm, cosr, comm)
        detected_set = set(all_detected)
        
        # Calculate true/false positives/negatives
        true_positives = len(gt_set & detected_set)
        false_positives = len(detected_set - gt_set)
        false_negatives = len(gt_set - detected_set)
        true_negatives = len(set(range(len(raw_data))) - gt_set - detected_set)
        
        # Calculate rates
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_dict.update({
            'min_delta_mismatches': len(mdm),
            'overlap_sign_reversals': len(cosr),
            'overlap_min_delta_mismatches': len(comm),
            'total_detected': len(all_detected),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        })
    except Exception as e:
        print(f"Warning: Error calculating detection metrics: {e}")
    
    return metrics_dict


def calculate_position_errors(level1_data: pd.DataFrame, level2_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate position-based error metrics.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with position error metrics
    """
    differences = calculate_swap_differences(level1_data, level2_data)
    
    errors = pd.DataFrame(index=differences.index)
    errors['head_position_error'] = differences['head_distance']
    errors['tail_position_error'] = differences['tail_distance']
    errors['centroid_position_error'] = differences['centroid_distance']
    errors['body_length_error'] = differences['body_length_diff']
    
    return errors


def calculate_motion_errors(level1_data: pd.DataFrame, level2_data: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    Calculate motion-based error metrics.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
    fps : int
        Frame rate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with motion error metrics
    """
    min_len = min(len(level1_data), len(level2_data))
    level1 = level1_data.iloc[:min_len].copy()
    level2 = level2_data.iloc[:min_len].copy()
    
    errors = pd.DataFrame(index=level1.index[:min_len])
    
    # Head speed
    h1_speed = metrics.get_speed_from_df(level1, 'head', fps=fps)
    h2_speed = metrics.get_speed_from_df(level2, 'head', fps=fps)
    errors['head_speed_error'] = h2_speed - h1_speed
    
    # Tail speed
    t1_speed = metrics.get_speed_from_df(level1, 'tail', fps=fps)
    t2_speed = metrics.get_speed_from_df(level2, 'tail', fps=fps)
    errors['tail_speed_error'] = t2_speed - t1_speed
    
    # Speed ratio
    h1_ratio = h1_speed / (t1_speed + 1e-6)  # Avoid division by zero
    h2_ratio = h2_speed / (t2_speed + 1e-6)
    errors['speed_ratio_error'] = h2_ratio - h1_ratio
    
    return errors


def calculate_geometric_errors(level1_data: pd.DataFrame, level2_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate geometric error metrics.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with geometric error metrics
    """
    min_len = min(len(level1_data), len(level2_data))
    level1 = level1_data.iloc[:min_len].copy()
    level2 = level2_data.iloc[:min_len].copy()
    
    errors = pd.DataFrame(index=level1.index[:min_len])
    
    # Body orientation
    orient1 = metrics.get_orientation(level1)
    orient2 = metrics.get_orientation(level2)
    # Calculate angular difference (accounting for wrap-around)
    angle_diff = orient2 - orient1
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize to [-pi, pi]
    errors['orientation_error'] = np.abs(angle_diff)
    
    # Cross-product sign consistency
    sign1 = metrics.get_ht_cross_sign(level1)
    sign2 = metrics.get_ht_cross_sign(level2)
    errors['cross_sign_match'] = (sign1 == sign2).astype(float)
    
    return errors


def calculate_error_statistics(level1_data: pd.DataFrame, level2_data: pd.DataFrame) -> Dict:
    """
    Calculate overall error statistics.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
        
    Returns:
    --------
    dict
        Dictionary containing error statistics
    """
    swapped_frames = identify_swapped_frames(level1_data, level2_data)
    swap_segments = get_swap_segments(level1_data, level2_data)
    
    total_frames = min(len(level1_data), len(level2_data))
    error_rate = len(swapped_frames) / total_frames if total_frames > 0 else 0
    
    segment_lengths = []
    if len(swap_segments) > 0:
        segment_lengths = [seg[1] - seg[0] + 1 for seg in swap_segments]
    
    stats = {
        'total_frames': total_frames,
        'swapped_frames': len(swapped_frames),
        'error_rate': error_rate,
        'num_swap_segments': len(swap_segments),
        'avg_segment_length': np.mean(segment_lengths) if segment_lengths else 0,
        'median_segment_length': np.median(segment_lengths) if segment_lengths else 0,
        'max_segment_length': np.max(segment_lengths) if segment_lengths else 0,
        'min_segment_length': np.min(segment_lengths) if segment_lengths else 0,
    }
    
    return stats


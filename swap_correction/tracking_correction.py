"""
Core tracking correction algorithms for head-tail swap detection and correction.

This module provides:
- Main tracking correction pipeline
- Swap detection algorithms (minimum delta, sign reversals, overlaps)
- Error removal and interpolation
- Data filtering (Gaussian, Savitzky-Golay, median)
- Segment-based validation
- Global swap correction
"""

import numpy as np
import pandas as pd
import scipy as sp
from swap_correction import utils, metrics
from swap_correction.kalman_filter import KalmanFilter

# Parameters
OVERLAP_THRESH = 0 # maximum distance between overlapping points


# ----- Tracking Correction -----

def tracking_correction(data : pd.DataFrame, fps : int, swapCorrection : bool = True,
            removeErrors : bool = True, interp : bool = False, validate : bool = True,
            filterData : bool = False, debug : bool = False,
            comprehensive_params : dict = None) -> pd.DataFrame:
    """
    Apply tracking corrections and filtering to raw data

    data (DataFrame): raw position data
    fps (int): frame rate
    swapCorrection (bool): correct head-tail swaps
    interp (bool): interpolate over position data in bad frames (of correctTracking)
    validate (bool): use assumption of forward movement to catch remaining swaps
    filterData (bool): apply a Savitzky-Golay filter to the position data
    debug (bool): print debug messages
    comprehensive_params (dict): parameters for comprehensive metrics approach (optional)
    """
    # correct tracking errors
    data = remove_edge_frames(data,debug=debug)
    if swapCorrection : data = correct_tracking_errors(data, fps=fps, debug=debug,
                                                       comprehensive_params=comprehensive_params)
    if validate : data = validate_corrected_data(data,fps,debug=debug)
    if removeErrors : data = remove_overlaps(data,fps,debug=debug)
    if interp : data = interpolate_gaps(data)
    if filterData: data = filter_data(data)
    data = data.round(1) # round out roundoff errors 
    return data


def remove_edge_frames(rawData : pd.DataFrame, debug : bool = False) -> pd.DataFrame:
    """
    Set values in frames with all zero position entries at the beginning / end of data to NaN
    """
    data = rawData.copy()
    cols = utils.flatten(metrics.POSDICT.values())
    xcols = [col for col in cols if 'x' in col] # sloppy, but functional
    ycols = [col for col in cols if 'y' in col]

    # find overlap frames at edges where raw position data set to zero
    # NOTE: data has been translated, so positions will not be zero; need to look see where all values identical
    xdata = data.loc[:,xcols]
    ydata = data.loc[:,ycols]
    x = xdata.eq(xdata.iloc[:, 0], axis=0).all(1) # check if all values in row equal to first
    y = ydata.eq(ydata.iloc[:, 0], axis=0).all(1)
    counts = np.logical_and(x,y)
    frames = np.where(counts)[0]
    if debug : print('Zeroed Frames:',frames)

    # get sequences at edges of data
    segs = utils.get_consecutive_ranges(frames)
    segs = [seg for seg in segs if seg[0] == 0 or seg[1] == data.shape[0]-1]
    if debug : print('Edge Segments:',segs)

    # set position data to NaN in target frames
    for a, b in segs:
        data.loc[a:b+1,cols] = np.nan

    return data


def detect_swaps_by_cross_sign_consistency(rawData : pd.DataFrame, fps : int = 30,
                                          window_size : int = 150,
                                          consistency_threshold : float = 0.6,
                                          global_threshold : float = 0.7,
                                          debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using cross-sign consistency analysis.
    
    Cross-sign is the most reliable indicator when speeds are similar. This function:
    1. Checks overall cross-sign consistency (for global swaps)
    2. Uses sliding windows to find regions with low consistency (swapped segments)
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    window_size : int
        Size of sliding window in frames (default: 150)
    consistency_threshold : float
        Minimum consistency rate to consider a window as not swapped (default: 0.6)
    global_threshold : float
        Minimum overall consistency to avoid global swap (default: 0.7)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Filter data for cross-sign calculation
    filtered = filter_data(rawData)
    
    # Calculate cross-sign
    cross_sign = metrics.get_ht_cross_sign(filtered)
    
    # Remove NaN values
    valid_mask = ~np.isnan(cross_sign)
    valid_signs = cross_sign[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_signs) == 0:
        if debug:
            print('No valid cross-sign data')
        return np.empty((0, 2), dtype=int)
    
    # Check overall consistency (for global swap detection)
    positive_ratio = np.sum(valid_signs > 0) / len(valid_signs)
    negative_ratio = np.sum(valid_signs < 0) / len(valid_signs)
    overall_consistency = max(positive_ratio, negative_ratio)
    
    if debug:
        print(f'Overall cross-sign consistency: {overall_consistency:.3f}')
        print(f'  Positive: {positive_ratio:.3f}, Negative: {negative_ratio:.3f}')
    
    n_frames = len(rawData)
    
    # If overall consistency is very low, check if swapping would improve it
    # This is more reliable than just checking consistency alone
    # (Some trajectories naturally have low consistency, not due to swaps)
    if overall_consistency < global_threshold:
        # Test if swapping would improve consistency
        # Create a test swapped version
        test_swapped = rawData.copy()
        test_swapped[['xhead','yhead','xtail','ytail']] = test_swapped[['xtail','ytail','xhead','yhead']]
        test_filtered = filter_data(test_swapped)
        test_signs = metrics.get_ht_cross_sign(test_filtered)
        test_valid = test_signs[~np.isnan(test_signs)]
        
        if len(test_valid) > 0:
            test_pos = np.sum(test_valid > 0) / len(test_valid)
            test_neg = np.sum(test_valid < 0) / len(test_valid)
            test_consistency = max(test_pos, test_neg)
            
            if debug:
                print(f'Original consistency: {overall_consistency:.3f}')
                print(f'Swapped consistency: {test_consistency:.3f}')
            
            # Only swap if it significantly improves consistency (>0.1 improvement)
            if test_consistency > overall_consistency + 0.1:
                if debug:
                    print(f'Swapping improves consistency by {test_consistency - overall_consistency:.3f} - returning entire trajectory')
                return np.array([[0, n_frames - 1]])
            else:
                if debug:
                    print(f'Swapping does not improve consistency - not swapping')
        else:
            if debug:
                print(f'Cannot test swapped consistency - not enough valid data')
    
    # Use sliding window to find regions with low consistency
    n_frames = len(rawData)
    swapped_windows = []
    
    # Slide window across trajectory
    for start in range(0, n_frames - window_size + 1, window_size // 2):  # 50% overlap
        end = min(start + window_size, n_frames)
        
        # Get cross-signs in this window
        window_mask = (valid_indices >= start) & (valid_indices < end)
        window_signs = valid_signs[window_mask]
        
        if len(window_signs) < window_size * 0.5:  # Need at least 50% valid data
            continue
        
        # Calculate consistency in this window
        pos_ratio = np.sum(window_signs > 0) / len(window_signs)
        neg_ratio = np.sum(window_signs < 0) / len(window_signs)
        window_consistency = max(pos_ratio, neg_ratio)
        
        # If consistency is very low, this window likely has swaps
        # Use stricter threshold (0.5 instead of 0.6) to reduce false positives
        # Very low consistency (<0.5) is a stronger indicator of swap
        if window_consistency < 0.5:  # Stricter threshold
            swapped_windows.append((start, end))
            if debug:
                print(f'Very low consistency window [{start}:{end}]: {window_consistency:.3f}')
    
    if len(swapped_windows) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping windows into segments
    swapped_windows = np.array(swapped_windows)
    # Sort by start frame
    swapped_windows = swapped_windows[swapped_windows[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_windows[0]
    
    for start, end in swapped_windows[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)  # Extend segment
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    if debug:
        print(f'Merged {len(swapped_windows)} windows into {len(merged_segments)} segments')
    
    return np.array(merged_segments)


def detect_swaps_by_velocity_ratios(rawData : pd.DataFrame, fps : int = 30,
                                    window_size : int = 50,
                                    min_window_size : int = 50,
                                    percentile_threshold : float = 0.6,
                                    debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using velocity-based analysis over windows.
    
    Calculates head/tail velocity ratios over sliding windows and flags windows
    where tail velocity consistently exceeds head velocity (indicating a swap).
    Uses median/percentile thresholds for robustness.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    window_size : int
        Size of sliding window in frames (default: 50)
    min_window_size : int
        Minimum window size required for detection (default: 50)
    percentile_threshold : float
        Percentile threshold for tail > head velocity (default: 0.6 = 60th percentile)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Filter data for velocity calculation
    filtered = filter_data(rawData)
    
    # Calculate speeds
    hspd = metrics.get_speed_from_df(filtered, 'head', fps=fps)
    tspd = metrics.get_speed_from_df(filtered, 'tail', fps=fps)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(hspd) | np.isnan(tspd))
    n_frames = len(rawData)
    
    # Use sliding window to find regions where tail > head velocity
    swapped_windows = []
    
    # Slide window across trajectory
    step_size = window_size // 2  # 50% overlap
    for start in range(0, n_frames - min_window_size + 1, step_size):
        end = min(start + window_size, n_frames)
        
        # Get speeds in this window
        window_mask = valid_mask[start:end]
        if np.sum(window_mask) < min_window_size * 0.5:  # Need at least 50% valid data
            continue
        
        h_window = hspd[start:end][window_mask]
        t_window = tspd[start:end][window_mask]
        
        # Calculate velocity ratio (tail/head)
        # Use median for robustness
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = t_window / np.maximum(h_window, 0.01)  # Avoid division by zero
        
        # Flag if tail velocity consistently exceeds head velocity
        # Check if percentile of ratios > 1.0 (tail faster than head)
        if len(ratio) > 0:
            percentile_value = np.percentile(ratio, percentile_threshold * 100)
            median_ratio = np.median(ratio)
            
            # Swap if median ratio significantly > 1.0 (tail consistently faster)
            # Use stricter threshold: median > 1.2 or percentile > 1.3 to reduce false positives
            if median_ratio > 1.2 or percentile_value > 1.3:
                swapped_windows.append((start, end))
                if debug:
                    print(f'Velocity swap window [{start}:{end}]: median_ratio={median_ratio:.3f}, percentile={percentile_value:.3f}')
    
    if len(swapped_windows) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping windows into segments
    swapped_windows = np.array(swapped_windows)
    swapped_windows = swapped_windows[swapped_windows[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_windows[0]
    
    for start, end in swapped_windows[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds (indices are 0 to n_frames-1)
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    if debug:
        print(f'Merged {len(swapped_windows)} windows into {len(merged_segments)} segments')
    
    return np.array(merged_segments)


def get_angular_velocity(data : pd.DataFrame, key : str, fps : int = 30, 
                        window_size : int = 5) -> np.ndarray:
    """
    Calculate angular velocity (rate of change of direction) for a keypoint.
    
    Angular velocity measures how quickly the direction of motion changes.
    Head typically has higher angular velocity than tail (head moves more, tail follows).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Position data
    key : str
        Keypoint ('head' or 'tail')
    fps : int
        Frame rate
    window_size : int
        Window size for calculating direction changes (default: 5 frames)
        
    Returns:
    --------
    np.ndarray
        Angular velocity in degrees per second
    """
    x, y = metrics.vectors_from_key(data, key)
    n_frames = len(data)
    
    angular_velocities = np.zeros(n_frames)
    angular_velocities[:] = np.nan
    
    for i in range(window_size, n_frames - window_size):
        # Get position vectors in window
        x_window = x[i-window_size:i+window_size+1]
        y_window = y[i-window_size:i+window_size+1]
        
        # Calculate direction vectors (motion vectors)
        dx = np.diff(x_window)
        dy = np.diff(y_window)
        
        # Calculate angles between consecutive direction vectors
        angles = []
        for j in range(len(dx) - 1):
            v1 = np.array([dx[j], dy[j]])
            v2 = np.array([dx[j+1], dy[j+1]])
            
            # Normalize vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0.01 and norm2 > 0.01:  # Avoid division by zero
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
        
        if len(angles) > 0:
            # Angular velocity = mean angle change per frame, converted to degrees per second
            angular_velocities[i] = np.mean(angles) * fps
    
    return angular_velocities


def detect_swaps_by_angular_velocity(rawData : pd.DataFrame, fps : int = 30,
                                     window_size : int = 50,
                                     min_window_size : int = 50,
                                     ratio_threshold : float = 0.8,
                                     consistency_window : int = 50,
                                     debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using angular velocity analysis.
    
    Head typically has higher angular velocity than tail (head moves more, tail follows).
    If tail angular velocity > head angular velocity, it suggests a swap.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    window_size : int
        Size of sliding window in frames (default: 50)
    min_window_size : int
        Minimum window size required for detection (default: 50)
    ratio_threshold : float
        Threshold for tail/head angular velocity ratio (default: 0.8)
        If tail_ang_vel / head_ang_vel > threshold, likely swapped
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Filter data for angular velocity calculation
    filtered = filter_data(rawData)
    
    # Calculate angular velocities
    head_ang_vel = get_angular_velocity(filtered, 'head', fps=fps, window_size=5)
    tail_ang_vel = get_angular_velocity(filtered, 'tail', fps=fps, window_size=5)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(head_ang_vel) | np.isnan(tail_ang_vel))
    n_frames = len(rawData)
    
    # Use sliding window to find regions where tail angular velocity > head
    swapped_windows = []
    
    # Slide window across trajectory
    step_size = window_size // 2  # 50% overlap
    for start in range(0, n_frames - min_window_size + 1, step_size):
        end = min(start + window_size, n_frames)
        
        # Get angular velocities in this window
        window_mask = valid_mask[start:end]
        if np.sum(window_mask) < min_window_size * 0.5:  # Need at least 50% valid data
            continue
        
        h_ang_vel_window = head_ang_vel[start:end][window_mask]
        t_ang_vel_window = tail_ang_vel[start:end][window_mask]
        
        # Calculate ratio (tail/head angular velocity)
        # Use median for robustness
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = t_ang_vel_window / np.maximum(h_ang_vel_window, 0.1)  # Avoid division by zero
        
        # Flag if tail angular velocity consistently exceeds head
        if len(ratio) > 0:
            median_ratio = np.median(ratio)
            percentile_75 = np.percentile(ratio, 75)
            
            # Enhanced detection: Check temporal consistency
            # Pattern must persist across consistency_window frames
            is_consistent = True
            if consistency_window > 0 and end - start >= consistency_window:
                # Check if pattern persists in larger window
                consistency_start = max(0, start - consistency_window // 2)
                consistency_end = min(n_frames, end + consistency_window // 2)
                consistency_mask = valid_mask[consistency_start:consistency_end]
                
                if np.sum(consistency_mask) >= consistency_window * 0.5:
                    h_consistency = head_ang_vel[consistency_start:consistency_end][consistency_mask]
                    t_consistency = tail_ang_vel[consistency_start:consistency_end][consistency_mask]
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        consistency_ratio = t_consistency / np.maximum(h_consistency, 0.1)
                    
                    # Pattern is consistent if >70% of frames in larger window show same pattern
                    consistent_frames = np.sum(consistency_ratio > ratio_threshold) / len(consistency_ratio)
                    is_consistent = consistent_frames > 0.7
            
            # Swap if median ratio > threshold AND pattern is temporally consistent
            # This indicates tail is moving more erratically than head (suggests swap)
            if is_consistent and (median_ratio > ratio_threshold or percentile_75 > 1.0):
                swapped_windows.append((start, end))
                if debug:
                    print(f'Angular velocity swap window [{start}:{end}]: median_ratio={median_ratio:.3f}, p75={percentile_75:.3f}, consistent={is_consistent}')
    
    if len(swapped_windows) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping windows into segments
    swapped_windows = np.array(swapped_windows)
    swapped_windows = swapped_windows[swapped_windows[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_windows[0]
    
    for start, end in swapped_windows[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    if debug:
        print(f'Merged {len(swapped_windows)} windows into {len(merged_segments)} segments')
    
    return np.array(merged_segments)


def _calculate_comprehensive_metrics(rawData : pd.DataFrame, start : int, end : int,
                                    fps : int = 30, validation_window : int = 20,
                                    angular_vel_ratio : float = 1.0,
                                    angular_var_ratio : float = 1.0,
                                    distance_ratio_threshold : float = 0.9,
                                    speed_ratio_threshold : float = 1.0,
                                    alignment_angle_threshold : float = 90.0) -> dict:
    """
    Calculate all 5 comprehensive metrics for a given segment.
    
    Parameters:
    -----------
    angular_vel_ratio : float
        Threshold ratio for angular velocity (tail/head). Default 1.0 means tail > head.
        Use >1.0 (e.g., 1.1, 1.2) to require stronger signal.
    angular_var_ratio : float
        Threshold ratio for angular variation (tail/head). Default 1.0 means tail > head.
        Use >1.0 (e.g., 1.1, 1.2) to require stronger signal.
    distance_ratio_threshold : float
        Threshold for distance ratio (tail/head). Default 0.9 means tail travels >90% of head.
        Use >0.9 (e.g., 0.95, 1.0, 1.05) for stricter requirement.
    speed_ratio_threshold : float
        Threshold for speed ratio (head/tail). Default 1.0 means head < tail.
        Use <1.0 (e.g., 0.9, 0.95) to require stronger signal.
    alignment_angle_threshold : float
        Threshold angle in degrees. Default 90.0 means >90° indicates swap.
        Use >90.0 (e.g., 100, 110, 120) for stricter requirement.
    
    Returns:
    --------
    dict
        Dictionary with metric values and swap indicators.
    """
    filtered = filter_data(rawData)
    segment_data = filtered.iloc[start:end+1]
    n_seg = len(segment_data)
    
    if n_seg < validation_window:
        return None
    
    metrics_result = {
        'votes': 0,
        'details': {}
    }
    
    # Metric 1: Absolute Angular Velocity
    head_ang_vel = get_angular_velocity(segment_data, 'head', fps=fps, window_size=5)
    tail_ang_vel = get_angular_velocity(segment_data, 'tail', fps=fps, window_size=5)
    
    valid_ang_vel = ~(np.isnan(head_ang_vel) | np.isnan(tail_ang_vel))
    if np.sum(valid_ang_vel) > 0:
        h_ang_abs = np.abs(head_ang_vel[valid_ang_vel])
        t_ang_abs = np.abs(tail_ang_vel[valid_ang_vel])
        
        median_h_ang = np.median(h_ang_abs)
        median_t_ang = np.median(t_ang_abs)
        
        # Swap if tail absolute angular velocity > head * ratio
        if median_h_ang > 0.001:  # Avoid division by zero
            ang_vel_ratio = median_t_ang / median_h_ang
            indicates_swap = ang_vel_ratio > angular_vel_ratio
        else:
            ang_vel_ratio = float('inf') if median_t_ang > 0 else 0.0
            indicates_swap = median_t_ang > median_h_ang * angular_vel_ratio
        
        if indicates_swap:
            metrics_result['votes'] += 1
        metrics_result['details']['angular_vel'] = {
            'head': median_h_ang,
            'tail': median_t_ang,
            'ratio': ang_vel_ratio if median_h_ang > 0.001 else 0.0,
            'indicates_swap': indicates_swap
        }
    
    # Metric 2: Angular Variation (standard deviation of direction changes)
    x_head, y_head = metrics.vectors_from_key(segment_data, 'head')
    x_tail, y_tail = metrics.vectors_from_key(segment_data, 'tail')
    
    # Calculate direction changes
    dx_head = np.diff(x_head)
    dy_head = np.diff(y_head)
    dx_tail = np.diff(x_tail)
    dy_tail = np.diff(y_tail)
    
    # Calculate angles between consecutive direction vectors
    head_angles = []
    tail_angles = []
    
    for i in range(len(dx_head) - 1):
        v1_h = np.array([dx_head[i], dy_head[i]])
        v2_h = np.array([dx_head[i+1], dy_head[i+1]])
        v1_t = np.array([dx_tail[i], dy_tail[i]])
        v2_t = np.array([dx_tail[i+1], dy_tail[i+1]])
        
        for v1, v2, angles in [(v1_h, v2_h, head_angles), (v1_t, v2_t, tail_angles)]:
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0.01 and norm2 > 0.01:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
    
    if len(head_angles) > 0 and len(tail_angles) > 0:
        head_variation = np.std(head_angles)
        tail_variation = np.std(tail_angles)
        
        # Swap if tail variation > head variation * ratio
        if head_variation > 0.001:  # Avoid division by zero
            var_ratio = tail_variation / head_variation
            indicates_swap = var_ratio > angular_var_ratio
        else:
            var_ratio = float('inf') if tail_variation > 0 else 0.0
            indicates_swap = tail_variation > head_variation * angular_var_ratio
        
        if indicates_swap:
            metrics_result['votes'] += 1
        metrics_result['details']['angular_variation'] = {
            'head': head_variation,
            'tail': tail_variation,
            'ratio': var_ratio if head_variation > 0.001 else 0.0,
            'indicates_swap': indicates_swap
        }
    
    # Metric 3: Distance Traveled
    # Calculate cumulative distance
    head_dist = 0
    tail_dist = 0
    
    for i in range(1, n_seg):
        h_dx = x_head[i] - x_head[i-1]
        h_dy = y_head[i] - y_head[i-1]
        t_dx = x_tail[i] - x_tail[i-1]
        t_dy = y_tail[i] - y_tail[i-1]
        
        if not (np.isnan(h_dx) or np.isnan(h_dy)):
            head_dist += np.sqrt(h_dx**2 + h_dy**2)
        if not (np.isnan(t_dx) or np.isnan(t_dy)):
            tail_dist += np.sqrt(t_dx**2 + t_dy**2)
    
    if head_dist > 0.001:  # Avoid division by zero
        distance_ratio = tail_dist / head_dist
        # Swap if tail travels more than threshold of head distance
        indicates_swap = distance_ratio > distance_ratio_threshold
        if indicates_swap:
            metrics_result['votes'] += 1
        metrics_result['details']['distance'] = {
            'head': head_dist,
            'tail': tail_dist,
            'ratio': distance_ratio,
            'indicates_swap': indicates_swap
        }
    
    # Metric 4: Speed Ratio
    hspd = metrics.get_speed_from_df(segment_data, 'head', fps=fps)
    tspd = metrics.get_speed_from_df(segment_data, 'tail', fps=fps)
    
    valid_speed = ~(np.isnan(hspd) | np.isnan(tspd))
    if np.sum(valid_speed) > 0:
        median_hspd = np.median(hspd[valid_speed])
        median_tspd = np.median(tspd[valid_speed])
        
        if median_hspd > 0.01:  # Avoid division by zero
            speed_ratio = median_hspd / median_tspd
            # Swap if head speed < tail speed * threshold (tail faster)
            indicates_swap = speed_ratio < speed_ratio_threshold
            if indicates_swap:
                metrics_result['votes'] += 1
            metrics_result['details']['speed'] = {
                'head': median_hspd,
                'tail': median_tspd,
                'ratio': speed_ratio,
                'indicates_swap': indicates_swap
            }
    
    # Metric 5: Alignment Angle
    tail_pos = segment_data[['xtail', 'ytail']].values
    mid_pos = segment_data[['xmid', 'ymid']].values
    
    # Body orientation vector (tail to midpoint)
    body_vec = mid_pos - tail_pos
    
    # Motion vector (tail displacement)
    tail_motion = np.diff(tail_pos, axis=0, prepend=tail_pos[0:1] - tail_pos[0:1])
    
    alignment_angles = []
    for i in range(1, len(body_vec)):
        bv = body_vec[i]
        tm = tail_motion[i]
        bv_norm = np.linalg.norm(bv)
        tm_norm = np.linalg.norm(tm)
        if bv_norm > 0.01 and tm_norm > 0.01:
            cos_angle = np.dot(bv, tm) / (bv_norm * tm_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            alignment_angles.append(angle)
    
    if len(alignment_angles) > 0:
        median_alignment = np.median(alignment_angles)
        # Swap if alignment angle > threshold (backwards motion)
        indicates_swap = median_alignment > alignment_angle_threshold
        if indicates_swap:
            metrics_result['votes'] += 1
        metrics_result['details']['alignment'] = {
            'angle': median_alignment,
            'indicates_swap': indicates_swap
        }
    
    return metrics_result


def detect_swaps_by_comprehensive_metrics(rawData : pd.DataFrame, fps : int = 30,
                                          window_size : int = 75,
                                          min_votes : int = 3,
                                          min_segment_size : int = 0,
                                          min_segment_duration : float = 0.0,
                                          angular_vel_ratio : float = 1.0,
                                          angular_var_ratio : float = 1.0,
                                          distance_ratio_threshold : float = 0.9,
                                          speed_ratio_threshold : float = 1.0,
                                          alignment_angle_threshold : float = 90.0,
                                          debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using comprehensive multi-metric analysis.
    
    Uses 5 complementary metrics about head vs tail behavior:
    1. Absolute angular velocity (head should be larger)
    2. Angular variation (head should have more variation)
    3. Distance traveled (head should travel farther)
    4. Speed ratio (head should be as fast or faster)
    5. Alignment angle (tail-midpoint should align with motion)
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    window_size : int
        Size of sliding window in frames (default: 75)
    min_votes : int
        Minimum number of metrics that must indicate swap (default: 3 out of 5)
    min_segment_size : int
        Minimum segment size in frames to keep (default: 0, no filtering)
    min_segment_duration : float
        Minimum segment duration in seconds to keep (default: 0.0, no filtering)
    angular_vel_ratio : float
        Threshold ratio for angular velocity metric (default: 1.0)
    angular_var_ratio : float
        Threshold ratio for angular variation metric (default: 1.0)
    distance_ratio_threshold : float
        Threshold for distance ratio metric (default: 0.9)
    speed_ratio_threshold : float
        Threshold for speed ratio metric (default: 1.0)
    alignment_angle_threshold : float
        Threshold angle in degrees for alignment metric (default: 90.0)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    filtered = filter_data(rawData)
    n_frames = len(rawData)
    
    # Use sliding window to find regions where multiple metrics agree on swap
    swapped_windows = []
    
    step_size = window_size // 2  # 50% overlap
    for start in range(0, n_frames - window_size + 1, step_size):
        end = min(start + window_size, n_frames)
        
        # Calculate comprehensive metrics for this window
        metrics_result = _calculate_comprehensive_metrics(
            filtered, start, end, fps=fps, validation_window=20,
            angular_vel_ratio=angular_vel_ratio,
            angular_var_ratio=angular_var_ratio,
            distance_ratio_threshold=distance_ratio_threshold,
            speed_ratio_threshold=speed_ratio_threshold,
            alignment_angle_threshold=alignment_angle_threshold
        )
        
        if metrics_result is None:
            continue
        
        # Check if enough metrics indicate swap
        if metrics_result['votes'] >= min_votes:
            swapped_windows.append((start, end))
            if debug:
                details = metrics_result['details']
                print(f'Comprehensive metrics swap window [{start}:{end}]: {metrics_result["votes"]}/5 votes')
                if 'angular_vel' in details:
                    print(f'  Angular vel: H={details["angular_vel"]["head"]:.2f}, T={details["angular_vel"]["tail"]:.2f}')
                if 'speed' in details:
                    print(f'  Speed ratio: {details["speed"]["ratio"]:.3f}')
                if 'alignment' in details:
                    print(f'  Alignment: {details["alignment"]["angle"]:.1f}°')
    
    if len(swapped_windows) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping windows into segments
    swapped_windows = np.array(swapped_windows)
    swapped_windows = swapped_windows[swapped_windows[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_windows[0]
    
    for start, end in swapped_windows[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    # Filter by minimum segment size and duration
    if min_segment_size > 0 or min_segment_duration > 0:
        filtered_segments = []
        min_frames = max(min_segment_size, int(min_segment_duration * fps)) if min_segment_duration > 0 else min_segment_size
        
        for start, end in merged_segments:
            segment_length = end - start + 1
            if segment_length >= min_frames:
                filtered_segments.append((start, end))
            elif debug:
                print(f'Filtered out segment [{start}:{end}] (length {segment_length} < {min_frames})')
        
        merged_segments = filtered_segments
    
    if debug:
        print(f'Merged {len(swapped_windows)} windows into {len(merged_segments)} segments')
    
    return np.array(merged_segments) if len(merged_segments) > 0 else np.empty((0, 2), dtype=int)


def refine_swap_boundaries_comprehensive(rawData : pd.DataFrame, segments : np.ndarray,
                                        fps : int = 30,
                                        validation_window : int = 20,
                                        min_votes : int = 2,
                                        max_expansion : int = 200,
                                        gap_threshold : int = 10,
                                        debug : bool = False) -> np.ndarray:
    """
    Refine swap boundaries to find exact swap edges.
    
    Expands detected segments frame-by-frame to capture full extent of swaps,
    not just window-aligned portions.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    segments : np.ndarray
        Nx2 array of detected swap segments (start, end frames)
    fps : int
        Frame rate
    validation_window : int
        Window size for validating each frame (default: 20)
    min_votes : int
        Minimum votes needed to include a frame (default: 2, more lenient than detection)
    max_expansion : int
        Maximum frames to expand in each direction (default: 200)
    gap_threshold : int
        Maximum gap between segments to merge (default: 10)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of refined segment boundaries
    """
    if len(segments) == 0:
        return segments
    
    filtered = filter_data(rawData)
    n_frames = len(rawData)
    refined_segments = []
    
    for seg_start, seg_end in segments:
        # Start with detected boundaries
        refined_start = seg_start
        refined_end = seg_end
        
        # Expand left (backward)
        expansion_count = 0
        while refined_start > 0 and expansion_count < max_expansion:
            # Check frame at refined_start - 1
            check_start = max(0, refined_start - validation_window)
            check_end = refined_start
            
            if check_end - check_start < validation_window // 2:
                break
            
            metrics_result = _calculate_comprehensive_metrics(
                filtered, check_start, check_end, fps=fps, validation_window=validation_window//2
            )
            
            if metrics_result is None:
                break
            
            # If enough votes indicate swap, include this frame
            if metrics_result['votes'] >= min_votes:
                refined_start -= 1
                expansion_count += 1
            else:
                break
        
        # Expand right (forward)
        expansion_count = 0
        while refined_end < n_frames - 1 and expansion_count < max_expansion:
            # Check frame at refined_end + 1
            check_start = refined_end + 1
            check_end = min(n_frames, refined_end + 1 + validation_window)
            
            if check_end - check_start < validation_window // 2:
                break
            
            metrics_result = _calculate_comprehensive_metrics(
                filtered, check_start, check_end, fps=fps, validation_window=validation_window//2
            )
            
            if metrics_result is None:
                break
            
            # If enough votes indicate swap, include this frame
            if metrics_result['votes'] >= min_votes:
                refined_end += 1
                expansion_count += 1
            else:
                break
        
        # Validate entire refined segment
        if refined_end > refined_start:
            final_validation = _calculate_comprehensive_metrics(
                filtered, refined_start, refined_end, fps=fps, validation_window=20
            )
            
            # Require 3+ votes over entire segment to keep it
            if final_validation is not None and final_validation['votes'] >= 3:
                refined_segments.append((refined_start, refined_end))
                if debug:
                    print(f'Refined segment [{seg_start}:{seg_end}] -> [{refined_start}:{refined_end}] ({refined_end - refined_start + 1} frames)')
    
    if len(refined_segments) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge segments that are close together
    refined_segments = np.array(refined_segments)
    refined_segments = refined_segments[refined_segments[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = refined_segments[0]
    
    for start, end in refined_segments[1:]:
        gap = start - current_end - 1
        if gap <= gap_threshold:  # Merge if within gap threshold
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    if debug:
        print(f'Merged {len(refined_segments)} refined segments into {len(merged_segments)} final segments')
    
    return np.array(merged_segments)


def detect_swaps_by_combined_metrics(rawData : pd.DataFrame, fps : int = 30,
                                     window_size : int = 50,
                                     angular_vel_weight : float = 1.0,
                                     speed_weight : float = 1.0,
                                     cross_sign_weight : float = 1.0,
                                     min_votes : int = 2,
                                     debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using combined metrics: angular velocity, speed ratio, and cross-sign consistency.
    
    Uses weighted voting system where multiple metrics must agree to detect a swap.
    This approach combines the strengths of different detection methods.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    window_size : int
        Size of sliding window in frames (default: 50)
    angular_vel_weight : float
        Weight for angular velocity metric (default: 1.0)
    speed_weight : float
        Weight for speed ratio metric (default: 1.0)
    cross_sign_weight : float
        Weight for cross-sign consistency metric (default: 1.0)
    min_votes : int
        Minimum number of metrics that must agree (default: 2, i.e., 2/3 must agree)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Filter data for analysis
    filtered = filter_data(rawData)
    n_frames = len(rawData)
    
    # Calculate all three metrics
    # Metric 1: Angular velocity
    head_ang_vel = get_angular_velocity(filtered, 'head', fps=fps, window_size=5)
    tail_ang_vel = get_angular_velocity(filtered, 'tail', fps=fps, window_size=5)
    
    # Metric 2: Speed ratio
    hspd = metrics.get_speed_from_df(filtered, 'head', fps=fps)
    tspd = metrics.get_speed_from_df(filtered, 'tail', fps=fps)
    
    # Metric 3: Cross-sign consistency
    cross_sign = metrics.get_ht_cross_sign(filtered)
    
    # Create valid mask (frames where all metrics are available)
    valid_mask = ~(np.isnan(head_ang_vel) | np.isnan(tail_ang_vel) | 
                   np.isnan(hspd) | np.isnan(tspd) | np.isnan(cross_sign))
    
    # Use sliding window to find regions where multiple metrics agree on swap
    swapped_windows = []
    
    step_size = window_size // 2  # 50% overlap
    for start in range(0, n_frames - window_size + 1, step_size):
        end = min(start + window_size, n_frames)
        
        # Get metrics for this window
        window_mask = valid_mask[start:end]
        if np.sum(window_mask) < window_size * 0.5:  # Need at least 50% valid data
            continue
        
        # Extract window data
        h_ang_vel_window = head_ang_vel[start:end][window_mask]
        t_ang_vel_window = tail_ang_vel[start:end][window_mask]
        hspd_window = hspd[start:end][window_mask]
        tspd_window = tspd[start:end][window_mask]
        cross_sign_window = cross_sign[start:end][window_mask]
        
        # Count votes for swap
        votes = 0
        
        # Vote 1: Angular velocity (tail > head indicates swap)
        if len(h_ang_vel_window) > 0 and len(t_ang_vel_window) > 0:
            with np.errstate(divide='ignore', invalid='ignore'):
                ang_vel_ratio = np.median(t_ang_vel_window) / np.maximum(np.median(h_ang_vel_window), 0.1)
            if ang_vel_ratio > 1.0:  # Tail angular velocity higher than head
                votes += angular_vel_weight
        
        # Vote 2: Speed ratio (tail > head indicates swap)
        if len(hspd_window) > 0 and len(tspd_window) > 0:
            speed_ratio = np.median(tspd_window) / np.maximum(np.median(hspd_window), 0.01)
            if speed_ratio > 1.2:  # Tail significantly faster
                votes += speed_weight
        
        # Vote 3: Cross-sign consistency (low consistency indicates swap)
        if len(cross_sign_window) > 0:
            positive_ratio = np.sum(cross_sign_window > 0) / len(cross_sign_window)
            negative_ratio = np.sum(cross_sign_window < 0) / len(cross_sign_window)
            consistency = max(positive_ratio, negative_ratio)
            if consistency < 0.6:  # Low consistency
                votes += cross_sign_weight
        
        # Check if enough metrics agree (weighted votes >= min_votes)
        if votes >= min_votes:
            swapped_windows.append((start, end))
            if debug:
                print(f'Combined metrics swap window [{start}:{end}]: votes={votes:.1f}/{angular_vel_weight + speed_weight + cross_sign_weight:.1f}')
    
    if len(swapped_windows) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping windows into segments
    swapped_windows = np.array(swapped_windows)
    swapped_windows = swapped_windows[swapped_windows[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_windows[0]
    
    for start, end in swapped_windows[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    if debug:
        print(f'Merged {len(swapped_windows)} windows into {len(merged_segments)} segments')
    
    return np.array(merged_segments)


def get_acceleration(data : pd.DataFrame, key : str, fps : int = 30, 
                    npoints : int = 3) -> np.ndarray:
    """
    Calculate acceleration (rate of change of speed) for a keypoint.
    
    Acceleration measures how quickly speed changes. Head typically has higher
    acceleration than tail (head moves more dynamically, tail follows).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Position data
    key : str
        Keypoint ('head' or 'tail')
    fps : int
        Frame rate
    npoints : int
        Number of points for numerical derivative (default: 3)
        
    Returns:
    --------
    np.ndarray
        Acceleration in mm/s²
    """
    # Get speed
    speed = metrics.get_speed_from_df(data, key, fps=fps, npoints=npoints)
    
    # Calculate acceleration as rate of change of speed
    # Use numerical derivative
    acceleration = np.zeros(len(speed))
    acceleration[:] = np.nan
    
    for i in range(npoints, len(speed) - npoints):
        # Calculate change in speed over time
        speed_window = speed[i-npoints:i+npoints+1]
        valid_speeds = speed_window[~np.isnan(speed_window)]
        
        if len(valid_speeds) >= npoints:
            # Use linear fit to estimate acceleration
            time_points = np.arange(len(valid_speeds)) / fps
            if len(time_points) > 1:
                coeffs = np.polyfit(time_points, valid_speeds, 1)
                acceleration[i] = coeffs[0]  # Slope = acceleration
    
    return acceleration


def detect_swaps_by_acceleration(rawData : pd.DataFrame, fps : int = 30,
                                  window_size : int = 50,
                                  min_window_size : int = 50,
                                  ratio_threshold : float = 1.0,
                                  debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using acceleration analysis.
    
    Head typically has higher acceleration than tail (head moves more dynamically).
    If tail acceleration > head acceleration, it suggests a swap.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    window_size : int
        Size of sliding window in frames (default: 50)
    min_window_size : int
        Minimum window size required for detection (default: 50)
    ratio_threshold : float
        Threshold for tail/head acceleration ratio (default: 1.0)
        If tail_accel / head_accel > threshold, likely swapped
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Filter data for acceleration calculation
    filtered = filter_data(rawData)
    
    # Calculate accelerations
    head_accel = get_acceleration(filtered, 'head', fps=fps, npoints=3)
    tail_accel = get_acceleration(filtered, 'tail', fps=fps, npoints=3)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(head_accel) | np.isnan(tail_accel))
    n_frames = len(rawData)
    
    # Use sliding window to find regions where tail acceleration > head
    swapped_windows = []
    
    # Slide window across trajectory
    step_size = window_size // 2  # 50% overlap
    for start in range(0, n_frames - min_window_size + 1, step_size):
        end = min(start + window_size, n_frames)
        
        # Get accelerations in this window
        window_mask = valid_mask[start:end]
        if np.sum(window_mask) < min_window_size * 0.5:  # Need at least 50% valid data
            continue
        
        h_accel_window = head_accel[start:end][window_mask]
        t_accel_window = tail_accel[start:end][window_mask]
        
        # Calculate ratio (tail/head acceleration)
        # Use median for robustness
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = t_accel_window / np.maximum(np.abs(h_accel_window), 0.01)  # Avoid division by zero, use abs for magnitude
        
        # Flag if tail acceleration consistently exceeds head
        if len(ratio) > 0:
            median_ratio = np.median(ratio)
            percentile_75 = np.percentile(ratio, 75)
            
            # Swap if median ratio > threshold (tail acceleration higher than head)
            # This indicates tail is accelerating more than head (suggests swap)
            if median_ratio > ratio_threshold or percentile_75 > 1.2:
                swapped_windows.append((start, end))
                if debug:
                    print(f'Acceleration swap window [{start}:{end}]: median_ratio={median_ratio:.3f}, p75={percentile_75:.3f}')
    
    if len(swapped_windows) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping windows into segments
    swapped_windows = np.array(swapped_windows)
    swapped_windows = swapped_windows[swapped_windows[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_windows[0]
    
    for start, end in swapped_windows[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    if debug:
        print(f'Merged {len(swapped_windows)} windows into {len(merged_segments)} segments')
    
    return np.array(merged_segments)


def get_path_curvature(data : pd.DataFrame, key : str, 
                       smoothing_window : int = 10) -> np.ndarray:
    """
    Calculate path curvature for a keypoint.
    
    Curvature measures how curved the path is. Head typically follows more
    curved paths than tail (head moves more, tail follows).
    
    Parameters:
    -----------
    data : pd.DataFrame
        Position data
    key : str
        Keypoint ('head' or 'tail')
    smoothing_window : int
        Window size for smoothing (default: 10)
        
    Returns:
    --------
    np.ndarray
        Curvature values (1/radius of curvature)
    """
    x, y = metrics.vectors_from_key(data, key)
    n_frames = len(data)
    
    curvature = np.zeros(n_frames)
    curvature[:] = np.nan
    
    # Calculate curvature using three-point method
    for i in range(smoothing_window, n_frames - smoothing_window):
        # Get smoothed positions
        x_window = x[i-smoothing_window:i+smoothing_window+1]
        y_window = y[i-smoothing_window:i+smoothing_window+1]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
        if np.sum(valid_mask) < 3:
            continue
        
        x_valid = x_window[valid_mask]
        y_valid = y_window[valid_mask]
        
        if len(x_valid) >= 3:
            # Use three consecutive points to calculate curvature
            # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            dx = np.diff(x_valid)
            dy = np.diff(y_valid)
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            
            if len(dx) >= 2 and len(ddx) >= 1:
                # Use middle points
                mid = len(dx) // 2
                x_prime = dx[mid] if mid < len(dx) else dx[-1]
                y_prime = dy[mid] if mid < len(dy) else dy[-1]
                x_double_prime = ddx[mid-1] if mid > 0 and mid-1 < len(ddx) else ddx[-1] if len(ddx) > 0 else 0
                y_double_prime = ddy[mid-1] if mid > 0 and mid-1 < len(ddy) else ddy[-1] if len(ddy) > 0 else 0
                
                numerator = abs(x_prime * y_double_prime - y_prime * x_double_prime)
                denominator = (x_prime**2 + y_prime**2)**1.5
                
                if denominator > 1e-6:  # Avoid division by zero
                    curvature[i] = numerator / denominator
    
    return curvature


def detect_swaps_by_curvature(rawData : pd.DataFrame, fps : int = 30,
                               window_size : int = 50,
                               curvature_threshold : float = 1.0,
                               smoothing_window : int = 10,
                               debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using curvature analysis.
    
    Head typically follows more curved paths than tail (head moves more, tail follows).
    If tail curvature > head curvature, it suggests a swap.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    window_size : int
        Size of sliding window in frames (default: 50)
    curvature_threshold : float
        Threshold for tail/head curvature ratio (default: 1.0)
        If tail_curvature / head_curvature > threshold, likely swapped
    smoothing_window : int
        Window size for curvature smoothing (default: 10)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Filter data for curvature calculation
    filtered = filter_data(rawData)
    
    # Calculate curvatures
    head_curvature = get_path_curvature(filtered, 'head', smoothing_window=smoothing_window)
    tail_curvature = get_path_curvature(filtered, 'tail', smoothing_window=smoothing_window)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(head_curvature) | np.isnan(tail_curvature))
    n_frames = len(rawData)
    
    # Use sliding window to find regions where tail curvature > head
    swapped_windows = []
    
    # Slide window across trajectory
    step_size = window_size // 2  # 50% overlap
    for start in range(0, n_frames - window_size + 1, step_size):
        end = min(start + window_size, n_frames)
        
        # Get curvatures in this window
        window_mask = valid_mask[start:end]
        if np.sum(window_mask) < window_size * 0.5:  # Need at least 50% valid data
            continue
        
        h_curv_window = head_curvature[start:end][window_mask]
        t_curv_window = tail_curvature[start:end][window_mask]
        
        # Calculate ratio (tail/head curvature)
        # Use median for robustness
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = t_curv_window / np.maximum(h_curv_window, 0.001)  # Avoid division by zero
        
        # Flag if tail curvature consistently exceeds head
        if len(ratio) > 0:
            median_ratio = np.median(ratio)
            percentile_75 = np.percentile(ratio, 75)
            
            # Swap if median ratio > threshold (tail curvature higher than head)
            # This indicates tail path is more curved than head (suggests swap)
            if median_ratio > curvature_threshold or percentile_75 > 1.2:
                swapped_windows.append((start, end))
                if debug:
                    print(f'Curvature swap window [{start}:{end}]: median_ratio={median_ratio:.3f}, p75={percentile_75:.3f}')
    
    if len(swapped_windows) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping windows into segments
    swapped_windows = np.array(swapped_windows)
    swapped_windows = swapped_windows[swapped_windows[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_windows[0]
    
    for start, end in swapped_windows[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    if debug:
        print(f'Merged {len(swapped_windows)} windows into {len(merged_segments)} segments')
    
    return np.array(merged_segments)


def detect_swaps_by_temporal_consistency(rawData : pd.DataFrame, fps : int = 30,
                                        baseline_window : int = 200,
                                        change_threshold : float = 0.3,
                                        validation_window : int = 50,
                                        debug : bool = False) -> np.ndarray:
    """
    Detect swap segments using temporal consistency patterns.
    
    Analyzes long-term patterns: if head angular velocity consistently > tail
    over long periods, sudden reversals indicate swaps. Uses change-point
    detection to identify when patterns reverse.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    baseline_window : int
        Window size for establishing baseline pattern (default: 200 frames)
    change_threshold : float
        Threshold for detecting pattern change (default: 0.3)
        If pattern changes by >threshold, likely a swap
    validation_window : int
        Window size for validating detected changes (default: 50)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Filter data for analysis
    filtered = filter_data(rawData)
    n_frames = len(rawData)
    
    # Calculate angular velocities
    head_ang_vel = get_angular_velocity(filtered, 'head', fps=fps, window_size=5)
    tail_ang_vel = get_angular_velocity(filtered, 'tail', fps=fps, window_size=5)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(head_ang_vel) | np.isnan(tail_ang_vel))
    
    # Calculate ratio over time
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = tail_ang_vel / np.maximum(head_ang_vel, 0.1)
    
    # Establish baseline pattern: head angular velocity should be > tail
    # So baseline ratio should be < 1.0
    baseline_ratio = []
    for i in range(0, min(baseline_window, n_frames), validation_window):
        end = min(i + validation_window, n_frames)
        window_mask = valid_mask[i:end]
        if np.sum(window_mask) > validation_window * 0.5:
            window_ratio = ratio[i:end][window_mask]
            if len(window_ratio) > 0:
                baseline_ratio.append(np.median(window_ratio))
    
    if len(baseline_ratio) == 0:
        return np.empty((0, 2), dtype=int)
    
    baseline_median = np.median(baseline_ratio)
    
    # Detect change points where pattern reverses
    swapped_segments = []
    
    # Slide window across trajectory
    step_size = validation_window // 2
    for start in range(baseline_window, n_frames - validation_window + 1, step_size):
        end = min(start + validation_window, n_frames)
        
        # Get ratio in this window
        window_mask = valid_mask[start:end]
        if np.sum(window_mask) < validation_window * 0.5:
            continue
        
        window_ratio = ratio[start:end][window_mask]
        if len(window_ratio) == 0:
            continue
        
        median_ratio = np.median(window_ratio)
        
        # Detect change: if ratio increases significantly from baseline, likely swap
        ratio_change = median_ratio - baseline_median
        
        if ratio_change > change_threshold:
            # Pattern reversed: tail angular velocity now > head (suggests swap)
            # Validate with additional metrics
            hspd = metrics.get_speed_from_df(filtered.iloc[start:end], 'head', fps=fps)
            tspd = metrics.get_speed_from_df(filtered.iloc[start:end], 'tail', fps=fps)
            
            valid_hspd = hspd[~np.isnan(hspd)]
            valid_tspd = tspd[~np.isnan(tspd)]
            
            # Additional validation: speed ratio should also indicate swap
            if len(valid_hspd) > 0 and len(valid_tspd) > 0:
                speed_ratio = np.median(valid_tspd) / np.maximum(np.median(valid_hspd), 0.01)
                if speed_ratio > 1.2:  # Tail faster confirms swap
                    swapped_segments.append((start, end))
                    if debug:
                        print(f'Temporal consistency swap [{start}:{end}]: ratio_change={ratio_change:.3f}, speed_ratio={speed_ratio:.3f}')
    
    if len(swapped_segments) == 0:
        return np.empty((0, 2), dtype=int)
    
    # Merge overlapping segments
    swapped_segments = np.array(swapped_segments)
    swapped_segments = swapped_segments[swapped_segments[:, 0].argsort()]
    
    merged_segments = []
    current_start, current_end = swapped_segments[0]
    
    for start, end in swapped_segments[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_segments.append((current_start, current_end))
    
    # Ensure segments don't exceed DataFrame bounds
    merged_segments = [(max(0, start), min(end, n_frames - 1)) for start, end in merged_segments]
    
    if debug:
        print(f'Merged {len(swapped_segments)} segments into {len(merged_segments)} segments')
    
    return np.array(merged_segments)


def correct_tracking_errors(rawData : pd.DataFrame, fps : int = 30, debug : bool = False,
                           comprehensive_params : dict = None) -> pd.DataFrame:
    """
    Remove tracking errors and correct head-tail swaps
    TODO: address errors where centroid overlaps head / tail

    rawData: dataFrame with imported piVR data
    fps: frame rate (default: 30)
    debug: print debug messages
    comprehensive_params: dict with parameters for comprehensive metrics approach
        If None, only baseline methods are used.
        Parameters: min_votes, window_size, min_segment_size, min_segment_duration,
                   angular_vel_ratio, angular_var_ratio, distance_ratio_threshold,
                   speed_ratio_threshold, alignment_angle_threshold
    """
    data = rawData.copy()

    # BASELINE: Flag frames where swaps appear to occur (frame-by-frame detection)
    swaps = flag_all_swaps(data,separate=False,debug=debug)

    # BASELINE: correct remaining head-tail swaps in segments
    segments = utils.indices_to_segments(swaps,nframes=data.shape[0],addBounds=True,inclusive=True,alternating=True)
    data = correct_swapped_segments(data,segments,debug=debug)
    
    # BASELINE: Apply original simple global swap detection
    data = correct_global_swap_simple(data, debug=debug)
    
    # OPTIONAL: Apply comprehensive metrics detection for remaining errors
    if comprehensive_params is not None:
        # Only look for new segments in regions not already corrected by baseline
        corrected_frames_mask = np.zeros(len(data), dtype=bool)
        for start, end in segments:
            corrected_frames_mask[start:end+1] = True
        
        # Find uncorrected regions
        uncorrected_segments = utils.indices_to_segments(
            ~corrected_frames_mask, nframes=len(data), addBounds=False, 
            inclusive=True, alternating=True
        )
        
        new_segments_found = []
        for start, end in uncorrected_segments:
            # Apply comprehensive detection to uncorrected segments
            comprehensive_swaps = detect_swaps_by_comprehensive_metrics(
                data.iloc[start:end+1], fps=fps, debug=debug, **comprehensive_params
            )
            # Adjust segment indices back to original data frame
            if len(comprehensive_swaps) > 0:
                adjusted_swaps = [(s + start, e + start) for s, e in comprehensive_swaps]
                new_segments_found.extend(adjusted_swaps)

        if len(new_segments_found) > 0:
            if debug:
                print(f'Correcting {len(new_segments_found)} new segments detected by comprehensive metrics')
            # Refine boundaries for these new segments
            refined_new_segments = []
            for seg_start, seg_end in new_segments_found:
                refined = refine_swap_boundaries_comprehensive(
                    data, np.array([[seg_start, seg_end]]), fps=fps, debug=debug
                )
                if len(refined) > 0:
                    refined_new_segments.append((refined[0][0], refined[0][1]))
            
            if len(refined_new_segments) > 0:
                # Merge overlapping segments before correction
                merged_new_segments = _merge_nearby_segments(refined_new_segments, gap=20)
                data = correct_swapped_segments(data, merged_new_segments, debug=debug)
    
    return data


def _merge_nearby_segments(segments : list[tuple[int, int]], gap : int = 20) -> list[tuple[int, int]]:
    """
    Merge segments that are within gap frames of each other.
    
    Parameters:
    -----------
    segments : list[tuple[int, int]]
        List of (start, end) segment tuples
    gap : int
        Maximum gap between segments to merge (default: 20 frames)
        
    Returns:
    --------
    list[tuple[int, int]]
        Merged segments
    """
    if len(segments) == 0:
        return []
    
    # Sort by start frame
    segments = sorted(segments, key=lambda seg: seg[0])
    
    merged = []
    current_start, current_end = segments[0]
    
    for start, end in segments[1:]:
        if start <= current_end + gap:  # Overlapping or within gap
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged.append((current_start, current_end))
    return merged


def expand_and_merge_flagged_frames(flagged_frames : np.ndarray, nframes : int,
                                     expand_window : int = 10, merge_gap : int = 50,
                                     debug : bool = False) -> np.ndarray:
    """
    Expand flagged frames to form proper segments.
    
    Problem: Frame-by-frame detection finds sparse frames (8-20 frames) but misses
    contiguous segments (1-3000+ frames). This function:
    1. Expands around each flagged frame (N-10 to N+10)
    2. Merges nearby flagged regions (within 50 frames)
    3. Uses temporal consistency: if 3+ frames in 10-frame window, flag entire window
    
    Parameters:
    -----------
    flagged_frames : np.ndarray
        Array of frame indices where swaps were detected
    nframes : int
        Total number of frames in trajectory
    expand_window : int
        Number of frames to expand around each flagged frame (default: 10)
    merge_gap : int
        Maximum gap between flagged regions to merge (default: 50)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Expanded array of flagged frame indices
    """
    if len(flagged_frames) == 0:
        return flagged_frames
    
    # Step 1: Expand around flagged frames, but only if there's strong temporal consistency
    # Very conservative: only expand if 3+ frames are nearby within a larger window
    # This reduces false positives while still catching contiguous segments
    expanded = set(flagged_frames)  # Start with original flagged frames
    
    # For each flagged frame, check if there are other flagged frames nearby
    # Only expand if we have strong evidence of a cluster (at least 3 frames within expand_window*3)
    for i, frame in enumerate(flagged_frames):
        # Count nearby flagged frames
        nearby_count = np.sum(np.abs(flagged_frames - frame) <= expand_window * 3)
        
        # Only expand if we have a strong cluster (at least 3 frames nearby)
        # Use smaller expansion window (5 instead of 10) to be more conservative
        if nearby_count >= 3:
            start = max(0, frame - 5)  # Smaller window
            end = min(nframes - 1, frame + 5)
            expanded.update(range(start, end + 1))
    
    expanded = np.array(sorted(expanded))
    
    if debug:
        print(f'After conservative expansion: {len(flagged_frames)} → {len(expanded)} frames')
    
    # Step 2: Find gaps and merge nearby regions
    if len(expanded) == 0:
        return expanded
    
    # Find consecutive ranges
    ranges = utils.get_consecutive_ranges(expanded)
    
    # Merge ranges that are close together
    merged_ranges = []
    if len(ranges) > 0:
        current_start, current_end = ranges[0]
        
        for start, end in ranges[1:]:
            gap = start - current_end - 1
            if gap <= merge_gap:
                # Merge: extend current range
                current_end = end
            else:
                # Gap too large: start new range
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_ranges.append((current_start, current_end))
    
    # Convert merged ranges back to frame indices
    merged_frames = []
    for start, end in merged_ranges:
        merged_frames.extend(range(start, end + 1))
    
    merged_frames = np.array(merged_frames)
    
    if debug:
        print(f'After merging: {len(ranges)} ranges → {len(merged_ranges)} segments')
        print(f'Final: {len(merged_frames)} frames')
    
    return merged_frames


def validate_corrected_data(rawData : pd.DataFrame, fps : int = 30, debug : bool = False) -> pd.DataFrame:
    '''
    Check for remaining head-tail swaps in segments between overlaps
    NOTE: this can only be done after the initial round of swap correction, as head-tail swaps within
    inter-overlap segments will throw off the segment-based swap detection
    '''
    missed = get_swapped_segments(rawData,fps,debug=debug)
    #missed = flag_swaps_after_curl(data,fps,debug=debug)
    data = correct_swapped_segments(rawData,missed,debug=debug)
    return data


def remove_overlaps(rawData : pd.DataFrame, fps : int, spdThresh : float = 20,
                    debug : bool = False) -> pd.DataFrame:
    """
    Set frames where head / tail overlap to NaN and optionally interpolate
    Attempts to determine which point is incorrectly-placed based on discontinuities in position
    Ex: in an overlap preceded or followed by a discontinuity in the head position (but not the tail position),
    only the head position data will be removed
    This behaviour can be suppressed by setting "spdThresh" to zero 
    
    rawData: DataFrame with position data
    fps: frame rate
    spdThresh: speed threshold (mm/s) for detecting head / tail discontinuities (0 -> remove all overlap data)
    interp: apply interpolation
    method: interpolation method (see pandas.DataFrame.interpolate)
    maxSegment: maximum number of consecutive frames to interpolate over
    debug: print debug messages
    """
    data = rawData.copy()

    # get overlaps & head / tail discontinuities
    edges = get_overlap_edges(data,debug=debug)
    dh = flag_discontinuities(data,'head',fps=fps,threshold=spdThresh,debug=False)
    dt = flag_discontinuities(data,'tail',fps=fps,threshold=spdThresh,debug=False)

    # find frames where start of overlap associated with head / tail discontinuity
    headErr = [np.arange(a,b+1) for a, b in edges if (a in dh) or (b+1 in dh)] # creates a nested list of frames
    tailErr = [np.arange(a,b+1) for a, b in edges if (a in dt) or (b+1 in dt)]
    headErrMerged = utils.flatten(headErr) # convert to 1D array
    tailErrMerged = utils.flatten(tailErr)

    # set overlap-discontinuity frames to NaN
    data.loc[headErrMerged,['xhead','yhead']] = np.nan # remove head errors
    data.loc[tailErrMerged,['xtail','ytail']] = np.nan # remove tail errors

    if debug:
        # redundant, but avoids additional computations if debug = False
        headSegs = [tuple(seg) for seg in edges if (seg[0] in dh) or (seg[1]+1 in dh)]
        tailSegs = [tuple(seg) for seg in edges if (seg[0] in dt) or (seg[1]+1 in dt)]
        print('Head Segments Removed: ({}) {}'.format(len(headSegs),headSegs))
        print('Tail Segments Removed: ({}) {}'.format(len(tailSegs),tailSegs))

    return data

def interpolate_gaps(rawData : pd.DataFrame, method : str = 'cubicspline', maxSegment : int = 15,
                    debug : bool = False) -> pd.DataFrame:
    '''
    Interpolate over short segments of NaN values in the position data

    rawData: dataframe with position data containing NaN values to interpolate over
    method: interpolation method
    maxSegment: maximum number of consecutive frames to interpolate over; larger gaps will be ignored
    debug: print debug messages
    '''
    data = rawData.copy()
    cols = ['xhead','yhead','xtail','ytail','xctr','yctr','xmid','ymid']

    # create mask indicating which values are either not NaNs or NaNs within short segments
    mask = data[cols].notnull() # boolean DataFrame indicating non-null values
    for col in cols:
        # locate gaps
        gaps = utils.get_value_segments(data[col],np.nan,inclusive=True)
        if gaps.size == 0 : continue

        # filter for gaps of appropriate length
        query = np.diff(gaps,axis=1)[0] <= maxSegment
        gaps = gaps[query,:]
        gapFrames = utils.ranges_to_list(gaps)
        mask.loc[gapFrames,col] = True

        # get a sub-Series of the interpolated vector based on the mask
        data[col] = data[col].interpolate(method=method)[mask[col]]

        if debug : print('Interpolation ({}): ({}) {}'.format(col,len(gaps),gaps))
    return data


# ----- Error Detection -----

def flag_all_swaps(rawData : pd.DataFrame, separate : bool = False,
            debug : bool = False) -> np.ndarray | tuple[np.ndarray]:
    """
    Flag all frames where swaps are detected

    rawData: DataFrame with raw position data
    fps: frame rate
    separate: return a tuple of separate vectors with flags of different types
    - sign reversal
    - minimum delta mismatch
    - overlap sign reversal
    - overlap minimum-delta mismatch
    - DEFAULT: return single vector of unique flags
    debug: print debug messages
    """
    # get flags
    olaps = flag_overlaps(rawData,debug=debug)
    #olap = get_overlap_edges(rawData,debug=debug)
    #sr = flag_sign_reversals(rawData,debug=debug)
    #dm = flag_delta_mismatches(rawData,debug=debug)
    mdm = flag_min_delta_mismatches(rawData,debug=debug)
    cosr = flag_overlap_sign_reversals(rawData,debug=debug)
    #com = flag_overlap_mismatches(rawData,debug=debug)
    comm = flag_overlap_minimum_mismatches(rawData,debug=debug)

    # filter out overlaps
    #filt = utils.merge(olap[:,0]-1,olap[:,0],olap[:,1],olap[:,1]+1)
    filt = utils.merge(olaps,olaps+1)
    mdm = utils.filter_array(mdm,filt)
    #sr = utils.filter_array(sr,filt)

    # finish
    flags = (mdm, cosr, comm)
    merged = utils.merge(*flags)

    if debug : print('All Flags: ({}) {}'.format(len(merged),merged))
    if separate : return flags
    else : return merged


def flag_discontinuities(data : pd.DataFrame, key : str, fps : int,
            threshold : float = 24, debug : bool = False) -> np.ndarray:
    """
    Flag frames where head / tail / midpoint move outside expected radius
    TODO: use a probability distribution and probability thresholds

    data: dataframe containing raw position data
    key: key in POSDICT indicating which point to check
    fps: frame rate
    threshold: minimum speed required to flag (mm/s)
    debug: print debug messages
    """
    delta = metrics.get_delta_between_frames(data,key,fps=fps) # behaves like np.diff()
    flag = np.where(delta > threshold)[0] + 1 # flag second frame of each pair used in diff()

    if debug : print('Discontinuities ({}): {}'.format(key,flag))
    return flag


def flag_delta_mismatches(data : pd.DataFrame, tolerance : float = 0.0, debug : bool = False) -> np.ndarray:
    """
    Flag frames where head and tail move shorter distance between frames if switched

    data: dataframe containing raw position data
    tolerance: minimum percent difference between distances required to flag
    debug: print debug messages
    """
    delta = get_all_deltas(data)
    dtt, dhh, dth, dht = delta

    query = dtt + dhh > (dht + dth) * (tolerance + 1)
    #query = dtt > dth * (tolerance + 1) and dhh > dht * (tolerance + 1) # this works less reliably?
    #query = dtt > dth * (tolerance + 1) # more stable, but assumes tail is correctly-labeled initially
    flag = np.where(query)[0] + 1

    if debug : print('Delta Mismatches: {}'.format(flag))
    return flag


def flag_min_delta_mismatches(data : pd.DataFrame, debug : bool = False) -> np.ndarray:
    """
    Flag frames where minimum distance between two frames is from head to tail or vice-versa
    TODO: add tolerance?

    data: dataframe containing raw position data
    debug: print debug messages
    """
    # get deltas between frames
    delta = get_all_deltas(data) # tt, hh, th, ht

    # find minimum deltas and check if index matches th or ht
    # NOTE: argmin <= 1 means tt or hh is minimum distance
    minidx = np.argmin(delta,axis=0) # index of minimum delta for each frame pair
    flag = np.where(minidx > 1)[0] + 1 # add one to revert index chage from diff()

    if debug : print('Minimum-Delta Mismatches: {}'.format(flag))
    return flag


def flag_sign_reversals(data : pd.DataFrame, threshold : float = np.pi/2, debug : bool = False) -> np.ndarray:
    """
    Flag frames where cross-product of tail-midpt and midpt-head vectors switches sign

    data: dataframe containing raw position data
    threshold: minimum internal angle required to flag (prevents excessive flagging when animal is straight)
    debug: print debug messages
    """
    # identify where z-component changes sign
    z = metrics.get_ht_cross_sign(data) # z-component of body vector cross-product
    dz = np.zeros_like(z)
    dz[1:] = np.abs(np.diff(z)) # flag frames with sign flip

    # get frames where current and preceding frames are above threshold
    ang = metrics.get_head_angle(data,halfAngle=True) # internal angle
    dang = np.ones_like(ang)
    dang[ang < threshold] = 0 # filter out frames with sub-threshold angle
    dang[1:] *= dang[:len(dang)-1] # filter out frames preceded by sub-threshold angle

    #flag = np.where(dz * ang > threshold)[0]
    flag = np.where(dz * dang > 0)[0]

    if debug : print('Sign Reversals: {}'.format(flag))
    return flag


def flag_overlaps(data : pd.DataFrame, tolerance : float = OVERLAP_THRESH,
                  pt1 : str = 'head', pt2: str = 'tail',
                  debug : bool = False) -> np.ndarray:
    '''
    Flag all frames where head, tail overlap

    data: dataFrame with raw position data
    tolerance: maximum allowed distance between "overlapping" points
    pt1: first point of interest
    pt2: second point of interest
    debug: print degbug messages
    '''
    if tolerance > 0:
        delta = metrics.get_delta_in_frame(data,pt1,pt2)
        overlaps = np.where(delta < tolerance)[0]
    else: # use faster method to find perfect overlaps
        overlaps = metrics.perfectly_overlapping(data,pt1,pt2,where=True)

    if debug:
        print('Overlaps ({}-{}): ({}) {}'.format(pt1[0],pt2[0],len(overlaps),overlaps))
        if tolerance > 0 : print('Overlap Deltas: {}'.format(delta[overlaps]))
    return overlaps


def flag_overlap_mismatches(data : pd.DataFrame, dtol : float = 0, otol : float = OVERLAP_THRESH,
                             debug : bool = False) -> np.ndarray:
    '''
    Flag frames with delta mismatch across a section where overlapping occurs
    NOTE: flags frame following overlap segment

    data: dataFrame with raw position data
    dtol: minimum percent difference between distances required to flag
    otol: maximum allowed distance between "overlapping" points (for edges = None)
    debug: print degbug messages
    '''
    edges = get_overlap_edges(data,tolerance=otol,offset=1,debug=False)
    if edges.size == 0 : return np.empty(0) # catch no-overlap condition
    deltas = get_all_deltas(data,edges)
    dtt, dhh, dth, dht = deltas

    query = dtt + dhh > (dht + dth) * (dtol + 1)
    #query = dtt > dth * (dtol + 1) # this assumes the tail is correctly-labeled on the first frame
    flag = np.array([fr[1] for i, fr in enumerate(edges) if query[i]])

    if debug : print('Cross-Overlap Mismatches: {}'.format(flag))
    return flag


def flag_overlap_minimum_mismatches(data : pd.DataFrame, otol : float = OVERLAP_THRESH,
                                    debug : bool = False) -> np.ndarray:
    '''
    Flag frames where minimum delta across overlap is between two differently-labeled points
    We assume that the tail moves the least distance; therefore, if the minimum distance is from
    head to tail or tail to head, there was likely a swap
    NOTE: flags frame following overlap segment

    data: dataFrame with raw position data
    dtol: minimum percent difference between distances required to flag
    otol: maximum allowed distance between "overlapping" points (for edges = None)
    debug: print degbug messages
    '''
    # get all deltas across frames on either end of overlap region
    edges = get_overlap_edges(data,tolerance=otol,offset=1,debug=False)
    if edges.size == 0 : return np.empty(0) # catch no-overlap condition
    delta = get_all_deltas(data,edges)

    # find minimum deltas and check if index matches th or ht
    minidx = np.argmin(delta,axis=0) # index of minimum delta for each frame pair
    flag = np.array([fr[1] for i, fr in enumerate(edges) if minidx[i] > 1])

    if debug:
        print('Cross-Overlap Minimum-Delta Mismatch: {}'.format(flag))

        # check if overlap is within swapped section
        swap = [fr for i, fr in enumerate(edges) if minidx[i] == 1]
        print('Overlaps within swaps: {}'.format(swap))

    return flag


def flag_overlap_sign_reversals(data : pd.DataFrame, tolerance : float = OVERLAP_THRESH,
                                threshold : float = np.pi/4,
                                debug : bool = False) -> np.ndarray:
    '''
    Flag frames with cross-product sign reversal across a section where overlapping occurs
    NOTE: flags frame following overlap segment
    Issue: larva can sometimes bend in opposite direction following overlap (semi-uncommon, but problematic)

    data: dataFrame with raw position data
    tolerance: maximum allowed distance between "overlapping" points (for edges = None)
    threshold: minimum internal angle required to flag (avoids false positives when animal is straight)
    debug: print degbug messages
    '''
    # get frames on either end of overlap region
    edges = get_overlap_edges(data,tolerance=tolerance,offset=1,debug=False)
    if edges.size == 0 : return np.empty(0) # catch no-overlap condition
    a, b = edges.T # get vectors of start, end frames

    # determie if cross-product signs mismatch across overlap
    z = metrics.get_ht_cross_sign(data) # z-component of body vector cross-product
    delta = np.abs(z[b] - z[a]) # nonzero (2 or 1) when sign changes

    # check for above-threshold angles on either side of overlap
    # NOTE: could also try checking for matching angles within certain tolerance, but
    # that would be riskier and would still require checking that the organism isn't straight
    ang = metrics.get_head_angle(data,halfAngle=True) # internal angle
    vang = np.array([0 if a < threshold else 1 for a in ang]) # flag valid angles
    valid = vang[b] * vang[a] # nonzero (1) if both angles valid
    
    # flag frames at end of overlaps where conditions met
    idx = np.where(valid * delta > 0)[0]
    flag = b[idx]

    if debug : print('Cross-Overlap Sign Reversals: {}'.format(flag))
    return flag


# ----- Collapsed Keypoints ------

def get_all_collapsed_frames(data : pd.DataFrame, tolerance : float = 0.1,
                              debug : bool = False) -> np.ndarray:
    """
    Get all frames with collapsed keypoints or missing data.
    
    Includes:
    1. Frames where keypoints physically collapse (head/centroid, tail/centroid, or 3+ keypoints)
    2. Frames where head or tail are NaN/empty (same tracking error source)
    3. First and last frame of trial (boundary conditions)
    
    These frames serve as "anchor points" that break the trajectory into reliable segments.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Position data with columns xhead, yhead, xtail, ytail, xmid, ymid, xctr, yctr
    tolerance : float
        Maximum distance (mm) between keypoints to consider them collapsed (default: 0.1mm)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates collapsed/missing frames
    """
    collapsed = detect_collapsed_keypoints(data, tolerance=tolerance, debug=False)
    
    # Add frames where head or tail are NaN/empty
    head_missing = data[['xhead', 'yhead']].isna().any(axis=1)
    tail_missing = data[['xtail', 'ytail']].isna().any(axis=1)
    missing = head_missing | tail_missing
    
    # Add first and last frame
    first_last = np.zeros(len(data), dtype=bool)
    if len(data) > 0:
        first_last[0] = True
        first_last[-1] = True
    
    # Combine all
    all_collapsed = collapsed | missing | first_last
    
    if debug:
        n_collapsed = np.sum(collapsed)
        n_missing = np.sum(missing)
        n_first_last = np.sum(first_last)
        n_total = np.sum(all_collapsed)
        print(f'All collapsed frames: {n_total} total')
        print(f'  Physical collapse: {n_collapsed} frames')
        print(f'  Missing data (NaN): {n_missing} frames')
        print(f'  First/last frame: {n_first_last} frames')
        if n_total > 0:
            collapsed_frames = np.where(all_collapsed)[0]
            collapsed_ranges = utils.get_consecutive_ranges(collapsed_frames)
            print(f'Collapsed regions: {len(collapsed_ranges)} segments')
            if len(collapsed_ranges) <= 10:
                print(f'  Segments: {collapsed_ranges}')
    
    return all_collapsed


def detect_collapsed_keypoints(data : pd.DataFrame, tolerance : float = 0.1,
                                debug : bool = False) -> np.ndarray:
    """
    Detect frames where keypoints collapse, indicating tracking errors.
    
    This occurs when tracking fails and cannot resolve different keypoints. Multiple scenarios
    are detected:
    1. Head/centroid collapse: head and centroid are very close (tracking can't resolve head)
    2. Tail/centroid collapse: tail and centroid are very close (tracking can't resolve tail)
    3. Three or more keypoints collapse: at least 3 of 4 keypoints are within tolerance
    
    When this persists for multiple frames, frame-by-frame detection methods fail because
    they rely on keypoint relationships.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Position data with columns xhead, yhead, xtail, ytail, xmid, ymid, xctr, yctr
    tolerance : float
        Maximum distance (mm) between keypoints to consider them collapsed (default: 0.1mm)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Boolean array where True indicates collapsed keypoints in that frame
    """
    # Get all keypoint positions
    head_pos = data[['xhead', 'yhead']].values
    tail_pos = data[['xtail', 'ytail']].values
    mid_pos = data[['xmid', 'ymid']].values
    ctr_pos = data[['xctr', 'yctr']].values
    
    # Calculate pairwise distances
    ht_dist = np.sqrt(np.sum((head_pos - tail_pos)**2, axis=1))
    hm_dist = np.sqrt(np.sum((head_pos - mid_pos)**2, axis=1))
    hc_dist = np.sqrt(np.sum((head_pos - ctr_pos)**2, axis=1))
    tm_dist = np.sqrt(np.sum((tail_pos - mid_pos)**2, axis=1))
    tc_dist = np.sqrt(np.sum((tail_pos - ctr_pos)**2, axis=1))
    mc_dist = np.sqrt(np.sum((mid_pos - ctr_pos)**2, axis=1))
    
    # Handle NaN values - set to large value so they don't match
    ht_dist = np.where(np.isnan(ht_dist), np.inf, ht_dist)
    hm_dist = np.where(np.isnan(hm_dist), np.inf, hm_dist)
    hc_dist = np.where(np.isnan(hc_dist), np.inf, hc_dist)
    tm_dist = np.where(np.isnan(tm_dist), np.inf, tm_dist)
    tc_dist = np.where(np.isnan(tc_dist), np.inf, tc_dist)
    mc_dist = np.where(np.isnan(mc_dist), np.inf, mc_dist)
    
    # Scenario 1: Head/centroid collapse (head tracking error)
    head_centroid_collapse = hc_dist < tolerance
    
    # Scenario 2: Tail/centroid collapse (tail tracking error)
    tail_centroid_collapse = tc_dist < tolerance
    
    # Scenario 3: Three or more keypoints collapse
    # Count how many pairs are within tolerance (out of 6 possible pairs)
    close_pairs = ((ht_dist < tolerance).astype(int) +
                   (hm_dist < tolerance).astype(int) +
                   (hc_dist < tolerance).astype(int) +
                   (tm_dist < tolerance).astype(int) +
                   (tc_dist < tolerance).astype(int) +
                   (mc_dist < tolerance).astype(int))
    
    # If 3+ pairs are close, at least 3 keypoints are collapsed
    # (3 pairs means 3 keypoints form a triangle, 4+ pairs means more collapse)
    three_or_more_collapse = close_pairs >= 3
    
    # Combine all scenarios: collapse if any scenario is true
    collapsed = head_centroid_collapse | tail_centroid_collapse | three_or_more_collapse
    
    if debug:
        n_collapsed = np.sum(collapsed)
        n_head_ctr = np.sum(head_centroid_collapse)
        n_tail_ctr = np.sum(tail_centroid_collapse)
        n_three_plus = np.sum(three_or_more_collapse)
        print(f'Collapsed keypoints: {n_collapsed} frames')
        print(f'  Head/centroid collapse: {n_head_ctr} frames')
        print(f'  Tail/centroid collapse: {n_tail_ctr} frames')
        print(f'  3+ keypoints collapse: {n_three_plus} frames')
        if n_collapsed > 0:
            collapsed_frames = np.where(collapsed)[0]
            collapsed_ranges = utils.get_consecutive_ranges(collapsed_frames)
            print(f'Collapsed regions: {len(collapsed_ranges)} segments')
            if len(collapsed_ranges) <= 10:
                print(f'  Segments: {collapsed_ranges}')
    
    return collapsed


# ----- Overlaps and Deltas ------

def get_overlap_edges(data : pd.DataFrame, offset : int = 0,
                      tolerance : float = OVERLAP_THRESH, pt1 : str = 'head', pt2 : str = 'tail',
                       debug : bool = False) -> np.ndarray:
    '''
    Get frames on either side of each overlap region
    Returns an array of start and end frames of shape (N x 2) for N overlaps

    data: DataFrame with raw position data
    offset: number of frames away from overlap
    (0 -> mark start / end frames of overlap; 1 -> mark end / start frames of non-overlap segments)
    tolerance: minimum distance between head / tail req. to be an overlap
    pt1: first point of interest
    pt2: second point of interest
    debug: print debug messages
    '''
    overlaps = flag_overlaps(data,tolerance,pt1,pt2)
    oranges = utils.get_consecutive_ranges(overlaps)
    edges = [(max(rng[0]-offset,0),min(rng[1]+offset,data.shape[0]-1)) for rng in oranges]
    if debug : print('Overlaps ({}-{}): ({}) {}'.format(pt1[0],pt2[0],len(edges),edges))
    return np.array(edges)


def get_all_overlap_edges(data : pd.DataFrame, offset : int = 0,
                      tolerance : float = OVERLAP_THRESH, debug : bool = False
                      ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    '''
    Get frames on either side of overlap regions of each type:
    - head-tail
    - head-midpt
    - tail-midpt
    - all three
    '''
    ht = flag_overlaps(data,tolerance,'head','tail')
    hm = flag_overlaps(data,tolerance,'head','mid')
    tm = flag_overlaps(data,tolerance,'tail','mid')

    htm = utils.match_arrays(ht,hm)
    htm = utils.match_arrays(htm,tm)

    htfilt = utils.filter_array(ht,hm)
    hmfilt = utils.filter_array(hm,ht)
    tmfilt = utils.filter_array(tm,ht)
    
    out = []
    flags = [htfilt,hmfilt,tmfilt,htm]
    labels = ['h-t','h-m','t-m','h-t-m']
    for flag, lab in zip(flags,labels):
        oranges = utils.get_consecutive_ranges(flag)
        edges = [(max(rng[0]-offset,0),min(rng[1]+offset,data.shape[0]-1)) for rng in oranges]
        out.append(np.array(edges))
        if debug : print('Overlaps ({}): ({}) {}'.format(lab,len(oranges),oranges))
    
    return tuple(out)


def get_all_deltas(data : pd.DataFrame, edges : np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    '''
    Get tt, hh, th, and ht deltas between frames on either side of each overlap as a 4xn array
    '''
    points = zip(['tail','head','tail','head'],['tail','head','head','tail']) # tt, hh, th, ht
    if edges is None : delta = np.array([metrics.get_delta_between_frames(data,a,b) for a, b in points])
    else : delta = np.array([metrics.get_cross_segment_deltas(data,edges,a,b) for a, b in points])
    return delta


# ----- Validation -----

def detect_swaps_between_collapsed_regions(rawData : pd.DataFrame, fps : int = 30,
                                           mode : str = 'alignment',
                                           minTime : float = 0.5,
                                           thresh : tuple[float,float] = (0.95,1.05),
                                           debug : bool = False) -> np.ndarray:
    """
    Detect head-tail swaps in segments between collapsed keypoint regions.
    
    Uses collapsed keypoints (including NaN/empty and first/last frame) as anchor points
    to identify reliable segments. For each segment, checks if motion direction indicates
    a swap (backwards motion).
    
    This addresses the observation that most persistent swaps in level1.csv are long segments
    that occur between tracking failures (collapsed keypoints).
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    mode : str
        Detection mode: 'alignment', 'speed', or 'distance'
        - 'alignment': Check alignment of tail-midpoint vector with motion vector (forward motion assumption)
        - 'speed': Check if tail speed > head speed (indicates swap)
        - 'distance': Check ratio of head/tail travel distances
    minTime : float
        Minimum segment duration (seconds) to analyze
    thresh : tuple[float,float]
        Thresholds for detection (lower bound for swap, upper bound for non-swap)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments
    """
    # Get all collapsed frames (including NaN, first/last)
    all_collapsed = get_all_collapsed_frames(rawData, tolerance=0.1, debug=debug)
    
    # Find collapsed regions (consecutive ranges)
    collapsed_frames = np.where(all_collapsed)[0]
    if len(collapsed_frames) == 0:
        # No collapsed frames - entire trajectory is one segment
        collapsed_ranges = np.array([[0, len(rawData)-1]])
    else:
        collapsed_ranges = utils.get_consecutive_ranges(collapsed_frames)
        collapsed_ranges = np.array(collapsed_ranges)
    
    # Get segments between collapsed regions
    # IMPORTANT: Only analyze segments that are BETWEEN collapsed regions in the middle
    # of the trajectory. Don't analyze segments at the very start or end (only first/last
    # frame collapsed), as these are unreliable and can cause false positives.
    segments = []
    n_frames = len(rawData)
    
    if len(collapsed_ranges) == 0:
        # No collapsed regions - skip detection (entire trajectory might be unreliable)
        if debug:
            print('No collapsed regions found - skipping segment-based detection')
        return np.empty((0, 2), dtype=int)
    
    # Check if collapsed regions are only at the edges (first and/or last frame)
    # If so, skip detection to avoid false positives from analyzing the entire trajectory
    n_frames = len(rawData)
    only_edges = True
    for collapsed_range in collapsed_ranges:
        start, end = collapsed_range
        # Check if this collapsed region is NOT at the very start (frame 0) or very end (last frame)
        if start > 0 and end < n_frames - 1:
            only_edges = False
            break
    
    if only_edges:
        # All collapsed regions are at edges - skip detection
        if debug:
            print('Collapsed regions only at edges - skipping segment-based detection to avoid false positives')
        return np.empty((0, 2), dtype=int)
    
    # Multiple collapsed regions with at least one in the middle - analyze segments BETWEEN them
    for i in range(len(collapsed_ranges) - 1):
        start = collapsed_ranges[i][1] + 1
        end = collapsed_ranges[i+1][0] - 1
        if start <= end:  # Valid segment
            segments.append((start, end))
    
    if len(segments) == 0:
        if debug:
            print('No valid segments between collapsed regions')
        return np.empty((0, 2), dtype=int)
    
    segments = np.array(segments)
    
    if debug:
        print(f'Segments between collapsed regions: {len(segments)}')
        print(f'  Segment ranges: {segments[:10]}')  # Show first 10
    
    # Filter data and prepare for analysis
    filt = filter_data(rawData)
    minFrames = int(minTime * fps)
    
    # Filter out segments that are too short
    seg_lengths = segments[:, 1] - segments[:, 0] + 1
    valid_segs = segments[seg_lengths >= minFrames]
    
    if len(valid_segs) == 0:
        if debug:
            print(f'No segments long enough (min {minFrames} frames)')
        return np.empty((0, 2), dtype=int)
    
    # Detect swaps in each segment using the specified mode
    match(mode):
        case 'alignment':
            med_angles, std_angles, mean_angles = _get_alignment_angles(filt, valid_segs)
            nvals = utils.segment_lengths(valid_segs)
            
            # Convert to degrees for threshold checking
            med_angles_deg = np.rad2deg(med_angles)
            std_angles_deg = np.rad2deg(std_angles)
            
            if debug:
                print(f'Alignment angles (median ± std, degrees):')
                for i, (med, std) in enumerate(zip(med_angles_deg, std_angles_deg)):
                    print(f'  Segment {i}: {med:.1f}° ± {std:.1f}°')
            
            # Focus on 90-180° angles (strong indicator of swap)
            # Use median as primary indicator, with std as margin of error
            # If median > 90°, it's likely a swap (90-180° range is strong indicator)
            # Use std to handle uncertainty: if std is very large relative to median, be cautious
            # If median < 70°, then it's clearly not swapped
            # Between 70-90° is ambiguous (could be turns or partial swaps)
            
            med_angles_norm = med_angles / (np.pi/4)  # normalize by 45° (90° = 2.0)
            std_angles_norm = std_angles / (np.pi/4)
            
            flag = np.zeros(len(valid_segs), dtype=int)
            for i in range(len(valid_segs)):
                if nvals[i] >= minFrames:
                    med_norm = med_angles_norm[i]
                    std_norm = std_angles_norm[i]
                    
                    # Clear swap: median > 90° (normalized > 2.0)
                    # If std is very large (> median), the segment might be too noisy
                    # But if median is clearly > 90°, it's still likely a swap
                    # Use a more lenient check: median > 90° AND std < 3.0 (135°) OR median > 100°
                    if med_norm > 2.0:  # median > 90°
                        # If std is reasonable (< 3.0 normalized = 135°) OR median is very high (> 100°)
                        if std_norm < 3.0 or med_norm > 2.22:  # std < 135° or median > 100°
                            flag[i] = 1  # Swapped
                        # If std is very large but median is only slightly > 90°, be cautious (ambiguous)
                        else:
                            flag[i] = 0  # Ambiguous
                    # Clear not swapped: median < 70° (normalized < 1.56)
                    elif med_norm < 1.56:  # 70° normalized
                        flag[i] = -1  # Not swapped
                    # Otherwise ambiguous (0) - between 70-90°, could be turns
        case 'speed':
            frac, nvals = _get_speed_ratios(filt, valid_segs)
            if debug:
                print(f'Speed ratios: {frac}')
                print(f'Frames per segment: {nvals}')
            flag = _flag_segment_metrics(frac, nvals, thresh, minFrames)
        case 'distance':
            frac = _get_travel_distance_ratios(filt, valid_segs)
            nvals = utils.segment_lengths(valid_segs)
            if debug:
                print(f'Distance ratios: {frac}')
            flag = _flag_segment_metrics(frac, nvals, thresh, minFrames)
        case _:
            raise ValueError(f"Unknown mode: {mode}. Must be 'alignment', 'speed', or 'distance'")
    
    # Extract swapped segments (flag == 1 indicates swap)
    swapped_segments = valid_segs[flag == 1]
    
    if debug:
        print(f'Swapped segments detected: {len(swapped_segments)}')
        if len(swapped_segments) > 0:
            print(f'  Segments: {swapped_segments}')
    
    return swapped_segments


def refine_swap_boundaries_in_segment(rawData : pd.DataFrame, segment_start : int, segment_end : int,
                                     fps : int = 30, window_size : int = 50,
                                     debug : bool = False) -> tuple[int, int] | None:
    """
    Refine swap boundaries within a segment to find the actual start and end of a swap.
    
    Takes a segment between collapsed regions and uses multiple metrics to find where
    the swap actually starts and ends, rather than assuming the entire segment is swapped.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    segment_start : int
        Start frame of the segment to analyze
    segment_end : int
        End frame of the segment to analyze
    fps : int
        Frame rate
    window_size : int
        Size of sliding window for analysis (default: 50 frames)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    tuple[int, int] | None
        Refined start and end frames of the swap, or None if no swap detected
    """
    if segment_end <= segment_start:
        return None
    
    segment_length = segment_end - segment_start + 1
    # No minimum size restriction - can refine segments of any size
    
    # Filter data for analysis
    filtered = filter_data(rawData)
    
    # Calculate metrics for the segment
    segment_data = filtered.iloc[segment_start:segment_end+1]
    
    # Metric 1: Alignment angles (backwards motion indicates swap)
    # Calculate alignment angle for each frame in segment
    tail_pos = segment_data[['xtail', 'ytail']].values
    mid_pos = segment_data[['xmid', 'ymid']].values
    
    # Body orientation vector (tail to midpoint)
    body_vec = mid_pos - tail_pos
    
    # Motion vector (tail displacement)
    tail_motion = np.diff(tail_pos, axis=0, prepend=tail_pos[0:1] - tail_pos[0:1])
    
    # Calculate angles
    alignment_angles = []
    for i in range(len(body_vec)):
        if i == 0:
            alignment_angles.append(np.nan)
            continue
        bv = body_vec[i]
        tm = tail_motion[i]
        # Normalize vectors
        bv_norm = np.linalg.norm(bv)
        tm_norm = np.linalg.norm(tm)
        if bv_norm > 0.01 and tm_norm > 0.01:  # Avoid division by zero
            cos_angle = np.dot(bv, tm) / (bv_norm * tm_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            alignment_angles.append(angle)
        else:
            alignment_angles.append(np.nan)
    
    alignment_angles = np.array(alignment_angles)
    
    # Metric 2: Cross-sign consistency
    cross_sign = metrics.get_ht_cross_sign(segment_data)
    
    # Metric 3: Speed ratios
    hspd = metrics.get_speed_from_df(segment_data, 'head', fps=fps)
    tspd = metrics.get_speed_from_df(segment_data, 'tail', fps=fps)
    
    # Use sliding window to find swap boundaries
    swapped_frames = np.zeros(segment_length, dtype=bool)
    
    step_size = window_size // 2  # 50% overlap
    for start_offset in range(0, segment_length - window_size + 1, step_size):
        end_offset = min(start_offset + window_size, segment_length)
        window_start = segment_start + start_offset
        window_end = segment_start + end_offset - 1
        
        # Get metrics for this window
        window_angles = alignment_angles[start_offset:end_offset]
        window_signs = cross_sign[start_offset:end_offset]
        window_hspd = hspd[start_offset:end_offset]
        window_tspd = tspd[start_offset:end_offset]
        
        # Remove NaN values
        valid_angles = window_angles[~np.isnan(window_angles)]
        valid_signs = window_signs[~np.isnan(window_signs)]
        valid_hspd = window_hspd[~np.isnan(window_hspd)]
        valid_tspd = window_tspd[~np.isnan(window_tspd)]
        
        if len(valid_angles) == 0 and len(valid_signs) == 0:
            continue
        
        # Vote-based detection: swap if multiple indicators agree
        votes = 0
        
        # Alignment angle: large angles (90-180°) indicate backwards motion (swap)
        if len(valid_angles) > 0:
            median_angle = np.median(valid_angles)
            if median_angle > 90:  # Backwards motion
                votes += 1
        
        # Cross-sign: low consistency indicates swap
        if len(valid_signs) > 0:
            positive_ratio = np.sum(valid_signs > 0) / len(valid_signs)
            negative_ratio = np.sum(valid_signs < 0) / len(valid_signs)
            consistency = max(positive_ratio, negative_ratio)
            if consistency < 0.6:  # Low consistency
                votes += 1
        
        # Speed ratio: tail > head indicates swap
        if len(valid_hspd) > 0 and len(valid_tspd) > 0:
            median_ratio = np.median(valid_tspd) / np.maximum(np.median(valid_hspd), 0.01)
            if median_ratio > 1.2:  # Tail significantly faster
                votes += 1
        
        # If 2+ votes, mark window as swapped
        if votes >= 2:
            swapped_frames[start_offset:end_offset] = True
    
    # Find contiguous swapped regions
    swapped_indices = np.where(swapped_frames)[0]
    if len(swapped_indices) == 0:
        return None
    
    # Get consecutive ranges
    swapped_ranges = utils.get_consecutive_ranges(swapped_indices)
    
    # Return the largest swapped region (or merge if close together)
    if len(swapped_ranges) == 0:
        return None
    
    # Merge nearby regions (within 50 frames)
    merged_ranges = []
    current_start, current_end = swapped_ranges[0]
    for start, end in swapped_ranges[1:]:
        gap = start - current_end - 1
        if gap <= 50:  # Merge if close
            current_end = end
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))
    
    # Return the largest merged region
    largest_range = max(merged_ranges, key=lambda r: r[1] - r[0])
    refined_start = segment_start + largest_range[0]
    refined_end = segment_start + largest_range[1]
    
    if debug:
        print(f'  Refined segment [{segment_start}:{segment_end}] ({segment_length} frames)')
        print(f'    -> Swap region [{refined_start}:{refined_end}] ({refined_end - refined_start + 1} frames)')
    
    return (refined_start, refined_end)


def validate_swap_segment(rawData : pd.DataFrame, segment_start : int, segment_end : int,
                         fps : int = 30, is_collapsed_region_segment : bool = False,
                         debug : bool = False) -> bool:
    """
    Validate a swap segment using multi-metric consensus.
    
    Uses three metrics to determine if a segment is truly swapped:
    1. Alignment angle: median > 90° (backwards motion)
    2. Cross-sign consistency: < 0.6 (low consistency)
    3. Speed ratio: tail/head > 1.2 (tail faster)
    
    Requires 2+ metrics to agree for validation (conservative approach).
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    segment_start : int
        Start frame of the segment
    segment_end : int
        End frame of the segment
    fps : int
        Frame rate
    debug : bool
        Print debug messages
        
    Returns:
    --------
    bool
        True if segment is validated as swapped (2+ metrics agree)
    """
    if segment_end <= segment_start:
        return False
    
    # Filter data for analysis
    filtered = filter_data(rawData)
    segment_data = filtered.iloc[segment_start:segment_end+1]
    
    # Metric 1: Alignment angles
    tail_pos = segment_data[['xtail', 'ytail']].values
    mid_pos = segment_data[['xmid', 'ymid']].values
    body_vec = mid_pos - tail_pos
    tail_motion = np.diff(tail_pos, axis=0, prepend=tail_pos[0:1] - tail_pos[0:1])
    
    alignment_angles = []
    for i in range(1, len(body_vec)):
        bv = body_vec[i]
        tm = tail_motion[i]
        bv_norm = np.linalg.norm(bv)
        tm_norm = np.linalg.norm(tm)
        if bv_norm > 0.01 and tm_norm > 0.01:
            cos_angle = np.dot(bv, tm) / (bv_norm * tm_norm)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            alignment_angles.append(angle)
    
    alignment_angles = np.array(alignment_angles)
    
    # Metric 2: Cross-sign consistency
    cross_sign = metrics.get_ht_cross_sign(segment_data)
    
    # Metric 3: Speed ratios
    hspd = metrics.get_speed_from_df(segment_data, 'head', fps=fps)
    tspd = metrics.get_speed_from_df(segment_data, 'tail', fps=fps)
    
    # Count votes
    votes = 0
    
    # Alignment angle check
    # For collapsed region segments, require stronger evidence (> 120°)
    if len(alignment_angles) > 0:
        median_angle = np.median(alignment_angles[~np.isnan(alignment_angles)])
        angle_threshold = 120 if is_collapsed_region_segment else 90
        if not np.isnan(median_angle) and median_angle > angle_threshold:
            votes += 1
            if debug:
                print(f'  Alignment angle: {median_angle:.1f}° (backwards motion, threshold={angle_threshold}°) ✓')
    
    # Cross-sign consistency check
    valid_signs = cross_sign[~np.isnan(cross_sign)]
    if len(valid_signs) > 0:
        positive_ratio = np.sum(valid_signs > 0) / len(valid_signs)
        negative_ratio = np.sum(valid_signs < 0) / len(valid_signs)
        consistency = max(positive_ratio, negative_ratio)
        if consistency < 0.6:
            votes += 1
            if debug:
                print(f'  Cross-sign consistency: {consistency:.3f} (low) ✓')
    
    # Speed ratio check
    # For collapsed region segments, require stronger evidence (> 1.5)
    valid_hspd = hspd[~np.isnan(hspd)]
    valid_tspd = tspd[~np.isnan(tspd)]
    if len(valid_hspd) > 0 and len(valid_tspd) > 0:
        median_ratio = np.median(valid_tspd) / np.maximum(np.median(valid_hspd), 0.01)
        ratio_threshold = 1.5 if is_collapsed_region_segment else 1.2
        if median_ratio > ratio_threshold:
            votes += 1
            if debug:
                print(f'  Speed ratio: {median_ratio:.3f} (tail faster, threshold={ratio_threshold}) ✓')
    
    # Require 2+ votes for validation
    # For segments between collapsed regions, use stricter thresholds but allow 2/3 votes
    # For small segments, require all 3 metrics
    segment_length = segment_end - segment_start + 1
    
    if is_collapsed_region_segment and segment_length >= 500:
        # Large segments between collapsed regions: require 2+ votes but with stronger evidence
        # Check if the evidence is strong enough (e.g., alignment > 100° or speed ratio > 1.3)
        has_strong_evidence = False
        if len(alignment_angles) > 0:
            median_angle = np.median(alignment_angles[~np.isnan(alignment_angles)])
            if not np.isnan(median_angle) and median_angle > 100:
                has_strong_evidence = True
        if not has_strong_evidence and len(valid_hspd) > 0 and len(valid_tspd) > 0:
            median_ratio = np.median(valid_tspd) / np.maximum(np.median(valid_hspd), 0.01)
            if median_ratio > 1.3:
                has_strong_evidence = True
        
        # Require 2+ votes AND strong evidence for large collapsed region segments
        is_valid = votes >= 2 and has_strong_evidence
        if debug:
            print(f'  Large collapsed region segment: {votes}/3 votes, strong_evidence={has_strong_evidence}')
    elif segment_length < 100:
        # Small segments: require all 3 metrics to agree (very conservative)
        is_valid = votes >= 3
    else:
        # Larger segments from sparse frames: require 2+ votes (standard)
        is_valid = votes >= 2
    
    if debug:
        result_str = "VALID" if is_valid else "REJECTED"
        print(f'  Validation: {votes}/3 votes, result: {result_str}')
    
    return is_valid


def expand_segments_with_refinement(rawData : pd.DataFrame, sparse_frames : np.ndarray,
                                    fps : int = 30, expand_window : int = 10,
                                    debug : bool = False) -> list[tuple[int, int]]:
    """
    Expand sparse frames into segments and refine their boundaries.
    
    Takes sparse frames from frame-by-frame detection, forms conservative
    initial segments, and applies boundary refinement to each.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    sparse_frames : np.ndarray
        Sparse frame indices from flag_all_swaps()
    fps : int
        Frame rate
    expand_window : int
        Number of frames to expand around each sparse frame (default: 10)
    debug : bool
        Print debug messages
        
    Returns:
    --------
    list[tuple[int, int]]
        List of refined segment boundaries (start, end)
    """
    if len(sparse_frames) == 0:
        return []
    
    n_frames = len(rawData)
    
    # Form initial segments from sparse frames with conservative expansion
    expanded = set()
    for frame in sparse_frames:
        start = max(0, frame - expand_window)
        end = min(n_frames - 1, frame + expand_window)
        expanded.update(range(start, end + 1))
    
    expanded = np.array(sorted(expanded))
    
    # Get consecutive ranges
    ranges = utils.get_consecutive_ranges(expanded)
    
    # Refine boundaries for each range
    refined_segments = []
    for start, end in ranges:
        refined = refine_swap_boundaries_in_segment(
            rawData, start, end, fps=fps, window_size=50, debug=debug
        )
        if refined is not None:
            refined_segments.append(refined)
    
    if debug:
        print(f'Expanded {len(sparse_frames)} sparse frames into {len(ranges)} segments')
        print(f'Refined to {len(refined_segments)} swap segments')
    
    return refined_segments


def detect_swaps_with_refined_boundaries(rawData : pd.DataFrame, fps : int = 30,
                                         mode : str = 'alignment',
                                         minTime : float = 0.5,
                                         thresh : tuple[float,float] = (0.95,1.05),
                                         debug : bool = False) -> np.ndarray:
    """
    Detect swaps between collapsed regions with refined boundaries.
    
    First identifies segments between collapsed regions that appear swapped,
    then refines the boundaries to find the actual swap region within each segment.
    
    Parameters:
    -----------
    rawData : pd.DataFrame
        Raw position data
    fps : int
        Frame rate
    mode : str
        Detection mode for initial segment detection
    minTime : float
        Minimum segment duration (seconds)
    thresh : tuple[float,float]
        Thresholds for detection
    debug : bool
        Print debug messages
        
    Returns:
    --------
    np.ndarray
        Nx2 array of start and end frames of swapped segments (with refined boundaries)
    """
    # First, get segments between collapsed regions that appear swapped
    candidate_segments = detect_swaps_between_collapsed_regions(
        rawData, fps=fps, mode=mode, minTime=minTime, thresh=thresh, debug=debug
    )
    
    if len(candidate_segments) == 0:
        return np.empty((0, 2), dtype=int)
    
    if debug:
        print(f'Found {len(candidate_segments)} candidate segments, refining boundaries...')
    
    # Refine boundaries for each candidate segment
    refined_segments = []
    for seg in candidate_segments:
        start, end = seg
        refined = refine_swap_boundaries_in_segment(
            rawData, start, end, fps=fps, window_size=50, debug=debug
        )
        if refined is not None:
            refined_segments.append(refined)
    
    if len(refined_segments) == 0:
        return np.empty((0, 2), dtype=int)
    
    if debug:
        print(f'Refined to {len(refined_segments)} swap segments')
    
    return np.array(refined_segments)


def correct_global_swap_simple(rawData : pd.DataFrame, debug : bool = False) -> pd.DataFrame:
    """
    Original simple global swap detection (baseline).
    Detect and correct global head-tail swap caused by misidentification on first frame.
    
    Simple check: if mean tail speed > mean head speed, swap entire trajectory.
    """
    # filter data
    data = rawData.copy()
    filtered = filter_data(rawData)

    # get speeds
    hspd = metrics.get_speed_from_df(filtered,'head')
    tspd = metrics.get_speed_from_df(filtered,'tail')

    # swap position data if tail speed varies more than head
    if np.nanmean(tspd) > np.nanmean(hspd):
        if debug : print('Correcting global head-tail reversal')
        data[['xhead','yhead','xtail','ytail']] = data[['xtail','ytail','xhead','yhead']]
    
    return data


def correct_global_swap(rawData : pd.DataFrame, debug : bool = False) -> pd.DataFrame:
    """
    Detect and correct global head-tail swap caused by misidentification on first frame.
    
    Uses multiple heuristics with consensus:
    1. Cross-sign consistency check (primary): If cross-sign match rate < 0.7, likely global swap
    2. Speed check (secondary): If mean tail speed > mean head speed, likely swap
    3. Head-leading motion check (tertiary): For forward motion, head should generally lead
    
    Swaps if 2 out of 3 heuristics indicate swap.
    
    IMPORTANT: Excludes collapsed keypoint frames from calculations, as these frames
    have unreliable keypoint relationships and would corrupt the detection.
    """
    # Use raw data for speed calculations (filtering can reduce speeds significantly)
    # Filter only for cross-sign calculation (needs clean data)
    data = rawData.copy()
    filtered = filter_data(rawData)

    # Detect collapsed keypoints and exclude them from calculations
    # Also exclude head-tail overlaps, as these have unreliable keypoint relationships
    collapsed = detect_collapsed_keypoints(rawData, tolerance=0.1, debug=debug)
    overlaps = flag_overlaps(rawData, tolerance=0.0, debug=False)  # Perfect overlaps
    overlap_mask = np.zeros(len(rawData), dtype=bool)
    if len(overlaps) > 0:
        overlap_mask[overlaps] = True
    
    # Exclude both collapsed keypoints and head-tail overlaps
    valid_mask = ~(collapsed | overlap_mask)
    
    if debug:
        n_valid = np.sum(valid_mask)
        n_collapsed = np.sum(collapsed)
        print(f'Global swap detection: using {n_valid} valid frames, excluding {n_collapsed} collapsed frames')

    # get speeds from raw data (more reliable for global swap detection)
    # Only use non-collapsed frames
    hspd = metrics.get_speed_from_df(rawData,'head')
    tspd = metrics.get_speed_from_df(rawData,'tail')
    
    # Filter out collapsed frames from speed calculations
    if np.sum(valid_mask) > 0:
        hspd_valid = hspd[valid_mask]
        tspd_valid = tspd[valid_mask]
    else:
        # If all frames are collapsed, can't detect - return unchanged
        if debug:
            print('Warning: All frames have collapsed keypoints, cannot detect global swap')
        return data
    
    # Heuristic 1: Cross-sign consistency check (PRIMARY)
    # Calculate cross-sign for the trajectory
    # Only use non-collapsed frames
    cross_sign = metrics.get_ht_cross_sign(filtered)
    # Remove NaN values and collapsed frames
    valid_signs = cross_sign[valid_mask & ~np.isnan(cross_sign)]
    
    if len(valid_signs) > 0:
        positive_ratio = np.sum(valid_signs > 0) / len(valid_signs)
        negative_ratio = np.sum(valid_signs < 0) / len(valid_signs)
        max_ratio = max(positive_ratio, negative_ratio)
        
        # Check start and end of trajectory as special cases
        # If swap persists from start or to end, we need to detect it
        n_valid = len(valid_signs)
        # Check both small window (for strong initial signal) and larger window (for persistence)
        start_window_small = min(100, n_valid // 10)  # First 100 frames or 10% - detects strong initial signal
        start_window_large = min(200, n_valid // 5)   # First 200 frames or 20% - checks persistence
        end_window_small = min(100, n_valid // 10)    # Last 100 frames
        end_window_large = min(200, n_valid // 5)     # Last 200 frames
        # Also check a middle portion to see if pattern persists
        mid_start = n_valid // 3
        mid_end = 2 * n_valid // 3
        mid_window = min(200, (mid_end - mid_start) // 2)
        
        # Check cross-sign at start (small and large windows), middle, and end
        start_signs_small = valid_signs[:start_window_small]
        start_signs_large = valid_signs[:start_window_large]
        end_signs_small = valid_signs[-end_window_small:]
        end_signs_large = valid_signs[-end_window_large:]
        mid_signs = valid_signs[mid_start:mid_start+mid_window] if mid_start+mid_window <= n_valid else np.array([])
        
        start_negative_small = np.sum(start_signs_small < 0) / len(start_signs_small) if len(start_signs_small) > 0 else 0
        start_negative_large = np.sum(start_signs_large < 0) / len(start_signs_large) if len(start_signs_large) > 0 else 0
        end_negative_small = np.sum(end_signs_small < 0) / len(end_signs_small) if len(end_signs_small) > 0 else 0
        end_negative_large = np.sum(end_signs_large < 0) / len(end_signs_large) if len(end_signs_large) > 0 else 0
        mid_negative_ratio = np.sum(mid_signs < 0) / len(mid_signs) if len(mid_signs) > 0 else 0
        
        # Use small window for detection (more sensitive to initial pattern)
        start_negative_ratio = start_negative_small
        end_negative_ratio = end_negative_small
        start_positive_ratio = 1 - start_negative_small
        end_positive_ratio = 1 - end_negative_small
        
        # But also check if pattern persists in larger window
        start_persists = start_negative_large > 0.6  # Pattern persists in larger window
        end_persists = end_negative_large > 0.6
        
        # CRITICAL: Check if start/end pattern suggests swap by comparing to expected pattern
        # If cross-sign is very consistent at start (>80% one sign), check if that's "wrong"
        # We can't compare to level2 directly, but we can check if the pattern is suspicious
        # A suspicious pattern: very consistent negative at start but mixed overall suggests swap
        start_consistently_negative = start_negative_ratio > 0.7
        end_consistently_negative = end_negative_ratio > 0.7
        start_consistently_positive = start_positive_ratio > 0.7
        end_consistently_positive = end_positive_ratio > 0.7
        
        # If start is very consistent (>80%) but overall is mixed (<70%), that's suspicious
        # This suggests the start has a different pattern than the rest, possibly indicating swap
        start_pattern_differs = (start_negative_ratio > 0.8 or start_positive_ratio > 0.8) and (max_ratio < 0.7)
        end_pattern_differs = (end_negative_ratio > 0.8 or end_positive_ratio > 0.8) and (max_ratio < 0.7)
        
        # Cross-sign should be consistent (mostly one sign) for normal trajectories
        # If cross-sign is very consistent (>70% one sign), check if it contradicts speed pattern
        # For forward motion with head leading, we'd expect positive cross-sign
        # If speeds suggest head should lead (head > tail) but cross-sign is mostly negative,
        # that's a contradiction indicating a swap
        mean_hspd = np.nanmean(hspd_valid)
        mean_tspd = np.nanmean(tspd_valid)
        speed_diff_ratio = abs(mean_hspd - mean_tspd) / max(mean_hspd, mean_tspd) if max(mean_hspd, mean_tspd) > 0 else 1.0
        
        # Check for swap at start or end (persistent swaps)
        # Require that the pattern persists: if start shows swap, it should persist in larger window
        # This prevents false positives from temporary patterns at the start
        start_shows_swap = start_consistently_negative and (mean_hspd > mean_tspd)
        end_shows_swap = end_consistently_negative and (mean_hspd > mean_tspd)
        mid_confirms_start = len(mid_signs) > 0 and (mid_negative_ratio > 0.6)  # Middle also negative
        
        # Swap at start if: 
        # - Start shows strong swap signal (>70% negative) AND
        # - Pattern persists in larger window (>60% negative) OR middle confirms OR pattern differs
        # This ensures it's not just a temporary pattern
        swap_at_start = start_shows_swap and (start_persists or mid_confirms_start or start_pattern_differs)
        # Swap at end if: end shows swap AND (pattern persists OR pattern differs)
        swap_at_end = end_shows_swap and (end_persists or end_pattern_differs)
        
        # If speeds are similar (within 20%), cross-sign consistency becomes more important
        if speed_diff_ratio < 0.2:
            # When speeds are similar, we can't rely on speed alone
            # If cross-sign is very consistent (>75% one sign), check if it's negative
            # Negative cross-sign with similar speeds might indicate swap
            # Lower threshold when speeds are ambiguous
            # OR if start/end shows consistent negative pattern
            cross_sign_indicates_swap = ((max_ratio > 0.75) and (negative_ratio > 0.6)) or \
                                        swap_at_start or swap_at_end
        else:
            # When speeds differ significantly, cross-sign should align with speed pattern
            # If head > tail (no swap expected) but cross-sign is mostly negative, that's contradictory
            # OR if start/end shows consistent negative pattern
            cross_sign_indicates_swap = ((mean_hspd > mean_tspd) and (negative_ratio > 0.7)) or \
                                        swap_at_start or swap_at_end
    else:
        cross_sign_indicates_swap = False
    
    # Heuristic 2: Speed check (SECONDARY)
    # Use only non-collapsed frames
    speed_indicates_swap = mean_tspd > mean_hspd
    
    # Heuristic 3: Head-leading motion check (TERTIARY)
    # For forward motion, head should generally lead (have higher speed)
    # Calculate ratio of frames where head speed > tail speed
    # Only use non-collapsed frames
    valid_speed_mask = valid_mask & ~(np.isnan(hspd) | np.isnan(tspd))
    if np.sum(valid_speed_mask) > 0:
        head_leading_ratio = np.sum(hspd[valid_speed_mask] > tspd[valid_speed_mask]) / np.sum(valid_speed_mask)
        # If head leads in <30% of frames, likely indicates swap
        head_leading_indicates_swap = head_leading_ratio < 0.3
    else:
        head_leading_indicates_swap = False
    
    # Consensus: swap if 2 out of 3 heuristics indicate swap
    # BUT: if start or end shows strong swap pattern (>80% negative), that's a very strong signal
    # In that case, we only need 1 additional vote (or the start/end signal alone if very strong)
    swap_votes = sum([cross_sign_indicates_swap, speed_indicates_swap, head_leading_indicates_swap])
    
    # SAFETY CHECK: Before swapping, verify that swapping would improve cross-sign consistency
    # If cross-sign is already consistent (mostly one sign), swapping might make it worse
    if len(valid_signs) > 0:
        # Check overall consistency
        overall_consistency = max(positive_ratio, negative_ratio)
        
        # If cross-sign is already very consistent (>85% one sign), be cautious
        # Only swap if we have very strong evidence (3 votes, or >90% at start/end)
        # Use swap_at_start and swap_at_end (which already include persistence checks) instead of raw ratios
        # This is more reliable because it checks if the pattern actually suggests a swap
        very_strong_signal = swap_at_start or swap_at_end
        
        if overall_consistency > 0.85:
            # Very consistent already - need stronger evidence to swap
            # Only swap if we have clear evidence from start/end AND additional confirmation
            if very_strong_signal:
                # Strong signal at start/end - need at least 1 additional vote
                should_swap = swap_votes >= 1
            else:
                # Need all 3 votes to swap when already consistent and no start/end signal
                should_swap = swap_votes >= 3
        else:
            # Cross-sign is mixed - use normal logic
            if very_strong_signal:
                # Strong signal at start/end - this is reliable, need 1 additional vote
                should_swap = swap_votes >= 1
            else:
                # Normal case - need 2 out of 3 votes
                should_swap = swap_votes >= 2
    else:
        should_swap = False
    
    if debug:
        print('Global swap detection:')
        if len(valid_signs) > 0:
            print(f'  Cross-sign: positive={positive_ratio:.3f}, negative={negative_ratio:.3f}')
            if 'start_negative_ratio' in locals():
                print(f'  Start window: negative={start_negative_ratio:.3f}, swap_at_start={swap_at_start}')
                print(f'  End window: negative={end_negative_ratio:.3f}, swap_at_end={swap_at_end}')
                if 'start_persists' in locals():
                    print(f'  Start persists in large window: {start_persists}, End persists: {end_persists}')
            print(f'  Cross-sign indicates swap: {cross_sign_indicates_swap}')
        print(f'  Speed: head={mean_hspd:.3f}, tail={mean_tspd:.3f}, indicates_swap={speed_indicates_swap}')
        if np.sum(valid_speed_mask) > 0:
            print(f'  Head-leading: ratio={head_leading_ratio:.3f}, indicates_swap={head_leading_indicates_swap}')
        print(f'  Votes: {swap_votes}/3, should_swap: {should_swap}')
    
    if should_swap:
        if debug:
            print('Correcting global head-tail reversal')
        data[['xhead','yhead','xtail','ytail']] = data[['xtail','ytail','xhead','yhead']]
    
    return data


def get_swapped_segments(rawData : pd.DataFrame, fps : int,
            mode : str = 'alignment',
            swapFollowing : bool = True, mergeAmbiguous : bool = False, ignoreEdges : bool = False,
            minOverlap : int = 1, minTime : float = 1,
            thresh : tuple[float,float] = (0.95,1.05),
            debug : bool = False) -> np.ndarray:
    '''
    Detect when head, tail swapped on trajectory segments between overlaps
    based on a given metric (set by "mode")
    Returns an Nx2 array of start and end frames of swapped segments

    data: corrected raw position data
    fps: frame rate

    mode: key of mode to use {'distance','speed','alignment'}
    - distance: check ratio of distance head traveled to distance tail traveled
    - speed: check ratio of head speed to tail speed
    - alignment: assume forward motion and check alignment of tail-midpt and tail motion vectors
    ratioThresh: maximum ratio of head metric to tail metric clearly indicative of a swap
    and minimum ratio indicative of a non-swap

    swapFollowing: swap all ambiguous segments following a non-ambiguous swapped segment
    up to the next non-ambiguous non-swap segment (NOTE: overrides mergeAmbiguous)
    mergeAmbiguous: mark ambiguous segments as swapped if sandwiched by swapped segments
    ignoreEdges: do not handle ambiguous edges (start for swapFollowing or start & end for mergeAmbiguous)

    minOverlap: minimum duration of overlaps to examine (in frames) -- can be used to focus only on
    longer overlaps where the initial swap detection fails most frequently
    minTime: minimum number of seconds between overlaps required to calculate metrics
    (NOTE: shorter segments will be treated as "ambiguous")

    thresh: minimum metric value to label a clear non-swap and maximum to label a clear swap
    NOTE: angular values are normalized by dividing by PI/4

    debug: print debug messages
    '''
    minFrames = int(minTime * fps)

    # filter data
    filt = remove_overlaps(rawData,fps,spdThresh=0)
    filt = filter_data(filt)

    # get overlaps of given minimum length
    olaps = get_overlap_edges(rawData,offset=0)
    if olaps.size == 0 : return np.empty(0) # catch no overlaps
    idx = olaps[:,1] - olaps[:,0] >= minOverlap - 1 # overlaps of sufficient length
    olaps = olaps[idx,:]

    # get inter-overlap segments
    segs = utils.invert_ranges(olaps,filt.shape[0],False)
    if debug:
        print(f'Overlaps: {olaps}')
        print(f'Non-Overlaps: {segs}')

    # use assumption of forward motion
    match(mode):
        case 'alignment':
            med_angles, std_angles, mean_angles = _get_alignment_angles(filt,segs)
            nvals = utils.segment_lengths(segs)
            if debug : 
                print('Alignment angles (median ± std, degrees):')
                for i, (med, std) in enumerate(zip(np.rad2deg(med_angles), np.rad2deg(std_angles))):
                    print(f'  Segment {i}: {med:.1f}° ± {std:.1f}°')
            # Use median for detection, normalize by π/4
            med_angles_norm = med_angles / (np.pi/4)
            std_angles_norm = std_angles / (np.pi/4)
            
            # For alignment angles: large angles (>90°) indicate swap
            # _flag_segment_metrics flags small values as swapped, large as not swapped
            # So we need to invert the logic or use inverted thresholds
            # Instead, let's use median directly and check if it's > 90°
            flag = np.zeros(len(segs), dtype=int)
            for i in range(len(segs)):
                if nvals[i] >= minFrames:
                    med_norm = med_angles_norm[i]
                    std_norm = std_angles_norm[i]
                    # Clear swap: median > 90° (normalized > 2.0)
                    # If std is reasonable (< 3.0 normalized = 135°) OR median is very high (> 100°)
                    if med_norm > 2.0:  # median > 90°
                        if std_norm < 3.0 or med_norm > 2.22:  # std < 135° or median > 100°
                            flag[i] = 1  # Swapped
                        else:
                            flag[i] = 0  # Ambiguous
                    # Clear not swapped: median < 70° (normalized < 1.56)
                    elif med_norm < 1.56:  # 70° normalized
                        flag[i] = -1  # Not swapped
                    # Otherwise ambiguous (0) - between 70-90°, could be turns
        case 'speed':
            # get speed ratios
            frac, nvals = _get_speed_ratios(filt,segs)
            if debug:
                print('Head-Tail Speed Ratio: {}'.format(frac))
                print('Frames per Segment: {}'.format(nvals))
            flag = _flag_segment_metrics(frac,nvals,thresh,minFrames)
        case 'distance':
            frac = _get_travel_distance_ratios(filt,segs)
            nvals = utils.segment_lengths(segs)
            if debug:
                print('Head-Tail Distance Ratio: {}'.format(frac))
                print('Frames per Segment: {}'.format(nvals))
            flag = _flag_segment_metrics(frac,nvals,thresh,minFrames)

    # handle ambiguous cases
    # TODO: try merging ambiguous segments until reach necessary length and operate on those??
    if swapFollowing : ambigFlag = _swap_following_ambiguous_flags(flag,ignoreEdges)
    elif mergeAmbiguous : ambigFlag = _merge_ambiguous_flags(flag,ignoreEdges)
    else : ambigFlag = np.zeros_like(flag)
    allFlags = flag + ambigFlag # "flag" should be zero where "ambigFlag" is nonzero
    swaps = segs[allFlags > 0,:]
    
    # finish
    if debug:
        print('Segment Flags: {}'.format(flag))
        if swapFollowing or mergeAmbiguous : print('Ambiguous Seg. Flags: {}'.format(ambigFlag))
        print('Swapped Segments: {}'.format(swaps))
    return swaps


def _swap_following_ambiguous_flags(flag : np.ndarray, ignoreEdges : bool = False) -> np.ndarray:
    '''
    assume all ambiguous segments following swapped segment (up to next non-swapped segment) are swapped

    flag: flags indicating clear and ambiguous segments
    ignoreEdges: do not try to assign a nonzero flag to first segment
    '''
    ambig = utils.get_value_segments(flag,0,inclusive=False)
    ambigFlag = np.zeros_like(flag)
    for a, b in ambig:
        # get flags before segment
        if a == 0:
            if b == flag.shape[0] or ignoreEdges : before = 0 # ignore ambiguous start segment
            else : before = flag[b] # assume start segment matches first non-ambiguous segment
        else:
            before = flag[a-1] # ambiguous segment matches preceding segment

        # set states of ambiguous segments
        ambigFlag[a:b] = before
    return ambigFlag


def _merge_ambiguous_flags(flag : np.ndarray, ignoreEdges : bool = False) -> np.ndarray:
    '''
    merge sandwiched segments
    ambiguous segments between swapped segments are likely also swapped
    edge cases rely on only single bordering segment
    segments with a swap on one side and a non-swap on the other are assumed to be fine
    swap flags invert the states of neighboring swaps on the associated side

    flag: flags indicating clear and ambiguous segments
    ignoreEdges: do not try to assign a nonzero flag to edge segments
    '''
    ambig = utils.get_value_segments(flag,0,inclusive=False)
    ambigFlag = np.zeros_like(flag)
    for a, b in ambig:
        # get flags before and after segment
        before = 0 if a == 0 else flag[a-1]
        after = 0 if b == flag.shape[0] else flag[b]

        # +1 if on edge with one adjacent swap or +2 if sandwiched by two swaps
        ambigFlag[a:b] = before + after

    # treat edges and finish
    if ignoreEdges:
        ambigFlag //= 2 # remove edge flags (1 -> 0, 2 -> 1)
    else:
        query = abs(ambigFlag) > 1
        ambigFlag[query] = np.sign(ambigFlag[query]) # (ensure annotations are 1, 0, or -1)
    return ambigFlag


def _flag_segment_metrics(metric : np.ndarray, nvals : np.ndarray,
                          thresh : tuple[float,float] = (0.9,1.1), minFrames : int = 0) -> np.ndarray:
    '''
    Returns flags indicating whether each segment is swapped given the segment length and some metric

    metric: 1D array of metric values
    thresh: lower and upper bound of metric for indicating a clear non-swap or clear swap (respectively)
    nvals: number of values considered for each segment
    minFrames: minimum segLength required to confirm a swap or non-swap
    '''
    flag = np.zeros_like(metric) # 0 -> ambiguous
    flag[np.logical_and(metric < thresh[0], nvals >= minFrames)] = 1 # swapped
    flag[np.logical_and(metric > thresh[1], nvals >= minFrames)] = -1 # not swapped
    return flag


def _get_travel_distance_ratios(data : pd.DataFrame, segs : np.ndarray) -> np.ndarray:
    '''
    Returns the ratio of head travel distance to tail travel distnce in the given semgments
    '''
    # calculate distances
    hdist = metrics.get_segment_distance(data,segs,'head')
    tdist = metrics.get_segment_distance(data,segs,'tail')
    
    # get ratio
    frac = hdist / (tdist + 10**-3) # prevent divide by zero
    return frac


def _get_speed_ratios(data : pd.DataFrame, segs : np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    '''
    Returns the ratio of head speed to tail speed in the given segments
    and the number of frames in each case being considered
    '''
    # calculate speeds
    hspd = metrics.get_speed_from_df(data,'head')
    tspd = metrics.get_speed_from_df(data,'tail')

    # get differences in head, tail speeds 
    frac = np.zeros((segs.shape[0],))
    nvals = np.zeros((segs.shape[0],))
    for i, seg in enumerate(segs):
        a, b = seg

        # get speeds in segment
        hs = hspd[a:b]
        ts = tspd[a:b]
        hs = hs[~np.isnan(hs)]
        ts = ts[~np.isnan(ts)]
        
        # get speed ratio and counts
        nvals[i] = min(len(hs),len(ts))
        if len(hs) > 0 and len(hs) > 0: # catch invalid division
            frac[i] = np.mean(hs) / np.mean(ts)
        else:
            frac[i] = 1 # mean speeds assumed identical (ambiguous case)
    return frac, nvals


def _get_alignment_angles(data : pd.DataFrame, segs : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate alignment angles for segments.
    
    Returns:
    -------
    tuple: (median_angles, std_angles, mean_angles)
        All in radians, not normalized
    """
    tvec = metrics.get_motion_vector(data,'tail')
    tmvec = metrics.get_orientation_vectors(data,head=False)
    tailAlignment = [utils.get_angle(tmvec[i],tvec[i],halfAngle=True) for i in range(data.shape[0])]
    tailAlignSeg = utils.metrics_by_segment(tailAlignment,segs) # mean, std, median
    return tailAlignSeg[:,2], tailAlignSeg[:,1], tailAlignSeg[:,0]  # median, std, mean


def _get_alignment_angles_legacy(data : pd.DataFrame, fps : int, segs : np.ndarray, minFrames : int,
                               minSpeed : float = 0, maxHeadAngle : float = np.pi/4) -> np.ndarray:
    '''
    Returns the mean angle between the midpoint-head and tail motion vectors for the given segments

    data: (filtered) position data
    fps: frame rate
    segs: segments to analyse
    minFrames: minimum valid segment length

    minSpeed: speed for detectung runs (forward-motion mode only)
    maxHeadAngle: maximum head angle for detecting runs (forward-motion mode only)
    '''
    # get tail-midpoint and centroid movement vectors
    cvec = metrics.get_motion_vector(data,'tail')
    tmvec = metrics.get_orientation_vectors(data,head=False)

    # check alignment of tail-midpoint and centroid movement vectors in non-curl regions
    ang = [utils.get_angle(tmvec[i],cvec[i],halfAngle=True) for i in range(data.shape[0])]
    ang = np.array(ang)

    # extract approximate run frames in each segment
    spd = metrics.get_speed_from_df(data,'ctr',fps,2)
    ha = metrics.get_head_angle(data,halfAngle=True)
    query = np.logical_and(spd > minSpeed, ha < maxHeadAngle)
    isRun = np.zeros_like(spd)
    isRun[query] = 1

    # estimate average angle during runs in each segment
    ang *= isRun # sets values in non-run frames to zeros
    vals = [(np.nansum(ang[a:b+1]),np.sum(isRun[a:b+1])) for a, b in segs] # get angle sums and counts
    avgang = [ang / ct if ct > minFrames else np.NaN for ang, ct in vals] # values in segments w/o runs -> NaN
    avgang = np.array(avgang)
    return avgang



# ----- Swap Correction -----

def correct_swapped_segments(rawData : pd.DataFrame, segments : np.ndarray,
                              debug : bool = False) -> pd.DataFrame:
    '''
    Swap head and tail in the given segments
    NOTE: end index is assumed to be inclusive

    rawData: DataFrame with raw position data
    segments: N x 2 array of start and end frames (inclusive) of swapped segments
    debug: print debug messages
    '''
    data = rawData.copy()

    # get list of swapped frames
    frames = [np.arange(a,b+1) for a, b in segments]
    frames = utils.flatten(frames)

    # correct swapped frames
    #TODO: do this more efficiently
    for i in frames:
        #data.loc[i, ['xhead','yhead','xtail','ytail']] = data.loc[i, ['xtail','ytail','xhead','yhead']].to_numpy()
        xh = data.at[i,'xhead']
        yh = data.at[i,'yhead']
        xt = data.at[i,'xtail']
        yt = data.at[i,'ytail']

        data.loc[i,'xhead'] = xt
        data.loc[i,'yhead'] = yt
        data.loc[i,'xtail'] = xh
        data.loc[i,'ytail'] = yh

    if debug:
        print('Swapped Segments: {}'.format(segments))
        print('Frames corrected:',len(frames))
    return data


# ----- Position Filtering -----

def filter_data(rawData : pd.DataFrame) -> pd.DataFrame:
    '''Apply the default filter used by the analysis pipeline'''
    return filter_gaussian(rawData,3)
    #return filter_sgolay(data,45,4)
    #return filter_median(data,10)


def filter_sgolay(rawData : pd.DataFrame, window : int = 45, order : int = 4) -> pd.DataFrame:
    '''Apply Savitzky-Golay filter to the position data'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.signal.savgol_filter(data[col].to_numpy(), window, order)
    
    return data


def filter_gaussian(rawData : pd.DataFrame, sigma : float = 3) -> pd.DataFrame:
    '''Apply Gaussian filter to the position data'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.ndimage.gaussian_filter1d(data[col].to_numpy(),sigma)
    
    return data


def filter_meanmed(rawData : pd.DataFrame, medWin : int = 15, meanWin : int | None = None) -> pd.DataFrame:
    '''Filter the position data by taking a rolling median followed by a rolling mean'''
    data = rawData.copy()

    if meanWin is None : meanWin = medWin
    for col in utils.flatten(metrics.POSDICT.values()):
        med = sp.ndimage.median_filter(data[col].to_numpy(),medWin)
        avg = sp.ndimage.uniform_filter(med,meanWin)
        data[col] = avg
    
    return data


def filter_median(rawData : pd.DataFrame, win : int = 5) -> pd.DataFrame:
    '''Filter the position data using a rolling median'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.ndimage.median_filter(data[col].to_numpy(),win)
    
    return data


# def filter_med_gaussian(rawData : pd.DataFrame, fps : int, window : int = 5, sigma : float = 1.0) -> pd.DataFrame:
#     '''
#     Apply a median filter followed by a Gaussian filter to the position data

#     rawData: unfiltered position data
#     fps: frame rate
#     window: window size for median filter
#     sigma: standard deviation for Gaussian filter
#     '''
#     data = rawData.copy()
#     cols = ['head','tail','mid','ctr'] # partial keys of columns of interest
#     for col in cols:
#         # extract data
#         xcol = 'x'+col
#         ycol = 'y'+col

#         start = max(data[xcol].first_valid_index(),data[ycol].first_valid_index())
#         end = min(data[xcol].last_valid_index(),data[ycol].last_valid_index()) + 1

#         vec = data.loc[start:end,[xcol,ycol]].to_numpy() # extract data, omitting NaNs at edges

#         # create and run filter
#         # TODO: ensure NaNs mtch between x, y vectors
#         data.loc[start:end,[xcol,ycol]] = filter_med_gaussian_vec(vec,window,sigma)
#     return data


# def filter_kalman(rawData : pd.DataFrame, fps : int, derivatives : int = 2, **kwargs) -> pd.DataFrame:
#     '''
#     Apply a Kalman filter to the position data

#     rawData: unfiltered position data
#     derivatives: number of derivativesto use in estimations
#     dt: time step
#     '''
#     data = rawData.copy()
#     dt = 1/fps # time step
#     ndim = 2 # number of dimensions; TODO: fully generalize?

#     kfilter = KalmanFilter(dt,ndim,derivatives,**kwargs) # set up Kalman filter
#     cols = ['head','tail','mid','ctr'] # partial keys of columns of interest
#     for col in cols:
#         # extract data
#         xcol = 'x'+col
#         ycol = 'y'+col

#         start = max(data[xcol].first_valid_index(),data[ycol].first_valid_index())
#         end = min(data[xcol].last_valid_index(),data[ycol].last_valid_index()) + 1

#         vec = data.loc[start:end,[xcol,ycol]].to_numpy() # extract data, omitting NaNs at edges

#         # create and run filter
#         # TODO: ensure NaNs mtch between x, y vectors
#         data.loc[start:end,[xcol,ycol]] = kfilter.filter(vec)
#     return data
            

"""
Machine learning feature extraction for swap detection - Version 2.

DEPRECATED: This is a legacy version kept for reference only.
The current default implementation is features_v4 (~36 features).

This version removed 10 redundant features from the original 56-feature set:
- 8 position features (head_x, head_y, tail_x, tail_y, mid_x, mid_y, centroid_x, centroid_y)
- 2 velocity magnitude features (head_velocity_magnitude, tail_velocity_magnitude)

Total: 46 features

This version removes redundant features:
- Removed: head_velocity_magnitude, tail_velocity_magnitude (redundant with speed)
- Removed: Raw position features (head_x, head_y, tail_x, tail_y, mid_x, mid_y, centroid_x, centroid_y)

Total features: ~48 (down from 56)

This module provides feature extraction functions that pre-compute expensive
operations and use vectorized NumPy operations for speed (400x faster than
naive implementations).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from swap_correction import metrics, utils, tracking_correction


def extract_all_frame_features_optimized(trial_data: pd.DataFrame, fps: int = 30,
                                        apply_filtering: bool = False, 
                                        filter_sigma: float = 4.5) -> pd.DataFrame:
    """
    Optimized version: Extract features for all frames in a trial (V2 - reduced redundancy).
    
    Pre-computes expensive operations once and uses vectorized operations.
    Expected speedup: 8-12x faster than original implementation.
    
    V2 Changes:
    - Removed redundant velocity_magnitude features (identical to speed)
    - Removed raw position features (x, y coordinates)
    
    Parameters:
    -----------
    trial_data : pd.DataFrame
        Tracking data for a trial
    fps : int
        Frame rate
    apply_filtering : bool
        If True, apply Gaussian filter to position data
    filter_sigma : float
        Standard deviation for Gaussian filter
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per frame, columns are features (~48 features)
    """
    from scipy import ndimage
    
    n_frames = len(trial_data)
    
    # Apply filtering if requested
    if apply_filtering:
        trial_data = trial_data.copy()
        position_cols = ['xhead', 'yhead', 'xtail', 'ytail', 'xmid', 'ymid', 'xctr', 'yctr']
        for col in position_cols:
            if col in trial_data.columns:
                valid_mask = ~pd.isna(trial_data[col])
                if valid_mask.sum() > 0:
                    filtered_values = trial_data[col].copy()
                    filtered_values[valid_mask] = ndimage.gaussian_filter1d(
                        trial_data[col][valid_mask].values, 
                        sigma=filter_sigma
                    )
                    trial_data[col] = filtered_values
    
    # ===== PRE-COMPUTE EXPENSIVE OPERATIONS ONCE =====
    
    # Convert to NumPy arrays for faster access (avoid pandas overhead)
    xhead = trial_data['xhead'].values
    yhead = trial_data['yhead'].values
    xtail = trial_data['xtail'].values
    ytail = trial_data['ytail'].values
    xmid = trial_data['xmid'].values
    ymid = trial_data['ymid'].values
    xctr = trial_data['xctr'].values
    yctr = trial_data['yctr'].values
    
    # Pre-compute speeds (called once instead of 10x per frame)
    hspd_all = metrics.get_speed_from_df(trial_data, 'head', fps=fps, npoints=2)
    tspd_all = metrics.get_speed_from_df(trial_data, 'tail', fps=fps, npoints=2)
    
    # Pre-compute cross-sign (called once instead of 1x per frame)
    cross_sign_all = metrics.get_ht_cross_sign(trial_data)
    
    # Pre-compute cumulative distances (O(n) instead of O(n²))
    cumulative_head_dist = np.zeros(n_frames)
    cumulative_tail_dist = np.zeros(n_frames)
    for i in range(1, n_frames):
        # Incremental calculation - just add distance from previous frame
        if not (np.isnan(xhead[i-1]) or np.isnan(yhead[i-1]) or np.isnan(xhead[i]) or np.isnan(yhead[i])):
            cumulative_head_dist[i] = cumulative_head_dist[i-1] + np.sqrt(
                (xhead[i] - xhead[i-1])**2 + (yhead[i] - yhead[i-1])**2
            )
        else:
            cumulative_head_dist[i] = cumulative_head_dist[i-1]
        
        if not (np.isnan(xtail[i-1]) or np.isnan(ytail[i-1]) or np.isnan(xtail[i]) or np.isnan(ytail[i])):
            cumulative_tail_dist[i] = cumulative_tail_dist[i-1] + np.sqrt(
                (xtail[i] - xtail[i-1])**2 + (ytail[i] - ytail[i-1])**2
            )
        else:
            cumulative_tail_dist[i] = cumulative_tail_dist[i-1]
    
    # Pre-compute alignment angles for all frames (for temporal context)
    alignment_angles_all = np.full(n_frames, np.nan)
    for i in range(1, n_frames):
        if not (np.isnan(xtail[i]) or np.isnan(ytail[i]) or np.isnan(xmid[i]) or np.isnan(ymid[i]) or
                np.isnan(xtail[i-1]) or np.isnan(ytail[i-1])):
            body_vec = np.array([xmid[i] - xtail[i], ymid[i] - ytail[i]])
            tail_motion = np.array([xtail[i] - xtail[i-1], ytail[i] - ytail[i-1]])
            
            body_norm = np.linalg.norm(body_vec)
            motion_norm = np.linalg.norm(tail_motion)
            
            if body_norm > 0.01 and motion_norm > 0.01:
                cos_angle = np.dot(body_vec, tail_motion) / (body_norm * motion_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                alignment_angles_all[i] = angle
    
    # Pre-compute velocity vectors (x, y components - kept for relative velocity)
    head_velocity_x = np.zeros(n_frames)
    head_velocity_y = np.zeros(n_frames)
    tail_velocity_x = np.zeros(n_frames)
    tail_velocity_y = np.zeros(n_frames)
    
    head_velocity_x[1:] = np.diff(xhead)
    head_velocity_y[1:] = np.diff(yhead)
    tail_velocity_x[1:] = np.diff(xtail)
    tail_velocity_y[1:] = np.diff(ytail)
    
    # Set NaN where original data was NaN
    head_velocity_x[np.isnan(xhead)] = np.nan
    head_velocity_y[np.isnan(yhead)] = np.nan
    tail_velocity_x[np.isnan(xtail)] = np.nan
    tail_velocity_y[np.isnan(ytail)] = np.nan
    
    # Pre-compute angular velocities (simplified)
    head_angular_velocity = np.full(n_frames, np.nan)
    tail_angular_velocity = np.full(n_frames, np.nan)
    
    for i in range(2, n_frames):
        # Head angular velocity
        if not (np.isnan(xhead[i-2]) or np.isnan(yhead[i-2]) or 
                np.isnan(xhead[i-1]) or np.isnan(yhead[i-1]) or
                np.isnan(xhead[i]) or np.isnan(yhead[i])):
            v1 = np.array([xhead[i-1] - xhead[i-2], yhead[i-1] - yhead[i-2]])
            v2 = np.array([xhead[i] - xhead[i-1], yhead[i] - yhead[i-1]])
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 0.01 and v2_norm > 0.01:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle) * 180 / np.pi
                head_angular_velocity[i] = angle_change * fps
        
        # Tail angular velocity
        if not (np.isnan(xtail[i-2]) or np.isnan(ytail[i-2]) or 
                np.isnan(xtail[i-1]) or np.isnan(ytail[i-1]) or
                np.isnan(xtail[i]) or np.isnan(ytail[i])):
            v1 = np.array([xtail[i-1] - xtail[i-2], ytail[i-1] - ytail[i-2]])
            v2 = np.array([xtail[i] - xtail[i-1], ytail[i] - ytail[i-1]])
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 0.01 and v2_norm > 0.01:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle) * 180 / np.pi
                tail_angular_velocity[i] = angle_change * fps
    
    # Pre-compute path curvature
    head_path_curvature = np.full(n_frames, np.nan)
    for i in range(2, n_frames):
        if not (np.isnan(xhead[i-2]) or np.isnan(yhead[i-2]) or 
                np.isnan(xhead[i-1]) or np.isnan(yhead[i-1]) or
                np.isnan(xhead[i]) or np.isnan(yhead[i])):
            v1 = np.array([xhead[i-1] - xhead[i-2], yhead[i-1] - yhead[i-2]])
            v2 = np.array([xhead[i] - xhead[i-1], yhead[i] - yhead[i-1]])
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 0.01 and v2_norm > 0.01:
                cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.arccos(cos_angle) * 180 / np.pi
                head_path_curvature[i] = angle_change
    
    # Window sizes for temporal context
    window_sizes = [5, 10, 20, 50]
    
    # Pre-compute temporal context statistics using sliding windows
    # Use pandas rolling for efficiency
    hspd_series = pd.Series(hspd_all)
    tspd_series = pd.Series(tspd_all)
    alignment_series = pd.Series(alignment_angles_all)
    
    temporal_stats = {}
    for window_size in window_sizes:
        # Speed statistics
        temporal_stats[f'head_speed_mean_{window_size}'] = hspd_series.rolling(
            window=window_size, min_periods=window_size//2, center=False
        ).mean().values
        temporal_stats[f'head_speed_std_{window_size}'] = hspd_series.rolling(
            window=window_size, min_periods=window_size//2, center=False
        ).std().values
        temporal_stats[f'tail_speed_mean_{window_size}'] = tspd_series.rolling(
            window=window_size, min_periods=window_size//2, center=False
        ).mean().values
        temporal_stats[f'tail_speed_std_{window_size}'] = tspd_series.rolling(
            window=window_size, min_periods=window_size//2, center=False
        ).std().values
        
        # Alignment angle statistics
        temporal_stats[f'alignment_angle_mean_{window_size}'] = alignment_series.rolling(
            window=window_size, min_periods=window_size//2, center=False
        ).mean().values
        temporal_stats[f'alignment_angle_std_{window_size}'] = alignment_series.rolling(
            window=window_size, min_periods=window_size//2, center=False
        ).std().values
    
    # ===== EXTRACT FEATURES FOR EACH FRAME (FAST - just indexing) =====
    
    all_features = []
    for i in range(n_frames):
        features = {}
        
        # V2: REMOVED Position features (raw coordinates)
        # These were: head_x, head_y, tail_x, tail_y, mid_x, mid_y, centroid_x, centroid_y
        # Rationale: Distance features capture spatial relationships; raw positions may be trial-specific
        
        # Distances
        if not (np.isnan(xhead[i]) or np.isnan(xtail[i])):
            features['head_tail_distance'] = np.sqrt((xhead[i] - xtail[i])**2 + (yhead[i] - ytail[i])**2)
        else:
            features['head_tail_distance'] = np.nan
        
        if not (np.isnan(xhead[i]) or np.isnan(xmid[i])):
            features['head_mid_distance'] = np.sqrt((xhead[i] - xmid[i])**2 + (yhead[i] - ymid[i])**2)
        else:
            features['head_mid_distance'] = np.nan
        
        if not (np.isnan(xtail[i]) or np.isnan(xmid[i])):
            features['tail_mid_distance'] = np.sqrt((xtail[i] - xmid[i])**2 + (ytail[i] - ymid[i])**2)
        else:
            features['tail_mid_distance'] = np.nan
        
        # Velocity features (from pre-computed arrays)
        features['head_speed'] = hspd_all[i] if i < len(hspd_all) and not np.isnan(hspd_all[i]) else np.nan
        features['tail_speed'] = tspd_all[i] if i < len(tspd_all) and not np.isnan(tspd_all[i]) else np.nan
        
        # Speed ratio
        if not (np.isnan(features['head_speed']) or np.isnan(features['tail_speed'])):
            if features['tail_speed'] > 0.01:
                features['speed_ratio'] = features['head_speed'] / features['tail_speed']
            else:
                features['speed_ratio'] = np.nan
        else:
            features['speed_ratio'] = np.nan
        
        # Velocity vectors (x, y components - kept for relative velocity calculation)
        features['head_velocity_x'] = head_velocity_x[i] if not np.isnan(head_velocity_x[i]) else np.nan
        features['head_velocity_y'] = head_velocity_y[i] if not np.isnan(head_velocity_y[i]) else np.nan
        
        features['tail_velocity_x'] = tail_velocity_x[i] if not np.isnan(tail_velocity_x[i]) else np.nan
        features['tail_velocity_y'] = tail_velocity_y[i] if not np.isnan(tail_velocity_y[i]) else np.nan
        
        # V2: REMOVED velocity_magnitude features
        # These were: head_velocity_magnitude, tail_velocity_magnitude
        # Rationale: These are identical to head_speed and tail_speed (sqrt(vx² + vy²) = speed)
        
        # Relative velocity
        if not (np.isnan(head_velocity_x[i]) or np.isnan(tail_velocity_x[i])):
            rel_vel_x = head_velocity_x[i] - tail_velocity_x[i]
            rel_vel_y = head_velocity_y[i] - tail_velocity_y[i]
            features['relative_velocity_magnitude'] = np.sqrt(rel_vel_x**2 + rel_vel_y**2)
        else:
            features['relative_velocity_magnitude'] = np.nan
        
        # Angular features
        if not (np.isnan(xtail[i]) or np.isnan(xmid[i])):
            body_vec = np.array([xmid[i] - xtail[i], ymid[i] - ytail[i]])
            features['body_orientation_angle'] = np.arctan2(body_vec[1], body_vec[0]) * 180 / np.pi
        else:
            features['body_orientation_angle'] = np.nan
        
        if not np.isnan(head_velocity_x[i]):
            features['head_motion_angle'] = np.arctan2(head_velocity_y[i], head_velocity_x[i]) * 180 / np.pi
        else:
            features['head_motion_angle'] = np.nan
        
        if not np.isnan(tail_velocity_x[i]):
            features['tail_motion_angle'] = np.arctan2(tail_velocity_y[i], tail_velocity_x[i]) * 180 / np.pi
        else:
            features['tail_motion_angle'] = np.nan
        
        # Alignment angle
        features['alignment_angle'] = alignment_angles_all[i] if not np.isnan(alignment_angles_all[i]) else np.nan
        
        # Angular velocities (from pre-computed)
        features['head_angular_velocity'] = head_angular_velocity[i] if not np.isnan(head_angular_velocity[i]) else np.nan
        features['tail_angular_velocity'] = tail_angular_velocity[i] if not np.isnan(tail_angular_velocity[i]) else np.nan
        
        # Geometric features
        features['cross_sign'] = cross_sign_all[i] if i < len(cross_sign_all) and not np.isnan(cross_sign_all[i]) else np.nan
        features['head_path_curvature'] = head_path_curvature[i] if not np.isnan(head_path_curvature[i]) else np.nan
        
        # Cumulative distances (from pre-computed)
        features['cumulative_head_distance'] = cumulative_head_dist[i]
        features['cumulative_tail_distance'] = cumulative_tail_dist[i]
        
        # Temporal context features (from pre-computed)
        for window_size in window_sizes:
            features[f'head_speed_mean_{window_size}'] = temporal_stats[f'head_speed_mean_{window_size}'][i] if not np.isnan(temporal_stats[f'head_speed_mean_{window_size}'][i]) else np.nan
            features[f'head_speed_std_{window_size}'] = temporal_stats[f'head_speed_std_{window_size}'][i] if not np.isnan(temporal_stats[f'head_speed_std_{window_size}'][i]) else np.nan
            features[f'tail_speed_mean_{window_size}'] = temporal_stats[f'tail_speed_mean_{window_size}'][i] if not np.isnan(temporal_stats[f'tail_speed_mean_{window_size}'][i]) else np.nan
            features[f'tail_speed_std_{window_size}'] = temporal_stats[f'tail_speed_std_{window_size}'][i] if not np.isnan(temporal_stats[f'tail_speed_std_{window_size}'][i]) else np.nan
            features[f'alignment_angle_mean_{window_size}'] = temporal_stats[f'alignment_angle_mean_{window_size}'][i] if not np.isnan(temporal_stats[f'alignment_angle_mean_{window_size}'][i]) else np.nan
            features[f'alignment_angle_std_{window_size}'] = temporal_stats[f'alignment_angle_std_{window_size}'][i] if not np.isnan(temporal_stats[f'alignment_angle_std_{window_size}'][i]) else np.nan
        
        # Context features
        features['position_in_trial'] = i / max(1, n_frames - 1)
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)


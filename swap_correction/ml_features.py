"""
Machine learning feature extraction for swap detection.

This module provides functions to extract frame-level and segment-level features
from tracking data for training ML models to detect head-tail swaps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from swap_correction import metrics, utils, tracking_correction


def extract_frame_features(data: pd.DataFrame, frame_idx: int, 
                          fps: int = 30, window_sizes: List[int] = [5, 10, 20, 50]) -> Dict:
    """
    Extract comprehensive features for a single frame.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Tracking data
    frame_idx : int
        Index of frame to extract features for
    fps : int
        Frame rate
    window_sizes : list
        Window sizes for temporal context features
        
    Returns:
    --------
    dict
        Dictionary of feature names to values
    """
    features = {}
    n_frames = len(data)
    
    if frame_idx < 0 or frame_idx >= n_frames:
        return features
    
    # Get frame data
    frame = data.iloc[frame_idx]
    
    # ===== Position Features =====
    features['head_x'] = frame.get('xhead', np.nan)
    features['head_y'] = frame.get('yhead', np.nan)
    features['tail_x'] = frame.get('xtail', np.nan)
    features['tail_y'] = frame.get('ytail', np.nan)
    features['mid_x'] = frame.get('xmid', np.nan)
    features['mid_y'] = frame.get('ymid', np.nan)
    features['centroid_x'] = frame.get('xctr', np.nan)
    features['centroid_y'] = frame.get('yctr', np.nan)
    
    # Distances
    if not (pd.isna(features['head_x']) or pd.isna(features['tail_x'])):
        head_tail_dist = np.sqrt(
            (features['head_x'] - features['tail_x'])**2 + 
            (features['head_y'] - features['tail_y'])**2
        )
        features['head_tail_distance'] = head_tail_dist
    else:
        features['head_tail_distance'] = np.nan
    
    if not (pd.isna(features['head_x']) or pd.isna(features['mid_x'])):
        head_mid_dist = np.sqrt(
            (features['head_x'] - features['mid_x'])**2 + 
            (features['head_y'] - features['mid_y'])**2
        )
        features['head_mid_distance'] = head_mid_dist
    else:
        features['head_mid_distance'] = np.nan
    
    if not (pd.isna(features['tail_x']) or pd.isna(features['mid_x'])):
        tail_mid_dist = np.sqrt(
            (features['tail_x'] - features['mid_x'])**2 + 
            (features['tail_y'] - features['mid_y'])**2
        )
        features['tail_mid_distance'] = tail_mid_dist
    else:
        features['tail_mid_distance'] = np.nan
    
    # ===== Velocity Features =====
    # Get speeds
    hspd = metrics.get_speed_from_df(data, 'head', fps=fps, npoints=2)
    tspd = metrics.get_speed_from_df(data, 'tail', fps=fps, npoints=2)
    
    if frame_idx < len(hspd):
        features['head_speed'] = hspd[frame_idx] if not np.isnan(hspd[frame_idx]) else np.nan
        features['tail_speed'] = tspd[frame_idx] if not np.isnan(tspd[frame_idx]) else np.nan
    else:
        features['head_speed'] = np.nan
        features['tail_speed'] = np.nan
    
    # Speed ratio
    if not (pd.isna(features['head_speed']) or pd.isna(features['tail_speed'])):
        if features['tail_speed'] > 0.01:
            features['speed_ratio'] = features['head_speed'] / features['tail_speed']
        else:
            features['speed_ratio'] = np.nan
    else:
        features['speed_ratio'] = np.nan
    
    # Velocity vectors (dx, dy)
    if frame_idx > 0:
        prev_frame = data.iloc[frame_idx - 1]
        if not (pd.isna(frame.get('xhead')) or pd.isna(prev_frame.get('xhead'))):
            features['head_velocity_x'] = frame.get('xhead') - prev_frame.get('xhead')
            features['head_velocity_y'] = frame.get('yhead') - prev_frame.get('yhead')
            features['head_velocity_magnitude'] = np.sqrt(
                features['head_velocity_x']**2 + features['head_velocity_y']**2
            )
        else:
            features['head_velocity_x'] = np.nan
            features['head_velocity_y'] = np.nan
            features['head_velocity_magnitude'] = np.nan
        
        if not (pd.isna(frame.get('xtail')) or pd.isna(prev_frame.get('xtail'))):
            features['tail_velocity_x'] = frame.get('xtail') - prev_frame.get('xtail')
            features['tail_velocity_y'] = frame.get('ytail') - prev_frame.get('ytail')
            features['tail_velocity_magnitude'] = np.sqrt(
                features['tail_velocity_x']**2 + features['tail_velocity_y']**2
            )
        else:
            features['tail_velocity_x'] = np.nan
            features['tail_velocity_y'] = np.nan
            features['tail_velocity_magnitude'] = np.nan
    else:
        features['head_velocity_x'] = np.nan
        features['head_velocity_y'] = np.nan
        features['head_velocity_magnitude'] = np.nan
        features['tail_velocity_x'] = np.nan
        features['tail_velocity_y'] = np.nan
        features['tail_velocity_magnitude'] = np.nan
    
    # Relative velocity
    if not (pd.isna(features['head_velocity_x']) or pd.isna(features['tail_velocity_x'])):
        rel_vel_x = features['head_velocity_x'] - features['tail_velocity_x']
        rel_vel_y = features['head_velocity_y'] - features['tail_velocity_y']
        features['relative_velocity_magnitude'] = np.sqrt(rel_vel_x**2 + rel_vel_y**2)
    else:
        features['relative_velocity_magnitude'] = np.nan
    
    # ===== Angular Features =====
    # Body orientation angle (tail to midpoint)
    if not (pd.isna(features['tail_x']) or pd.isna(features['mid_x'])):
        body_vec = np.array([features['mid_x'] - features['tail_x'], 
                            features['mid_y'] - features['tail_y']])
        body_angle = np.arctan2(body_vec[1], body_vec[0]) * 180 / np.pi
        features['body_orientation_angle'] = body_angle
    else:
        features['body_orientation_angle'] = np.nan
    
    # Motion direction angles
    if not pd.isna(features['head_velocity_x']):
        head_motion_angle = np.arctan2(features['head_velocity_y'], 
                                      features['head_velocity_x']) * 180 / np.pi
        features['head_motion_angle'] = head_motion_angle
    else:
        features['head_motion_angle'] = np.nan
    
    if not pd.isna(features['tail_velocity_x']):
        tail_motion_angle = np.arctan2(features['tail_velocity_y'], 
                                       features['tail_velocity_x']) * 180 / np.pi
        features['tail_motion_angle'] = tail_motion_angle
    else:
        features['tail_motion_angle'] = np.nan
    
    # Alignment angle (body orientation vs tail motion)
    if not (pd.isna(features['body_orientation_angle']) or pd.isna(features['tail_motion_angle'])):
        # Calculate angle between body vector and tail motion vector
        if not (pd.isna(features['tail_x']) or pd.isna(features['mid_x']) or 
                pd.isna(features['tail_velocity_x'])):
            body_vec = np.array([features['mid_x'] - features['tail_x'], 
                                features['mid_y'] - features['tail_y']])
            tail_motion_vec = np.array([features['tail_velocity_x'], 
                                       features['tail_velocity_y']])
            
            body_norm = np.linalg.norm(body_vec)
            motion_norm = np.linalg.norm(tail_motion_vec)
            
            if body_norm > 0.01 and motion_norm > 0.01:
                cos_angle = np.dot(body_vec, tail_motion_vec) / (body_norm * motion_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                alignment_angle = np.arccos(cos_angle) * 180 / np.pi
                features['alignment_angle'] = alignment_angle
            else:
                features['alignment_angle'] = np.nan
        else:
            features['alignment_angle'] = np.nan
    else:
        features['alignment_angle'] = np.nan
    
    # Angular velocity (simplified calculation)
    # Calculate as change in direction angle over time
    try:
        if frame_idx > 0 and frame_idx < len(data):
            # Get head positions for angular velocity calculation
            if frame_idx >= 2:
                hx_prev = data.iloc[frame_idx-2].get('xhead')
                hy_prev = data.iloc[frame_idx-2].get('yhead')
                hx_curr = data.iloc[frame_idx-1].get('xhead')
                hy_curr = data.iloc[frame_idx-1].get('yhead')
                hx_next = data.iloc[frame_idx].get('xhead')
                hy_next = data.iloc[frame_idx].get('yhead')
                
                if not (pd.isna(hx_prev) or pd.isna(hy_prev) or pd.isna(hx_curr) or 
                       pd.isna(hy_curr) or pd.isna(hx_next) or pd.isna(hy_next)):
                    v1 = np.array([hx_curr - hx_prev, hy_curr - hy_prev])
                    v2 = np.array([hx_next - hx_curr, hy_next - hy_curr])
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    if v1_norm > 0.01 and v2_norm > 0.01:
                        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle_change = np.arccos(cos_angle) * 180 / np.pi
                        # Convert to angular velocity (degrees per frame, then per second)
                        features['head_angular_velocity'] = angle_change * fps
                    else:
                        features['head_angular_velocity'] = np.nan
                else:
                    features['head_angular_velocity'] = np.nan
            else:
                features['head_angular_velocity'] = np.nan
            
            # Tail angular velocity
            if frame_idx >= 2:
                tx_prev = data.iloc[frame_idx-2].get('xtail')
                ty_prev = data.iloc[frame_idx-2].get('ytail')
                tx_curr = data.iloc[frame_idx-1].get('xtail')
                ty_curr = data.iloc[frame_idx-1].get('ytail')
                tx_next = data.iloc[frame_idx].get('xtail')
                ty_next = data.iloc[frame_idx].get('ytail')
                
                if not (pd.isna(tx_prev) or pd.isna(ty_prev) or pd.isna(tx_curr) or 
                       pd.isna(ty_curr) or pd.isna(tx_next) or pd.isna(ty_next)):
                    v1 = np.array([tx_curr - tx_prev, ty_curr - ty_prev])
                    v2 = np.array([tx_next - tx_curr, ty_next - ty_curr])
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    if v1_norm > 0.01 and v2_norm > 0.01:
                        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle_change = np.arccos(cos_angle) * 180 / np.pi
                        features['tail_angular_velocity'] = angle_change * fps
                    else:
                        features['tail_angular_velocity'] = np.nan
                else:
                    features['tail_angular_velocity'] = np.nan
            else:
                features['tail_angular_velocity'] = np.nan
        else:
            features['head_angular_velocity'] = np.nan
            features['tail_angular_velocity'] = np.nan
    except:
        features['head_angular_velocity'] = np.nan
        features['tail_angular_velocity'] = np.nan
    
    # ===== Geometric Features =====
    # Cross-sign consistency
    try:
        cross_sign = metrics.get_ht_cross_sign(data)
        if frame_idx < len(cross_sign):
            features['cross_sign'] = cross_sign[frame_idx] if not np.isnan(cross_sign[frame_idx]) else np.nan
        else:
            features['cross_sign'] = np.nan
    except:
        features['cross_sign'] = np.nan
    
    # Path curvature (simplified - using change in direction)
    if frame_idx > 1:
        try:
            # Get head positions for last 3 frames
            head_positions = []
            for i in range(max(0, frame_idx-2), frame_idx+1):
                if i < len(data):
                    hx = data.iloc[i].get('xhead')
                    hy = data.iloc[i].get('yhead')
                    if not (pd.isna(hx) or pd.isna(hy)):
                        head_positions.append([hx, hy])
            
            if len(head_positions) >= 3:
                v1 = np.array(head_positions[1]) - np.array(head_positions[0])
                v2 = np.array(head_positions[2]) - np.array(head_positions[1])
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                if v1_norm > 0.01 and v2_norm > 0.01:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.arccos(cos_angle) * 180 / np.pi
                    features['head_path_curvature'] = angle_change
                else:
                    features['head_path_curvature'] = np.nan
            else:
                features['head_path_curvature'] = np.nan
        except:
            features['head_path_curvature'] = np.nan
    else:
        features['head_path_curvature'] = np.nan
    
    # Cumulative distance traveled (head and tail)
    if frame_idx > 0:
        head_dist = 0
        tail_dist = 0
        for i in range(1, frame_idx + 1):
            if i < len(data):
                prev = data.iloc[i-1]
                curr = data.iloc[i]
                
                hx_prev = prev.get('xhead')
                hy_prev = prev.get('yhead')
                hx_curr = curr.get('xhead')
                hy_curr = curr.get('yhead')
                
                if not (pd.isna(hx_prev) or pd.isna(hy_prev) or pd.isna(hx_curr) or pd.isna(hy_curr)):
                    head_dist += np.sqrt((hx_curr - hx_prev)**2 + (hy_curr - hy_prev)**2)
                
                tx_prev = prev.get('xtail')
                ty_prev = prev.get('ytail')
                tx_curr = curr.get('xtail')
                ty_curr = curr.get('ytail')
                
                if not (pd.isna(tx_prev) or pd.isna(ty_prev) or pd.isna(tx_curr) or pd.isna(ty_curr)):
                    tail_dist += np.sqrt((tx_curr - tx_prev)**2 + (ty_curr - ty_prev)**2)
        
        features['cumulative_head_distance'] = head_dist
        features['cumulative_tail_distance'] = tail_dist
    else:
        features['cumulative_head_distance'] = 0.0
        features['cumulative_tail_distance'] = 0.0
    
    # ===== Temporal Context Features =====
    # Mean and std of key features over sliding windows
    for window_size in window_sizes:
        start_idx = max(0, frame_idx - window_size + 1)
        end_idx = min(n_frames, frame_idx + 1)
        window_data = data.iloc[start_idx:end_idx]
        
        if len(window_data) >= window_size // 2:  # Require at least half window
            # Speed statistics
            window_hspd = metrics.get_speed_from_df(window_data, 'head', fps=fps, npoints=2)
            window_tspd = metrics.get_speed_from_df(window_data, 'tail', fps=fps, npoints=2)
            
            valid_hspd = window_hspd[~np.isnan(window_hspd)]
            valid_tspd = window_tspd[~np.isnan(window_tspd)]
            
            if len(valid_hspd) > 0:
                features[f'head_speed_mean_{window_size}'] = np.mean(valid_hspd)
                features[f'head_speed_std_{window_size}'] = np.std(valid_hspd)
            else:
                features[f'head_speed_mean_{window_size}'] = np.nan
                features[f'head_speed_std_{window_size}'] = np.nan
            
            if len(valid_tspd) > 0:
                features[f'tail_speed_mean_{window_size}'] = np.mean(valid_tspd)
                features[f'tail_speed_std_{window_size}'] = np.std(valid_tspd)
            else:
                features[f'tail_speed_mean_{window_size}'] = np.nan
                features[f'tail_speed_std_{window_size}'] = np.nan
            
            # Alignment angle statistics
            try:
                alignment_angles = []
                for i in range(len(window_data) - 1):
                    if i + start_idx < len(data) - 1:
                        tail_pos = window_data.iloc[i][['xtail', 'ytail']].values
                        mid_pos = window_data.iloc[i][['xmid', 'ymid']].values
                        next_tail_pos = window_data.iloc[i+1][['xtail', 'ytail']].values
                        
                        if not (np.any(np.isnan(tail_pos)) or np.any(np.isnan(mid_pos)) or 
                                np.any(np.isnan(next_tail_pos))):
                            body_vec = mid_pos - tail_pos
                            tail_motion = next_tail_pos - tail_pos
                            
                            body_norm = np.linalg.norm(body_vec)
                            motion_norm = np.linalg.norm(tail_motion)
                            
                            if body_norm > 0.01 and motion_norm > 0.01:
                                cos_angle = np.dot(body_vec, tail_motion) / (body_norm * motion_norm)
                                cos_angle = np.clip(cos_angle, -1, 1)
                                angle = np.arccos(cos_angle) * 180 / np.pi
                                alignment_angles.append(angle)
                
                if len(alignment_angles) > 0:
                    features[f'alignment_angle_mean_{window_size}'] = np.mean(alignment_angles)
                    features[f'alignment_angle_std_{window_size}'] = np.std(alignment_angles)
                else:
                    features[f'alignment_angle_mean_{window_size}'] = np.nan
                    features[f'alignment_angle_std_{window_size}'] = np.nan
            except:
                features[f'alignment_angle_mean_{window_size}'] = np.nan
                features[f'alignment_angle_std_{window_size}'] = np.nan
        else:
            # Not enough data for window
            features[f'head_speed_mean_{window_size}'] = np.nan
            features[f'head_speed_std_{window_size}'] = np.nan
            features[f'tail_speed_mean_{window_size}'] = np.nan
            features[f'tail_speed_std_{window_size}'] = np.nan
            features[f'alignment_angle_mean_{window_size}'] = np.nan
            features[f'alignment_angle_std_{window_size}'] = np.nan
    
    # Position in trial (normalized)
    features['position_in_trial'] = frame_idx / max(1, n_frames - 1)
    
    return features


def extract_all_frame_features(trial_data: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """
    Extract features for all frames in a trial.
    
    Parameters:
    -----------
    trial_data : pd.DataFrame
        Tracking data for a trial
    fps : int
        Frame rate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per frame, columns are features
    """
    n_frames = len(trial_data)
    all_features = []
    
    for i in range(n_frames):
        features = extract_frame_features(trial_data, i, fps=fps)
        all_features.append(features)
    
    return pd.DataFrame(all_features)


def extract_segment_features(data: pd.DataFrame, start_frame: int, end_frame: int,
                            fps: int = 30) -> Dict:
    """
    Extract aggregate features for a segment.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Tracking data
    start_frame : int
        Start frame of segment
    end_frame : int
        End frame of segment
    fps : int
        Frame rate
        
    Returns:
    --------
    dict
        Dictionary of segment-level features
    """
    features = {}
    
    if start_frame < 0 or end_frame >= len(data) or start_frame > end_frame:
        return features
    
    segment_data = data.iloc[start_frame:end_frame+1]
    segment_length = end_frame - start_frame + 1
    
    features['segment_length'] = segment_length
    features['segment_duration'] = segment_length / fps
    
    # Extract frame-level features for the segment
    frame_features_list = []
    for i in range(start_frame, end_frame + 1):
        frame_feat = extract_frame_features(data, i, fps=fps)
        frame_features_list.append(frame_feat)
    
    frame_features_df = pd.DataFrame(frame_features_list)
    
    # Aggregate statistics for numeric features
    numeric_cols = frame_features_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        values = frame_features_df[col].dropna()
        if len(values) > 0:
            features[f'{col}_mean'] = np.mean(values)
            features[f'{col}_median'] = np.median(values)
            features[f'{col}_std'] = np.std(values)
            features[f'{col}_min'] = np.min(values)
            features[f'{col}_max'] = np.max(values)
            if len(values) > 1:
                features[f'{col}_q25'] = np.percentile(values, 25)
                features[f'{col}_q75'] = np.percentile(values, 75)
                features[f'{col}_q90'] = np.percentile(values, 90)
    
    # Pattern features
    # Consistency (how stable are features across segment)
    if 'alignment_angle' in frame_features_df.columns:
        alignment_angles = frame_features_df['alignment_angle'].dropna()
        if len(alignment_angles) > 0:
            features['alignment_consistency'] = 1.0 / (1.0 + np.std(alignment_angles))
        else:
            features['alignment_consistency'] = np.nan
    
    # Transition features (how segment differs from surrounding)
    n_frames = len(data)
    context_before = max(0, start_frame - 50)
    context_after = min(n_frames, end_frame + 50)
    
    if start_frame > 0:
        before_data = data.iloc[context_before:start_frame]
        if len(before_data) > 0:
            before_speeds = metrics.get_speed_from_df(before_data, 'head', fps=fps)
            before_speeds = before_speeds[~np.isnan(before_speeds)]
            if len(before_speeds) > 0:
                segment_speeds = metrics.get_speed_from_df(segment_data, 'head', fps=fps)
                segment_speeds = segment_speeds[~np.isnan(segment_speeds)]
                if len(segment_speeds) > 0:
                    features['speed_change_from_before'] = np.mean(segment_speeds) - np.mean(before_speeds)
    
    if end_frame < n_frames - 1:
        after_data = data.iloc[end_frame+1:context_after]
        if len(after_data) > 0:
            after_speeds = metrics.get_speed_from_df(after_data, 'head', fps=fps)
            after_speeds = after_speeds[~np.isnan(after_speeds)]
            if len(after_speeds) > 0:
                segment_speeds = metrics.get_speed_from_df(segment_data, 'head', fps=fps)
                segment_speeds = segment_speeds[~np.isnan(segment_speeds)]
                if len(segment_speeds) > 0:
                    features['speed_change_to_after'] = np.mean(after_speeds) - np.mean(segment_speeds)
    
    # Context features
    features['position_in_trial_start'] = start_frame / max(1, n_frames - 1)
    features['position_in_trial_end'] = end_frame / max(1, n_frames - 1)
    features['is_near_start'] = (start_frame < n_frames * 0.1)
    features['is_near_end'] = (end_frame > n_frames * 0.9)
    
    return features


def extract_all_segment_features(trial_data: pd.DataFrame, segments: np.ndarray,
                                fps: int = 30) -> pd.DataFrame:
    """
    Extract features for all segments in a trial.
    
    Parameters:
    -----------
    trial_data : pd.DataFrame
        Tracking data for a trial
    segments : np.ndarray
        Nx2 array of [start_frame, end_frame] for each segment
    fps : int
        Frame rate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with one row per segment, columns are features
    """
    all_features = []
    
    for seg in segments:
        start, end = seg
        features = extract_segment_features(trial_data, start, end, fps=fps)
        all_features.append(features)
    
    return pd.DataFrame(all_features)


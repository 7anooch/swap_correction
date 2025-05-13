"""Metrics module for calculating movement metrics."""

from .metrics import (
    MovementMetrics,
    get_speed_from_df,
    get_delta_in_frame,
    get_delta_between_frames,
    get_head_angle,
    get_ht_cross_sign,
    get_orientation_vectors,
    get_motion_vector,
    get_vectors_between,
    perfectly_overlapping,
    get_cross_segment_deltas,
    get_all_deltas,
    flag_overlap_sign_reversals,
    flag_overlaps,
    get_overlap_edges,
    get_consecutive_ranges
)

__all__ = [
    'MovementMetrics',
    'get_speed_from_df',
    'get_delta_in_frame',
    'get_delta_between_frames',
    'get_head_angle',
    'get_ht_cross_sign',
    'get_orientation_vectors',
    'get_motion_vector',
    'get_vectors_between',
    'perfectly_overlapping',
    'get_cross_segment_deltas',
    'get_all_deltas',
    'flag_overlap_sign_reversals',
    'flag_overlaps',
    'get_overlap_edges',
    'get_consecutive_ranges'
]

# Position column mappings
POSDICT = {
    'head': ('X-Head', 'Y-Head'),
    'tail': ('X-Tail', 'Y-Tail'),
    'ctr': ('X-Centroid', 'Y-Centroid'),
    'mid': ('X-Midpoint', 'Y-Midpoint')
}

# Alternative column names (for backward compatibility)
ALT_POSDICT = {
    'head': ('xhead', 'yhead'),
    'tail': ('xtail', 'ytail'),
    'ctr': ('xctr', 'yctr'),
    'mid': ('xmid', 'ymid')
}

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to uppercase with hyphen format.
    
    Args:
        data: DataFrame with position columns
        
    Returns:
        DataFrame with normalized column names
    """
    data = data.copy()
    
    # Define column mappings
    column_mapping = {
        'xhead': 'X-Head', 'yhead': 'Y-Head',
        'xtail': 'X-Tail', 'ytail': 'Y-Tail',
        'xctr': 'X-Centroid', 'yctr': 'Y-Centroid',
        'xmid': 'X-Midpoint', 'ymid': 'Y-Midpoint'
    }
    
    # Only rename columns that exist
    existing_cols = {k: v for k, v in column_mapping.items() if k in data.columns}
    if existing_cols:
        data = data.rename(columns=existing_cols)
    
    return data

def get_speed_from_df(data: pd.DataFrame, key: str, fps: int) -> np.ndarray:
    """Calculate speed from position data.
    
    Args:
        data: DataFrame with position columns
        key: Point to calculate speed for ('head', 'tail', etc.)
        fps: Frame rate
        
    Returns:
        Speed values
    """
    if key not in POSDICT:
        raise ValueError(f"Invalid key: {key}")
        
    x_col, y_col = POSDICT[key]
    dx = np.diff(data[x_col].values)
    dy = np.diff(data[y_col].values)
    speed = np.sqrt(dx*dx + dy*dy) * fps
    return np.concatenate([[0], speed])  # Add 0 for first frame

def perfectly_overlapping(data: pd.DataFrame, pt1: str, pt2: str, where: bool = False) -> np.ndarray:
    """Find frames where two points perfectly overlap.
    
    Args:
        data: DataFrame with position columns
        pt1: First point ('head', 'tail', etc.)
        pt2: Second point
        where: Return indices where True if True, else boolean array
        
    Returns:
        Boolean array or indices where points overlap
    """
    if pt1 not in POSDICT or pt2 not in POSDICT:
        raise ValueError(f"Invalid points: {pt1}, {pt2}")
        
    x1, y1 = POSDICT[pt1]
    x2, y2 = POSDICT[pt2]
    
    overlaps = (data[x1] == data[x2]) & (data[y1] == data[y2])
    return np.where(overlaps)[0] if where else overlaps 

def get_delta_in_frame(data: pd.DataFrame, pt1: str, pt2: str) -> np.ndarray:
    """Calculate distance between two points in each frame.
    
    Args:
        data: DataFrame with position columns
        pt1: First point ('head', 'tail', etc.)
        pt2: Second point
        
    Returns:
        Array of distances
    """
    if pt1 not in POSDICT or pt2 not in POSDICT:
        raise ValueError(f"Invalid points: {pt1}, {pt2}")
        
    x1, y1 = POSDICT[pt1]
    x2, y2 = POSDICT[pt2]
    
    dx = data[x1] - data[x2]
    dy = data[y1] - data[y2]
    return np.sqrt(dx*dx + dy*dy)

def get_delta_between_frames(data: pd.DataFrame, key: str, fps: int = 1) -> np.ndarray:
    """Calculate distance between consecutive frames for a point.
    
    Args:
        data: DataFrame with position columns
        key: Point to calculate for ('head', 'tail', etc.)
        fps: Frame rate
        
    Returns:
        Array of distances
    """
    if key not in POSDICT:
        raise ValueError(f"Invalid key: {key}")
        
    x_col, y_col = POSDICT[key]
    x = data[x_col].values
    y = data[y_col].values
    
    dx = np.diff(x)
    dy = np.diff(y)
    delta = np.sqrt(dx*dx + dy*dy)
    if isinstance(fps, (int, float)):
        delta *= fps
    return np.concatenate([[0], delta])  # Add 0 for first frame

def get_head_angle(data: pd.DataFrame) -> np.ndarray:
    """Calculate head angle relative to body axis.
    
    Args:
        data: DataFrame with position columns
        
    Returns:
        Array of angles in radians
    """
    # Get head and tail positions
    x_head = data['X-Head'].values
    y_head = data['Y-Head'].values
    x_tail = data['X-Tail'].values
    y_tail = data['Y-Tail'].values
    
    # Calculate vectors
    dx = x_head - x_tail
    dy = y_head - y_tail
    
    # Calculate angles
    angles = np.arctan2(dy, dx)
    return angles 

def get_ht_cross_sign(data: pd.DataFrame) -> np.ndarray:
    """Calculate the cross sign between head and tail.
    
    Args:
        data: DataFrame with position columns
        
    Returns:
        Array of cross signs
    """
    # Get head and tail positions
    x_head = data['X-Head'].values
    y_head = data['Y-Head'].values
    x_tail = data['X-Tail'].values
    y_tail = data['Y-Tail'].values
    
    # Calculate cross sign
    cross_sign = (x_head - x_tail) * (y_head + y_tail)
    return cross_sign 

def get_orientation(data: pd.DataFrame, ref: list[float] = [1,0],
                   halfAngle: bool = True, fromMotion: bool = False, head: bool = False) -> np.ndarray:
    """Calculate the global body angle in radians relative to a reference vector.
    
    Args:
        data: DataFrame with position columns
        ref: Reference vector [x, y]
        halfAngle: If true, return the half-angle on [0,pi]
        fromMotion: Use centroid velocity to estimate angle
        head: Use midpoint-head vector instead of tail-midpoint vector
        
    Returns:
        Array of angles in radians
    """
    # Get orientation vectors
    if fromMotion:
        v = get_motion_vector(data)
    else:
        if head:
            v = get_vectors_between(data, 'mid', 'head')
        else:
            v = get_vectors_between(data, 'tail', 'mid')
    
    # Calculate angles
    angles = []
    for i in range(len(data)):
        # Calculate angle between reference vector and orientation vector
        dot = np.dot(ref, v[i])
        norms = np.linalg.norm(ref) * np.linalg.norm(v[i])
        
        if norms == 0:
            angles.append(0)
            continue
            
        cos_angle = dot / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if halfAngle:
            angle /= 2
            
        angles.append(angle)
    
    return np.array(angles) 
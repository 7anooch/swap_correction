from .metrics import MovementMetrics

__all__ = ['MovementMetrics']

"""Metrics module for calculating movement metrics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Position column mappings
POSDICT = {
    'head': ['X-Head', 'Y-Head'],
    'tail': ['X-Tail', 'Y-Tail'],
    'mid': ['X-Midpoint', 'Y-Midpoint'],
    'ctr': ['X-Centroid', 'Y-Centroid']
}

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
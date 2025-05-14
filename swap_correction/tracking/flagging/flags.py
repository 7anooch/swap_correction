"""
Functions for detecting potential head-tail swaps and tracking errors.
"""

import numpy as np
import pandas as pd
from swap_correction import utils, metrics

def flag_all_swaps(data: pd.DataFrame, fps: float, debug: bool = False) -> np.ndarray:
    """Flag all potential head-tail swaps using multiple detection methods."""
    # Get all flags
    olap = flag_overlaps(data, debug=debug)
    sr = flag_sign_reversals(data, debug=debug)
    dm = flag_delta_mismatches(data, debug=debug)
    cosr = flag_overlap_sign_reversals(data, debug=debug)
    comm = flag_overlap_minimum_mismatches(data, debug=debug)
    
    # Filter out overlaps
    filt = utils.merge(olap, olap+1)
    dm = utils.filter_array(dm, filt)
    sr = utils.filter_array(sr, filt)
    
    # Combine all flags
    all_flags = np.unique(np.concatenate([olap, sr, dm, cosr, comm]))
    return all_flags

def flag_discontinuities(data: pd.DataFrame, point: str, fps: float, debug: bool = False) -> np.ndarray:
    """Flag frames where there are discontinuities in position data."""
    # Get position columns for the point
    pos_cols = metrics.POSDICT[point]
    
    # Calculate velocity
    vel = np.zeros((len(data), 2))
    for i, col in enumerate(pos_cols):
        vel[:, i] = np.gradient(data[col].to_numpy()) * fps
    
    # Calculate speed
    speed = np.sqrt(np.sum(vel**2, axis=1))
    
    # Flag frames where speed exceeds threshold
    threshold = 5.0  # mm/s
    flags = np.where(speed > threshold)[0]
    
    if debug:
        print(f"Found {len(flags)} discontinuities in {point} data")
    
    return flags

def flag_delta_mismatches(data: pd.DataFrame, debug: bool = False) -> np.ndarray:
    """Flag frames where head-tail delta direction changes abruptly."""
    # Get deltas
    deltas = get_all_deltas(data)
    
    # Calculate delta direction changes
    delta_changes = np.diff(deltas, axis=0)
    change_magnitude = np.sqrt(np.sum(delta_changes**2, axis=1))
    
    # Flag frames where change magnitude exceeds threshold
    threshold = 2.0  # mm
    flags = np.where(change_magnitude > threshold)[0] + 1  # +1 because of diff
    
    if debug:
        print(f"Found {len(flags)} delta mismatches")
    
    return flags

def flag_sign_reversals(data: pd.DataFrame, debug: bool = False) -> np.ndarray:
    """Flag frames where head-tail delta direction reverses."""
    # Get deltas
    deltas = get_all_deltas(data)
    
    # Calculate dot product between consecutive deltas
    dot_products = np.sum(deltas[1:] * deltas[:-1], axis=1)
    
    # Flag frames where dot product is negative (reversal)
    flags = np.where(dot_products < 0)[0] + 1  # +1 because of diff
    
    if debug:
        print(f"Found {len(flags)} sign reversals")
    
    return flags

def flag_overlaps(data: pd.DataFrame, debug: bool = False) -> np.ndarray:
    """Flag frames where head and tail positions overlap."""
    # Get head-tail separation
    dist = metrics.get_delta_in_frame(data, 'head', 'tail')
    
    # Flag frames where separation is below threshold
    threshold = 0.5  # mm
    flags = np.where(dist < threshold)[0]
    
    if debug:
        print(f"Found {len(flags)} overlaps")
    
    return flags

def flag_overlap_sign_reversals(data: pd.DataFrame, debug: bool = False) -> np.ndarray:
    """Flag frames where head-tail delta direction reverses during overlaps."""
    # Get overlap frames
    olap = flag_overlaps(data, debug=debug)
    
    # Get deltas
    deltas = get_all_deltas(data)
    
    # Calculate dot product between consecutive deltas in overlap regions
    dot_products = np.sum(deltas[olap[1:]] * deltas[olap[:-1]], axis=1)
    
    # Flag frames where dot product is negative (reversal)
    flags = olap[np.where(dot_products < 0)[0] + 1]  # +1 because of diff
    
    if debug:
        print(f"Found {len(flags)} overlap sign reversals")
    
    return flags

def get_overlap_edges(overlaps: np.ndarray) -> tuple:
    """Get the start and end frames of overlap segments."""
    if len(overlaps) == 0:
        return np.array([]), np.array([])
    
    # Find where gaps in overlap frames occur
    gaps = np.where(np.diff(overlaps) > 1)[0]
    
    # Get start and end frames
    starts = np.concatenate([[overlaps[0]], overlaps[gaps + 1]])
    ends = np.concatenate([overlaps[gaps], [overlaps[-1]]])
    
    return starts, ends

def get_all_overlap_edges(data: pd.DataFrame, debug: bool = False) -> tuple:
    """Get all overlap segment edges in the data."""
    overlaps = flag_overlaps(data, debug=debug)
    return get_overlap_edges(overlaps)

def get_all_deltas(data: pd.DataFrame) -> np.ndarray:
    """Get all head-tail delta vectors."""
    xdelta = data['xhead'].to_numpy() - data['xtail'].to_numpy()
    ydelta = data['yhead'].to_numpy() - data['ytail'].to_numpy()
    return np.column_stack((xdelta, ydelta))

def flag_overlap_minimum_mismatches(data: pd.DataFrame, debug: bool = False) -> np.ndarray:
    """Flag frames where minimum head-tail separation occurs at unexpected times."""
    # Get overlap segments
    starts, ends = get_all_overlap_edges(data, debug=debug)
    
    # Get head-tail separation
    dist = metrics.get_delta_in_frame(data, 'head', 'tail')
    
    # For each overlap segment, find where minimum separation occurs
    flags = []
    for start, end in zip(starts, ends):
        segment = dist[start:end+1]
        min_idx = np.argmin(segment)
        
        # Flag if minimum is not at start or end
        if min_idx != 0 and min_idx != len(segment) - 1:
            flags.append(start + min_idx)
    
    if debug:
        print(f"Found {len(flags)} overlap minimum mismatches")
    
    return np.array(flags) 
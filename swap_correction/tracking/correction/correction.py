"""
Core functions for correcting head-tail swaps and tracking errors.
"""

import numpy as np
import pandas as pd
from swap_correction import utils, metrics
from swap_correction.tracking import flagging
import logging
import copy

logging.basicConfig(level=logging.INFO)  # or logging.DEBUG for more detail

def tracking_correction(data: pd.DataFrame, fps: float,
                       swapCorrection: bool = True, validate: bool = False,
                       removeErrors: bool = True, interp: bool = True,
                       debug: bool = False) -> pd.DataFrame:
    """
    Main function for correcting tracking data.
    Clean pipeline: just calls helpers in sequence.
    """
    data = data.copy()
    if removeErrors:
        data = remove_edge_frames(data)
    if swapCorrection:
        data = correct_tracking_errors(data, fps, swapCorrection, validate, debug)
    if removeErrors:
        data = remove_overlaps(data, debug)
    if interp:
        data = interpolate_gaps(data, debug)
    return data

# --- Helper functions below ---
def remove_edge_frames(data: pd.DataFrame) -> pd.DataFrame:
    """Remove frames where head or tail is at the edge of the frame."""
    width = 1920  # TODO: Get from settings
    height = 1080  # TODO: Get from settings
    edge_frames = np.where(
        (data['X-Head'] <= 0) | (data['X-Head'] >= width) |
        (data['Y-Head'] <= 0) | (data['Y-Head'] >= height) |
        (data['X-Tail'] <= 0) | (data['X-Tail'] >= width) |
        (data['Y-Tail'] <= 0) | (data['Y-Tail'] >= height)
    )[0]
    edge_frames = [int(i) for i in edge_frames]
    if len(edge_frames) == 0:
        return data
    for col in utils.flatten(metrics.POSDICT.values()):
        if col in data.columns:
            data.loc[edge_frames, col] = np.nan
    return data

def correct_tracking_errors(data: pd.DataFrame, fps: float = None,
                          swapCorrection: bool = True, validate: bool = False,
                          debug: bool = False, flags: dict = None) -> pd.DataFrame:
    """Correct tracking errors in the data."""
    if flags is not None:
        data = data.copy()
        if 'tracking_errors' not in flags:
            return data
        for start, end in flags['tracking_errors']:
            for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail', 'X-Midpoint', 'Y-Midpoint', 'X-Centroid', 'Y-Centroid', 'angle']:
                if col in data.columns and start > 0 and end < len(data) - 1:
                    data.loc[start:end, col] = np.interp(
                        np.arange(start, end + 1),
                        [start - 1, end + 1],
                        [data.loc[start - 1, col], data.loc[end + 1, col]]
                    )
        return data
    # Pipeline case: correct tracking errors
    swap_frames = flagging.flag_all_swaps(data, fps, debug)
    if len(swap_frames) == 0:
        return data
    segments = utils.get_consecutive_ranges(swap_frames)
    for start, end in segments:
        data = correct_swapped_segments(data, start, end, debug)
    if validate:
        data = validate_corrected_data(data, fps, debug)
    return data

def correct_swapped_segments(data: pd.DataFrame, start: int, end: int,
                           debug: bool = False) -> pd.DataFrame:
    """Correct a segment of swapped frames."""
    if debug:
        logging.info(f"Correcting segment {start}-{end}")
    for hcol, tcol in zip(metrics.POSDICT['head'], metrics.POSDICT['tail']):
        if hcol in data.columns and tcol in data.columns:
            data.loc[start:end, hcol], data.loc[start:end, tcol] = (
                data.loc[start:end, tcol], data.loc[start:end, hcol]
            )
    # Shift angle by pi and wrap, only if 'angle' exists
    if 'angle' in data.columns:
        data.loc[start:end, 'angle'] = (data.loc[start:end, 'angle'] + np.pi) % (2 * np.pi)
    return data

def remove_overlaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Remove frames where head and tail overlap."""
    overlaps = flagging.flag_overlaps(data, debug)
    overlaps = [int(i) for i in overlaps]
    if len(overlaps) == 0:
        return data
    for col in utils.flatten(metrics.POSDICT.values()):
        if col in data.columns:
            data.loc[overlaps, col] = np.nan
    return data

def interpolate_gaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Interpolate over gaps in the data."""
    cols = utils.flatten(metrics.POSDICT.values())
    for col in cols:
        if col in data.columns:
            data[col] = data[col].interpolate(method='linear')
    return data

def validate_corrected_data(data: pd.DataFrame, fps: float,
                          debug: bool = False) -> pd.DataFrame:
    """Validate corrections by checking for remaining errors."""
    # Get all potential swap frames
    swap_frames = flagging.flag_all_swaps(data, fps, debug)
    
    if len(swap_frames) == 0:
        return data
    
    # Get segments of consecutive swap frames
    segments = utils.get_consecutive_ranges(swap_frames)
    
    # Correct each segment
    for start, end in segments:
        data = correct_swapped_segments(data, start, end, debug)
    
    return data

def correct_global_swap(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Correct a global head-tail swap."""
    # Get head-tail separation
    dist = metrics.get_delta_in_frame(data, 'head', 'tail')
    
    # If mean separation is negative, swap is needed
    if np.nanmean(dist) < 0:
        if debug:
            logging.info("Correcting global swap")
        
        # Swap head and tail columns
        for hcol, tcol in zip(metrics.POSDICT['head'], metrics.POSDICT['tail']):
            data[hcol], data[tcol] = data[tcol], data[hcol]
    
    return data

def get_swapped_segments(data: pd.DataFrame, fps: float,
                        debug: bool = False) -> list:
    """Get segments of frames that need to be swapped."""
    # Get all potential swap frames
    swap_frames = flagging.flag_all_swaps(data, fps, debug)
    
    if len(swap_frames) == 0:
        return []
    
    # Get segments of consecutive swap frames
    return utils.get_consecutive_ranges(swap_frames)

def correct_no_flags(data: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """Return data unchanged when no flags are present."""
    return data.copy()

def correct_swaps(data: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """Correct head-tail swaps in the data."""
    data = data.copy()
    if 'swaps' not in flags:
        return data
        
    for start, end in flags['swaps']:
        # Swap head and tail coordinates
        for hcol, tcol in zip(metrics.POSDICT['head'], metrics.POSDICT['tail']):
            data.loc[start:end, hcol], data.loc[start:end, tcol] = (
                data.loc[start:end, tcol], data.loc[start:end, hcol]
            )
        # Shift angle by pi and wrap
        data.loc[start:end, 'angle'] = (data.loc[start:end, 'angle'] + np.pi) % (2 * np.pi)
    
    return data

def correct_both_flags(data: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """Correct both swaps and tracking errors."""
    # First correct swaps
    data = correct_swaps(data, flags)
    # Then correct tracking errors
    data = correct_tracking_errors(data, flags)
    return data

def correct_empty_df(data: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """Handle empty DataFrame case."""
    return data.copy() 
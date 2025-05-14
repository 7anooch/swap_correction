"""
Core functions for correcting head-tail swaps and tracking errors.
"""

import numpy as np
import pandas as pd
from swap_correction import utils, metrics
from swap_correction.tracking import flagging
from swap_correction.tracking import filtering

def tracking_correction(data: pd.DataFrame, fps: float, filterData: bool = False,
                       swapCorrection: bool = True, validate: bool = False,
                       removeErrors: bool = True, interp: bool = True,
                       debug: bool = False) -> pd.DataFrame:
    """
    Main function for correcting tracking data.
    
    Args:
        data: Raw tracking data
        fps: Frames per second
        filterData: Whether to filter the data
        swapCorrection: Whether to correct head-tail swaps
        validate: Whether to validate corrections
        removeErrors: Whether to remove error frames
        interp: Whether to interpolate over gaps
        debug: Whether to print debug messages
    
    Returns:
        Corrected tracking data
    """
    # Remove edge frames
    data = remove_edge_frames(data)
    
    # Filter data if requested
    if filterData:
        data = filtering.filter_data(data)
    
    # Correct tracking errors
    data = correct_tracking_errors(data, fps, swapCorrection, validate, debug)
    
    # Remove overlaps if requested
    if removeErrors:
        data = remove_overlaps(data, debug)
    
    # Interpolate over gaps if requested
    if interp:
        data = interpolate_gaps(data, debug)
    
    return data

def remove_edge_frames(data: pd.DataFrame) -> pd.DataFrame:
    """Remove frames where head or tail is at the edge of the frame."""
    # Get frame dimensions from settings
    width = 1920  # TODO: Get from settings
    height = 1080  # TODO: Get from settings
    
    # Find frames where head or tail is at edge
    edge_frames = np.where(
        (data['xhead'] <= 0) | (data['xhead'] >= width) |
        (data['yhead'] <= 0) | (data['yhead'] >= height) |
        (data['xtail'] <= 0) | (data['xtail'] >= width) |
        (data['ytail'] <= 0) | (data['ytail'] >= height)
    )[0]
    
    # Set these frames to NaN
    for col in utils.flatten(metrics.POSDICT.values()):
        data.loc[edge_frames, col] = np.nan
    
    return data

def correct_tracking_errors(data: pd.DataFrame, fps: float,
                          swapCorrection: bool = True, validate: bool = False,
                          debug: bool = False) -> pd.DataFrame:
    """Correct tracking errors in the data."""
    # Get all potential swap frames
    swap_frames = flagging.flag_all_swaps(data, fps, debug)
    
    if len(swap_frames) == 0:
        return data
    
    # Get segments of consecutive swap frames
    segments = utils.get_consecutive_ranges(swap_frames)
    
    # Correct each segment
    for start, end in segments:
        data = correct_swapped_segments(data, start, end, debug)
    
    # Validate corrections if requested
    if validate:
        data = validate_corrected_data(data, fps, debug)
    
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

def remove_overlaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Remove frames where head and tail overlap."""
    # Get overlap frames
    overlaps = flagging.flag_overlaps(data, debug)
    
    # Set these frames to NaN
    for col in utils.flatten(metrics.POSDICT.values()):
        data.loc[overlaps, col] = np.nan
    
    return data

def interpolate_gaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Interpolate over gaps in the data."""
    # Get columns to interpolate
    cols = utils.flatten(metrics.POSDICT.values())
    
    # Interpolate each column
    for col in cols:
        data[col] = data[col].interpolate(method='linear')
    
    return data

def correct_global_swap(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Correct a global head-tail swap."""
    # Get head-tail separation
    dist = metrics.get_delta_in_frame(data, 'head', 'tail')
    
    # If mean separation is negative, swap is needed
    if np.nanmean(dist) < 0:
        if debug:
            print("Correcting global swap")
        
        # Swap head and tail columns
        for hcol, tcol in zip(metrics.POSDICT['head'], metrics.POSDICT['tail']):
            data[hcol], data[tcol] = data[tcol], data[hcol]
    
    return data

def correct_swapped_segments(data: pd.DataFrame, start: int, end: int,
                           debug: bool = False) -> pd.DataFrame:
    """Correct a segment of swapped frames."""
    if debug:
        print(f"Correcting segment {start}-{end}")
    
    # Swap head and tail columns for the segment
    for hcol, tcol in zip(metrics.POSDICT['head'], metrics.POSDICT['tail']):
        data.loc[start:end, hcol], data.loc[start:end, tcol] = (
            data.loc[start:end, tcol], data.loc[start:end, hcol]
        )
    
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
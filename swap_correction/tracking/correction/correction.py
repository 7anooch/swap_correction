"""
Core functions for correcting head-tail swaps and tracking errors.
"""

import numpy as np
import pandas as pd
from swap_correction import utils, metrics
from swap_correction.tracking import flagging
import logging
import copy

logging.basicConfig(level=logging.DEBUG)  # or logging.DEBUG for more detail

def tracking_correction(data: pd.DataFrame, fps: float, resolution: str,
                        validate: bool = False,
                        removeErrors: bool = True, interp: bool = True,
                        debug: bool = False) -> pd.DataFrame:
    """
    Main function for correcting tracking data.
    Clean pipeline: just calls helpers in sequence.
    """

    data = data.copy()
    if removeErrors:
        data = remove_edge_frames(data, resolution)

    data = correct_tracking_errors(data, fps, validate, debug)
    if removeErrors:
        data = remove_overlaps(data, debug)
    if interp:
        data = interpolate_gaps(data, debug)
    return data

# --- Helper functions below ---
def remove_edge_frames(data: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Remove frames where head or tail is at the edge of the frame."""

    res_x, res_y = resolution.split('x')
    width = int(res_x)
    height = int(res_y)

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


## def correct_tracking_errors(rawData : pd.DataFrame, debug : bool = False) -> pd.DataFrame:
    # """
    # Remove tracking errors and correct head-tail swaps
    # TODO: address errors where centroid overlaps head / tail

    # rawData: dataFrame with imported piVR data
    # debug: print debug messages
    # """
    # data = rawData.copy()

    # # flag frames where swaps appear to occur
    # swaps = flag_all_swaps(rawData,separate=False,debug=debug)

    # # correct head-tail swaps
    # segments = utils.indices_to_segments(swaps,nframes=data.shape[0],addBounds=True,inclusive=True,alternating=True)
    # data = correct_swapped_segments(data,segments,debug=debug)
    # data = correct_global_swap(data,debug=debug)
    # return data

def correct_tracking_errors(data: pd.DataFrame, fps: float, validate: bool = False, debug: bool = False) -> pd.DataFrame:
    """
    Correct tracking errors in the data using the provided flags or by detecting them.
    
    Args:
        data: DataFrame containing tracking data with PiVR column names
        fps: Frame rate (required for some flagging operations)
        validate: Whether to validate corrections
        debug: Whether to print debug messages
        
    Returns:
        DataFrame with corrected tracking data
    """
    data = data.copy()
    flags = []
    
    # Auto-detect swaps
    swap_flags = flagging.flag_all_swaps(data, fps=fps, debug=debug)
    print(swap_flags.astype(int))
    print(len(swap_flags))
    if len(swap_flags) > 0:
        segments = utils.get_consecutive_ranges(swap_flags)
        flags.extend(segments)

    print(len(flags))
    
    # # Process each error region
    # for start, end in flags:
    #     # Set relevant columns to NaN in the main DataFrame
    #     cols = utils.flatten(metrics.POSDICT.values())
    #     for col in cols:
    #         if col in data.columns:
    #             data.loc[start:end, col] = np.nan
    
    # Process swap regions
    for start, end in flags:
        # logging.debug(f'Correcting swap {start} to {end}')
        data = correct_swapped_segments(data, segments=np.array([[start, end]]))
    
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
    """Interpolate over gaps in the data, including angle if present."""
    cols = utils.flatten(metrics.POSDICT.values())
    for col in cols:
        if col in data.columns:
            data[col] = data[col].interpolate(method='linear')
    # Interpolate angle as well, if present
    if 'angle' in data.columns:
        data['angle'] = pd.to_numeric(data['angle'], errors='coerce').interpolate(method='linear')
    return data

def validate_corrected_data(data: pd.DataFrame, fps: float, debug: bool = False) -> pd.DataFrame:
    """Validate corrections by checking for remaining errors."""
    swap_frames = flagging.flag_all_swaps(data, fps, debug)
    if len(swap_frames) == 0:
        return data
    segments = utils.get_consecutive_ranges(swap_frames)
    for start, end in segments:
        data = correct_swapped_segments(data, segments=np.array([[start, end]]), debug=debug)
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

def correct_swapped_segments(data: pd.DataFrame, segments: np.ndarray = None, start: int = None, end: int = None, debug: bool = False) -> pd.DataFrame:
    '''
    Swap head and tail in the given segments
    NOTE: end index is assumed to be inclusive
    Args:
        data: DataFrame with raw position data using PiVR column names
        segments: N x 2 array of start and end frames (inclusive) of swapped segments
        start: Single start frame (alternative to segments)
        end: Single end frame (alternative to segments)
        debug: print debug messages
    Returns:
        DataFrame with corrected head-tail positions
    '''
    data = data.copy()
    # Accept (data, start, end, debug) as well as (data, segments, ...)
    if segments is None and start is not None and end is not None:
        segments = np.array([[start, end]])
    elif segments is None:
        return data
    # If segments is a tuple or list of two ints, convert to 2D array
    if isinstance(segments, (tuple, list)) and len(segments) == 2 and all(isinstance(x, (int, np.integer, np.floating, float)) for x in segments):
        segments = np.array([segments], dtype=int)
    # If segments is a single pair (not 2D), convert
    if isinstance(segments, np.ndarray) and segments.ndim == 1 and len(segments) == 2:
        segments = np.array([segments], dtype=int)
    # get list of swapped frames
    frames = []
    for seg in segments:
        a, b = int(seg[0]), int(seg[1])
        frames.extend(np.arange(a, b+1))
    # correct swapped frames
    for i in frames:
        xh = data.at[i, 'X-Head']
        yh = data.at[i, 'Y-Head']
        xt = data.at[i, 'X-Tail']
        yt = data.at[i, 'Y-Tail']
        data.loc[i, 'X-Head'] = xt
        data.loc[i, 'Y-Head'] = yt
        data.loc[i, 'X-Tail'] = xh
        data.loc[i, 'Y-Tail'] = yh
    if debug:
        print('Swapped Segments: {}'.format(segments))
        print('Frames corrected:', len(frames))
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
    data = correct_swaps(data, flags)
    data = correct_tracking_errors(data, flags)
    return data

def correct_empty_df(data: pd.DataFrame, flags: dict) -> pd.DataFrame:
    """Handle empty DataFrame case."""
    return data.copy() 
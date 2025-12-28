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
            filterData : bool = False, debug : bool = False) -> pd.DataFrame:
    """
    Apply tracking corrections and filtering to raw data

    data (DataFrame): raw position data
    fps (int): frame rate
    swapCorrection (bool): correct head-tail swaps
    interp (bool): interpolate over position data in bad frames (of correctTracking)
    validate (bool): use assumption of forward movement to catch remaining swaps
    filterData (bool): apply a Savitzky-Golay filter to the position data
    debug (bool): print debug messages
    """
    # correct tracking errors
    data = remove_edge_frames(data,debug=debug)
    if swapCorrection : data = correct_tracking_errors(data,debug=debug)
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
    
    # If overall consistency is very low, this might be a global swap
    # But we'll let correct_global_swap handle that, so we skip it here
    if overall_consistency < global_threshold:
        if debug:
            print(f'Overall consistency < {global_threshold}, likely global swap (handled separately)')
        return np.empty((0, 2), dtype=int)
    
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
        
        # If consistency is low, this window likely has swaps
        if window_consistency < consistency_threshold:
            swapped_windows.append((start, end))
            if debug:
                print(f'Low consistency window [{start}:{end}]: {window_consistency:.3f}')
    
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


def correct_tracking_errors(rawData : pd.DataFrame, debug : bool = False) -> pd.DataFrame:
    """
    Remove tracking errors and correct head-tail swaps
    TODO: address errors where centroid overlaps head / tail

    rawData: dataFrame with imported piVR data
    fps: frame rate (needed for segment-based detection)
    debug: print debug messages
    """
    data = rawData.copy()

    # Flag frames where swaps appear to occur (frame-by-frame detection)
    swaps = flag_all_swaps(data,separate=False,debug=debug)

    # correct remaining head-tail swaps in segments
    segments = utils.indices_to_segments(swaps,nframes=data.shape[0],addBounds=True,inclusive=True,alternating=True)
    data = correct_swapped_segments(data,segments,debug=debug)
    
    # Apply original simple global swap detection (baseline)
    data = correct_global_swap_simple(data, debug=debug)
    return data


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
    
    # Step 1: Expand around each flagged frame
    expanded = set()
    for frame in flagged_frames:
        start = max(0, frame - expand_window)
        end = min(nframes - 1, frame + expand_window)
        expanded.update(range(start, end + 1))
    
    expanded = np.array(sorted(expanded))
    
    if debug:
        print(f'After expansion: {len(flagged_frames)} → {len(expanded)} frames')
    
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
            

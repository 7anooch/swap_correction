import numpy as np
import pandas as pd
import scipy as sp
from pivr_tools import utils, metrics
from pivr_tools.kalman_filter import KalmanFilter

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


def correct_tracking_errors(rawData : pd.DataFrame, debug : bool = False) -> pd.DataFrame:
    """
    Remove tracking errors and correct head-tail swaps
    TODO: address errors where centroid overlaps head / tail

    rawData: dataFrame with imported piVR data
    debug: print debug messages
    """
    data = rawData.copy()

    # flag frames where swaps appear to occur
    swaps = flag_all_swaps(rawData,separate=False,debug=debug)

    # correct head-tail swaps
    segments = utils.indices_to_segments(swaps,nframes=data.shape[0],addBounds=True,inclusive=True,alternating=True)
    data = correct_swapped_segments(data,segments,debug=debug)
    data = correct_global_swap(data,debug=debug)
    return data


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

def correct_global_swap(rawData : pd.DataFrame, debug : bool = False) -> pd.DataFrame:
    """
    Detect and correct global head-tail swap caused by misidentification on first frame
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
            avgang = _get_alignment_angles(filt,segs)
            nvals = utils.segment_lengths(segs)
            if debug : print('Mean Run Angles: {}'.format(np.rad2deg(avgang)))
            avgang /= (np.pi/4) # normalize
            flag = _flag_segment_metrics(avgang,nvals,thresh,minFrames)
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


def _get_alignment_angles(data : pd.DataFrame, segs : np.ndarray) -> np.ndarray:
    tvec = metrics.get_motion_vector(data,'tail')
    tmvec = metrics.get_orientation_vectors(data,head=False)
    tailAlignment = [utils.get_angle(tmvec[i],tvec[i],halfAngle=True) for i in range(data.shape[0])]
    tailAlignSeg = utils.metrics_by_segment(tailAlignment,segs) # mean, med, std
    return tailAlignSeg[:,0] # mean


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


def filter_butter(rawData : pd.DataFrame, fps : int, cutoff : float = 0, order : int = 5) -> pd.DataFrame:
    '''Apply a Butterworth low-pass filter to the position data'''
    data = rawData.copy()

    b, a = sp.signal.butter(order,cutoff,fs=fps,btype='low',analog=False)
    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.signal.lfilter(b, a, data[col].to_numpy())
    
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


def filter_med_gaussian(rawData : pd.DataFrame, sigma : float = 3, win : int = 5) -> pd.DataFrame:
    '''Apply Median and Gaussian filter to the position data'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        med = sp.ndimage.median_filter(data[col].to_numpy(),win)
        gauss = sp.ndimage.gaussian_filter1d(med,sigma)
        data[col] = gauss
    
    return data


def filter_median(rawData : pd.DataFrame, win : int = 5) -> pd.DataFrame:
    '''Filter the position data using a rolling median'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.ndimage.median_filter(data[col].to_numpy(),win)
    
    return data


def filter_mean(rawData : pd.DataFrame, win : int = 5) -> pd.DataFrame:
    '''Filter the position data using a rolling mean'''
    data = rawData.copy()

    for col in utils.flatten(metrics.POSDICT.values()):
        data[col] = sp.ndimage.uniform_filter(data[col].to_numpy(),win)
    
    return data


def filter_kalman(rawData : pd.DataFrame, fps : int, derivatives : int = 2, **kwargs) -> pd.DataFrame:
    '''
    Apply a Kalman filter to the position data

    rawData: unfiltered position data
    derivatives: number of derivativesto use in estimations
    dt: time step
    '''
    data = rawData.copy()
    dt = 1/fps # time step
    ndim = 2 # number of dimensions; TODO: fully generalize?

    kfilter = KalmanFilter(dt,ndim,derivatives,**kwargs) # set up Kalman filter
    cols = ['head','tail','mid','ctr'] # partial keys of columns of interest
    for col in cols:
        # extract data
        xcol = 'x'+col
        ycol = 'y'+col

        start = max(data[xcol].first_valid_index(),data[ycol].first_valid_index())
        end = min(data[xcol].last_valid_index(),data[ycol].last_valid_index()) + 1

        vec = data.loc[start:end,[xcol,ycol]].to_numpy() # extract data, omitting NaNs at edges

        # create and run filter
        # TODO: ensure NaNs mtch between x, y vectors
        data.loc[start:end,[xcol,ycol]] = kfilter.filter(vec)
    return data
            

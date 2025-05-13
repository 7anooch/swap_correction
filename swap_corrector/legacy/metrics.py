import numpy as np
import pandas as pd
from swap_corrector.legacy import utils


# Position column mappings
POSDICT = {
    'head' : ('xhead','yhead'),
    'tail' : ('xtail','ytail'),
    'ctr' : ('xctr','yctr'),
    'mid' : ('xmid','ymid')
}

# Alternative column names (for backward compatibility)
ALT_POSDICT = {
    'head': ('X-Head', 'Y-Head'),
    'tail': ('X-Tail', 'Y-Tail'),
    'ctr': ('X-Centroid', 'Y-Centroid'),
    'mid': ('X-Midpoint', 'Y-Midpoint')
}


def vectors_from_key(data : pd.DataFrame, key : str, transpose : bool = True) -> np.ndarray:
    """
    Get x, y position vectors from the given dataframe using the given key

    data: dataframe containing position data
    key: key for feature of interest
    transpose : apply transpose such that x, y vectors can be unpacked directly
    """
    if key not in POSDICT:
        raise ValueError(f"Invalid key: {key}")
        
    # Try lowercase columns first
    kx, ky = POSDICT[key]
    if kx in data.columns and ky in data.columns:
        vec = data.loc[:, [kx, ky]].to_numpy()
        return vec.T if transpose else vec
        
    # Try uppercase columns with hyphens
    kx, ky = ALT_POSDICT[key]
    if kx in data.columns and ky in data.columns:
        vec = data.loc[:, [kx, ky]].to_numpy()
        return vec.T if transpose else vec
        
    raise ValueError(f"Could not find columns for key {key} in data")


def detect_stops_via_threshold(spd : np.ndarray, minSpeed : float,
                    minTime : float, fps : int) -> tuple[np.ndarray]:
    """
    Returns an binary array indicating the state of the larva across time
    and an array containing activity durations on the first frame of the activity
    0: run
    1: stop
    """
    npts = len(spd)
    stopWidth = minTime * fps # minimum number of frames for stop
    isStop = spd < minSpeed # boolean array of stops (True) and runs (False)

    stops = np.zeros(npts) # binary array indicating stop frames
    times = np.zeros(npts) # record activity durations on start frames
    runStart = 0 # track frame where current activity begins
    stopStart = 0
    state = isStop[0] # if true, is stopped
    delta = -1 # duration of current activity

    for i in range(npts):
        
        if state != isStop[i]:
            # end a possible stop
            if state:
                # confirm stop if of sufficient duration
                if delta > stopWidth:
                    stops[stops == 2] = 1
                    times[stopStart] += delta
                    runStart = i # mark the beginning of a run
                
                # retroactively label stop as run if too short
                else:
                    stops[stops == 2] = 0
                    times[runStart] += delta # add duration onto previous run
            
            # tentatively end a run
            else:
                times[runStart] += delta
                stopStart = i # mark the beginning of a possible stop
            
            # update state
            state = isStop[i]
            delta = -1
        
        # update activity duration
        delta += 1
        
        # flag possible stop frame
        if isStop[i]:
            stops[i] = 2
    
    stops[stops == 2] = 0 # clear unconfirmed stops
    durations = times / fps # convert durations from frames to seconds
    return (stops, durations)


# ----- DataFrame Shortcuts -----

def get_df_bounds(dfs : list[pd.DataFrame], cols : list[str], buffer : float = 0.05) -> tuple[float]:
    '''
    Get bounds (extrema) of data in given columns across all given dataframes with a buffer

    dfs: list of DataFrames
    cols: list of columns to extract data ranges from
    buffer: amount of padding to add to ranges
    '''
    a = np.min(list([np.nanmin(df[cols].min(axis=0,skipna=True)) for df in dfs]))
    b = np.max(list([np.nanmax(df[cols].max(axis=0,skipna=True)) for df in dfs]))
    a -= (b - a) * buffer
    b += (b - a) * buffer
    return (a,b)


def get_speed_from_df(data : pd.DataFrame, key : str, fps : int = 1, npoints : int = 2) -> np.ndarray:
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
        
    # Try lowercase columns first
    kx, ky = POSDICT[key]
    if kx in data.columns and ky in data.columns:
        x = data[kx].values
        y = data[ky].values
    else:
        # Try uppercase columns with hyphens
        kx, ky = ALT_POSDICT[key]
        if kx in data.columns and ky in data.columns:
            x = data[kx].values
            y = data[ky].values
        else:
            raise ValueError(f"Could not find columns for key {key} in data")
    
    dx = np.diff(x)
    dy = np.diff(y)
    speed = np.sqrt(dx*dx + dy*dy) * fps
    return np.concatenate([[0], speed])  # Add 0 for first frame


def get_dist_from_df(data : pd.DataFrame, key : str, origin : tuple[float] = (0,0)) -> np.ndarray:
    '''get distance from source based on x, y position data in dataframe'''
    x, y = vectors_from_key(data,key)
    return utils.get_distance(x, y, origin)


def get_event_vector(data : pd.DataFrame, col = 'duration', stop : bool = True) -> np.ndarray:
    """Get a vector of values from the given column when the given activity begins"""
    i = 1 if stop else 0
    dur = data[col][(data['duration'] > 0) & (data['stop'] == i)]
    return dur


# ----- Angle Calculations -----

def get_head_angle(data : pd.DataFrame, halfAngle : bool = False) -> np.ndarray:
    """Calculate head angle relative to body axis.
    
    Args:
        data: DataFrame with position columns
        
    Returns:
        Array of angles in radians
    """
    # Try lowercase columns first
    if all(col in data.columns for col in ['xhead', 'yhead', 'xtail', 'ytail']):
        x_head = data['xhead'].values
        y_head = data['yhead'].values
        x_tail = data['xtail'].values
        y_tail = data['ytail'].values
    else:
        # Try uppercase columns with hyphens
        if all(col in data.columns for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']):
            x_head = data['X-Head'].values
            y_head = data['Y-Head'].values
            x_tail = data['X-Tail'].values
            y_tail = data['Y-Tail'].values
        else:
            raise ValueError("Could not find head/tail position columns in data")
    
    # Calculate vectors
    dx = x_head - x_tail
    dy = y_head - y_tail
    
    # Calculate angles
    angles = np.arctan2(dy, dx)
    return angles


def get_orientation(data : pd.DataFrame, ref : list[float] = [1,0],
            halfAngle : bool = True, fromMotion : bool = False, head : bool = False) -> np.ndarray:
    '''
    Returns the global body angle in radians relative to a reference vector
    data (pd.DataFrame): source dataframe
    ref (list[float]): reference vector
    halfAngle (bool): if true, return the half-angle on [0,pi]; otherwise, return the full angle on [-pi,pi]
    fromMotion (bool): use the velocity of the centroid to estimate angle
    head (bool): use the midpt-head vector instead of the tail-midpt vector (global head angle)
    '''
    v = get_orientation_vectors(data,head,fromMotion)
    ba = [utils.get_angle(ref,v[i],halfAngle=halfAngle) for i in range(data.shape[0])]
    return np.array(ba)


def get_custom_orientation(data : pd.DataFrame, p1 : str = 'mid', p2 : str = 'head',
            ref : list[float] = [1,0], halfAngle : bool = True, ) -> np.ndarray:
    """
    Returns the global angle between two points of interest with respect to a reference vector

    data (pd.DataFrame): input dataframe
    ref (list[float]): reference vector
    halfAngle (bool): if true, return the half-angle on [0,pi]; otherwise, return the full angle on [-pi,pi]
    p1 (str): partial key for first point of reference
    p2 (str): partial key for second point of reference
    """
    v = get_vectors_between(data,p1,p2)
    ba = [utils.get_angle(ref,v[i],halfAngle=halfAngle) for i in range(data.shape[0])]
    return np.array(ba)
    

def get_bearing(data : pd.DataFrame, source : list[float] = [0,0],
            halfAngle : bool = True, head : bool = False, fromMotion : bool = False) -> np.ndarray:
    """
    Get the angle of the organism relative to the vector between it and the source (or global centre)
    """
    v = get_orientation_vectors(data,head,fromMotion)
    ref = [[source[0]-data.at[i,'xctr'],source[1]-data.at[i,'yctr']] for i in range(data.shape[0])]
    ba = np.array(list([utils.get_angle(ref[i],v[i],halfAngle=halfAngle) for i in range(data.shape[0])]))
    return ba


def get_ht_cross_sign(data : pd.DataFrame) -> np.ndarray:
    """Calculate the cross sign between head and tail.
    
    Args:
        data: DataFrame with position columns
        
    Returns:
        Array of cross signs
    """
    # Try lowercase columns first
    if all(col in data.columns for col in ['xhead', 'yhead', 'xtail', 'ytail']):
        x_head = data['xhead'].values
        y_head = data['yhead'].values
        x_tail = data['xtail'].values
        y_tail = data['ytail'].values
    else:
        # Try uppercase columns with hyphens
        if all(col in data.columns for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']):
            x_head = data['X-Head'].values
            y_head = data['Y-Head'].values
            x_tail = data['X-Tail'].values
            y_tail = data['Y-Tail'].values
        else:
            raise ValueError("Could not find head/tail position columns in data")
    
    # Calculate cross sign
    cross_sign = (x_head - x_tail) * (y_head + y_tail)
    return cross_sign


def get_orientation_vectors(data : pd.DataFrame, head : bool = False, fromMotion : bool = False) -> np.ndarray:
    """
    Returns the coordinates of the desired orientation vector as a numpy array of shape (n,2)
    Default: body orientation (tail-midpoint vector)

    data (pd.DataFrame): source dataframe
    head (bool): return head orientation (midpoint-head vector) instead
    fromMotion (bool): determine body orientation using centroid velocity (overrides head = True)
    """
    if fromMotion : return get_motion_vector(data)
    elif head : return get_vectors_between(data,'mid','head')
    else : return get_vectors_between(data,'tail','mid')


def get_vectors_between(data : pd.DataFrame, key1 : str, key2 : str) -> np.ndarray:
    """
    Returns the coordinates of the vectors between two points of interest

    data (pd.DataFrame): source dataframe
    k1 (str): key for first point of reference
    k2 (str): key for second point of reference
    """
    x1, y1 = vectors_from_key(data,key1)
    x2, y2 = vectors_from_key(data,key2)
    relx = x2 - x1
    rely = y2 - y1
    return np.array([relx,rely]).T


def get_motion_vector(data : pd.DataFrame, key : str = 'ctr', npoints : int = 3) -> np.ndarray:
    """
    Get coordinates of vector indicating orientation of body based on movement of centroid

    data: source dataframe
    key: key of of point to use
    npoints: nuber of values to use in numerical derivative when calculating velocity
    """
    x, y = vectors_from_key(data,key)
    relx = utils.ddt(x,npoints)
    rely = utils.ddt(y,npoints)
    return np.array([relx,rely]).T


# ----- Delta Calculations -----

def get_local_tortuosity(data : pd.DataFrame, window : float = 30, key : str = 'mid',
                   step : int = 1, fps : int | None = None) -> np.ndarray:
    '''
    Get the local tortuosity of the trajectory within a window of the given width
    Values near edges are calculated using a truncated window

    data: position data
    window: window length in frames (seconds if fps != None)
    key: feature of interest
    step: step between points used to calculate local distances (for downsampling)
    fps: frame rate (if None, window is assumed to be in frames instead of seconds)
    '''
    # get window length in frames
    win = int(window) if fps is None else int(window * fps)
    win //= step # adjust for downsampling
    hw = win // 2

    # extract position values
    x, y = vectors_from_key(data,key)

    # calculate tortuisity
    tort = np.zeros(x.size)
    for i in range(tort.size):
        # get segment limits
        a = max(0,i-hw)
        b = min(tort.size,i+hw+1)

        # downsample and calculate local distance
        xx = x[a:b:step]
        yy = y[a:b:step]
        dx = xx[1:] - xx[:xx.size-1]
        dy = yy[1:] - yy[:yy.size-1]
        delta = np.sqrt(dx**2 + dy**2)

        # calculate tortuosity
        dist = np.sum(delta) # gross distance traveled
        diff = np.sqrt((x[b-1] - x[a])**2 + (y[b-1] - y[a])**2) # net path distance
        if diff == 0 : diff = 10e-5 # prevent divide-by-zero error
        tort[i] = dist / diff
    return tort


def get_segment_tortuosity(data : pd.DataFrame, segments : np.ndarray, key : str = 'mid',
                           step : int = 1, combined : bool = False) -> np.ndarray:
    '''
    Get the tortuosity of the trajectory within the given segments

    data: position data
    segments: Nx2 array of start & end frames of segments of interest
    key: feature of interest
    step: downsampling step
    combined: use median tortuosity from each possible downsample
    '''
    # extract position values and deltas
    x, y = vectors_from_key(data,key)
    delta = get_delta_between_frames(data,key)

    # calculate tortuosity
    tort = np.zeros(segments.shape[0])
    for i, seg in enumerate(segments):
        a, b = seg

        if step == 1 or not combined:
            dist = np.sum(delta[a:b:step])
        else:
            vec = [np.sum(delta[a+j:b:step]) for j in range(step)]
            dist = np.median(vec)

        diff = np.sqrt((x[b-1] - x[a])**2 + (y[b-1] - y[a])**2)
        if diff == 0 : diff = 10e-5 # prevent divide-by-zero error

        tort[i] = dist / diff
    return tort


def get_segment_distance(data : pd.DataFrame, segments : np.ndarray, key : str = 'mid',
                           step : int = 1, combined : bool = False) -> np.ndarray:
    '''
    Get the gross distance traveled by the point of interest within the given segments

    data: position data
    segments: Nx2 array of start & end frames of segments of interest
    key: feature of interest
    step: downsampling step
    combined: use median distance from each possible downsample
    '''
    # extract position values and deltas
    delta = get_delta_between_frames(data,key)

    # calculate distance
    dist = np.zeros(segments.shape[0])
    for i, seg in enumerate(segments):
        a, b = seg

        if step == 1 or not combined:
            dist[i] = np.nansum(delta[a:b:step])
        else:
            vec = [np.nansum(delta[a+j:b:step]) for j in range(step)]
            dist[i] = np.median(vec)

    return dist


def get_delta_between_frames(data : pd.DataFrame,
            key1 : str, key2 : str | None = None, fps : int = 1) -> np.ndarray:
    """Calculate distance between consecutive frames for a point.
    
    Args:
        data: DataFrame with position columns
        key1: Point to calculate for ('head', 'tail', etc.)
        key2: Second point
        fps: Frame rate
        
    Returns:
        Array of distances
    """
    if not key2 : key2 = key1 # calculate delta for same point if nosecond point specified
    x1, y1 = vectors_from_key(data,key1)
    x2, y2 = vectors_from_key(data,key2)
    npts = data.shape[0]

    dx = x2[1:] - x1[:npts-1]
    dy = y2[1:] - y1[:npts-1]
    delta = np.sqrt(dx**2 + dy**2)
    if isinstance(fps, (int, float)):
        delta *= fps
    return np.concatenate([[0], delta])  # Add 0 for first frame


def get_delta_in_frame(data : pd.DataFrame, pt1 : str, pt2 : str) -> np.ndarray:
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
        
    # Try lowercase columns first
    x1, y1 = POSDICT[pt1]
    x2, y2 = POSDICT[pt2]
    if all(col in data.columns for col in [x1, y1, x2, y2]):
        dx = data[x1] - data[x2]
        dy = data[y1] - data[y2]
    else:
        # Try uppercase columns with hyphens
        x1, y1 = ALT_POSDICT[pt1]
        x2, y2 = ALT_POSDICT[pt2]
        if all(col in data.columns for col in [x1, y1, x2, y2]):
            dx = data[x1] - data[x2]
            dy = data[y1] - data[y2]
        else:
            raise ValueError(f"Could not find columns for points {pt1}, {pt2} in data")
    
    return np.sqrt(dx*dx + dy*dy)


def perfectly_overlapping(data : pd.DataFrame, pt1 : str, pt2 : str, where : bool = False) -> np.ndarray:
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
        
    # Try lowercase columns first
    x1, y1 = POSDICT[pt1]
    x2, y2 = POSDICT[pt2]
    if all(col in data.columns for col in [x1, y1, x2, y2]):
        overlaps = (data[x1] == data[x2]) & (data[y1] == data[y2])
    else:
        # Try uppercase columns with hyphens
        x1, y1 = ALT_POSDICT[pt1]
        x2, y2 = ALT_POSDICT[pt2]
        if all(col in data.columns for col in [x1, y1, x2, y2]):
            overlaps = (data[x1] == data[x2]) & (data[y1] == data[y2])
        else:
            raise ValueError(f"Could not find columns for points {pt1}, {pt2} in data")
    
    return np.where(overlaps)[0] if where else overlaps


def get_cross_segment_deltas(data : pd.DataFrame, segments : np.ndarray,
                           key1 : str, key2 : str, offset : int = 0) -> np.ndarray:
    """
    Get distance between two points across specified frames
    Note: assumes segment start & end are inclusive

    data: dataFrame with raw position data
    segments: (N x 2) array containing pairs of frames to compare
    p1: colums corresponding to coordinates of point in first frame
    p2: colums corresponding to coordinates of point in second frame
    offset: distance to offset from input frames (positive -> outward, negative -> inward)
    """
    # get coordinates of points of interest
    x1, y1 = vectors_from_key(data,key1)
    x2, y2 = vectors_from_key(data,key2)

    a, b = segments.T + np.array([-1,1])[:,None] * offset # TODO: check for valid bounds
    delta = np.sqrt((x2[b] - x1[a])**2 + (y2[b] - y1[a])**2)
    return delta

import numpy as np
import pandas as pd
from . import utils


POSDICT = {
    'head' : ('xhead','yhead'),
    'tail' : ('xtail','ytail'),
    'ctr' : ('xctr','yctr'),
    'mid' : ('xmid','ymid')
}


def vectors_from_key(data : pd.DataFrame, key : str, transpose : bool = True) -> np.ndarray:
    """
    Get x, y position vectors from the given dataframe using the given key

    data: dataframe containing position data
    key: key for feature of interest
    transpose : apply transpose such that x, y vectors can be unpacked directly
    """
    kx, ky = POSDICT[key]
    vec = data.loc[:,[kx,ky]].to_numpy()
    return vec.T if transpose else vec


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
    '''get speed based on x, y position data in dataframe'''
    x, y = vectors_from_key(data,key)
    return utils.get_speed(x, y, fps, npoints)


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
    '''
    Returns the internal angle of the animal in radians
    (angle of midpt-head vector relative to tail-midpt vector)

    data (pd.DataFrame): source dataframe
    halfAngle (bool): if true, return the half-angle on [0,pi]; otherwise, return the full angle on [-pi,pi]
    '''
    npts = data.shape[0]
    u = get_vectors_between(data,'mid','head') # midpt-head
    v = get_vectors_between(data,'tail','mid') # tail-midpt

    ha = [utils.get_angle(v[i],u[i],halfAngle=halfAngle) for i in range(npts)]
    return np.array(ha)


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
    """
    Returns the signs of the z-components of the cross-products between tail-midpt and midpt-head vectors
    """
    u = get_orientation_vectors(data,True) # midpt-head
    v = get_orientation_vectors(data,False) # tail-midpt
    return utils.get_cross_sign(v,u)


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
    '''
    Get delta from (x1,y1) to (x2,y2) between frames
    Note: not recommended for speed calculations; use get_speed_from_df() instead
    '''
    if not key2 : key2 = key1 # calculate delta for same point if nosecond point specified
    x1, y1 = vectors_from_key(data,key1)
    x2, y2 = vectors_from_key(data,key2)
    npts = data.shape[0]

    vx = x2[1:] - x1[:npts-1]
    vy = y2[1:] - y1[:npts-1]
    delta = np.sqrt(vx**2 + vy**2) * fps
    return delta


def get_delta_in_frame(data : pd.DataFrame, key1 : str, key2 : str) -> np.ndarray:
    '''get distance from (x1,y1) to (x2,y2) each frame'''
    x1, y1 = vectors_from_key(data,key1)
    x2, y2 = vectors_from_key(data,key2)

    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist


def perfectly_overlapping(data : pd.DataFrame, key1 : str, key2 : str, where : bool = True) -> np.ndarray:
    '''
    Check if two points perfectly overlap in frame (faster than get_delta_in_frame)

    data: position data
    key1, key2: keys of points of interest
    where: return frames where overlaps occur instead of boolean array
    '''
    x1, y1 = vectors_from_key(data,key1)
    x2, y2 = vectors_from_key(data,key2)

    query = np.logical_and(x2 == x1, y2 == y1)
    if where : return np.where(query)[0]
    else : return query


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


def calculate_velocity(data: pd.DataFrame, fps: int, npoints: int = 2) -> pd.DataFrame:
    """
    Calculate velocity for all points in the trajectory.
    
    Args:
        data: DataFrame containing position data
        fps: Frame rate
        npoints: Number of points to use for velocity calculation
        
    Returns:
        DataFrame containing velocity data
    """
    velocity = pd.DataFrame(index=data.index)
    for key in POSDICT:
        x, y = vectors_from_key(data, key)
        v = utils.get_speed(x, y, fps, npoints)
        velocity[f'v_{key}_x'] = np.gradient(x, 1/fps)
        velocity[f'v_{key}_y'] = np.gradient(y, 1/fps)
    return velocity


def calculate_acceleration(data: pd.DataFrame, fps: int, npoints: int = 2) -> pd.DataFrame:
    """
    Calculate acceleration for all points in the trajectory.
    
    Args:
        data: DataFrame containing position data
        fps: Frame rate
        npoints: Number of points to use for acceleration calculation
        
    Returns:
        DataFrame containing acceleration data
    """
    velocity = calculate_velocity(data, fps, npoints)
    acceleration = pd.DataFrame(index=data.index)
    for key in POSDICT:
        vx = velocity[f'v_{key}_x'].values
        vy = velocity[f'v_{key}_y'].values
        acceleration[f'a_{key}_x'] = np.gradient(vx, 1/fps)
        acceleration[f'a_{key}_y'] = np.gradient(vy, 1/fps)
    return acceleration


def calculate_curvature(data: pd.DataFrame, fps: int) -> np.ndarray:
    """
    Calculate curvature of the trajectory.
    
    Args:
        data: DataFrame containing position data
        fps: Frame rate
        
    Returns:
        Array containing curvature values
    """
    x, y = vectors_from_key(data, 'mid')
    dx = np.gradient(x, 1/fps)
    dy = np.gradient(y, 1/fps)
    ddx = np.gradient(dx, 1/fps)
    ddy = np.gradient(dy, 1/fps)
    
    # Calculate denominator
    denom = (dx * dx + dy * dy) ** 1.5
    
    # Handle edge cases where denominator is zero
    curvature = np.zeros_like(x)
    mask = denom > 1e-10  # Threshold for numerical stability
    curvature[mask] = np.abs(dx[mask] * ddy[mask] - dy[mask] * ddx[mask]) / denom[mask]
    
    return curvature


def calculate_angular_velocity(data: pd.DataFrame, fps: int) -> np.ndarray:
    """
    Calculate angular velocity of the trajectory.
    
    Args:
        data: DataFrame containing position data
        fps: Frame rate
        
    Returns:
        Array containing angular velocity values
    """
    angles = get_head_angle(data)
    angular_velocity = np.gradient(angles, 1/fps)
    return angular_velocity


def calculate_all_metrics(data: pd.DataFrame, fps: int) -> dict:
    """
    Calculate all metrics for the trajectory.
    
    Args:
        data: DataFrame containing position data
        fps: Frame rate
        
    Returns:
        Dictionary containing all calculated metrics
    """
    return {
        'velocity': calculate_velocity(data, fps),
        'acceleration': calculate_acceleration(data, fps),
        'curvature': calculate_curvature(data, fps),
        'angular_velocity': calculate_angular_velocity(data, fps)
    }


def compare_metrics(raw_metrics: dict, processed_metrics: dict) -> dict:
    """
    Compare metrics between raw and processed data.
    
    Args:
        raw_metrics: Dictionary containing raw data metrics
        processed_metrics: Dictionary containing processed data metrics
        
    Returns:
        Dictionary containing comparison results
    """
    comparison = {}
    for key in raw_metrics:
        if isinstance(raw_metrics[key], pd.DataFrame):
            # For DataFrames, calculate correlation for each column
            corr = raw_metrics[key].corrwith(processed_metrics[key])
            # Replace NaN values with 0
            corr = corr.fillna(0.0)
            comparison[key] = corr
        else:
            # For arrays, calculate correlation coefficient
            try:
                corr = np.corrcoef(raw_metrics[key], processed_metrics[key])[0, 1]
                # Replace NaN with 0
                if np.isnan(corr):
                    corr = 0.0
            except:
                # If correlation calculation fails, return 0.0
                corr = 0.0
            comparison[key] = float(corr)
    return comparison

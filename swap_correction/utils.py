import numpy as np
import pandas as pd
import os
import json
import warnings


# ----- File System Shortcuts -----

def get_dirs(sourceDir : str, exclude : list[str] = ['Omit']) -> list[str]:
    '''Returns a list of all directories inside the given directory'''
    files = sorted(os.listdir(sourceDir)) # ensure order preserved
    for val in exclude:
        files = [nm for nm in files if val not in nm]
    paths = [os.path.join(sourceDir,file) for file in files]
    dirs = [path for path in paths if os.path.isdir(path)]
    return dirs


def find_file(source : str, substr : str, exclude : list[str] = ['Omit']) -> str:
    '''Returns the full path of first file in the given directory containing the specified substring'''
    initial = os.listdir(source)
    final = [nm for nm in initial if substr in nm]
    for val in exclude:
        final = [nm for nm in final if val not in nm]
    return os.path.join(source,final[0])


def read_csv(path : str) -> pd.DataFrame:
    '''Read a csv file into a Pandas dataFrame, recording missing values'''
    data =  pd.read_csv(path,na_values=['',' ']) # catch missing entries
    #data = data.fillna(0) # replace missing values with zero
    #data.columns = data.columns.str.replace(' ','') # remove spaces from column names
    return data


def export_json(sourceDir : str, fileName : str, data : dict) -> None:
    '''Export a dictionary to a json file'''
    path = os.path.join(sourceDir,fileName)
    with open(path, "w") as fp:
        json.dump(data, fp)


def load_json(sourceDir : str, fileName : str) -> dict:
    '''Load a json file to a dictionary'''
    path = os.path.join(sourceDir,fileName)
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def file_exists(sourceDir : str, fileName : str) -> bool:
    '''Check if file exists in specified folder'''
    path = os.path.join(sourceDir,fileName)
    return os.path.isfile(path)


# ----- Array Manipulation -----

def get_bounds(data : np.ndarray, buffer : float = 0.05, floor : float = None, ceil : float = None) -> tuple[float]:
    '''
    Get bounds (extrema) of input array with a buffer

    data (list): 1D list of values
    buffer (float): amount of padding to use
    floor (float): minimum permitted value
    ceil (float): maximum permitted value
    '''
    # remove NaNs and catch all-NaN case
    vec = data[~np.isnan(data)] 
    if not np.any(vec) : return (0,0)

    a = np.min(vec)
    b = np.max(vec)
    a -= (b - a) * buffer
    b += (b - a) * buffer
    if floor is not None : a = max(floor,a)
    if ceil is not None : b = min(ceil,b)
    return (a,b)


def create_sample_matrix(samples : list[np.ndarray],
                         length : int | None = None, toMin : bool = True) -> np.ndarray:
    '''
    Create a 2D (nsamples x nentries) array from a jagged list of samples
    
    samples: jagged list of arrays
    length: length to fill / truncate arrays to; None -> use min / max length of list
    toMin: truncate to minimum array length; otherwise, fill to max (requires length = None)
    '''
    # get sample and target lengths
    sampleLength = [len(sample) for sample in samples]
    if length is None : length = min(sampleLength) if toMin else max(sampleLength)

    vec = np.zeros((len(samples),length))
    for i in range(len(samples)):
        if sampleLength[i] < length: # fill sample with NaNs to desired length
            vec[i,:] = list(samples[i]) + [np.NaN] * (length - sampleLength[i])
        else: # truncate sample
            vec[i,:] = samples[i][:length]
    return vec


def flatten(mtx : list) -> np.ndarray:
    '''
    Reshape a jagged 2D list to a 1D array while preserving order

    Ex: [[0,1,2],[3,4],[5,6,7],[8]] -> [1,2,3,4,5,6,7,8]
    '''
    out = [val for row in mtx for val in row]
    return np.array(out)


def merge(*args : list) -> np.ndarray:
    '''Merge multiple numeric lists into an array with non-redundant values'''
    return np.unique(flatten([arg for arg in args]))


def filter_array(a : list, b : list) -> np.ndarray:
    '''filter the values in list b from list a; return the filtered list as a numpy array'''
    filt = [i for i in a if i not in b]
    return np.array(filt)


def match_arrays(a : list, b : list) -> np.ndarray:
    '''return values shared by two lists as numpy array'''
    filt = [i for i in a if i in b]
    return np.array(filt)


def mismatch_arrays(a : list, b : list) -> np.ndarray:
    '''return all values unique to either array'''
    ab = filter_array(a,b)
    ba = filter_array(b,a)
    return merge(ab,ba)


# ----- Intervals -----

def get_time_axis(nframes : int, fps : int) -> np.ndarray:
    '''get a time axis based on the frame rate and frame count'''
    return np.linspace(0,float(nframes)/fps,nframes)


def where(cond : np.ndarray) -> np.ndarray:
    '''
    Get the coordinates where the input condition is met
    Returns an N x dim array of indices
    '''
    # catch 1D case where would otherwise return an Nx1 vector
    if cond.ndim == 1:
        return np.where(cond)[0]

    # all other cases
    widx = np.where(cond) # tuple of arrays
    idx = np.array(list(zip(*widx))) # convert where output to indices
    return idx


def get_consecutive_values(nums : list[int]) -> list[np.ndarray]:
    '''
    Returns consecutive values in a vector as a list of ndarrays 
    Ex: [1,2,3,15,20,21] -> [[1,2,3],[15],[20,21]]
    '''
    ranges = get_consecutive_ranges(nums)
    sets = [np.arange(rng[0],rng[1]+1) for rng in ranges]
    return sets


def ranges_to_list(ranges : list[tuple[int,int]]) -> list:
    '''
    Converts ranges of values to a list of integers
    Ex: [(1,3),(15,15),(20,21)] -> [1,2,3,15,20,21]]
    TODO: update to use ranges that use Python indexing conventions
    '''
    return flatten([np.arange(rng[0],rng[1]+1) for rng in ranges])


def ranges_to_edges(swapRanges) -> np.ndarray:
    """
    Convert list of tuples indicating ranges to 1D list of frames where ranges start / end
    Ex: [(3,5),(11,15)] -> [3,6,11,16]
    """
    swaps = np.array(swapRanges)
    if swaps.size == 0 : return np.empty(0) # catch empthy input
    return merge(swaps[:,0],swaps[:,1]+1)


def invert_ranges(ranges : list[tuple[int,int]], length : int, inclusive : bool = False) -> np.ndarray:
    """
    Invert a list of index ranges
    Ex: [(1,2),(5,7)], 10, False -> [[0,0],[3,4],[8,9]]
    Ex: [(1,2),(5,7)], 10, True -> [[0,1],[2,5],[7,9]]

    ranges: intial list of index ranges
    length: length of indexed vector
    inclusive: include original indices
    """
    # catch case of empty input
    rangeArray = np.array(ranges)
    if rangeArray.shape[0] == 0 : return np.array([(0,length)])

    # invert ranges
    delta = 0 if inclusive else 1
    start, end = rangeArray.T
    inverted = list(zip(end[:-1]+delta,start[1:]-delta))

    # catch edge cases
    if start[0] > 0 : inverted.insert(0,(0,start[0]-delta)) # first edge case; append to front of list
    if end[-1] < length-1 : inverted.append((end[-1]+delta,length-1)) # last edge case

    return np.array(inverted)


def fuse_ranges(*ranges : list[tuple[int,int]] | np.ndarray, inclusive : bool = True) -> np.ndarray:
    '''
    Sort and merge a collection of overlapping ranges
    ranges: lists of ranges to fuse
    inclusive: upper bound of ranges is inclusive

    Ex: [(7,9),(2,3),(16,18)], [(14,15),(1,4),(9,12)], True -> [[1,4],[7,12],[14,18]]
    Ex: [(7,9),(2,3),(16,18)], [(14,15),(1,4),(9,12)], False -> [[1,4],[7,12],[14,15],[16,18]]
    '''
    # convert numpy arrays to lists
    rangeLists = [rng.tolist() if isinstance(rng,np.ndarray) else rng for rng in ranges]

    # merge lists into single, sorted array
    raw = []
    for rng in rangeLists : raw += rng
    raw = np.array(raw)
    raw.sort(axis=0)

    # fuse overlapping arrays
    step = 1 if inclusive else 0
    fused = []
    start, end = raw[0,:]
    for a, b in raw[1:]:
        if a <= end+step: # fuse
            end = max(end,b)
        else: # store new segment
            fused.append((start,end))
            start, end = a, b
    fused.append((start,end)) # add "unfinished" segment

    return np.array(fused)


def filter_ranges(base : list[tuple[int,int]] | np.ndarray, filt : list[tuple[int,int]] | np.ndarray,
                  inclusive : bool = True) -> np.ndarray:
    '''
    Sort and filter one set of ranges from the other
    base: ranges of interest
    filt: ranges to filter out
    inclusive: upper bound of range is inclusive

    Ex: [(1,5),(7,10),(13,20)], [(10,15),(2,3),(19,20)], True -> [(1,1),(4,5),(7,9),(16,18)]
    Unimplemented: Ex: [(1,5),(7,10),(13,20)], [(10,15),(2,3),(19,20)], False -> [(1,2),(3,5),(7,10),(15,19)]
    TODO: make this more efficient and introduce inclusivity
    '''
    a = ranges_to_list(base)
    b = ranges_to_list(filt)
    c = filter_array(a,b)
    filt = get_consecutive_ranges(c)
    return filt


def segment_lengths(segments : np.ndarray, inclusive: bool = False) -> np.ndarray:
    '''
    Get the lengths of each segment

    segments: Nx2 array of segment start / end indices
    inclusive: upper index of segment is inclusive
    '''
    length = np.diff(segments,axis=1).reshape((segments.shape[0],))
    if inclusive : length += 1
    return length


def segments_of_length(segments : np.ndarray, lower : int = 1, upper : int | None = None,
                       inclusive : bool = False) -> np.ndarray:
    '''
    Returns a reduced list of segments of of lengths between the specified limits
    segments: Nx2 array of segments
    lower: lower length limit (inclusive)
    upper: upper length limit (non-inclusive)
    inclusive: upper bound of segment is inclusive
    '''
    length = segment_lengths(segments,inclusive)
    idx = length >= lower if upper is None else np.logical_and(length >= lower, length < upper)
    return segments[idx,:]


def get_consecutive_ranges(nums : list[int]) -> list[tuple]:
    '''
    Returns ranges of consecutive values in a vector as a list of tuples
    Ex: [1,2,3,15,20,21] -> [(1,3),(15,15),(20,21)]
    TODO: make upper bound non-inclusive to match Python indexing conventions
    '''
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    ranges = list(zip(edges, edges))
    return ranges


def get_value_segments(vec : np.ndarray, val, equalTo : bool = True, inclusive : bool = True) -> np.ndarray:
    '''
    Get the start and end indices of segments of a given value in a vector
    Ex: [2,0,1,1,1,0,1], val = 1, inclusive = True -> [[2,4],[6,6]]
    Ex: [2,0,1,1,1,0,1], val = 1, inclusive = False -> [[2,5],[6,7]]

    vec: vector of interest
    val: value of interest
    equalTo: check for segments equal to that value; if false, look for segments not equal to that value
    inclusive: upper bound is inclusive
    '''
    # flag where condition met
    valid = np.zeros((vec.shape[0],))
    valid[vec == val] = 1 # binary array
    if not equalTo : valid = 1 - valid # invert

    # get start and end indices of segments
    a = np.array([0] + valid.tolist())
    b = np.array(valid.tolist() + [0])
    start = np.where(np.diff(a) == 1)[0]
    end = np.where(np.diff(b) == -1)[0]

    # merge into single N x 2 array
    step = 0 if inclusive else 1
    segs = np.array([start,end+step]).T
    return segs


def get_indices_steps(idxs : list[int]) -> tuple[np.ndarray]:
    '''Get starting indices and lengths of consecutive sections'''
    ranges = get_consecutive_ranges(idxs)
    idx = [rng[0] for rng in ranges]
    steps = [rng[1] - rng[0] for rng in ranges]
    return (np.array(idx),np.array(steps))


def get_intervals(idxs : list[int], nframes : int = -1, addBounds : bool = True,
                  inclusive : bool = False, alternating : bool = False) -> list[np.ndarray]:
    '''
    Returns a nested list of indices between each pair of input indices
    nframes: upper bound (for use with addBounds)
    addBounds: add 0 and the end frame index as bounds
    alternating: get every second range (note: may cause undesired behaviour if addBounds == False)

    Ex: idxs = [3,5,8], nframes = 10, addBounds = True, staggered = False -> [[0,1,2],[3,4],[5,6,7],[8,9]]
    Ex: idxs = [3,5,8], nframes = 10, addBounds = True, staggered = True -> [[3,4],[8,9]]
    '''
    idx = indices_to_segments(idxs,nframes,addBounds,inclusive,alternating)
    ivls = [np.arange(*i) for i in idx]
    if alternating : return ivls[1::2]
    else : return ivls


def indices_to_segments(idxs : list[int], nframes : int = -1, addBounds : bool = True,
                        inclusive : bool = False, alternating : bool = False) -> list[tuple]:
    '''
    Returns a list of interval bounds (start,end) for a given list of input indices
    nframes: upper bound (for use with addBounds)
    addBounds: add 0 and end frame as indices
    inclusive: end index of interval is inclusive
    alternating: get every second range (note: may cause undesired behaviour if addBounds == False)

    Ex: idxs = [3,5,8], nframes = 10 -> [(0,3),(3,5),(5,8),(8,10)]
    Ex: idxs = [3,5,8], nframes = 10, inclusive = True -> [(0,2),(3,4),(5,7),(8,9)]
    Ex: idxs = [3,5,8], nframes = 10, alternating = True-> [(3,5),(8,10)]
    '''
    if addBounds:
        endFrame = nframes if nframes > 1 else idxs[-1] + 1 # define upper bound
        idx = np.unique(list(idxs) + [0,endFrame])
    else:
        idx = np.unique(idxs)

    delta = 1 if inclusive else 0
    segments = list(zip(idx[:len(idx)-1],idx[1:]-delta))

    if alternating : return segments[1::2]
    else : return segments


def metrics_by_segment(data : np.ndarray, segs : np.ndarray, buffer : int = 0) -> np.ndarray:
    """
    Calculate statistics of value within each segment defined by the input flags
    Returns an N x 3 array of mean, std, median

    data: vector of interet
    segs: segments of interest
    buffer: size of buffer at edges of segment to reduce edge artifacts
    """
    with warnings.catch_warnings(): # suppress "mean, etc of empty slice"
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg = [np.nanmean(data[a+buffer:b-buffer]) for a, b in segs]
        std = [np.nanstd(data[a+buffer:b-buffer]) for a, b in segs]
        med = [np.nanmedian(data[a+buffer:b-buffer]) for a, b in segs]
    return np.array(list(zip(avg,std,med)))


# ----- Geometry -----

def unit_vector(vector):
    '''Returns the unit vector of the input vector'''
    if np.linalg.norm(vector) != 0:
        out = vector / np.linalg.norm(vector)
    else:
        out = np.zeros_like(vector)
    return out


def get_angle(ref : np.ndarray, test : np.ndarray, halfAngle : bool = True, degrees : bool = False) -> float:
    '''
    Returns the angle in radians between two vectors
    TODO: modify to enable vector calculations?

    ref: reference vector
    test: vector to compare against reference
    halfAngle: calculate positive angle on [0,pi] (if False, returns angle on [-pi,pi])
    degrees: return value in degrees
    '''
    u1 = unit_vector(ref)
    u2 = unit_vector(test)
    theta = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))

    if not halfAngle:
        theta *= get_cross_sign(u1,u2)

    if degrees : return np.rad2deg(theta)
    else : return theta


def get_cross_sign(v1 : np.ndarray, v2 : np.ndarray) -> float | np.ndarray:
    """
    Returns the signs of the z-components of the cross-products between two vectors or sets of vectors
    v1: referece vector of shape (2) or set of reference vectors of shape (n,2)
    v1: comparison vector of shape (2) or set of coparison vectors of shape (n,2)
    """
    if v1.ndim == 1:
        z = v1[0] * v2[1] - v1[1] * v2[0] # third component of cross-product
    else:
        z = v1[:,0] * v2[:,1] - v1[:,1] * v2[:,0]
    return np.sign(z)


# ----- Data Analysis -----

def get_speed(x : np.ndarray, y : np.ndarray, fps : int, npoints : int) -> np.ndarray:
    '''
    Get speed based on x, y position data
    '''
    npts = len(x)
    vx = ddt(x,npoints)
    vy = ddt(y,npoints)

    spd = np.sqrt(vx**2 + vy**2) * fps
    return spd


def ddt(data : np.ndarray, npoints : int = 2, fps : int = 1,
        absolute : bool = False, angular : bool = False) -> np.ndarray:
    '''
    Take the numerical derivative of a vector
    Note: 2-point behaves like np.diff(), but the last value is duplicated
    to match the size of the input vector
    
    data: data vector
    npoints: number of points to use in estimate (2, 3, or 5)
    fps: frame rate / sampling frequency
    absolute: return the absolute value
    angular: input is angular data (in radians)
    '''
    npts = len(data)
    delta = np.zeros(npts)
    vec = np.copy(data)
    if angular:
        idx = ~np.isnan(vec)
        vec[idx] = np.unwrap(vec[idx]) # prevents unwrap from returning all NaN's

    # take n-point derivative
    match(npoints):
        case 2:
            delta[:npts-1] = np.diff(vec)
            delta[npts-1] = delta[npts-2] # estimate last value as preceding value
        case 3:
            delta[1:npts-1] = (vec[2:npts] - vec[:npts-2])/2
        case 5:
            delta[2:npts-2] = (-vec[4:npts] + 8*vec[3:npts-1] - 8*vec[1:npts-3] + vec[:npts-4])/12
        case _:
            raise ValueError("Invalid number of points in derivative. Must be 2, 3, or 5.")

    # estimate edges with two-point derivative
    if npoints > 2:
        delta[0] = vec[1] - vec[0]
        delta[npts-1] = vec[npts-1] - vec[npts-2]
    
    # estimate points adjacent to edges with 3-point derivative
    if npoints > 3:
        delta[1] = (vec[2] - vec[0])/2
        delta[npts-2] = (vec[npts-1] - vec[npts-3])/2

    # finish
    delta *= fps
    if absolute : return np.abs(delta)
    else : return delta


def get_distance(x : np.ndarray, y : np.ndarray, origin : tuple = (0,0)) -> np.ndarray:
    '''Calculate distance from origin'''
    return np.sqrt((x-origin[0])**2 + (y-origin[1])**2)


def get_cross_segment_deltas(segments : list[tuple[int,int]] | np.ndarray,
                             vec : np.ndarray, offset : int = 1,
                             absolute : bool = False, angular : bool = False) -> np.ndarray:
    '''
    Get the change in the value from the input vector between the start and end of the specified segments
    segments: Nx2 list of start and end frames of segments
    vec: 1D vector of interest
    offset: how many frames away from actual segment start / end to retrieve values from
    (positive -> away from segment, negative -> into segment)
    absolute: return absolute difference
    angular: operating on angular values (in rad) that need to be unwrapped

    Returns NaN values if offset segment start / end are out of bounds
    '''
    # setup
    segs = np.array(segments)
    pad = np.array([-1,1]) * offset
    vals = np.unwrap(vec) if angular else vec.copy() # unswap if necessary

    # extract deltas
    delta = np.full((segs.shape[0]),np.NaN,dtype=float)
    for i, seg in enumerate(segs):
        a, b = seg + pad # get segment start / end and apply offset
        if a >= 0 and b < vals.shape[0]:
            delta[i] = vals[b] - vals[a]
    
    # finish
    if absolute : delta = np.abs(delta)
    return delta


def rolling_mean(x : np.ndarray, w : int) -> np.ndarray:
    '''Calculates a moving averageof the same length as the input vector'''
    hw = w // 2
    npts = len(x)
    avg = [np.nanmean(x[max(0,i-hw):min(npts-1,i+hw)]) for i in range(npts)]
    #std = [np.nanstd(x[max(0,i-hw):min(npts-1,i+hw)]) for i in range(npts)]
    return np.array(avg)#, np.array(std)


def rolling_median(x : np.ndarray, w : int) -> np.ndarray:
    '''Calculates a moving median of the same length as the input vector'''
    hw = w // 2
    npts = len(x)
    med = [np.nanmedian(x[max(0,i-hw):min(npts-1,i+hw)]) for i in range(npts)]
    return np.array(med)
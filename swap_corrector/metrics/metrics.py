"""Metrics module for calculating movement metrics."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

# Constants
OVERLAP_THRESH = 0  # Maximum distance between overlapping points

# Position column mappings - using uppercase with hyphen format
POSDICT = {
    'head': ('X-Head', 'Y-Head'),
    'tail': ('X-Tail', 'Y-Tail'),
    'ctr': ('X-Centroid', 'Y-Centroid'),
    'mid': ('X-Midpoint', 'Y-Midpoint')
}

# Alternative column names (for backward compatibility)
ALT_POSDICT = {
    'head': ('xhead', 'yhead'),
    'tail': ('xtail', 'ytail'),
    'ctr': ('xctr', 'yctr'),
    'mid': ('xmid', 'ymid')
}

def vectors_from_key(data: pd.DataFrame, key: str, transpose: bool = True) -> np.ndarray:
    """Get x, y position vectors from the given dataframe using the given key.
    
    Args:
        data: DataFrame containing position data
        key: Key for feature of interest
        transpose: Apply transpose such that x, y vectors can be unpacked directly
        
    Returns:
        Array of position vectors
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

def get_vectors_between(data: pd.DataFrame, key1: str, key2: str, fps: int = None) -> np.ndarray:
    """Returns the coordinates of the vectors between two points of interest.
    
    Args:
        data: DataFrame containing position data
        key1: Key for first point of reference
        key2: Key for second point of reference
        fps: Frame rate (optional, only used for speed calculations)
        
    Returns:
        Array of vectors between points
    """
    x1, y1 = vectors_from_key(data, key1)
    x2, y2 = vectors_from_key(data, key2)
    relx = x2 - x1
    rely = y2 - y1
    
    # If fps is provided, scale the vectors
    if fps is not None:
        relx *= fps
        rely *= fps
    
    return np.array([relx, rely]).T

def get_orientation_vectors(data: pd.DataFrame, head: bool = False, fromMotion: bool = False) -> np.ndarray:
    """Returns the coordinates of the desired orientation vector.
    
    Args:
        data: DataFrame containing position data
        head: Return head orientation (midpoint-head vector) instead
        fromMotion: Determine body orientation using centroid velocity
        
    Returns:
        Array of orientation vectors
    """
    if fromMotion:
        return get_motion_vector(data)
    elif head:
        return get_vectors_between(data, 'mid', 'head')
    else:
        return get_vectors_between(data, 'tail', 'mid')

def get_motion_vector(data: pd.DataFrame, key: str = 'ctr', npoints: int = 3) -> np.ndarray:
    """Get coordinates of vector indicating orientation of body based on movement.
    
    Args:
        data: DataFrame containing position data
        key: Key of point to use
        npoints: Number of points for numerical derivative
        
    Returns:
        Array of motion vectors
    """
    x, y = vectors_from_key(data, key)
    relx = np.gradient(x, npoints)
    rely = np.gradient(y, npoints)
    return np.array([relx, rely]).T

def get_head_angle(data: pd.DataFrame, halfAngle: bool = True) -> np.ndarray:
    """Returns the internal angle of the animal in radians.
    
    Args:
        data: DataFrame containing position data
        halfAngle: If true, return the half-angle on [0,pi]
        
    Returns:
        Array of head angles
    """
    # Calculate midpoints if not present
    if 'X-Midpoint' not in data.columns or 'Y-Midpoint' not in data.columns:
        data = data.copy()
        data['X-Midpoint'] = (data['X-Head'] + data['X-Tail']) / 2
        data['Y-Midpoint'] = (data['Y-Head'] + data['Y-Tail']) / 2
    
    npts = data.shape[0]
    u = get_vectors_between(data, 'mid', 'head')  # midpt-head
    v = get_vectors_between(data, 'tail', 'mid')  # tail-midpt
    
    ha = []
    for i in range(npts):
        # Calculate angle between vectors
        dot = np.dot(v[i], u[i])
        norms = np.linalg.norm(v[i]) * np.linalg.norm(u[i])
        
        if norms == 0:
            ha.append(0)
            continue
            
        cos_angle = dot / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if halfAngle:
            ha.append(angle / 2)
        else:
            ha.append(angle)
    
    return np.array(ha)

def get_ht_cross_sign(data: pd.DataFrame) -> np.ndarray:
    """Returns the signs of the cross-products between tail-midpt and midpt-head vectors.
    
    Args:
        data: DataFrame containing position data
        
    Returns:
        Array of cross signs
    """
    u = get_orientation_vectors(data, True)  # midpt-head
    v = get_orientation_vectors(data, False)  # tail-midpt
    
    # Calculate cross product signs
    cross_signs = np.zeros(len(data))
    for i in range(len(data)):
        cross = np.cross(v[i], u[i])
        cross_signs[i] = np.sign(cross)
    
    return cross_signs

def get_delta_in_frame(data1: pd.DataFrame, key1: str, key2: str, data2: pd.DataFrame = None) -> np.ndarray:
    """Calculate the distance between two points in each frame.
    
    Args:
        data1: DataFrame containing position data for first point
        key1: Key for first point
        key2: Key for second point
        data2: Optional DataFrame containing position data for second point
              If None, use data1 for both points
        
    Returns:
        Array of distances
    """
    x1, y1 = vectors_from_key(data1, key1)
    if data2 is None:
        x2, y2 = vectors_from_key(data1, key2)
    else:
        x2, y2 = vectors_from_key(data2, key2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_speed_from_df(data: pd.DataFrame, key: str, fps: int = 30, npoints: int = 2) -> np.ndarray:
    """Calculate speed from position data.
    
    Args:
        data: DataFrame with position columns
        key: Point to calculate speed for ('head', 'tail', etc.)
        fps: Frame rate (default: 30)
        npoints: Number of points for numerical derivative
        
    Returns:
        Speed values
    """
    if key not in POSDICT:
        raise ValueError(f"Invalid key: {key}")
        
    x, y = vectors_from_key(data, key)
    dx = np.gradient(x, npoints)
    dy = np.gradient(y, npoints)
    speed = np.sqrt(dx*dx + dy*dy) * fps
    return speed

def perfectly_overlapping(data: pd.DataFrame, key1: str, key2: str, where: bool = True) -> np.ndarray:
    """Find frames where two points perfectly overlap.
    
    Args:
        data: DataFrame containing position data
        key1: Key for first point
        key2: Key for second point
        where: Return indices where condition is true
        
    Returns:
        Boolean array or indices
    """
    delta = get_delta_in_frame(data, key1, key2)
    overlapping = delta < 1e-10
    return np.where(overlapping)[0] if where else overlapping

def get_delta_between_frames(data: pd.DataFrame, key1: str, key2: str | None = None, fps: int = 1) -> np.ndarray:
    """Get delta from (x1,y1) to (x2,y2) between frames.
    
    Args:
        data: DataFrame containing position data
        key1: First point key
        key2: Second point key (if None, use key1)
        fps: Frame rate
        
    Returns:
        Array of deltas between frames
    """
    if not key2:
        key2 = key1  # Calculate delta for same point if no second point specified
    x1, y1 = vectors_from_key(data, key1)
    x2, y2 = vectors_from_key(data, key2)
    npts = data.shape[0]

    vx = x2[1:] - x1[:npts-1]
    vy = y2[1:] - y1[:npts-1]
    delta = np.sqrt(vx**2 + vy**2) * fps
    return delta

def get_cross_segment_deltas(data: pd.DataFrame, segments: np.ndarray,
                           key1: str, key2: str, offset: int = 0) -> np.ndarray:
    """Get distance between two points across specified frames.
    
    Args:
        data: DataFrame containing position data
        segments: (N x 2) array containing pairs of frames to compare
        key1: First point key
        key2: Second point key
        offset: Distance to offset from input frames
        
    Returns:
        Array of deltas across segments
    """
    x1, y1 = vectors_from_key(data, key1)
    x2, y2 = vectors_from_key(data, key2)

    a, b = segments.T + np.array([-1, 1])[:, None] * offset
    delta = np.sqrt((x2[b] - x1[a])**2 + (y2[b] - y1[a])**2)
    return delta

def get_all_deltas(data: pd.DataFrame, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get tt, hh, th, and ht deltas between frames.
    
    Args:
        data: DataFrame with position columns
        edges: Optional array of frame pairs to calculate deltas between
        
    Returns:
        Tuple of (tail-tail, head-head, tail-head, head-tail) deltas
    """
    points = [('tail', 'tail'), ('head', 'head'), ('tail', 'head'), ('head', 'tail')]
    if edges is None:
        delta = [get_delta_between_frames(data, a, b) for a, b in points]
    else:
        delta = [get_cross_segment_deltas(data, edges, a, b) for a, b in points]
    return tuple(delta)

class MovementMetrics:
    """Class for calculating movement-related metrics from tracking data."""
    
    def __init__(self, data: pd.DataFrame, fps: int):
        """Initialize the metrics calculator.
        
        Args:
            data: DataFrame with position columns
            fps: Frames per second of the recording
        """
        self.data = normalize_column_names(data)
        self.fps = fps
        self._validate_data()
    
    def _validate_data(self):
        """Validate that the data contains required columns."""
        required_cols = [
            'X-Head', 'Y-Head', 'X-Tail', 'Y-Tail',
            'X-Midpoint', 'Y-Midpoint'
        ]
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def get_position(self, point: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get x, y coordinates for a given point.
        
        Args:
            point: One of 'head', 'tail', or 'mid'
            
        Returns:
            Tuple of (x_coordinates, y_coordinates)
        """
        point = point.lower()
        if point not in ['head', 'tail', 'mid']:
            raise ValueError("point must be one of: head, tail, mid")
            
        x_col = POSDICT[point][0]
        y_col = POSDICT[point][1]
        
        return (
            self.data[x_col].values,
            self.data[y_col].values
        )
    
    def get_speed(self, point: str) -> np.ndarray:
        """Calculate speed for head/tail/midpoint.
        
        Args:
            point: One of 'head', 'tail', or 'mid'
            
        Returns:
            Array of speeds in pixels/second
        """
        x, y = self.get_position(point)
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.sqrt(dx**2 + dy**2) * self.fps
        # Repeat last value to match array length
        return np.concatenate([speed, [speed[-1]]])
    
    def get_acceleration(self, point: str) -> np.ndarray:
        """Calculate acceleration for a given point.
        
        Args:
            point: One of 'head', 'tail', or 'mid'
            
        Returns:
            Array of accelerations in pixels/second²
        """
        speed = self.get_speed(point)
        return np.gradient(speed) * self.fps
    
    def get_angular_velocity(self) -> np.ndarray:
        """Calculate angular velocity between head and tail.
        
        Returns:
            Array of angular velocities in radians/second
        """
        head_x, head_y = self.get_position('head')
        tail_x, tail_y = self.get_position('tail')
        
        # Calculate angles
        angles = np.arctan2(head_y - tail_y, head_x - tail_x)
        
        # Unwrap angles to handle discontinuities
        angles = np.unwrap(angles)
        
        # Calculate angular velocity
        angular_velocity = np.gradient(angles) * self.fps
        
        # Handle potential outliers
        median_vel = np.median(angular_velocity)
        mad = np.median(np.abs(angular_velocity - median_vel))
        outliers = np.abs(angular_velocity - median_vel) > 3 * mad
        
        if np.any(outliers):
            # Replace outliers with local median
            window = 5
            for i in np.where(outliers)[0]:
                start = max(0, i - window)
                end = min(len(angular_velocity), i + window + 1)
                angular_velocity[i] = np.median(angular_velocity[start:end])
        
        return angular_velocity
    
    def get_curvature(self) -> np.ndarray:
        """Calculate path curvature for turn detection.
        
        Returns:
            Array of curvature values (1/pixel)
        """
        x, y = self.get_position('mid')
        
        # First derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx * dx + dy * dy) ** 1.5
        
        # Handle potential division by zero
        curvature = np.zeros_like(numerator)
        nonzero = denominator > 1e-10
        curvature[nonzero] = numerator[nonzero] / denominator[nonzero]
        
        return curvature
    
    def get_body_length(self) -> np.ndarray:
        """Calculate body length (distance between head and tail).
        
        Returns:
            Array of body lengths in pixels
        """
        head_x, head_y = self.get_position('head')
        tail_x, tail_y = self.get_position('tail')
        
        return np.sqrt(
            (head_x - tail_x)**2 + 
            (head_y - tail_y)**2
        )

def flag_overlap_sign_reversals(data: pd.DataFrame, tolerance: float = OVERLAP_THRESH,
                                debug: bool = False) -> np.ndarray:
    """Flag frames where sign of head-tail vector reverses across an overlap.
    
    Args:
        data: DataFrame containing position data
        tolerance: Maximum distance between "overlapping" points
        debug: Whether to print debug messages
        
    Returns:
        Array of flagged frames
    """
    # Get overlap edges
    edges = get_overlap_edges(data, tolerance=tolerance, offset=1, debug=False)
    if edges.size == 0:
        return np.empty(0)  # No overlaps found
    
    # Get orientation vectors
    vecs = get_orientation_vectors(data)
    
    # Check for sign reversals across overlaps
    flag = []
    for a, b in edges:
        # Get vectors before and after overlap
        v1 = vecs[a]
        v2 = vecs[b]
        
        # Calculate angle between vectors
        dot = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            continue
            
        cos_angle = dot / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Flag if angle is greater than pi/2
        if angle > np.pi/2:
            flag.append(b)
    
    if debug:
        print(f'Overlap Sign Reversals: {flag}')
    
    return np.array(flag)

def get_overlap_edges(data: pd.DataFrame, offset: int = 0,
                      tolerance: float = OVERLAP_THRESH, pt1: str = 'head', pt2: str = 'tail',
                      debug: bool = False) -> np.ndarray:
    """Get frames on either side of each overlap region.
    
    Args:
        data: DataFrame containing position data
        offset: Number of frames away from overlap
        tolerance: Maximum distance between points to be considered overlapping
        pt1: First point of interest
        pt2: Second point of interest
        debug: Whether to print debug messages
        
    Returns:
        Array of start and end frames of shape (N x 2) for N overlaps
    """
    overlaps = flag_overlaps(data, tolerance, pt1, pt2)
    ranges = get_consecutive_ranges(overlaps)
    edges = [(max(rng[0]-offset, 0), min(rng[1]+offset, data.shape[0]-1)) for rng in ranges]
    
    if debug:
        print(f'Overlaps ({pt1[0]}-{pt2[0]}): ({len(edges)}) {edges}')
    
    return np.array(edges)

def flag_overlaps(data: pd.DataFrame, tolerance: float = OVERLAP_THRESH,
                 pt1: str = 'head', pt2: str = 'tail', debug: bool = False) -> np.ndarray:
    """Flag frames where two points overlap.
    
    Args:
        data: DataFrame containing position data
        tolerance: Maximum distance between points to be considered overlapping
        pt1: First point of interest
        pt2: Second point of interest
        debug: Whether to print debug messages
        
    Returns:
        Array of flagged frames
    """
    # Calculate distance between points
    dist = get_delta_in_frame(data, pt1, pt2)
    
    # Flag frames where distance is less than tolerance
    flag = np.where(dist <= tolerance)[0]
    
    if debug:
        print(f'Overlaps ({pt1[0]}-{pt2[0]}): {flag}')
    
    return flag

def get_consecutive_ranges(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Get start and end frames of consecutive ranges.
    
    Args:
        indices: Array of frame indices
        
    Returns:
        List of (start, end) tuples for each consecutive range
    """
    if len(indices) == 0:
        return []
    
    # Find breaks in consecutive sequence
    breaks = np.where(np.diff(indices) > 1)[0]
    
    # Convert to ranges
    ranges = []
    start = 0
    for end in breaks:
        ranges.append((indices[start], indices[end]))
        start = end + 1
    ranges.append((indices[start], indices[-1]))
    
    return ranges

def flag_overlap_minimum_mismatches(data: pd.DataFrame, tolerance: float = OVERLAP_THRESH,
                                  debug: bool = False) -> np.ndarray:
    """Flag frames where minimum distance between frames occurs during overlap.
    
    Args:
        data: DataFrame containing position data
        tolerance: Maximum distance between points to be considered overlapping
        debug: Whether to print debug messages
        
    Returns:
        Array of flagged frames
    """
    # Get overlap edges
    edges = get_overlap_edges(data, tolerance=tolerance, offset=1, debug=False)
    if edges.size == 0:
        return np.empty(0)  # No overlaps found
    
    # Get all deltas between frames
    deltas = get_all_deltas(data, edges)  # tt, hh, th, ht
    
    # Find minimum deltas and check if index matches th or ht
    min_idx = np.argmin(deltas, axis=0)  # index of minimum delta for each frame pair
    flag = np.where(min_idx > 1)[0]  # indices where th or ht is minimum
    
    if debug:
        print(f'Overlap Minimum Mismatches: {flag}')
    
    return flag

def get_orientation(data: pd.DataFrame, ref: list[float] = [1,0],
                   halfAngle: bool = True, fromMotion: bool = False, head: bool = False) -> np.ndarray:
    """Calculate the global body angle in radians relative to a reference vector.
    
    Args:
        data: DataFrame with position columns
        ref: Reference vector [x, y]
        halfAngle: If true, return the half-angle on [0,pi]
        fromMotion: Use centroid velocity to estimate angle
        head: Use midpoint-head vector instead of tail-midpoint vector
        
    Returns:
        Array of angles in radians
    """
    # Get orientation vectors
    if fromMotion:
        v = get_motion_vector(data)
    else:
        if head:
            v = get_vectors_between(data, 'mid', 'head')
        else:
            v = get_vectors_between(data, 'tail', 'mid')
    
    # Calculate angles
    angles = []
    for i in range(len(data)):
        # Calculate angle between reference vector and orientation vector
        dot = np.dot(ref, v[i])
        norms = np.linalg.norm(ref) * np.linalg.norm(v[i])
        
        if norms == 0:
            angles.append(0)
            continue
            
        cos_angle = dot / norms
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if halfAngle:
            angle /= 2
            
        angles.append(angle)
    
    return np.array(angles)

def get_segment_distance(data: pd.DataFrame, segments: np.ndarray, key: str) -> np.ndarray:
    """Calculate total distance traveled in each segment.
    
    Args:
        data: DataFrame containing position data
        segments: N x 2 array of start and end frames (inclusive)
        key: Point to calculate distance for ('head', 'tail', etc.)
        
    Returns:
        Array of distances for each segment
    """
    # Get position vectors
    x, y = vectors_from_key(data, key)
    
    # Calculate distances for each segment
    distances = []
    for start, end in segments:
        # Get position differences
        dx = np.diff(x[start:end+1])
        dy = np.diff(y[start:end+1])
        
        # Calculate total distance
        dist = np.sum(np.sqrt(dx*dx + dy*dy))
        distances.append(dist)
    
    return np.array(distances)

def calculate_metrics(data: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    """Calculate metrics comparing tracking data with ground truth.
    
    Args:
        data: DataFrame containing tracking data to evaluate
        ground_truth: DataFrame containing ground truth data
        
    Returns:
        Dictionary containing various comparison metrics
    """
    # Normalize column names
    data = normalize_column_names(data)
    ground_truth = normalize_column_names(ground_truth)
    
    # Calculate position errors
    head_error = get_delta_in_frame(data, 'head', 'head', ground_truth)
    tail_error = get_delta_in_frame(data, 'tail', 'tail', ground_truth)
    
    # Calculate speed errors
    data_head_speed = get_speed_from_df(data, 'head')
    data_tail_speed = get_speed_from_df(data, 'tail')
    gt_head_speed = get_speed_from_df(ground_truth, 'head')
    gt_tail_speed = get_speed_from_df(ground_truth, 'tail')
    
    head_speed_error = np.abs(data_head_speed - gt_head_speed)
    tail_speed_error = np.abs(data_tail_speed - gt_tail_speed)
    
    # Calculate angle errors
    data_angles = get_head_angle(data)
    gt_angles = get_head_angle(ground_truth)
    angle_error = np.abs(data_angles - gt_angles)
    
    # Calculate metrics
    metrics = {
        'mean_head_position_error': np.nanmean(head_error),
        'mean_tail_position_error': np.nanmean(tail_error),
        'mean_head_speed_error': np.nanmean(head_speed_error),
        'mean_tail_speed_error': np.nanmean(tail_speed_error),
        'mean_angle_error': np.nanmean(angle_error),
        'max_head_position_error': np.nanmax(head_error),
        'max_tail_position_error': np.nanmax(tail_error),
        'max_head_speed_error': np.nanmax(head_speed_error),
        'max_tail_speed_error': np.nanmax(tail_speed_error),
        'max_angle_error': np.nanmax(angle_error),
        'std_head_position_error': np.nanstd(head_error),
        'std_tail_position_error': np.nanstd(tail_error),
        'std_head_speed_error': np.nanstd(head_speed_error),
        'std_tail_speed_error': np.nanstd(tail_speed_error),
        'std_angle_error': np.nanstd(angle_error)
    }
    
    return metrics

def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to uppercase with hyphen format.
    
    Args:
        data: DataFrame with position columns
        
    Returns:
        DataFrame with normalized column names
    """
    data = data.copy()
    
    # Define column mappings for both formats
    column_mapping = {
        # Lowercase format
        'xhead': 'X-Head', 'yhead': 'Y-Head',
        'xtail': 'X-Tail', 'ytail': 'Y-Tail',
        'xctr': 'X-Centroid', 'yctr': 'Y-Centroid',
        'xmid': 'X-Midpoint', 'ymid': 'Y-Midpoint',
        # Uppercase format
        'X_Head': 'X-Head', 'Y_Head': 'Y-Head',
        'X_Tail': 'X-Tail', 'Y_Tail': 'Y-Tail',
        'X_Centroid': 'X-Centroid', 'Y_Centroid': 'Y-Centroid',
        'X_Midpoint': 'X-Midpoint', 'Y_Midpoint': 'Y-Midpoint',
        # Already normalized format (no change needed)
        'X-Head': 'X-Head', 'Y-Head': 'Y-Head',
        'X-Tail': 'X-Tail', 'Y-Tail': 'Y-Tail',
        'X-Centroid': 'X-Centroid', 'Y-Centroid': 'Y-Centroid',
        'X-Midpoint': 'X-Midpoint', 'Y-Midpoint': 'Y-Midpoint'
    }
    
    # Only rename columns that exist
    existing_cols = {k: v for k, v in column_mapping.items() if k in data.columns}
    if existing_cols:
        data = data.rename(columns=existing_cols)
    
    return data 
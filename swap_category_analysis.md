# Swap Category Analysis and Improvement Proposals

## Overview
This document analyzes the current implementation and proposes improvements for each category of head-tail swaps in Drosophila tracking data.

## 1. Close Proximity Swaps (747 instances, 84.9% correction rate)

### Current Implementation
- Uses `flag_overlaps` with tolerance parameter
- Detects when head and tail points are within threshold distance
- Relies on `get_delta_in_frame` for distance calculations

### Limitations
- Fixed distance thresholds regardless of movement context
- Limited temporal context consideration
- May miss swaps when points are close but not overlapping

### Proposed Improvements
```python
def adaptive_proximity_threshold(data: pd.DataFrame, fps: int) -> float:
    """Dynamic threshold based on movement context."""
    speed = metrics.get_speed_from_df(data, 'midpoint', fps)
    accel = np.gradient(speed) * fps
    
    if np.mean(speed) > 10: return 5.0
    elif np.mean(np.abs(accel)) > 5: return 3.0
    else: return 2.0
```

## 2. Near Overlap Swaps (673 instances, 85.6% correction rate)

### Current Implementation
- Uses `flag_overlap_mismatches` for detection
- Considers both distance and movement patterns
- Applies `remove_overlaps` for correction

### Limitations
- Fixed overlap thresholds
- Doesn't consider movement direction
- May misinterpret legitimate overlaps

### Proposed Improvements
```python
def detect_near_overlap(data: pd.DataFrame, fps: int) -> np.ndarray:
    """Enhanced near overlap detection."""
    # Calculate distances
    dist = metrics.get_delta_in_frame(data, 'head', 'tail')
    
    # Calculate movement vectors
    head_vec = metrics.get_motion_vector(data, 'head')
    tail_vec = metrics.get_motion_vector(data, 'tail')
    
    # Calculate angle between movement vectors
    angle = np.arccos(np.dot(head_vec, tail_vec) / 
                     (np.linalg.norm(head_vec) * np.linalg.norm(tail_vec)))
    
    # Combine evidence
    is_near_overlap = (dist < 5.0) & (angle > np.pi/2)
    return is_near_overlap
```

## 3. Body Shape Changes (565 instances, 84.4% correction rate)

### Current Implementation
- Uses `flag_sign_reversals` for detection
- Considers body angle changes
- Applies `correct_swapped_segments` for correction

### Limitations
- Fixed angle thresholds
- Doesn't consider body length changes
- May miss gradual shape changes

### Proposed Improvements
```python
def detect_body_shape_changes(data: pd.DataFrame, fps: int) -> np.ndarray:
    """Enhanced body shape change detection."""
    # Calculate body metrics
    body_length = metrics.get_delta_in_frame(data, 'head', 'tail')
    body_angle = metrics.get_orientation(data)
    
    # Calculate changes
    length_change = np.gradient(body_length) * fps
    angle_change = np.gradient(body_angle) * fps
    
    # Detect significant changes
    is_shape_change = (
        (np.abs(length_change) > 2.0) |  # Rapid length change
        (np.abs(angle_change) > np.pi/4)  # Rapid angle change
    )
    return is_shape_change
```

## 4. Rapid Turn Swaps (19 instances, 84.2% correction rate)

### Current Implementation
- Uses `flag_discontinuities` for detection
- Considers speed and angle changes
- Applies `validate_corrected_data` for correction

### Limitations
- Fixed speed thresholds
- Doesn't consider turn radius
- May miss gradual turns

### Proposed Improvements
```python
def detect_rapid_turns(data: pd.DataFrame, fps: int) -> np.ndarray:
    """Enhanced rapid turn detection."""
    # Calculate movement metrics
    speed = metrics.get_speed_from_df(data, 'midpoint', fps)
    curvature = metrics.calculate_curvature(data, fps)
    angular_velocity = metrics.calculate_angular_velocity(data, fps)
    
    # Detect rapid turns
    is_rapid_turn = (
        (speed > 10) &  # High speed
        (curvature > 0.1) &  # High curvature
        (np.abs(angular_velocity) > np.pi/2)  # Rapid rotation
    )
    return is_rapid_turn
```

## 5. Curled Body Swaps (9 instances, 66.7% correction rate)

### Current Implementation
- Uses `flag_overlap_minimum_mismatches` for detection
- Considers body length and overlap
- Applies `correct_swapped_segments` for correction

### Limitations
- Fixed length thresholds
- Doesn't consider curl direction
- May miss partial curls

### Proposed Improvements
```python
def detect_curled_body(data: pd.DataFrame, fps: int) -> np.ndarray:
    """Enhanced curled body detection."""
    # Calculate body metrics
    body_length = metrics.get_delta_in_frame(data, 'head', 'tail')
    body_angle = metrics.get_orientation(data)
    
    # Calculate curl metrics
    length_ratio = body_length / np.mean(body_length)
    angle_change = np.gradient(body_angle) * fps
    
    # Detect curls
    is_curled = (
        (length_ratio < 0.7) &  # Shortened body
        (np.abs(angle_change) > np.pi/6)  # Rapid angle change
    )
    return is_curled
```

## 6. High Speed/Sudden Movement Swaps (3 instances, 33.3% correction rate)

### Current Implementation
- Uses `flag_discontinuities` with speed threshold
- Considers acceleration patterns
- Applies `validate_corrected_data` for correction

### Limitations
- Fixed speed thresholds
- Doesn't consider movement context
- May miss gradual accelerations

### Proposed Improvements
```python
def detect_high_speed_movement(data: pd.DataFrame, fps: int) -> np.ndarray:
    """Enhanced high speed movement detection."""
    # Calculate movement metrics
    speed = metrics.get_speed_from_df(data, 'midpoint', fps)
    accel = np.gradient(speed) * fps
    jerk = np.gradient(accel) * fps
    
    # Detect high speed movements
    is_high_speed = (
        (speed > 20) |  # High speed
        (np.abs(accel) > 10) |  # High acceleration
        (np.abs(jerk) > 5)  # High jerk
    )
    return is_high_speed
```

## Common Improvements Across Categories

### 1. Multi-Scale Analysis
```python
def multi_scale_analysis(data: pd.DataFrame, fps: int) -> dict:
    """Analyze movement patterns at different temporal scales."""
    scales = {
        'short_term': {'window': 5, 'metrics': ['speed', 'acceleration']},
        'medium_term': {'window': 15, 'metrics': ['path_tortuosity']},
        'long_term': {'window': 30, 'metrics': ['behavioral_state']}
    }
    return {scale: calculate_scale_metrics(data, fps, params) 
            for scale, params in scales.items()}
```

### 2. Confidence Scoring
```python
def calculate_swap_confidence(data: pd.DataFrame, category: str) -> float:
    """Calculate confidence score for swap detection."""
    scores = {
        'proximity': calculate_proximity_score(data),
        'movement': calculate_movement_score(data),
        'temporal': calculate_temporal_score(data),
        'behavioral': calculate_behavioral_score(data)
    }
    
    weights = get_category_weights(category)
    return sum(scores[k] * weights[k] for k in scores)
```

## Implementation Strategy

1. **Phase 1: Core Improvements**
   - Implement adaptive thresholds
   - Add movement context analysis
   - Enhance validation logic

2. **Phase 2: Category-Specific Enhancements**
   - Add specialized detection for each category
   - Implement category-specific validation
   - Add confidence scoring

3. **Phase 3: Integration and Optimization**
   - Combine all detection methods
   - Optimize performance
   - Add comprehensive testing

## Testing Requirements

1. **Category-Specific Tests**
   - Test cases for each swap type
   - Edge cases and complex scenarios
   - Validation against ground truth

2. **Performance Testing**
   - Processing time impact
   - Memory usage
   - Scalability

3. **Accuracy Testing**
   - False positive rates
   - False negative rates
   - Correction success rates

## Future Considerations

1. **Machine Learning Integration**
   - Category-specific classifiers
   - Feature engineering
   - Model training and validation

2. **Real-time Processing**
   - Streaming analysis
   - Predictive capabilities
   - Performance optimization

3. **Behavioral Context**
   - Species-specific parameters
   - Environmental factors
   - Behavioral state integration 
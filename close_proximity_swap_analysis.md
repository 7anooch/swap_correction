# Close Proximity Swap Analysis and Improvement Proposals

## Current Implementation

### 1. Detection System
The current system detects close proximity swaps through several mechanisms:

- **Distance-Based Detection**:
  - Uses `flag_overlaps` with a tolerance parameter
  - Calculates distances between head and tail points using `get_delta_in_frame`
  - Detects when points are within a specified threshold distance

- **Movement Pattern Analysis**:
  - Uses `flag_delta_mismatches` to detect speed differences
  - Applies `flag_sign_reversals` for sudden orientation changes
  - Considers `flag_overlap_mismatches` for complex cases

### 2. Correction Process
The correction pipeline follows these steps:

1. **Edge Frame Removal**:
   - `remove_edge_frames` handles frames with zero position data
   - Ensures clean data boundaries

2. **Error Correction**:
   - `correct_tracking_errors` applies initial corrections
   - Uses multiple criteria to identify problematic segments
   - Corrects segments using `correct_swapped_segments`

3. **Validation**:
   - `validate_corrected_data` checks for remaining swaps
   - Uses alignment angles and movement patterns
   - Applies additional corrections if needed

4. **Overlap Handling**:
   - `remove_overlaps` manages cases where points are too close
   - Uses speed thresholds to determine incorrect placements
   - Can remove data from either head or tail based on discontinuities

## Current Limitations

### 1. Fixed Threshold Issues
- Uses constant distance thresholds regardless of movement context
- Doesn't adapt to different behavioral states
- May miss swaps when points are close but not overlapping

### 2. Temporal Context Limitations
- Limited consideration of movement patterns before/after close proximity
- Doesn't use acceleration or jerk information
- May misinterpret legitimate close movements as swaps

### 3. Validation Weaknesses
- Relies heavily on movement continuity
- May fail during complex movements
- Doesn't consider behavioral context

## Proposed Improvements

### 1. Adaptive Thresholds
```python
def adaptive_proximity_threshold(data: pd.DataFrame, fps: int) -> float:
    """Calculate dynamic threshold based on movement context."""
    # Calculate local speed
    speed = metrics.get_speed_from_df(data, 'midpoint', fps)
    
    # Calculate local acceleration
    accel = np.gradient(speed) * fps
    
    # Base threshold on movement characteristics
    if np.mean(speed) > 10:  # High speed
        return 5.0  # More lenient threshold
    elif np.mean(np.abs(accel)) > 5:  # High acceleration
        return 3.0  # Medium threshold
    else:
        return 2.0  # Strict threshold for slow movements
```

### 2. Enhanced Movement Analysis
```python
def analyze_movement_context(data: pd.DataFrame, fps: int) -> dict:
    """Analyze movement patterns around close proximity events."""
    context = {
        'pre_movement': metrics.get_speed_from_df(data.iloc[:-5], 'midpoint', fps),
        'post_movement': metrics.get_speed_from_df(data.iloc[5:], 'midpoint', fps),
        'curvature': metrics.calculate_curvature(data, fps),
        'angular_velocity': metrics.calculate_angular_velocity(data, fps)
    }
    return context
```

### 3. Improved Validation
```python
def validate_close_proximity_swap(data: pd.DataFrame, fps: int) -> bool:
    """Enhanced validation for close proximity swaps."""
    # Check movement continuity
    speed_ratio = metrics.get_speed_ratios(data)
    angle_changes = metrics.get_orientation(data)
    
    # Check body shape
    body_length = metrics.get_delta_in_frame(data, 'head', 'tail')
    
    # Check temporal patterns
    pre_context = analyze_movement_context(data.iloc[:-5], fps)
    post_context = analyze_movement_context(data.iloc[5:], fps)
    
    # Combine evidence
    is_valid = (
        speed_ratio_within_bounds(speed_ratio) and
        angle_changes_consistent(angle_changes) and
        body_length_plausible(body_length) and
        movement_context_consistent(pre_context, post_context)
    )
    
    return is_valid
```

### 4. Multi-Scale Analysis
```python
def multi_scale_analysis(data: pd.DataFrame, fps: int) -> dict:
    """Analyze movement patterns at different temporal scales."""
    scales = {
        'short_term': {
            'window': 5,
            'metrics': ['speed', 'acceleration', 'curvature']
        },
        'medium_term': {
            'window': 15,
            'metrics': ['path_tortuosity', 'direction_change']
        },
        'long_term': {
            'window': 30,
            'metrics': ['overall_direction', 'behavioral_state']
        }
    }
    
    analysis = {}
    for scale, params in scales.items():
        analysis[scale] = calculate_scale_metrics(data, fps, params)
    
    return analysis
```

### 5. Confidence Scoring
```python
def calculate_swap_confidence(data: pd.DataFrame, fps: int) -> float:
    """Calculate confidence score for swap detection."""
    scores = {
        'proximity': calculate_proximity_score(data),
        'movement': calculate_movement_score(data, fps),
        'temporal': calculate_temporal_score(data, fps),
        'behavioral': calculate_behavioral_score(data)
    }
    
    weights = {
        'proximity': 0.3,
        'movement': 0.3,
        'temporal': 0.2,
        'behavioral': 0.2
    }
    
    confidence = sum(scores[k] * weights[k] for k in scores)
    return confidence
```

## Implementation Considerations

### 1. Performance Impact
- Adaptive thresholds may increase computation time
- Multi-scale analysis requires careful optimization
- Need to balance accuracy with processing speed

### 2. Integration Strategy
1. Start with adaptive thresholds
2. Add movement context analysis
3. Implement improved validation
4. Add multi-scale analysis
5. Finally implement confidence scoring

### 3. Testing Requirements
- Need test cases for different movement patterns
- Should include edge cases and complex scenarios
- Must validate against ground truth data

## Next Steps

1. **Initial Implementation**:
   - Implement adaptive thresholds
   - Add basic movement context analysis
   - Test with existing datasets

2. **Validation**:
   - Compare results with current implementation
   - Measure improvement in swap detection
   - Assess impact on processing time

3. **Refinement**:
   - Adjust thresholds based on results
   - Optimize performance
   - Add more sophisticated analysis

4. **Documentation**:
   - Update documentation with new features
   - Create usage examples
   - Document best practices

## Future Considerations

1. **Machine Learning Integration**:
   - Train models on labeled swap data
   - Use for confidence scoring
   - Improve pattern recognition

2. **Real-time Processing**:
   - Optimize for live tracking
   - Implement streaming analysis
   - Add predictive capabilities

3. **Behavioral Context**:
   - Incorporate behavioral state information
   - Add species-specific parameters
   - Consider environmental factors 
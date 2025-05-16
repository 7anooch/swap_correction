# Core Concepts

This guide explains the fundamental concepts and principles behind the Swap Correction package.

## Data Structure

### Position Data

The package works with tracking data stored in a pandas DataFrame with the following structure:

```python
# Required columns
xhead, yhead  # Head position coordinates
xtail, ytail  # Tail position coordinates

# Optional columns
xctr, yctr    # Center position coordinates
xmid, ymid    # Midpoint position coordinates
```

### Data Types

- All position coordinates should be numeric (float or int)
- Missing values should be represented as NaN
- Time should be represented in frames (integer indices)

## Swap Detection

### What is a Head-Tail Swap?

A head-tail swap occurs when the tracking system incorrectly identifies which end of the animal is the head and which is the tail. This can happen due to:

- Rapid movements
- Overlapping positions
- Tracking errors
- Edge cases

### Detection Methods

1. **Overlap Detection**
   - Identifies frames where head and tail positions overlap
   - Uses a threshold of 0.5mm for separation
   - Most basic form of swap detection

2. **Sign Reversal Detection**
   - Detects sudden reversals in head-tail delta direction
   - Uses dot product between consecutive deltas
   - Good for detecting rapid swaps

3. **Delta Mismatch Detection**
   - Identifies frames where head-tail delta changes abruptly
   - Uses magnitude of delta changes
   - Effective for detecting gradual swaps

4. **Overlap Sign Reversal Detection**
   - Specialized detection for reversals during overlaps
   - Combines overlap and sign reversal detection
   - More sensitive to swap detection during overlaps

5. **Overlap Minimum Mismatch Detection**
   - Detects unexpected minimum separations during overlaps
   - Helps identify frames where swaps occur during overlaps

## Data Processing Pipeline

### 1. Edge Frame Removal

```python
data = remove_edge_frames(data)
```

- Removes frames where head or tail is at the edge of the frame
- Prevents false swap detections
- Sets edge frames to NaN

### 2. Data Filtering

```python
if filterData:
    data = filtering.filter_data(data)
```

Available filters:
- Gaussian filter (default)
- Savitzky-Golay filter
- Mean-median filter
- Median filter

### 3. Swap Correction

```python
data = correct_tracking_errors(data, fps, swapCorrection, validate, debug)
```

Process:
1. Detect potential swap frames
2. Group into consecutive segments
3. Apply corrections to each segment
4. Validate corrections if requested

### 4. Error Frame Removal

```python
if removeErrors:
    data = remove_overlaps(data, debug)
```

- Removes frames with head-tail overlaps
- Sets problematic frames to NaN

### 5. Gap Interpolation

```python
if interp:
    data = interpolate_gaps(data, debug)
```

- Fills gaps in the data using linear interpolation
- Maintains data continuity

## Metrics and Analysis

### Position Metrics

1. **Separation**
   ```python
   dist = metrics.get_delta_in_frame(data, 'head', 'tail')
   ```
   - Calculates separation between head and tail
   - Used for overlap detection

2. **Vectors**
   ```python
   vec = metrics.get_vectors_between(data, 'head', 'tail')
   ```
   - Calculates vectors between points
   - Used for direction analysis

### Angle Metrics

1. **Head Angle**
   ```python
   angle = metrics.get_head_angle(data)
   ```
   - Calculates internal angle of the animal
   - Used for orientation analysis

2. **Orientation**
   ```python
   orient = metrics.get_orientation(data)
   ```
   - Calculates global body angle
   - Used for movement analysis

### Motion Metrics

1. **Speed**
   ```python
   speed = metrics.get_speed_from_df(data, 'ctr', fps)
   ```
   - Calculates speed from position data
   - Used for movement analysis

2. **Tortuosity**
   ```python
   tort = metrics.get_local_tortuosity(data)
   ```
   - Calculates local path tortuosity
   - Used for movement pattern analysis

## Error Handling

### Common Errors

1. **Missing Data**
   - Handled by interpolation
   - Can be configured to skip or fill

2. **Invalid Data**
   - Checked during validation
   - Can be removed or corrected

3. **Edge Cases**
   - Handled by edge frame removal
   - Can be configured to keep or remove

### Debug Mode

```python
corrected_data = swap_corrector(data, fps=30.0, debug=True)
```

- Provides detailed logging
- Helps identify issues
- Useful for development

## Best Practices

1. **Data Preparation**
   - Clean data before processing
   - Handle missing values appropriately
   - Verify data types and formats

2. **Parameter Selection**
   - Choose appropriate filter parameters
   - Adjust thresholds based on data
   - Validate results

3. **Error Handling**
   - Use try-except blocks
   - Log errors appropriately
   - Handle edge cases

4. **Performance**
   - Use appropriate data structures
   - Optimize for large datasets
   - Consider memory usage

## Next Steps

- Read the [Swap Detection](swap_detection.md) guide
- Explore the [Data Processing](data_processing.md) guide
- Check out the [API Reference](api/main.md) 
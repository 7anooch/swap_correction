# Advanced Usage Examples

This document provides advanced examples of using the Swap Correction package.

## Custom Filtering Pipeline

### Creating a Custom Filter

```python
import numpy as np
from swap_correction.tracking import filtering

def custom_filter(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Custom filter that combines median and mean filtering."""
    # Apply median filter
    median_filtered = filtering.filter_median(data, win=window)
    # Apply mean filter
    mean_filtered = np.convolve(median_filtered, np.ones(window)/window, mode='same')
    return mean_filtered

# Apply custom filter to data
filtered_data = filtering.apply_filter_to_column(
    data=data['xhead'].values,
    filter_func=custom_filter,
    window=5
)
```

### Custom Filtering Pipeline

```python
from swap_correction.tracking import filtering, utils

def custom_filtering_pipeline(data: pd.DataFrame) -> pd.DataFrame:
    """Custom filtering pipeline with multiple steps."""
    # Create a copy of the data
    filtered = data.copy()
    
    # Apply different filters to different columns
    filtered['xhead'] = filtering.filter_sgolay(data['xhead'], window=45, order=4)
    filtered['yhead'] = filtering.filter_sgolay(data['yhead'], window=45, order=4)
    filtered['xtail'] = filtering.filter_gaussian(data['xtail'], sigma=3)
    filtered['ytail'] = filtering.filter_gaussian(data['ytail'], sigma=3)
    
    # Apply custom filter to center points
    if 'xctr' in data.columns:
        filtered['xctr'] = custom_filter(data['xctr'].values)
    if 'yctr' in data.columns:
        filtered['yctr'] = custom_filter(data['yctr'].values)
    
    return filtered
```

## Custom Swap Detection

### Implementing Custom Detection Method

```python
from swap_correction.tracking import flagging, metrics

def custom_swap_detection(
    data: pd.DataFrame,
    threshold: float = 2.0,
    window: int = 10
) -> np.ndarray:
    """Custom swap detection using speed and angle changes."""
    # Calculate speed and angle
    speed = metrics.calculate_speed(data, fps=30.0)
    angle = metrics.calculate_head_angle(data)
    
    # Find frames with high speed and angle changes
    speed_threshold = np.mean(speed) + threshold * np.std(speed)
    angle_diff = np.abs(np.diff(angle))
    angle_threshold = np.mean(angle_diff) + threshold * np.std(angle_diff)
    
    # Flag frames meeting both criteria
    swap_frames = np.where(
        (speed > speed_threshold) & 
        (np.append(angle_diff, 0) > angle_threshold)
    )[0]
    
    return swap_frames
```

### Combining Multiple Detection Methods

```python
def comprehensive_swap_detection(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> np.ndarray:
    """Combines multiple swap detection methods."""
    # Get frames from different detection methods
    overlap_frames = flagging.flag_overlaps(data)
    sign_reversal_frames = flagging.flag_sign_reversals(data)
    delta_mismatch_frames = flagging.flag_delta_mismatches(data)
    custom_frames = custom_swap_detection(data)
    
    # Combine all frames
    all_frames = np.unique(np.concatenate([
        overlap_frames,
        sign_reversal_frames,
        delta_mismatch_frames,
        custom_frames
    ]))
    
    if debug:
        print(f"Found {len(all_frames)} potential swap frames")
    
    return all_frames
```

## Custom Metrics Calculation

### Implementing Custom Metrics

```python
def calculate_custom_metrics(
    data: pd.DataFrame,
    fps: float,
    window: int = 10
) -> pd.DataFrame:
    """Calculates custom metrics for the tracking data."""
    # Create a copy of the data
    metrics_data = data.copy()
    
    # Calculate basic metrics
    metrics_data['separation'] = metrics.calculate_separation(data)
    metrics_data['head_angle'] = metrics.calculate_head_angle(data)
    metrics_data['speed'] = metrics.calculate_speed(data, fps)
    
    # Calculate custom metrics
    # 1. Angular velocity
    angle_diff = np.diff(metrics_data['head_angle'])
    metrics_data['angular_velocity'] = np.append(angle_diff, 0) * fps
    
    # 2. Acceleration
    speed_diff = np.diff(metrics_data['speed'])
    metrics_data['acceleration'] = np.append(speed_diff, 0) * fps
    
    # 3. Curvature
    metrics_data['curvature'] = np.abs(metrics_data['angular_velocity']) / metrics_data['speed']
    metrics_data['curvature'] = metrics_data['curvature'].replace([np.inf, -np.inf], np.nan)
    
    return metrics_data
```

## Complete Advanced Example

Here's a complete example that combines all the advanced functionality:

```python
import pandas as pd
import numpy as np
from swap_correction.tracking import (
    utils,
    tracking_correction,
    filtering,
    metrics,
    flagging
)

# Load and validate data
data = utils.load_data('tracking_data.csv')
utils.validate_input_data(data)

# Apply custom filtering
filtered_data = custom_filtering_pipeline(data)

# Apply swap correction with custom detection
corrected_data = tracking_correction(
    data=filtered_data,
    fps=30.0,
    filterData=False,  # Already filtered
    swapCorrection=True,
    validate=True,
    removeErrors=True,
    interp=True
)

# Calculate custom metrics
data_with_metrics = calculate_custom_metrics(
    data=corrected_data,
    fps=30.0
)

# Save results
utils.save_data(data_with_metrics, 'processed_data.csv')
```

## Notes

1. These examples demonstrate advanced usage of the package.
2. Custom functions can be created to extend functionality.
3. Multiple detection methods can be combined for better results.
4. Custom metrics can be calculated based on specific needs.

## See Also

- [Basic Usage Examples](basic_usage.md)
- [API Reference](../api/main.md)
- [Core Concepts](../guides/core_concepts.md) 
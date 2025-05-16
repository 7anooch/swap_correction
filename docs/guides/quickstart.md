# Quick Start Guide

This guide provides a quick introduction to using the Swap Correction package.

## Basic Usage

### Loading Data

```python
import pandas as pd
from swap_correction.tracking import utils

# Load data from a CSV file
data = utils.load_data('tracking_data.csv')

# Validate the data
is_valid = utils.validate_input_data(
    data=data,
    required_columns=['xhead', 'yhead', 'xtail', 'ytail']
)
```

### Applying Swap Correction

```python
from swap_correction.tracking import tracking_correction

# Apply swap correction
corrected_data = tracking_correction(
    data=data,
    fps=30.0,
    filterData=True,
    swapCorrection=True,
    validate=True,
    removeErrors=True,
    interp=True,
    debug=True
)

# Save the corrected data
utils.save_data(corrected_data, 'corrected_data.csv')
```

## Advanced Usage

### Custom Filtering

```python
from swap_correction.tracking import filtering

# Apply Gaussian filter
gaussian_filtered = filtering.filter_gaussian(
    data=data,
    sigma=3
)

# Apply Savitzky-Golay filter
sgolay_filtered = filtering.filter_sgolay(
    data=data,
    window=45,
    order=4
)
```

### Manual Swap Detection

```python
from swap_correction.tracking import flagging

# Flag all potential swaps
swap_frames = flagging.flag_all_swaps(
    data=data,
    fps=30.0
)

# Flag specific types of swaps
overlap_frames = flagging.flag_overlaps(data)
sign_reversal_frames = flagging.flag_sign_reversals(data)
```

### Custom Correction Pipeline

```python
from swap_correction.tracking import (
    correction,
    filtering,
    flagging
)

# Create a custom correction pipeline
def custom_correction(data: pd.DataFrame, fps: float) -> pd.DataFrame:
    # Remove edge frames
    data = correction.remove_edge_frames(data)
    
    # Apply custom filtering
    data = filtering.filter_gaussian(data, sigma=3)
    
    # Correct tracking errors
    data = correction.correct_tracking_errors(data)
    
    # Remove errors
    data = correction.remove_overlaps(data)
    
    # Interpolate gaps
    data = correction.interpolate_gaps(data)
    
    return data
```

## Example Workflow

Here's a complete example workflow:

```python
import pandas as pd
import matplotlib.pyplot as plt
from swap_correction.tracking import (
    utils,
    tracking_correction,
    metrics
)

# Load and validate data
data = utils.load_data('tracking_data.csv')
utils.validate_input_data(data)

# Apply swap correction
corrected_data = tracking_correction(
    data=data,
    fps=30.0,
    filterData=True,
    swapCorrection=True,
    validate=True,
    removeErrors=True,
    interp=True
)

# Calculate metrics
data_with_metrics = metrics.calculate_metrics(
    data=corrected_data,
    fps=30.0
)

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(data_with_metrics['xhead'], data_with_metrics['yhead'], 'b-', label='Head')
plt.plot(data_with_metrics['xtail'], data_with_metrics['ytail'], 'r-', label='Tail')
plt.legend()
plt.title('Animal Trajectory')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()

# Save results
utils.save_data(data_with_metrics, 'processed_data.csv')
```

## Common Patterns

### Batch Processing

```python
import glob
from pathlib import Path

# Process multiple files
for file_path in glob.glob('data/*.csv'):
    # Load data
    data = utils.load_data(file_path)
    
    # Apply correction
    corrected_data = tracking_correction(
        data=data,
        fps=30.0,
        filterData=True,
        swapCorrection=True
    )
    
    # Save results
    output_path = Path('output') / Path(file_path).name
    utils.save_data(corrected_data, output_path)
```

### Error Handling

```python
try:
    # Load and process data
    data = utils.load_data('tracking_data.csv')
    corrected_data = tracking_correction(
        data=data,
        fps=30.0,
        filterData=True,
        swapCorrection=True
    )
except Exception as e:
    print(f"Error processing data: {e}")
```

### Debug Mode

```python
# Enable debug mode for detailed logging
corrected_data = tracking_correction(
    data=data,
    fps=30.0,
    filterData=True,
    swapCorrection=True,
    debug=True
)
```

## Next Steps

1. Read the [Core Concepts](core_concepts.md) guide
2. Explore the [API Reference](api/main.md)
3. Check out more [Examples](examples/basic_usage.md)

## Notes

1. Always validate your data before processing
2. Use appropriate filter parameters for your data
3. Enable debug mode for troubleshooting
4. Save intermediate results for long processing

## See Also

- [Core Concepts](core_concepts.md)
- [API Reference](api/main.md)
- [Examples](examples/basic_usage.md)
- [FAQ](faq.md) 
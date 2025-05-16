# Basic Usage Examples

This document provides basic examples of using the Swap Correction package.

## Loading and Processing Data

### Basic Data Loading

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

if is_valid:
    print("Data is valid")
else:
    print("Data validation failed")
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

## Filtering Data

### Using Different Filters

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

# Apply median filter
median_filtered = filtering.filter_median(
    data=data,
    win=5
)
```

## Calculating Metrics

### Basic Metrics

```python
from swap_correction.tracking import metrics

# Calculate all metrics
data_with_metrics = metrics.calculate_metrics(
    data=corrected_data,
    fps=30.0
)

# Calculate specific metrics
separation = metrics.calculate_separation(data)
head_angle = metrics.calculate_head_angle(data)
speed = metrics.calculate_speed(data, fps=30.0)
```

## Flagging Swaps

### Using Different Detection Methods

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
delta_mismatch_frames = flagging.flag_delta_mismatches(data)
```

## Complete Example

Here's a complete example that combines all the basic functionality:

```python
import pandas as pd
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

# Save results
utils.save_data(data_with_metrics, 'processed_data.csv')
```

## Notes

1. These examples demonstrate basic usage of the package.
2. More advanced examples are available in the Advanced Usage guide.
3. Custom configurations can be used for more specific needs.
4. Debug mode can be enabled for detailed logging.

## See Also

- [Advanced Usage Examples](advanced_usage.md)
- [API Reference](../api/main.md)
- [Core Concepts](../guides/core_concepts.md) 
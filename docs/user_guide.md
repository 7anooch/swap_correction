# User Guide

## Introduction

The Swap Correction package is designed to handle identity swaps in tracking data, particularly in behavioral experiments. This guide will walk you through the main features and how to use them effectively.

## Installation

### Requirements
- Python 3.8 or higher
- NumPy
- Pandas
- Matplotlib
- SciPy

### Installation Steps
1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Install the package

See the [README.md](../README.md) for detailed installation instructions.

## Basic Usage

### Loading Data

The package supports various data formats:

```python
from swap_correction import tracking_correction

# Load from CSV
data = tracking_correction.load_data("tracking.csv")

# Load from JSON
data = tracking_correction.load_data("tracking.json")

# Load from HDF5
data = tracking_correction.load_data("tracking.h5")
```

### Data Structure

The tracking data should be organized as follows:

```python
# Example data structure
{
    'frame': [0, 1, 2, ...],
    'x1': [x1_0, x1_1, x1_2, ...],  # x-coordinates for object 1
    'y1': [y1_0, y1_1, y1_2, ...],  # y-coordinates for object 1
    'x2': [x2_0, x2_1, x2_2, ...],  # x-coordinates for object 2
    'y2': [y2_0, y2_1, y2_2, ...],  # y-coordinates for object 2
    ...
}
```

### Correcting Swaps

Basic swap correction:

```python
from swap_correction import tracking_correction

# Load data
data = tracking_correction.load_data("tracking.csv")

# Correct swaps with default parameters
corrected = tracking_correction.correct_swaps(data)

# Correct swaps with custom parameters
params = {
    'min_swap_distance': 15,  # minimum distance for swap detection
    'max_swap_angle': 30,     # maximum angle for swap detection
    'smoothing_window': 5     # window size for smoothing
}
corrected = tracking_correction.correct_swaps(data, params)
```

### Computing Metrics

Calculate various behavioral metrics:

```python
from swap_correction import metrics

# Compute all metrics
results = metrics.compute_metrics(corrected)

# Compute specific metrics
speed = metrics.get_speed(corrected['x1'], corrected['y1'], fps=30)
distance = metrics.get_distance(corrected['x1'], corrected['y1'])
```

### Visualization

Plot tracking data and correction results:

```python
from swap_correction import plotting

# Plot trajectories
plotting.plot_trajectories(corrected)

# Plot with detected swaps
plotting.plot_swaps(corrected, swaps)

# Customize plots
plotting.plot_trajectories(
    corrected,
    title="Corrected Trajectories",
    xlabel="X Position",
    ylabel="Y Position",
    show_legend=True
)
```

## Advanced Usage

### Kalman Filtering

Apply Kalman filtering to smooth trajectories:

```python
from swap_correction import kalman_filter

# Filter trajectories
filtered = kalman_filter.filter_trajectory(
    corrected,
    params={
        'process_noise': 0.1,
        'measurement_noise': 0.1
    }
)
```

### Custom Metrics

Create custom metrics using the utility functions:

```python
from swap_correction import utils

# Calculate segment statistics
segments = np.array([[0, 100], [200, 300]])
stats = utils.metrics_by_segment(
    corrected['x1'],
    segments,
    buffer=5
)

# Get consecutive ranges
ranges = utils.get_consecutive_ranges([1, 2, 3, 5, 6, 8, 9, 10])
```

### Batch Processing

Process multiple files:

```python
import glob
from swap_correction import tracking_correction

# Process all CSV files in a directory
for file in glob.glob("data/*.csv"):
    data = tracking_correction.load_data(file)
    corrected = tracking_correction.correct_swaps(data)
    corrected.to_csv(f"corrected_{file}")
```

## Best Practices

### Data Preprocessing

1. Check for missing values:
```python
# Remove frames with missing values
data = data.dropna()

# Or interpolate missing values
data = data.interpolate()
```

2. Filter outliers:
```python
# Remove points with unrealistic speeds
speed = metrics.get_speed(data['x1'], data['y1'], fps=30)
data = data[speed < max_speed]
```

### Parameter Tuning

1. Start with default parameters
2. Adjust based on your data:
   - Increase `min_swap_distance` for noisy data
   - Decrease `max_swap_angle` for more conservative detection
   - Adjust `smoothing_window` based on frame rate

### Validation

Always validate your results:

```python
# Check correction quality
if corrector.validate_correction():
    print("Correction successful")
else:
    print("Correction may need review")

# Compare original and corrected data
plotting.plot_comparison(original, corrected)
```

## Troubleshooting

### Common Issues

1. **Missing Data**
   - Check file format
   - Verify data structure
   - Use appropriate preprocessing

2. **Poor Correction**
   - Adjust detection parameters
   - Check data quality
   - Consider manual review

3. **Performance Issues**
   - Use appropriate data types
   - Process in batches
   - Optimize parameters

### Getting Help

- Check the [API Reference](api.md)
- Open an issue on GitHub
- Contact the maintainers

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. 
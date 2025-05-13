# API Reference

## Core Modules

### tracking_correction

The main module for detecting and correcting identity swaps in tracking data.

#### Functions

##### `load_data(file_path: str) -> pd.DataFrame`
Load tracking data from a CSV file.

**Parameters:**
- `file_path`: Path to the CSV file containing tracking data

**Returns:**
- DataFrame containing the tracking data

##### `correct_swaps(data: pd.DataFrame, params: dict = None) -> pd.DataFrame`
Detect and correct identity swaps in tracking data.

**Parameters:**
- `data`: DataFrame containing tracking data
- `params`: Dictionary of correction parameters (optional)

**Returns:**
- DataFrame with corrected tracking data

### metrics

Module for computing behavioral metrics from tracking data.

#### Functions

##### `compute_metrics(data: pd.DataFrame) -> dict`
Compute various behavioral metrics from tracking data.

**Parameters:**
- `data`: DataFrame containing tracking data

**Returns:**
- Dictionary containing computed metrics

##### `get_speed(x: np.ndarray, y: np.ndarray, fps: int, npoints: int) -> np.ndarray`
Calculate speed from x,y position data.

**Parameters:**
- `x`: Array of x-coordinates
- `y`: Array of y-coordinates
- `fps`: Frames per second
- `npoints`: Number of points to use in calculation

**Returns:**
- Array of speed values

### plotting

Module for visualizing tracking data and correction results.

#### Functions

##### `plot_trajectories(data: pd.DataFrame, **kwargs) -> None`
Plot trajectories from tracking data.

**Parameters:**
- `data`: DataFrame containing tracking data
- `**kwargs`: Additional plotting parameters

##### `plot_swaps(data: pd.DataFrame, swaps: list, **kwargs) -> None`
Visualize detected identity swaps.

**Parameters:**
- `data`: DataFrame containing tracking data
- `swaps`: List of detected swap points
- `**kwargs`: Additional plotting parameters

### utils

Utility functions for data processing and analysis.

#### Functions

##### `get_bounds(data: np.ndarray, buffer: float = 0.05, floor: float = None, ceil: float = None) -> tuple`
Calculate bounds of data with optional buffer and limits.

**Parameters:**
- `data`: Input data array
- `buffer`: Buffer size as fraction of range
- `floor`: Minimum value
- `ceil`: Maximum value

**Returns:**
- Tuple of (min, max) values

##### `create_sample_matrix(samples: list[np.ndarray], length: int = None, toMin: bool = True) -> np.ndarray`
Create a matrix from jagged arrays.

**Parameters:**
- `samples`: List of arrays to combine
- `length`: Target length (optional)
- `toMin`: Whether to truncate to minimum length

**Returns:**
- 2D array containing the samples

##### `metrics_by_segment(data: np.ndarray, segs: np.ndarray, buffer: int = 0) -> np.ndarray`
Calculate statistics for each segment.

**Parameters:**
- `data`: Input data array
- `segs`: Array of segment boundaries
- `buffer`: Buffer size at segment edges

**Returns:**
- Array of statistics (mean, std, median) for each segment

### kalman_filter

Module for Kalman filtering of tracking data.

#### Functions

##### `filter_trajectory(data: np.ndarray, params: dict = None) -> np.ndarray`
Apply Kalman filter to smooth trajectory.

**Parameters:**
- `data`: Array of position data
- `params`: Filter parameters (optional)

**Returns:**
- Filtered trajectory data

## Data Structures

### TrackingData

Class representing tracking data with metadata.

#### Attributes
- `positions`: Array of position data
- `timestamps`: Array of timestamps
- `metadata`: Dictionary of metadata

#### Methods
- `load(file_path: str) -> TrackingData`
- `save(file_path: str) -> None`
- `get_metrics() -> dict`

### SwapCorrection

Class for managing swap correction operations.

#### Attributes
- `data`: TrackingData object
- `params`: Correction parameters
- `swaps`: List of detected swaps

#### Methods
- `detect_swaps() -> list`
- `correct_swaps() -> TrackingData`
- `validate_correction() -> bool`

## Constants

### Configuration

```python
DEFAULT_PARAMS = {
    'min_swap_distance': 10,
    'max_swap_angle': 45,
    'smoothing_window': 5
}
```

### File Formats

```python
SUPPORTED_FORMATS = [
    '.csv',
    '.json',
    '.h5'
]
```

## Error Handling

The package uses custom exceptions for better error handling:

### `TrackingError`
Base exception for tracking-related errors.

### `SwapDetectionError`
Raised when swap detection fails.

### `CorrectionError`
Raised when swap correction fails.

## Examples

### Basic Usage

```python
from swap_correction import tracking_correction, metrics, plotting

# Load data
data = tracking_correction.load_data("tracking.csv")

# Correct swaps
corrected = tracking_correction.correct_swaps(data)

# Compute metrics
results = metrics.compute_metrics(corrected)

# Plot results
plotting.plot_trajectories(corrected)
```

### Advanced Usage

```python
from swap_correction import SwapCorrection, TrackingData

# Create tracking data object
tracking = TrackingData.load("tracking.csv")

# Initialize correction
corrector = SwapCorrection(tracking, params={
    'min_swap_distance': 15,
    'max_swap_angle': 30
})

# Detect and correct swaps
swaps = corrector.detect_swaps()
corrected = corrector.correct_swaps()

# Validate results
if corrector.validate_correction():
    corrected.save("corrected_tracking.csv")
``` 
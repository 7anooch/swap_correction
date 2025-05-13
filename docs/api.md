# API Documentation

## Core Components

### Tracking Correction

The main module for detecting and correcting head-tail swaps in tracking data.

```python
from swap_corrector import tracking_correction as tc

corrected_data = tc.tracking_correction(
    data,
    fps,
    filterData=False,
    swapCorrection=True,
    validate=False,
    removeErrors=True,
    interp=True,
    debug=False
)
```

#### Parameters
- `data` (pd.DataFrame): DataFrame containing tracking data
- `fps` (float): Frames per second
- `filterData` (bool): Whether to filter data before correction
- `swapCorrection` (bool): Whether to correct head-tail swaps
- `validate` (bool): Whether to validate corrections
- `removeErrors` (bool): Whether to remove tracking errors
- `interp` (bool): Whether to interpolate gaps
- `debug` (bool): Whether to print debug messages

#### Returns
- DataFrame with corrected tracking data

### Configuration

#### SwapCorrectionConfig
Configuration for the swap correction pipeline.

```python
from swap_corrector import config

cfg = config.SwapCorrectionConfig(
    filter_data=False,      # Disable filtering
    fix_swaps=True,         # Enable swap correction
    validate=False,         # Disable validation
    remove_errors=True,     # Remove tracking errors
    interpolate=True,       # Interpolate gaps
    debug=False,            # Enable debug output
    diagnostic_plots=True,  # Generate plots
    show_plots=False,       # Display plots
    times=None             # Time range for plots
)
```

### Data Loading

#### PiVR Loader
Tools for loading and saving PiVR tracking data.

```python
from swap_corrector import pivr_loader as loader

# Load raw data
data = loader.load_raw_data('path/to/data')

# Save processed data
loader.save_data(data, 'path/to/output')
```

#### Methods

##### load_raw_data(path: str, filename: str = None) -> pd.DataFrame
Load tracking data from a file.

##### save_data(data: pd.DataFrame, path: str, filename: str = None) -> None
Save tracking data to a file.

##### get_all_settings(path: str) -> dict
Get all settings for a data file.

### Metrics

#### Metrics Calculation
Tools for calculating various metrics on tracking data.

```python
from swap_corrector import metrics

# Calculate orientation
orientation = metrics.get_orientation(data)

# Calculate head-tail separation
separation = metrics.get_delta_in_frame(data, 'head', 'tail')
```

#### Methods

##### get_orientation(data: pd.DataFrame) -> np.ndarray
Calculate body orientation.

##### get_delta_in_frame(data: pd.DataFrame, point1: str, point2: str) -> np.ndarray
Calculate separation between two points.

##### get_df_bounds(dataframes: List[pd.DataFrame], columns: List[str]) -> Tuple[float, float]
Get bounds of data across multiple dataframes.

### Visualization

#### Plotting
Tools for visualizing swap detection and correction results.

```python
from swap_corrector import plotting

# Compare trajectories
plotting.compare_filtered_trajectories(
    'path/to/data',
    output_path='results',
    file_name='trajectories.png'
)

# Compare distributions
plotting.compare_filtered_distributions(
    'path/to/data',
    output_path='results',
    file_name='distributions.png'
)

# Examine flags
plotting.examine_flags(
    'path/to/data',
    output_path='results',
    file_name='flags.png'
)
```

#### Methods

##### compare_filtered_trajectories(main_path: str, output_path: str = None, file_name: str = 'compare_trajectories.png', times: Tuple[float, float] = None, show: bool = True) -> None
Compare raw and processed trajectories.

##### compare_filtered_distributions(main_path: str, output_path: str = None, file_name: str = 'compare_distributions.png', show: bool = True) -> None
Compare distributions of raw and processed data.

##### examine_flags(main_path: str, output_path: str = None, show: bool = True, file_name: str = 'flags.png', times: Tuple[float, float] = None, label_frames: bool = False) -> None
Examine flagged frames and verified swap frames.

### Utilities

#### Utility Functions
Common utility functions used throughout the package.

```python
from swap_corrector import utils

# Get time axis
time = utils.get_time_axis(n_frames, fps)

# Get consecutive ranges
ranges = utils.get_consecutive_ranges(array)

# Filter array
filtered = utils.filter_array(array, filter)
```

#### Methods

##### get_time_axis(n_frames: int, fps: float) -> np.ndarray
Create time axis for plotting.

##### get_consecutive_ranges(array: np.ndarray) -> List[Tuple[int, int]]
Get ranges of consecutive True values.

##### filter_array(array: np.ndarray, filter: np.ndarray) -> np.ndarray
Filter array using boolean mask.

## Usage Examples

### Basic Usage

```python
from swap_corrector import tracking_correction as tc
from swap_corrector import pivr_loader as loader

# Load data
data = loader.load_raw_data('path/to/data')

# Process data
corrected_data = tc.tracking_correction(
    data,
    fps=30,
    filterData=False,
    swapCorrection=True,
    validate=False,
    removeErrors=True,
    interp=True
)

# Save results
loader.save_data(corrected_data, 'path/to/output')
```

### With Visualization

```python
from swap_corrector import plotting

# Create diagnostic plots
plotting.compare_filtered_trajectories(
    'path/to/data',
    output_path='results',
    file_name='trajectories.png'
)

plotting.compare_filtered_distributions(
    'path/to/data',
    output_path='results',
    file_name='distributions.png'
)

plotting.examine_flags(
    'path/to/data',
    output_path='results',
    file_name='flags.png'
)
```

### Custom Configuration

```python
from swap_corrector import config
from swap_corrector import tracking_correction as tc

# Create configuration
cfg = config.SwapCorrectionConfig(
    filter_data=False,
    fix_swaps=True,
    validate=False,
    remove_errors=True,
    interpolate=True,
    debug=False,
    diagnostic_plots=True,
    show_plots=False
)

# Use configuration
corrected_data = tc.tracking_correction(
    data,
    fps=30,
    filterData=cfg.filter_data,
    swapCorrection=cfg.fix_swaps,
    validate=cfg.validate,
    removeErrors=cfg.remove_errors,
    interp=cfg.interpolate,
    debug=cfg.debug
)
``` 
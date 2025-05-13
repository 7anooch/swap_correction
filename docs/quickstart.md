# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/swap_correction.git
cd swap_correction

# Install package
pip install -e .
```

## Basic Usage

### 1. Simple Correction

```python
from swap_corrector import tracking_correction as tc
from swap_corrector import pivr_loader as loader
import pandas as pd

# Load your tracking data
data = loader.load_raw_data('path/to/data')

# Process data with default settings
corrected_data = tc.tracking_correction(
    data,
    fps=30,  # Your frame rate
    filterData=False,
    swapCorrection=True,
    validate=False,
    removeErrors=True,
    interp=True
)

# Save results
loader.save_data(corrected_data, 'path/to/output')
```

### 2. With Visualization

```python
from swap_corrector import tracking_correction as tc
from swap_corrector import plotting, utils, metrics
from pathlib import Path

# Load data
data = loader.load_raw_data('path/to/data')

# Process data
corrected_data = tc.tracking_correction(data, fps=30)

# Create diagnostic plots
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

# Compare trajectories
plotting.compare_filtered_trajectories(
    'path/to/data',
    output_path=output_dir,
    file_name='trajectories.png'
)

# Compare distributions
plotting.compare_filtered_distributions(
    'path/to/data',
    output_path=output_dir,
    file_name='distributions.png'
)

# Examine flags
plotting.examine_flags(
    'path/to/data',
    output_path=output_dir,
    file_name='flags.png'
)
```

### 3. Custom Configuration

```python
from swap_corrector import config

# Create configuration
cfg = config.SwapCorrectionConfig(
    filter_data=False,      # Disable filtering
    fix_swaps=True,         # Enable swap correction
    validate=False,         # Disable validation
    remove_errors=True,     # Remove tracking errors
    interpolate=True,       # Interpolate gaps
    debug=False,            # Disable debug output
    diagnostic_plots=True,  # Generate plots
    show_plots=False        # Don't display plots
)

# Use configuration in tracking correction
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

## Data Format

Your input data should be a CSV file with the following columns:

```
X-Head,Y-Head,X-Tail,Y-Tail[,X-Midpoint,Y-Midpoint]
```

Example:
```csv
X-Head,Y-Head,X-Tail,Y-Tail
100.5,200.3,120.4,220.1
101.2,201.1,121.0,221.2
...
```

Note: Midpoint columns are optional. If not provided, they will be calculated automatically.

## Common Operations

### 1. Batch Processing

```python
from pathlib import Path
import pandas as pd

# Process multiple files
data_dir = Path('data')
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

for data_file in data_dir.glob('*.csv'):
    # Load data
    data = loader.load_raw_data(data_file)
    
    # Process
    corrected_data = tc.tracking_correction(data, fps=30)
    
    # Save results
    loader.save_data(corrected_data, output_dir / f"corrected_{data_file.name}")
```

### 2. Diagnostic Visualization

```python
from swap_corrector import plotting

# Create comprehensive report
plotting.compare_filtered_trajectories(
    'path/to/data',
    output_path='diagnostics',
    file_name='trajectories.png'
)

plotting.compare_filtered_distributions(
    'path/to/data',
    output_path='diagnostics',
    file_name='distributions.png'
)

plotting.examine_flags(
    'path/to/data',
    output_path='diagnostics',
    file_name='flags.png'
)
```

## Configuration Options

### Processing Options

```python
from swap_corrector import config

cfg = config.SwapCorrectionConfig(
    # Processing settings
    filter_data=False,      # Disable filtering
    fix_swaps=True,         # Enable swap correction
    validate=False,         # Disable validation
    remove_errors=True,     # Remove tracking errors
    interpolate=True,       # Interpolate gaps
    
    # Debug settings
    debug=False,            # Enable debug output
    diagnostic_plots=True,  # Generate plots
    show_plots=False,       # Display plots
    times=None             # Time range for plots (None -> show all)
)
```

## Troubleshooting

### Common Issues

1. **No Swaps Detected**
   - Check threshold values
   - Verify data format
   - Enable debug mode

2. **Too Many False Positives**
   - Adjust thresholds
   - Enable validation
   - Check data quality

### Getting Help

1. Check the [API Documentation](api.md)
2. Review the [Developer Guide](developer/developer_guide.md)
3. File an issue on GitHub 
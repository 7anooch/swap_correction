# PiVR Swap Correction Pipeline

A Python package for correcting tracking errors in PiVR (Pi-based Virtual Reality) data, specifically focused on head-tail swap corrections in animal tracking data.

## Features

- Detection and correction of head-tail swaps
- Removal of tracking errors
- Interpolation over short overlap segments
- Generation of diagnostic plots
- Comparison of raw and processed trajectories
- Configurable processing pipeline

## Installation

```bash
pip install -e .
```

## Usage

### Basic Usage

```python
from pivr_analysis_pipeline.swap_corrector import config, logger
from swap_correct import process_sample

# Configure processing
cfg = config.SwapCorrectionConfig(
    fix_swaps=True,
    remove_errors=True,
    interpolate=True,
    diagnostic_plots=True
)

# Process a single sample
process_sample("path/to/sample", cfg)
```

### Command Line Interface

```bash
python swap_correct.py
```

This will open a file dialog to select a PiVR trial folder or a parent folder containing multiple trials.

## Configuration

The processing pipeline can be configured using the `SwapCorrectionConfig` class:

```python
from pivr_analysis_pipeline.swap_corrector import config

cfg = config.SwapCorrectionConfig(
    # File settings
    filtered_data_filename="filtered_data.csv",
    
    # Processing settings
    fix_swaps=True,          # Correct head-tail swaps
    validate=False,          # Validate corrections
    remove_errors=True,      # Remove tracking errors
    interpolate=True,        # Interpolate over gaps
    filter_data=False,       # Apply data filtering
    
    # Debug settings
    debug=False,             # Enable debug messages
    diagnostic_plots=True,   # Generate diagnostic plots
    show_plots=False,        # Display plots
    
    # Plot settings
    times=None,              # Time range for plots
    
    # Logging settings
    log_level="INFO",        # Logging level
    log_file=None            # Log file path
)
```

## Project Structure

- `swap_correct.py`: Main entry point and high-level functions
- `swap_corrector/`:
  - `config.py`: Configuration management
  - `logger.py`: Logging configuration
  - `pivr_loader.py`: Data loading and file management
  - `plotting.py`: Visualization functions
  - `tracking_correction.py`: Core swap correction algorithms
  - `utils.py`: Utility functions
  - `kalman_filter.py`: Kalman filtering implementation
  - `metrics.py`: Metrics and calculations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

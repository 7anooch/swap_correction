# swap_correction

A Python package for correcting head-tail swaps in animal tracking data from PiVR (a virtual reality system for studying animal behavior, typically used with Drosophila larvae).

## Overview

This package detects and corrects tracking errors in PiVR data, specifically focusing on:
- **Head-tail swap detection**: Identifies frames where the head and tail labels have been swapped
- **Tracking error correction**: Removes overlaps and discontinuities in position data
- **Data interpolation**: Fills gaps in tracking data over short segments
- **Diagnostic visualization**: Generates plots to verify correction quality

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.10
- numpy >= 1.20.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0

## Quick Start

### Basic Usage

```python
from swap_correction import tracking_correction, pivr_loader

# Load raw data
data = pivr_loader.load_raw_data('/path/to/pivr/trial')
fps = pivr_loader.get_all_settings('/path/to/pivr/trial')['Framerate']

# Apply corrections
corrected_data = tracking_correction.tracking_correction(
    data, fps,
    swapCorrection=True,
    removeErrors=True,
    interp=True,
    validate=False
)

# Export corrected data
pivr_loader.export_to_PiVR('/path/to/pivr/trial', corrected_data)
```

### Command Line Usage

Run the main script to process one or more PiVR trials:

```bash
python -m swap_correction.swap_correct
```

This will:
1. Open a file dialog to select a PiVR trial folder (or parent folder with multiple trials)
2. Apply tracking corrections to all trials
3. Export corrected data as `*_level1.csv` files
4. Generate diagnostic trajectory comparison plots (if enabled)

## Configuration

Configuration flags are currently set at the top of `swap_correction/swap_correct.py`:

- `FIX_SWAPS`: Correct head-tail swaps using single-frame flags (default: `True`)
- `VALIDATE`: Attempt to correct missed swaps using segment-based metrics (default: `False`, not recommended)
- `REMOVE_ERRORS`: Set position values in frames where head/tail overlap to NaN (default: `True`)
- `INTERPOLATE`: Interpolate over short overlap segments (default: `True`)
- `FILTER_DATA`: Filter data before exporting (default: `False`, not recommended)
- `DEBUG`: Print debug messages (default: `False`)
- `DIAGNOSTIC_PLOTS`: Generate and save diagnostic figures (default: `True`)
- `SHOW_PLOTS`: Display diagnostic figures after saving (default: `True`)
- `TIMES`: Time range to show on plots, `None` for entire trajectory (default: `None`)

A comprehensive `config.yaml` file documents all parameters and magic numbers used throughout the codebase. This file is currently for documentation only and will be wired into the code in a future update.

## Algorithm Overview

### Swap Detection Methods

The package uses multiple complementary heuristics to detect head-tail swaps:

1. **Minimum Delta Mismatch**: Flags frames where the minimum distance between consecutive frames is between differently-labeled points (head-to-tail or tail-to-head)

2. **Sign Reversals**: Detects when the cross-product of tail-midpoint and midpoint-head vectors switches sign, indicating a swap

3. **Overlap-based Detection**: 
   - Overlap sign reversals: Detects sign changes across overlap regions
   - Overlap minimum-delta mismatches: Finds swaps at the end of overlap segments

4. **Segment-based Validation**: Uses assumptions about forward motion to detect remaining swaps in segments between overlaps (optional, currently not recommended)

### Correction Pipeline

1. **Remove edge frames**: Sets frames with zeroed positions at data edges to NaN
2. **Correct tracking errors**: Applies swap detection and correction
3. **Validate** (optional): Segment-based validation for missed swaps
4. **Remove overlaps**: Sets overlapping head/tail positions to NaN, with intelligent detection of which point is incorrectly placed
5. **Interpolate**: Fills gaps over short segments (default: ≤15 frames)
6. **Filter** (optional): Applies smoothing filter (default: Gaussian with σ=3)
7. **Round**: Rounds position data to 1 decimal place to remove roundoff errors

## Project Structure

```
swap_correction/
├── __init__.py              # Package initialization
├── swap_correct.py          # Main entry point script
├── tracking_correction.py   # Core correction algorithms
├── metrics.py               # Metrics and calculations
├── utils.py                 # Utility functions
├── pivr_loader.py           # PiVR data I/O
├── plotting.py              # Visualization functions
├── kalman_filter.py         # Kalman filter (currently unused)
└── legacy_code.py           # Deprecated code archive
```

## Testing

Run tests with pytest:

```bash
pytest
```

With coverage:

```bash
pytest --cov=swap_correction --cov-report=term-missing
```

## Development

### Code Style

The project uses:
- **Black** for code formatting (line length: 88)
- **isort** for import sorting (Black profile)
- **mypy** for type checking

Format code:

```bash
black swap_correction/
isort swap_correction/
```

## Documentation

- See `config.yaml` for comprehensive documentation of all configuration parameters
- Function docstrings provide detailed parameter descriptions
- Module-level docstrings describe each module's purpose

## License

[Add license information here]

## Citation

[Add citation information here]

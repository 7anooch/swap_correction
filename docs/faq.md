# Frequently Asked Questions

This document provides answers to common questions about the Swap Correction package.

## General Questions

### What is Swap Correction?

Swap Correction is a Python package for correcting head-tail swaps and tracking errors in animal tracking data. It provides functions for detecting and correcting swaps, filtering data, and calculating various metrics.

### What are head-tail swaps?

Head-tail swaps occur when the tracking system incorrectly identifies which end of the animal is the head and which is the tail. This can happen due to various factors such as:
- Overlapping frames
- Sudden changes in direction
- Tracking errors
- Low-quality video

### What types of data does Swap Correction support?

Swap Correction supports tracking data stored in pandas DataFrames with the following required columns:
- `xhead`, `yhead`: Head position coordinates
- `xtail`, `ytail`: Tail position coordinates

Optional columns include:
- `xctr`, `yctr`: Center position coordinates
- `xmid`, `ymid`: Midpoint position coordinates

## Installation

### How do I install Swap Correction?

You can install Swap Correction using pip:
```bash
pip install swap-correction
```

For development installation:
```bash
git clone https://github.com/your-username/swap-correction.git
cd swap-correction
pip install -e ".[dev]"
```

### What are the dependencies?

Required dependencies:
- Python 3.8 or higher
- numpy
- pandas
- scipy

Optional dependencies:
- matplotlib (for visualization)

## Usage

### How do I use Swap Correction?

Basic usage:
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
    interp=True
)
```

### What filtering methods are available?

Available filtering methods:
- Gaussian filter
- Savitzky-Golay filter
- Median filter
- Mean-median filter

### How do I choose the right filter?

The choice of filter depends on your data:
- Gaussian filter: Good for general smoothing
- Savitzky-Golay filter: Good for preserving higher moments
- Median filter: Good for removing outliers
- Mean-median filter: Good for both smoothing and outlier removal

## Troubleshooting

### My data has NaN values. What should I do?

You can handle NaN values in several ways:
1. Remove rows with NaN values:
   ```python
   from swap_correction.tracking import utils
   data = utils.remove_nan_rows(data)
   ```
2. Interpolate NaN values:
   ```python
   data = utils.interpolate_nan(data)
   ```

### The swap correction is not working well. What can I do?

Try these steps:
1. Check your data quality
2. Adjust filter parameters
3. Try different swap detection methods
4. Enable debug mode for detailed logging
5. Validate your data before processing

### How do I debug issues?

Enable debug mode:
```python
corrected_data = tracking_correction(
    data=data,
    fps=30.0,
    debug=True
)
```

## Performance

### How can I improve performance?

1. Use appropriate filter parameters
2. Process data in chunks
3. Use efficient data structures
4. Enable parallel processing where available

### How do I handle large datasets?

1. Process data in chunks
2. Use efficient data structures
3. Enable parallel processing
4. Optimize memory usage

## Contributing

### How can I contribute?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Update documentation
6. Create a pull request

### What are the coding standards?

We follow:
- PEP 8 style guide
- Type hints
- Docstrings
- Test coverage

## Support

### Where can I get help?

1. Check the documentation
2. Search existing issues
3. Create a new issue
4. Contact the maintainers

### How do I report bugs?

Create an issue with:
1. Clear description
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details

## Notes

1. This FAQ is a living document and will be updated as needed.
2. If you have a question not answered here, please create an issue.
3. Contributions to this FAQ are welcome.

## See Also

- [Installation Guide](guides/installation.md)
- [Quick Start Guide](guides/quickstart.md)
- [API Reference](api/main.md)
- [Examples](examples/basic_usage.md)
- [Contributing Guide](contributing.md) 
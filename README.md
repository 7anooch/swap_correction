# Swap Correction

A Python package for correcting tracking data in behavioral experiments, particularly focused on handling identity swaps and tracking artifacts.

## Overview

This package provides tools for:
- Loading and processing tracking data
- Detecting and correcting identity swaps
- Computing behavioral metrics
- Visualizing tracking and correction results
- Kalman filtering for smooth trajectories

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/swap_correction.git
cd swap_correction

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from swap_correction import tracking_correction, metrics, plotting

# Load tracking data
data = tracking_correction.load_data("path/to/tracking_data.csv")

# Detect and correct swaps
corrected_data = tracking_correction.correct_swaps(data)

# Compute behavioral metrics
metrics = metrics.compute_metrics(corrected_data)

# Visualize results
plotting.plot_trajectories(corrected_data)
```

## Features

### Tracking Correction
- Automatic detection of identity swaps
- Correction using multiple strategies (e.g., distance-based, velocity-based)
- Support for multiple tracked objects
- Configurable correction parameters

### Metrics
- Speed and velocity calculations
- Distance and angle measurements
- Behavioral state classification
- Custom metric computation

### Visualization
- Trajectory plotting
- Swap detection visualization
- Metric time series
- Interactive plots

### Utilities
- File I/O operations
- Data preprocessing
- Array manipulation
- Statistical analysis

## Documentation

Detailed documentation is available in the `docs` directory:
- [API Reference](docs/api.md)
- [User Guide](docs/user_guide.md)
- [Examples](docs/examples.md)
- [Contributing Guidelines](docs/contributing.md)

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=swap_correction
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:
```
@software{swap_correction2024,
  author = {Your Name},
  title = {Swap Correction},
  year = {2024},
  url = {https://github.com/yourusername/swap_correction}
}
```

## Contact

For questions and support, please open an issue on GitHub or contact [your-email@example.com].

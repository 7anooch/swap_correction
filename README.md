# Swap Correction

A Python package for correcting head-tail swaps and tracking errors in animal tracking data.

## Overview

Swap Correction provides functions for:
- Detecting and correcting head-tail swaps
- Filtering tracking data
- Calculating various metrics
- Handling common tracking errors

## Installation

```bash
pip install swap-correction
```

For development installation:
```bash
git clone https://github.com/your-username/swap-correction.git
cd swap-correction
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from swap_correction.tracking import tracking_correction

# Load your tracking data
data = pd.read_csv('tracking_data.csv')

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

# Save the corrected data
corrected_data.to_csv('corrected_data.csv', index=False)
```

## Features

- Multiple swap detection methods:
  - Overlap detection
  - Sign reversal detection
  - Delta mismatch detection
  - Overlap sign reversal detection
  - Overlap minimum mismatch detection

- Data filtering options:
  - Gaussian filter
  - Savitzky-Golay filter
  - Median filter
  - Mean-median filter

- Metrics calculation:
  - Position metrics (separation, vectors)
  - Angle metrics (head angle, orientation)
  - Motion metrics (speed, tortuosity)

## Documentation

- [Installation Guide](docs/guides/installation.md)
- [Quick Start Guide](docs/guides/quickstart.md)
- [Core Concepts](docs/guides/core_concepts.md)
- [API Reference](docs/api/main.md)
- [Examples](docs/examples/basic_usage.md)
- [Contributing Guide](docs/contributing.md)
- [FAQ](docs/faq.md)

## Requirements

- Python 3.8 or higher
- numpy
- pandas
- scipy

Optional:
- matplotlib (for visualization)

## Contributing

Contributions are welcome! Please see our [Contributing Guide](docs/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{swap_correction,
  author = {Your Name},
  title = {Swap Correction},
  year = {2024},
  url = {https://github.com/your-username/swap-correction}
}
```

## Acknowledgments

- Thanks to all contributors
- Inspired by various tracking analysis tools
- Built with support from the community

## Contact

- GitHub Issues: [Create an issue](https://github.com/your-username/swap-correction/issues)
- Email: [your-email@example.com](mailto:your-email@example.com)

## Notes

1. This package is under active development
2. Bug reports and feature requests are welcome
3. Documentation is continuously updated
4. Community contributions are encouraged

## See Also

- [Documentation](docs/index.md)
- [Changelog](docs/changelog.md)
- [Code of Conduct](docs/code_of_conduct.md)
- [Contributing Guide](docs/contributing.md)
- [FAQ](docs/faq.md)

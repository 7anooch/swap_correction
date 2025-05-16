# Swap Correction Documentation

Welcome to the Swap Correction documentation! This comprehensive guide will help you understand and use the Swap Correction package effectively.

## Overview

The Swap Correction package is designed to correct head-tail swaps and tracking errors in animal tracking data. It provides a robust set of tools for:

- Detecting and correcting head-tail swaps
- Filtering and smoothing tracking data
- Handling edge cases and errors
- Processing and analyzing tracking data

## Key Features

- Multiple swap detection methods
- Configurable filtering options
- Comprehensive error handling
- Flexible data processing pipeline
- Extensive helper functions for analysis

## Quick Links

- [Installation Guide](guides/installation.md)
- [Quick Start Tutorial](guides/quickstart.md)
- [API Reference](api/main.md)
- [Examples](examples/basic_usage.md)

## Getting Started

To get started with Swap Correction, follow these steps:

1. Install the package:
   ```bash
   pip install swap-correction
   ```

2. Import the main function:
   ```python
   from swap_correction import swap_corrector
   ```

3. Process your tracking data:
   ```python
   corrected_data = swap_corrector(
       data=your_data,
       fps=30.0,
       filterData=True,
       swapCorrection=True
   )
   ```

## Documentation Structure

This documentation is organized into several sections:

- **Getting Started**: Installation and basic usage
- **User Guide**: Detailed explanations of features and concepts
- **API Reference**: Complete API documentation
- **Examples**: Code examples and tutorials
- **Legacy Code**: Information about deprecated features
- **Contributing**: Guidelines for contributing to the project

## Support

If you encounter any issues or have questions:

- Check the [FAQ](guides/faq.md)
- Open an issue on [GitHub](https://github.com/yourusername/swap_correction)
- Contact the maintainers

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing/development.md) for details on how to:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation 
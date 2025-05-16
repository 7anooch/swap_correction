# Installation Guide

This guide provides detailed instructions for installing and setting up the Swap Correction package.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for development installation)

## Dependencies

Required dependencies:
- numpy>=1.20.0
- pandas>=1.3.0
- scipy>=1.7.0

Optional dependencies:
- matplotlib>=3.4.0 (for visualization)

## Installation Methods

### 1. Using pip

The simplest way to install Swap Correction is using pip:

```bash
pip install swap-correction
```

### 2. Installing from Source

To install from source:

```bash
# Clone the repository
git clone https://github.com/your-username/swap-correction.git
cd swap-correction

# Install the package
pip install -e .
```

### 3. Development Installation

For development installation:

```bash
# Clone the repository
git clone https://github.com/your-username/swap-correction.git
cd swap-correction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Verifying Installation

To verify the installation:

```python
import swap_correction
print(swap_correction.__version__)
```

## Common Issues

### Missing Dependencies

If you encounter missing dependency errors:

```bash
pip install -r requirements.txt
```

### Version Conflicts

If you encounter version conflicts:

```bash
pip install --upgrade swap-correction
```

### Permission Issues

If you encounter permission issues:

```bash
pip install --user swap-correction
```

## Environment Setup

### Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install swap-correction
```

### Conda Environment

```bash
# Create a conda environment
conda create -n swap-correction python=3.8

# Activate the environment
conda activate swap-correction

# Install the package
pip install swap-correction
```

## Configuration

After installation, you can create a configuration file:

```python
from swap_correction.tracking import utils

# Create default configuration
config = utils.create_default_config()

# Save configuration
utils.save_config(config, 'config.json')
```

## Next Steps

1. Read the [Quick Start Guide](quickstart.md)
2. Check out the [Examples](examples/basic_usage.md)
3. Review the [API Reference](api/main.md)

## Notes

1. Always use a virtual environment for installation
2. Keep your dependencies up to date
3. Check the [FAQ](faq.md) for common issues
4. Report any installation problems on GitHub

## See Also

- [Quick Start Guide](quickstart.md)
- [API Reference](api/main.md)
- [Examples](examples/basic_usage.md)
- [FAQ](faq.md) 
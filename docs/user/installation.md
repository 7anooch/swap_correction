# Installation Guide

## System Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Optional Dependencies
- CUDA (for GPU acceleration)
- OpenCV (for advanced visualization)

## Installation Methods

### 1. Installation from PyPI (Recommended)

```bash
pip install swap-corrector
```

### 2. Installation from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/swap_correction.git
cd swap_correction
```

2. Create and activate a virtual environment (recommended):
```bash
# On Unix/macOS
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

### 3. Development Installation

For development purposes, install with development dependencies:

```bash
pip install -e ".[dev]"
```

This will install additional tools for development:
- pytest (testing)
- black (code formatting)
- mypy (type checking)
- flake8 (linting)

## Verifying Installation

To verify your installation, run:

```bash
python -c "import swap_corrector; print(swap_corrector.__version__)"
```

## Configuration

After installation, you may want to configure the following:

1. **Data Directory**
   - Set up a directory for your tracking data
   - Configure paths in your environment

2. **Performance Settings**
   - Adjust memory usage limits
   - Configure GPU usage if available

3. **Logging**
   - Configure log levels
   - Set up log file locations

## Troubleshooting Installation

### Common Issues

1. **Missing Dependencies**
   - Error: "ModuleNotFoundError: No module named 'numpy'"
   - Solution: `pip install numpy`

2. **Version Conflicts**
   - Error: "Version conflict for package X"
   - Solution: Create a fresh virtual environment

3. **Permission Issues**
   - Error: "Permission denied"
   - Solution: Use `pip install --user` or run with sudo

### Getting Help

If you encounter issues during installation:

1. Check the [FAQ](faq.md)
2. Review the [Troubleshooting Guide](troubleshooting.md)
3. File an issue on GitHub

## Next Steps

After installation, proceed to:
- [Quick Start Guide](../quickstart.md)
- [Basic Usage Tutorial](tutorials/basic_usage.md)
- [Configuration Guide](../api/configuration.md) 
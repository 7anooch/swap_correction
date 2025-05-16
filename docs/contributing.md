# Contributing to Swap Correction

Thank you for your interest in contributing to the Swap Correction package! This guide will help you get started with contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/swap-correction.git
   cd swap-correction
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We follow the PEP 8 style guide for Python code. Please ensure your code follows these guidelines:

1. Use 4 spaces for indentation
2. Maximum line length of 88 characters
3. Use descriptive variable names
4. Add docstrings to all functions and classes
5. Use type hints for function parameters and return values

We use `black` for code formatting and `flake8` for linting. You can run these tools with:

```bash
black .
flake8
```

## Testing

We use `pytest` for testing. Please ensure all new code is covered by tests:

1. Write tests in the `tests` directory
2. Run tests with:
   ```bash
   pytest
   ```
3. Check test coverage with:
   ```bash
   pytest --cov=swap_correction
   ```

## Documentation

We use MkDocs with the Material theme for documentation. Please ensure all new features are documented:

1. Add docstrings to all functions and classes
2. Update relevant documentation files in the `docs` directory
3. Build and check documentation with:
   ```bash
   mkdocs serve
   ```

## Pull Request Process

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests and ensure they pass
4. Update documentation
5. Commit your changes with a descriptive message
6. Push to your fork
7. Create a pull request

## Code Review

All pull requests will be reviewed. Please ensure your code:

1. Follows the code style guidelines
2. Has appropriate test coverage
3. Is well documented
4. Addresses the issue or adds the feature as described

## Issue Reporting

When reporting issues, please include:

1. A clear description of the issue
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment details (Python version, OS, etc.)

## Feature Requests

When requesting features, please include:

1. A clear description of the feature
2. Use cases for the feature
3. Any relevant examples or references

## Development Workflow

1. Create an issue for your feature or bug fix
2. Create a branch for your changes
3. Make your changes
4. Write tests
5. Update documentation
6. Create a pull request
7. Address any review comments
8. Once approved, your changes will be merged

## Release Process

1. Update version number in `setup.py`
2. Update changelog
3. Create a release tag
4. Build and upload to PyPI

## Notes

1. Be respectful and considerate of other contributors
2. Follow the project's code of conduct
3. Ask for help if you need it
4. Have fun!

## See Also

- [Installation Guide](guides/installation.md)
- [Quick Start Guide](guides/quickstart.md)
- [API Reference](api/main.md)
- [Examples](examples/basic_usage.md) 
# Developer Guide

Welcome to the Swap Correction developer guide! This document provides detailed instructions and best practices for contributing to the development of the package.

## Table of Contents

- [Local Setup](#local-setup)
- [Code Structure](#code-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Code Style](#code-style)
- [Debugging](#debugging)
- [Release Process](#release-process)
- [Tips for Contributors](#tips-for-contributors)
- [Resources](#resources)

---

## Local Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/swap-correction.git
   cd swap-correction
   ```
2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```
4. **Pre-commit Hooks**
   ```bash
   pre-commit install
   ```
   This will enable automatic code formatting and linting on commit.

---

## Code Structure

- `swap_correction/` — Main package code
  - `tracking/` — Core tracking correction modules
    - `correction/`, `flagging/`, `filtering/`, `metrics/`, `utils/`
  - `cli.py` — Command-line interface
- `tests/` — Unit and integration tests
- `docs/` — Documentation (MkDocs)
- `examples/` — Example scripts and notebooks
- `setup.py`, `pyproject.toml` — Build and packaging configuration
- `.pre-commit-config.yaml` — Pre-commit hooks

---

## Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make Your Changes**
   - Add or modify code in the appropriate module
   - Add or update tests in `tests/`
   - Update documentation in `docs/` as needed
3. **Run Tests and Linting**
   ```bash
   pytest
   black .
   flake8
   mypy swap_correction/
   ```
4. **Commit and Push**
   ```bash
   git add .
   git commit -m "Describe your changes"
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**
   - Go to GitHub and open a PR against the `main` branch
   - Fill out the PR template and link related issues

---

## Testing

- Use `pytest` for all tests
- Place tests in the `tests/` directory, mirroring the package structure
- Run tests with:
  ```bash
  pytest
  ```
- Check coverage:
  ```bash
  pytest --cov=swap_correction
  ```
- Write tests for new features and bug fixes

---

## Documentation

- All public functions and classes must have docstrings (NumPy or Google style)
- Update or add documentation in `docs/` as needed
- Build and preview docs locally:
  ```bash
  mkdocs serve
  ```
- Documentation is published via MkDocs (see `mkdocs.yml` for structure)

---

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for all function signatures
- Format code with `black`:
  ```bash
  black .
  ```
- Lint code with `flake8`:
  ```bash
  flake8
  ```
- Sort imports with `isort`:
  ```bash
  isort .
  ```
- Check types with `mypy`:
  ```bash
  mypy swap_correction/
  ```

---

## Debugging

- Use the `debug=True` flag in main functions for verbose output
- Add logging as needed for new features
- Use breakpoints and IDE debuggers for step-through debugging
- Write tests for edge cases and error handling

---

## Release Process

1. Update the version in `setup.py` and `pyproject.toml`
2. Update `docs/changelog.md` with new changes
3. Commit and tag the release:
   ```bash
   git tag vX.Y.Z
   git push --tags
   ```
4. Build and upload to PyPI:
   ```bash
   python setup.py sdist bdist_wheel
   twine upload dist/*
   ```
5. Announce the release and update documentation as needed

---

## Tips for Contributors

- Write clear, descriptive commit messages
- Keep pull requests focused and small
- Add tests and documentation for all new features
- Ask questions and request reviews early
- Be respectful and follow the [Code of Conduct](../code_of_conduct.md)

---

## Resources

- [Contributing Guide](../contributing.md)
- [API Reference](../api/main.md)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [pytest Documentation](https://docs.pytest.org/)
- [black Documentation](https://black.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)

---

Happy coding! 
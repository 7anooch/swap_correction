# Developer Guide

## Project Structure

```
swap_corrector/
├── swap_corrector/
│   ├── __init__.py
│   ├── processor.py        # Main processing pipeline
│   ├── config.py          # Configuration management
│   ├── metrics.py         # Movement metrics calculation
│   ├── filtering.py       # Trajectory filtering
│   ├── visualization.py   # Visualization tools
│   ├── profiling.py       # Performance profiling
│   ├── detectors/         # Swap detection implementations
│   │   ├── __init__.py
│   │   ├── base.py       # Base detector class
│   │   ├── proximity.py  # Proximity-based detection
│   │   ├── speed.py      # Speed-based detection
│   │   └── turn.py       # Turn-based detection
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── docs/                # Documentation
├── scripts/             # Analysis and utility scripts
└── data/               # Example datasets
```

## Development Setup

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/swap_correction.git
cd swap_correction
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install -e .  # Install package in development mode
```

4. **Install Development Dependencies**
```bash
pip install pytest pytest-cov black mypy flake8
```

## Development Workflow

### 1. Code Style

We follow PEP 8 guidelines with these additions:
- Line length: 88 characters (Black default)
- Docstrings: Google style
- Type hints: Required for function arguments and return values

Example:
```python
def process_data(data: pd.DataFrame, config: Optional[Config] = None) -> pd.DataFrame:
    """Process tracking data.
    
    Args:
        data: Input tracking data
        config: Optional configuration object
        
    Returns:
        Processed tracking data
        
    Raises:
        ValueError: If data is empty
    """
    if data.empty:
        raise ValueError("Input data is empty")
    # Implementation
```

### 2. Testing

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=swap_corrector

# Run specific test file
pytest tests/test_processor.py
```

#### Writing Tests

1. **Test Organization**
   - One test file per module
   - Test class per component class
   - Test function per method/functionality

2. **Test Structure**
```python
def test_function_name():
    # Arrange
    input_data = ...
    expected_output = ...
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output
```

3. **Using Fixtures**
```python
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(...)

def test_with_fixture(sample_data):
    result = process_data(sample_data)
    assert result.shape == sample_data.shape
```

### 3. Performance Optimization

1. **Profiling**
```bash
# Profile complete pipeline
python scripts/profile_pipeline.py --data-dir data/

# Profile specific component
python -m cProfile -o output.prof your_script.py
```

2. **Memory Usage**
```python
@profile  # From memory_profiler
def memory_intensive_function():
    # Implementation
```

3. **Optimization Guidelines**
- Use NumPy operations instead of loops
- Implement caching for expensive calculations
- Consider using Numba for compute-intensive functions
- Profile before and after optimizations

### 4. Adding New Features

1. **New Detector**
```python
from .base import SwapDetector

class NewDetector(SwapDetector):
    """New swap detection method."""
    
    def __init__(self, config: Optional[SwapConfig] = None):
        super().__init__(config)
        # Additional initialization
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Implement detection logic."""
        # Implementation
```

2. **New Metric**
```python
def calculate_new_metric(data: pd.DataFrame) -> np.ndarray:
    """Calculate new movement metric.
    
    Args:
        data: Tracking data
        
    Returns:
        Array of metric values
    """
    # Implementation
```

### 5. Documentation

1. **Docstrings**
- Required for all public modules, classes, methods
- Include type hints
- Document exceptions
- Provide examples for complex functionality

2. **API Documentation**
- Update api.md when adding/modifying public interfaces
- Include usage examples
- Document configuration options

3. **Implementation Notes**
- Document complex algorithms
- Explain key design decisions
- Note performance considerations

### 6. Pull Request Process

1. **Before Submitting**
- Run full test suite
- Check code style (black, flake8)
- Update documentation
- Profile if performance-critical

2. **PR Template**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Added unit tests
- [ ] Updated existing tests
- [ ] Tested with real data

## Performance Impact
- [ ] Profiled changes
- [ ] No significant impact
- [ ] Performance improvement
```

## Common Tasks

### Adding a New Detector

1. Create new file in `detectors/`
2. Implement detector class inheriting from `SwapDetector`
3. Add tests in `tests/`
4. Update documentation
5. Add to `SwapProcessor`

### Modifying Configuration

1. Update `SwapConfig` or `SwapThresholds` in `config.py`
2. Update default values
3. Update documentation
4. Add migration guide if breaking

### Performance Optimization

1. Profile target component
2. Identify bottlenecks
3. Implement optimizations
4. Verify improvements
5. Document changes

## Troubleshooting

### Common Issues

1. **Memory Usage**
- Use `memory_profiler` to identify leaks
- Consider batch processing
- Clear cached results

2. **Performance**
- Profile specific components
- Check for unnecessary computations
- Consider parallelization

3. **Test Failures**
- Check test data
- Verify assumptions
- Update expected results

### Debug Tools

1. **Logging**
```python
import logging

logging.debug("Detailed information")
logging.info("General information")
logging.warning("Warning message")
```

2. **Visualization**
```python
from swap_corrector.visualization import SwapVisualizer

visualizer = SwapVisualizer()
visualizer.plot_trajectories(...)
```

3. **Profiling**
```python
from swap_corrector.profiling import profile_function

@profile_function("component_name")
def function_to_profile():
    # Implementation
``` 
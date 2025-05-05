# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/swap_correction.git
cd swap_correction

# Install package
pip install -e .
```

## Basic Usage

### 1. Simple Correction

```python
from swap_corrector.processor import SwapProcessor
import pandas as pd

# Load your tracking data
data = pd.read_csv('tracking_data.csv')

# Create processor with default settings
processor = SwapProcessor()

# Process data
corrected_data = processor.process(data)

# Save results
corrected_data.to_csv('corrected_data.csv', index=False)
```

### 2. With Visualization

```python
from swap_corrector.processor import SwapProcessor
from swap_corrector.visualization import SwapVisualizer
from pathlib import Path

# Load data
data = pd.read_csv('tracking_data.csv')

# Create processor and visualizer
processor = SwapProcessor()
visualizer = SwapVisualizer()

# Process data
corrected_data = processor.process(data)

# Create diagnostic plots
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

visualizer.plot_trajectories(
    data,
    corrected_data,
    processor.swap_segments,
    save_path=output_dir / 'trajectories.png'
)
```

### 3. Custom Configuration

```python
from swap_corrector.config import SwapConfig, SwapThresholds

# Create custom thresholds
thresholds = SwapThresholds(
    proximity=3.0,      # More lenient proximity threshold
    speed=15.0,        # Higher speed threshold
    angle=np.pi/3,     # Different angle threshold
    curvature=0.15     # Different curvature threshold
)

# Create configuration
config = SwapConfig(
    fps=30,            # Your frame rate
    thresholds=thresholds
)

# Create processor with custom config
processor = SwapProcessor(config=config)
```

### 4. With Filtering

```python
from swap_corrector.processor import SwapProcessor
from swap_corrector.filtering import TrajectoryFilter

# Load data
data = pd.read_csv('tracking_data.csv')

# Create components
filter = TrajectoryFilter()
processor = SwapProcessor()

# Pre-process data
filtered_data = filter.preprocess(data)

# Process data
corrected_data = processor.process(filtered_data)

# Post-process data
final_data = filter.postprocess(corrected_data, processor.swap_segments)
```

## Data Format

Your input data should be a CSV file with the following columns:

```
X-Head,Y-Head,X-Tail,Y-Tail[,X-Midpoint,Y-Midpoint]
```

Example:
```csv
X-Head,Y-Head,X-Tail,Y-Tail
100.5,200.3,120.4,220.1
101.2,201.1,121.0,221.2
...
```

Note: Midpoint columns are optional. If not provided, they will be calculated automatically.

## Common Operations

### 1. Batch Processing

```python
from pathlib import Path
import pandas as pd

# Process multiple files
data_dir = Path('data')
output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

processor = SwapProcessor()

for data_file in data_dir.glob('*.csv'):
    # Load data
    data = pd.read_csv(data_file)
    
    # Process
    corrected_data = processor.process(data)
    
    # Save results
    output_path = output_dir / f"corrected_{data_file.name}"
    corrected_data.to_csv(output_path, index=False)
```

### 2. Performance Analysis

```python
from swap_corrector.profiling import PerformanceProfiler

# Create profiler
profiler = PerformanceProfiler()

# Profile processing
processed_data, results = profiler.profile_pipeline(data)

# Analyze results
print(f"Processing time: {results['timing']['total_time']:.2f}s")
print(f"Memory usage: {results['memory']['peak_usage']:.1f}MB")
```

### 3. Diagnostic Visualization

```python
from swap_corrector.visualization import SwapVisualizer

visualizer = SwapVisualizer()

# Create comprehensive report
visualizer.create_diagnostic_report(
    raw_data=data,
    processed_data=corrected_data,
    swap_segments=processor.swap_segments,
    metrics=processor.metrics,
    results=processor.results,
    output_dir=Path('diagnostics')
)
```

## Configuration Options

### 1. Detection Thresholds

```python
thresholds = SwapThresholds(
    # Base thresholds
    proximity=2.0,      # Minimum distance between head and tail
    speed=10.0,        # Maximum expected speed
    angle=np.pi/4,     # Maximum angle change
    curvature=0.1,     # Maximum path curvature
    
    # High-speed thresholds
    acceleration_threshold=5.0,
    jerk_threshold=10.0,
    
    # Turn detection
    turn_radius_threshold=5.0,
    path_tortuosity_threshold=1.5
)
```

### 2. Processing Options

```python
correction_config = SwapCorrectionConfig(
    # Processing settings
    fix_swaps=True,          # Enable swap correction
    validate=True,           # Validate corrections
    remove_errors=True,      # Remove tracking errors
    interpolate=True,        # Interpolate gaps
    
    # Debug settings
    debug=False,             # Enable debug output
    diagnostic_plots=True    # Generate plots
)
```

## Troubleshooting

### Common Issues

1. **No Swaps Detected**
   - Check threshold values
   - Verify data format
   - Enable debug mode

2. **Too Many False Positives**
   - Adjust thresholds
   - Enable validation
   - Check data quality

3. **Performance Issues**
   - Use filtering
   - Process in batches
   - Profile bottlenecks

### Getting Help

1. Check the [API Documentation](api.md)
2. Review the [Developer Guide](developer_guide.md)
3. File an issue on GitHub 
# API Documentation

## Core Components

### SwapProcessor

The main class for detecting and correcting head-tail swaps in tracking data.

```python
from swap_corrector.processor import SwapProcessor

processor = SwapProcessor(config=None, correction_config=None)
corrected_data = processor.process(data)
```

#### Parameters
- `config` (SwapConfig, optional): Configuration for swap detection
- `correction_config` (SwapCorrectionConfig, optional): Configuration for correction pipeline

#### Methods

##### process(data: pd.DataFrame) -> pd.DataFrame
Process tracking data to detect and correct swaps.

**Args:**
- `data`: DataFrame containing tracking data with columns:
  - X-Head, Y-Head: Head coordinates
  - X-Tail, Y-Tail: Tail coordinates
  - X-Midpoint, Y-Midpoint: Midpoint coordinates

**Returns:**
- DataFrame with corrected tracking data

### Detectors

#### Base Detector
```python
from swap_corrector.detectors.base import SwapDetector

class CustomDetector(SwapDetector):
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        # Implementation
        pass
```

#### ProximityDetector
Detects swaps based on head-tail proximity.

```python
from swap_corrector.detectors.proximity import ProximityDetector

detector = ProximityDetector(config)
swaps = detector.detect(data)
```

#### SpeedDetector
Detects swaps based on speed anomalies.

```python
from swap_corrector.detectors.speed import SpeedDetector

detector = SpeedDetector(config)
swaps = detector.detect(data)
```

#### TurnDetector
Detects swaps during turning movements.

```python
from swap_corrector.detectors.turn import TurnDetector

detector = TurnDetector(config)
swaps = detector.detect(data)
```

### Configuration

#### SwapConfig
Configuration for swap detection parameters.

```python
from swap_corrector.config import SwapConfig

config = SwapConfig(
    fps=30,
    thresholds=SwapThresholds(),
    window_sizes={'speed': 5, 'acceleration': 7}
)
```

#### SwapThresholds
Thresholds for different types of swap detection.

```python
from swap_corrector.config import SwapThresholds

thresholds = SwapThresholds(
    proximity=2.0,      # pixels
    speed=10.0,        # pixels/frame
    angle=np.pi/4,     # radians
    curvature=0.1      # 1/pixel
)
```

### Filtering

#### TrajectoryFilter
Filters for smoothing and predicting trajectories.

```python
from swap_corrector.filtering import TrajectoryFilter

filter = TrajectoryFilter(config)
filtered_data = filter.preprocess(data)
smoothed_data = filter.postprocess(data, swap_segments)
```

#### Methods

##### preprocess(data: pd.DataFrame) -> pd.DataFrame
Apply pre-processing filtering to raw data.

##### postprocess(data: pd.DataFrame, swap_segments: List[Tuple]) -> pd.DataFrame
Apply post-processing filtering after swap correction.

##### predict_positions(data: pd.DataFrame, frames_ahead: int = 5) -> Tuple[np.ndarray, np.ndarray]
Predict future positions for validation.

### Visualization

#### SwapVisualizer
Tools for visualizing swap detection and correction results.

```python
from swap_corrector.visualization import SwapVisualizer

visualizer = SwapVisualizer(config)
visualizer.plot_trajectories(raw_data, processed_data, swap_segments)
```

#### Methods

##### plot_trajectories(raw_data, processed_data, swap_segments)
Plot raw and processed trajectories with swap segments highlighted.

##### plot_metrics(data, metrics, swap_segments)
Plot movement metrics with swap segments highlighted.

##### create_diagnostic_report(raw_data, processed_data, swap_segments, metrics, results, output_dir)
Create comprehensive diagnostic report with all plots.

### Profiling

#### PerformanceProfiler
Tools for profiling and optimizing performance.

```python
from swap_corrector.profiling import PerformanceProfiler

profiler = PerformanceProfiler(output_dir)
processed_data, results = profiler.profile_pipeline(data)
```

#### Methods

##### profile_pipeline(data, config=None, correction_config=None)
Profile the complete processing pipeline.

##### profile_component(component_name, func, *args, **kwargs)
Profile a specific component or function.

##### analyze_bottlenecks()
Analyze profiling results to identify bottlenecks.

## Usage Examples

### Basic Usage

```python
from swap_corrector.processor import SwapProcessor
from swap_corrector.config import SwapConfig
import pandas as pd

# Load data
data = pd.read_csv('tracking_data.csv')

# Create processor
processor = SwapProcessor()

# Process data
corrected_data = processor.process(data)

# Save results
corrected_data.to_csv('corrected_data.csv', index=False)
```

### Custom Configuration

```python
from swap_corrector.config import SwapConfig, SwapThresholds

# Create custom thresholds
thresholds = SwapThresholds(
    proximity=3.0,
    speed=15.0,
    angle=np.pi/3,
    curvature=0.15
)

# Create configuration
config = SwapConfig(
    fps=30,
    thresholds=thresholds,
    window_sizes={
        'speed': 7,
        'acceleration': 9,
        'curvature': 7
    }
)

# Create processor with custom config
processor = SwapProcessor(config=config)
```

### With Filtering and Visualization

```python
from swap_corrector.filtering import TrajectoryFilter
from swap_corrector.visualization import SwapVisualizer
from pathlib import Path

# Create components
filter = TrajectoryFilter(config)
visualizer = SwapVisualizer()

# Pre-process data
filtered_data = filter.preprocess(data)

# Process data
processor = SwapProcessor(config)
corrected_data = processor.process(filtered_data)

# Create visualizations
output_dir = Path('results')
visualizer.create_diagnostic_report(
    data,
    corrected_data,
    processor.swap_segments,
    processor.metrics,
    processor.results,
    output_dir
)
```

### Performance Profiling

```python
from swap_corrector.profiling import PerformanceProfiler
from pathlib import Path

# Create profiler
profiler = PerformanceProfiler(output_dir=Path('profiling_results'))

# Profile pipeline
processed_data, results = profiler.profile_pipeline(data)

# Analyze bottlenecks
bottlenecks = profiler.analyze_bottlenecks()

# Generate optimization report
profiler.generate_optimization_report(bottlenecks)
``` 
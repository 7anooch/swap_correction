# Streamlined Swap Correction Implementation Plan

## Phase 1: Core Metrics (1-2 weeks)

### 1.1 Essential Metrics Module
```python
# metrics.py
class MovementMetrics:
    def __init__(self, data: pd.DataFrame, fps: int):
        self.data = data
        self.fps = fps
        
    def get_speed(self, point: str) -> np.ndarray:
        """Calculate speed for head/tail/midpoint"""
        dx = np.diff(self.data[f'X-{point}'])
        dy = np.diff(self.data[f'Y-{point}'])
        speed = np.sqrt(dx**2 + dy**2) * self.fps
        return np.concatenate([speed, [speed[-1]]])
        
    def get_acceleration(self, point: str) -> np.ndarray:
        """Calculate acceleration"""
        speed = self.get_speed(point)
        return np.gradient(speed) * self.fps
        
    def get_angular_velocity(self) -> np.ndarray:
        """Calculate angular velocity between head and tail"""
        angles = np.arctan2(
            self.data['Y-Head'] - self.data['Y-Tail'],
            self.data['X-Head'] - self.data['X-Tail']
        )
        return np.gradient(angles) * self.fps
        
    def get_curvature(self) -> np.ndarray:
        """Calculate path curvature for turn detection"""
        dx = np.gradient(self.data['X-Midpoint'])
        dy = np.gradient(self.data['Y-Midpoint'])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        return np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy)**1.5
```

### 1.2 Configuration
```python
# config.py
SWAP_THRESHOLDS = {
    'proximity': 2.0,  # pixels
    'speed': 10.0,    # pixels/frame
    'angle': np.pi/4, # radians
    'curvature': 0.1  # 1/pixel
}

def adjust_thresholds(speed: np.ndarray) -> dict:
    """Adjust thresholds based on movement speed"""
    mean_speed = np.mean(speed)
    return {
        'proximity': SWAP_THRESHOLDS['proximity'] * (1 + 0.1 * mean_speed),
        'speed': SWAP_THRESHOLDS['speed'] * (1 + 0.05 * mean_speed),
        'angle': SWAP_THRESHOLDS['angle'],
        'curvature': SWAP_THRESHOLDS['curvature']
    }
```

## Phase 2: Swap Detection (2-3 weeks)

### 2.1 Detector Base
```python
# detectors/base.py
class SwapDetector:
    def __init__(self):
        self.metrics = None
        
    def setup(self, data: pd.DataFrame, fps: int):
        """Initialize metrics for detection"""
        self.metrics = MovementMetrics(data, fps)
        self.thresholds = adjust_thresholds(
            self.metrics.get_speed('Midpoint')
        )
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect swaps in data"""
        raise NotImplementedError
```

### 2.2 Category Implementations
```python
# detectors/categories.py
class ProximityDetector(SwapDetector):
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect close proximity swaps"""
        dist = np.sqrt(
            (data['X-Head'] - data['X-Tail'])**2 +
            (data['Y-Head'] - data['Y-Tail'])**2
        )
        return dist < self.thresholds['proximity']

class SpeedDetector(SwapDetector):
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect high-speed swaps"""
        speed = self.metrics.get_speed('Midpoint')
        return speed > self.thresholds['speed']

class TurnDetector(SwapDetector):
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect turn-related swaps"""
        curvature = self.metrics.get_curvature()
        ang_vel = self.metrics.get_angular_velocity()
        return (curvature > self.thresholds['curvature']) & \
               (np.abs(ang_vel) > self.thresholds['angle'])
```

## Phase 3: Integration and Testing (2-3 weeks)

### 3.1 Main Processing
```python
# processor.py
class SwapProcessor:
    def __init__(self):
        self.detectors = [
            ProximityDetector(),
            SpeedDetector(),
            TurnDetector()
        ]
    
    def process_experiment(self, data: pd.DataFrame, fps: int) -> dict:
        """Process single experiment data"""
        results = {}
        for detector in self.detectors:
            detector.setup(data, fps)
            swaps = detector.detect(data)
            results[detector.__class__.__name__] = swaps
        return results
```

### 3.2 Testing Framework
```python
# tests/test_detectors.py
def test_detector(detector_class, test_data, expected_swaps):
    """Test single detector with real data"""
    detector = detector_class()
    detector.setup(test_data, fps=30)
    detected = detector.detect(test_data)
    assert np.array_equal(detected, expected_swaps)

# tests/test_integration.py
def test_full_pipeline(experiment_data):
    """Test complete processing pipeline"""
    processor = SwapProcessor()
    results = processor.process_experiment(experiment_data, fps=30)
    validate_results(results, experiment_data)
```

## Timeline

1. **Week 1-2: Core Metrics**
   - Implement metrics module
   - Basic threshold configuration
   - Unit tests for metrics

2. **Week 3-5: Detectors**
   - Implement detectors for each category
   - Detector validation
   - Integration with metrics

3. **Week 6-7: Integration**
   - Complete processing pipeline
   - Full system testing
   - Performance optimization

## Success Metrics

1. **Correction Rates**
   - Close Proximity: >90% (from 84.9%)
   - High Speed: >70% (from 33.3%)
   - Turn-related: >90% (from 84.2%)

2. **Performance**
   - Processing time: <100ms per frame
   - False positive rate: <1%
   - False negative rate: <5%

## Dependencies
```requirements.txt
numpy>=1.21.0
pandas>=1.3.0
pytest>=6.2.0
``` 
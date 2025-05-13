"""Tests for swap detectors."""

import pytest
import numpy as np
import pandas as pd
from swap_corrector.config import SwapConfig
from swap_corrector.detectors.base import SwapDetector
from typing import Dict, Any, List, Tuple

class MockDetector(SwapDetector):
    """Mock detector for testing."""
    
    def __init__(self, config: SwapConfig):
        """Initialize mock detector."""
        super().__init__(config)
        self.metrics = None
        
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate mock metrics."""
        metrics = {
            'test_metric': 1.0,
            'speed': np.zeros(len(data)),
            'acceleration': np.zeros(len(data)),
            'turn_angle': np.zeros(len(data))
        }
        self.metrics = metrics
        return metrics
        
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect swaps every 10th frame."""
        if data.empty:
            return np.array([], dtype=bool)
            
        if self.data is None:
            self.setup(data)
            
        # For test data with all zeros, return no swaps
        if np.all(data['X-Head'] == 0) and np.all(data['Y-Head'] == 0) and \
           np.all(data['X-Tail'] == 0) and np.all(data['Y-Tail'] == 0):
            return np.zeros(len(data), dtype=bool)
            
        swaps = np.zeros(len(data), dtype=bool)
        swaps[::10] = True  # Flag every 10th frame
        return swaps
        
    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate mock confidence."""
        return 0.5
        
    def get_confidence(self, data: pd.DataFrame) -> np.ndarray:
        """Get confidence scores."""
        if data.empty:
            return np.array([], dtype=float)
            
        if self.data is None:
            self.setup(data)
            
        confidences = np.zeros(len(data))
        confidences[::10] = 1.0  # High confidence for every 10th frame
        return confidences
        
    def get_swap_segments(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Get swap segments."""
        if data.empty:
            return []
            
        swaps = self.detect(data)
        segments = []
        start_idx = None
        
        for i in range(len(swaps)):
            if swaps[i]:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                segments.append((start_idx, i-1))
                start_idx = None
                
        if start_idx is not None:
            segments.append((start_idx, len(swaps)-1))
            
        return segments
        
    def validate_swap(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """Validate a potential swap."""
        if self.data is None:
            self.setup(data)
            
        # Mock validation: always return True for frames divisible by 10
        return start % 10 == 0

def test_detector_initialization():
    """Test detector initialization."""
    config = SwapConfig()
    detector = MockDetector(config)
    
    assert isinstance(detector.config, SwapConfig)
    assert isinstance(detector.metrics, dict)
    assert isinstance(detector.thresholds, dict)

def test_detector_setup(mock_detector, sample_data):
    """Test detector setup with data."""
    data, _ = sample_data
    mock_detector.setup(data)
    
    assert isinstance(mock_detector.metrics, dict)
    assert 'test_metric' in mock_detector.metrics
    assert mock_detector.metrics['test_metric'] == 1.0

def test_detector_detect(mock_detector, sample_data):
    """Test detector detection method."""
    data, _ = sample_data
    mock_detector.setup(data)
    result = mock_detector.detect(data)
    
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    assert len(result) == len(data)

def test_detector_validate_swap(mock_detector, sample_data):
    """Test detector swap validation."""
    data, _ = sample_data
    mock_detector.setup(data)
    result = mock_detector.validate_swap(data, 0, 10)
    
    assert isinstance(result, bool)

def test_detector_get_confidence(mock_detector, sample_data):
    """Test detector confidence calculation."""
    data, _ = sample_data
    mock_detector.setup(data)
    confidences = mock_detector.get_confidence(data)
    
    assert isinstance(confidences, np.ndarray)
    assert len(confidences) == len(data)
    assert np.all((confidences >= 0) & (confidences <= 1))

def test_detector_calculate_metrics(mock_detector, sample_data):
    """Test detector metrics calculation."""
    data, _ = sample_data
    metrics = mock_detector._calculate_metrics(data)
    
    assert isinstance(metrics, dict)
    assert 'test_metric' in metrics
    assert metrics['test_metric'] == 1.0

def test_detector_calculate_confidence(mock_detector):
    """Test detector confidence calculation."""
    metrics = {'test_metric': 1.0}
    confidence = mock_detector._calculate_confidence(metrics)
    
    assert isinstance(confidence, float)
    assert confidence == 0.5

@pytest.fixture
def mock_detector():
    """Create a mock detector instance."""
    config = SwapConfig()
    return MockDetector(config)

def test_detector_initialization(mock_detector):
    """Test detector initialization."""
    assert isinstance(mock_detector.config, SwapConfig)
    assert mock_detector.metrics is None
    assert mock_detector._current_thresholds is None

def test_detector_setup(mock_detector, sample_data):
    """Test detector setup with data."""
    data, fps = sample_data
    mock_detector.config.fps = fps
    mock_detector.setup(data)
    
    assert mock_detector.metrics is not None
    assert mock_detector._current_thresholds is not None
    assert isinstance(mock_detector._current_thresholds, dict)
    
    # Check that thresholds were updated
    assert 'proximity' in mock_detector._current_thresholds
    assert 'speed' in mock_detector._current_thresholds
    assert 'angle' in mock_detector._current_thresholds
    assert 'curvature' in mock_detector._current_thresholds
    assert 'body_length' in mock_detector._current_thresholds

def test_update_thresholds_without_setup(mock_detector):
    """Test that updating thresholds without setup raises an error."""
    with pytest.raises(RuntimeError):
        mock_detector._update_thresholds()

def test_get_swap_segments():
    """Test conversion of frame-by-frame detections to segments."""
    detector = MockDetector(SwapConfig())
    
    # Test empty case
    empty_data = pd.DataFrame()
    assert detector.get_swap_segments(empty_data) == []
    
    # Test no swaps case
    no_swaps_data = pd.DataFrame({'X-Head': np.zeros(10), 'Y-Head': np.zeros(10),
                                 'X-Tail': np.zeros(10), 'Y-Tail': np.zeros(10)})
    assert detector.get_swap_segments(no_swaps_data) == []
    
    # Test single segment
    single_segment_data = pd.DataFrame({
        'X-Head': np.ones(10), 'Y-Head': np.ones(10),
        'X-Tail': np.ones(10), 'Y-Tail': np.ones(10)
    })
    segments = detector.get_swap_segments(single_segment_data)
    assert len(segments) == 1
    assert segments[0][0] == 0  # First swap should be at frame 0
    
    # Test multiple segments
    multi_segment_data = pd.DataFrame({
        'X-Head': np.ones(20), 'Y-Head': np.ones(20),
        'X-Tail': np.ones(20), 'Y-Tail': np.ones(20)
    })
    segments = detector.get_swap_segments(multi_segment_data)
    assert len(segments) == 2  # Should have swaps at frames 0 and 10
    assert segments[0][0] == 0  # First swap should be at frame 0
    assert segments[1][0] == 10  # Second swap should be at frame 10

def test_mock_detector_methods(mock_detector, sample_data):
    """Test the mock detector's implementations."""
    data, fps = sample_data
    mock_detector.config.fps = fps
    mock_detector.setup(data)
    
    # Test detection
    swaps = mock_detector.detect(data)
    assert isinstance(swaps, np.ndarray)
    assert swaps.dtype == bool
    assert len(swaps) == len(data)
    # Every 10th frame should be flagged (0, 10, 20, ..., 90)
    assert np.sum(swaps) == 10
    
    # Test validation
    assert mock_detector.validate_swap(data, 0, 10) is True
    
    # Test confidence
    confidences = mock_detector.get_confidence(data)
    assert isinstance(confidences, np.ndarray)
    assert len(confidences) == len(data)
    assert np.all((confidences >= 0) & (confidences <= 1))
    # Every 10th frame should have high confidence
    assert np.all(confidences[::10] == 1.0) 
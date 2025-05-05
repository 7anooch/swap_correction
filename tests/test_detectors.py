"""Tests for swap detectors."""

import pytest
import numpy as np
import pandas as pd
from swap_corrector.config import SwapConfig
from swap_corrector.detectors import SwapDetector

class MockDetector(SwapDetector):
    """Mock detector for testing base class functionality."""
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Mock detection that flags every 10th frame."""
        return np.arange(len(data)) % 10 == 0
    
    def validate_swap(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """Mock validation that always returns True."""
        return True
    
    def get_confidence(self, data: pd.DataFrame, start: int, end: int) -> float:
        """Mock confidence that always returns 0.8."""
        return 0.8

@pytest.fixture
def mock_detector():
    """Create a mock detector instance."""
    config = SwapConfig(fps=30)
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
    assert detector.get_swap_segments(np.array([])) == []
    assert detector.get_swap_segments(np.zeros(10, dtype=bool)) == []
    
    # Test single segment
    swaps = np.array([0, 0, 1, 1, 1, 0, 0], dtype=bool)
    segments = detector.get_swap_segments(swaps)
    assert segments == [(2, 5)]
    
    # Test multiple segments
    swaps = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
    segments = detector.get_swap_segments(swaps)
    assert segments == [(0, 2), (4, 7)]
    
    # Test edge cases
    swaps = np.array([1, 0, 0, 1], dtype=bool)
    segments = detector.get_swap_segments(swaps)
    assert segments == [(0, 1), (3, 4)]

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
    assert mock_detector.get_confidence(data, 0, 10) == 0.8 
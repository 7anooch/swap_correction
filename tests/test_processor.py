"""Tests for the swap processor."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from swap_corrector.config import SwapConfig, SwapCorrectionConfig
from swap_corrector.processor import SwapProcessor
from swap_corrector.metrics import MovementMetrics
from swap_corrector.detectors.base import SwapDetector

@pytest.fixture
def sample_data():
    """Create sample data with known swaps."""
    # Create a sequence of frames
    n_frames = 200
    data = pd.DataFrame()
    
    # Create circular movement
    t = np.linspace(0, 4*np.pi, n_frames)
    radius = 10
    
    # Base movement
    x_head = radius * np.cos(t)
    y_head = radius * np.sin(t)
    x_tail = radius * np.cos(t + np.pi)
    y_tail = radius * np.sin(t + np.pi)
    
    # Add different types of swaps
    
    # 1. Proximity swap (frames 50-55)
    x_head[50:55] = x_tail[50:55]
    y_head[50:55] = y_tail[50:55]
    x_tail[50:55] = x_head[50:55]
    y_tail[50:55] = y_head[50:55]
    
    # 2. Speed-based swap (frames 100-105)
    speed_factor = 3.0
    t_speed = t[100:105] * speed_factor
    x_head[100:105] = radius * np.cos(t_speed)
    y_head[100:105] = radius * np.sin(t_speed)
    x_tail[100:105] = radius * np.cos(t_speed + np.pi)
    y_tail[100:105] = radius * np.sin(t_speed + np.pi)
    
    # 3. Turn-based swap (frames 150-155)
    turn_factor = 2.0
    t_turn = t[150:155] * turn_factor
    x_head[150:155] = radius * np.cos(t_turn)
    y_head[150:155] = radius * np.sin(t_turn)
    x_tail[150:155] = radius * np.cos(t_turn + np.pi)
    y_tail[150:155] = radius * np.sin(t_turn + np.pi)
    
    # Create DataFrame
    data['X-Head'] = x_head
    data['Y-Head'] = y_head
    data['X-Tail'] = x_tail
    data['Y-Tail'] = y_tail
    data['X-Midpoint'] = (x_head + x_tail) / 2
    data['Y-Midpoint'] = (y_head + y_tail) / 2
    
    # Return tuple of (data, fps)
    return data, 30  # 30 fps is a common frame rate

@pytest.fixture
def processor(config):
    """Create a processor instance."""
    correction_config = SwapCorrectionConfig(
        validate=True,
        debug=True
    )
    return SwapProcessor(config, correction_config)

@pytest.fixture
def initialized_processor(processor, sample_data):
    """Create an initialized processor instance."""
    data, fps = sample_data
    processor.config.fps = fps
    processor.config.debug = True  # Ensure debug mode is active
    processor.metrics = MovementMetrics(data, fps)
    for detector in processor.detectors.values():
        detector.setup(data)
    return processor

def test_processor_initialization(processor):
    """Test processor initialization."""
    assert isinstance(processor.config, SwapConfig)
    assert processor.metrics is None
    assert all(isinstance(detector, SwapDetector) for detector in processor.detectors.values())
    assert all(detector.metrics is None for detector in processor.detectors.values())
    assert all(detector._current_thresholds is None for detector in processor.detectors.values())

def test_detector_setup(processor, sample_data):
    """Test detector setup."""
    # Process data to trigger setup
    processor.process(sample_data)
    
    # Check that all detectors are set up
    for detector in processor.detectors.values():
        assert detector.metrics is not None
        assert detector._current_thresholds is not None

def test_swap_detection(initialized_processor, sample_data):
    """Test swap detection."""
    data, _ = sample_data
    # Get segments per detector
    segments_dict = initialized_processor.get_detector_segments(data)
    # Check that detectors found swaps
    assert any(len(segments) > 0 for segments in segments_dict.values())

def test_detection_combination(initialized_processor, sample_data):
    """Test combination of detector results."""
    data, _ = sample_data
    # Run detectors
    detector_results = initialized_processor._run_detectors(data)
    
    # Combine detections
    combined_segments = initialized_processor._combine_detections(detector_results)
    
    # Check that segments are combined correctly
    assert len(combined_segments) > 0
    for start, end, detector, confidence in combined_segments:
        assert start < end
        assert detector in initialized_processor.detectors
        assert 0 <= confidence <= 1

def test_correction_application(initialized_processor, sample_data):
    """Test application of corrections."""
    data, _ = sample_data
    # Get segments to correct
    detector_results = initialized_processor._run_detectors(data)
    combined_segments = initialized_processor._combine_detections(detector_results)
    
    # Apply corrections
    corrected_data = initialized_processor._apply_corrections(data, combined_segments)
    
    # Check that corrections were applied
    assert not corrected_data.equals(data)
    assert not corrected_data.isna().any().any()

def test_correction_validation(processor, sample_data):
    """Test validation of corrections."""
    # Process data with validation
    corrected_data = processor.process(sample_data)
    
    # Check that midpoints are correctly updated
    assert np.allclose(
        corrected_data['X-Midpoint'],
        (corrected_data['X-Head'] + corrected_data['X-Tail']) / 2
    )
    assert np.allclose(
        corrected_data['Y-Midpoint'],
        (corrected_data['Y-Head'] + corrected_data['Y-Tail']) / 2
    )

def test_full_pipeline(processor, sample_data):
    """Test the complete processing pipeline."""
    # Process data
    corrected_data = processor.process(sample_data)
    # Check that output has same shape as input
    assert corrected_data.shape == sample_data[0].shape
    # Check that all required columns are present
    required_columns = [
        'X-Head', 'Y-Head',
        'X-Tail', 'Y-Tail',
        'X-Midpoint', 'Y-Midpoint'
    ]
    assert all(col in corrected_data.columns for col in required_columns)
    # Check that no NaN values were introduced
    assert not corrected_data.isna().any().any()

def test_error_handling(initialized_processor):
    """Test error handling."""
    # Test with empty DataFrame
    with pytest.raises(ValueError, match="Empty data"):
        initialized_processor.process(pd.DataFrame())
    
    # Test with missing columns
    bad_data = pd.DataFrame({'X-Head': [0], 'Y-Head': [0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        initialized_processor.process(bad_data)

def test_confidence_thresholding(initialized_processor, sample_data):
    """Test confidence thresholding in detection combination."""
    data, _ = sample_data
    # Set very high confidence threshold
    initialized_processor.config.thresholds.confidence_threshold = 0.99
    
    # Process data
    detector_results = initialized_processor._run_detectors(data)
    combined_segments = initialized_processor._combine_detections(detector_results)
    
    # Check that high confidence threshold filters segments
    assert len(combined_segments) == 0

def test_multiple_detector_agreement(initialized_processor, sample_data):
    """Test handling of cases where multiple detectors identify same swap."""
    data, _ = sample_data
    # Process data
    detector_results = initialized_processor._run_detectors(data)
    combined_segments = initialized_processor._combine_detections(detector_results)
    
    # Check that overlapping segments are handled correctly
    for i, (start1, end1, _, _) in enumerate(combined_segments):
        for start2, end2, _, _ in combined_segments[i+1:]:
            # No overlapping segments
            assert not (start1 <= end2 and end1 >= start2) 
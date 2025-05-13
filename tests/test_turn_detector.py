"""Tests for turn-based swap detector."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from swap_corrector.config import SwapConfig
from swap_corrector.detectors.turn import TurnDetector

@pytest.fixture
def turn_detector(config):
    """Create a turn detector instance."""
    return TurnDetector(config)

@pytest.fixture
def initialized_turn_detector(turn_detector, sample_data):
    """Create an initialized turn detector instance."""
    data, fps = sample_data
    turn_detector.config.fps = fps
    turn_detector.setup(data)
    return turn_detector

@pytest.fixture
def artificial_data():
    """Create artificial data with known turn-related swaps."""
    # Create a sequence of frames
    n_frames = 100
    data = pd.DataFrame()
    
    # Create circular movement
    t = np.linspace(0, 2*np.pi, n_frames)
    radius = 10
    x_head = radius * np.cos(t)
    y_head = radius * np.sin(t)
    x_tail = radius * np.cos(t + np.pi)
    y_tail = radius * np.sin(t + np.pi)
    
    # Add a turn-related swap at frame 50
    x_head[50] = x_tail[50]  # Swap positions
    y_head[50] = y_tail[50]
    x_tail[50] = x_head[50]
    y_tail[50] = y_head[50]
    
    # Add a rapid turn at frame 75
    t[75:80] = t[75:80] * 2  # Double the angular velocity
    x_head[75:80] = radius * np.cos(t[75:80])
    y_head[75:80] = radius * np.sin(t[75:80])
    x_tail[75:80] = radius * np.cos(t[75:80] + np.pi)
    y_tail[75:80] = radius * np.sin(t[75:80] + np.pi)
    
    data['X-Head'] = x_head
    data['Y-Head'] = y_head
    data['X-Tail'] = x_tail
    data['Y-Tail'] = y_tail
    data['X-Midpoint'] = (x_head + x_tail) / 2
    data['Y-Midpoint'] = (y_head + y_tail) / 2
    
    return data

@pytest.fixture
def real_data():
    """Load real experimental data with known turn-related swaps."""
    data_dir = Path("data")
    # Get first experiment as a test case
    exp_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    exp_dir = exp_dirs[0]
    
    # Load raw and corrected data
    timestamp = exp_dir.name[:19]  # YYYY.MM.DD_HH-MM-SS
    raw_data = pd.read_csv(exp_dir / f"{timestamp}_data.csv")
    level2_data = pd.read_csv(exp_dir / f"{timestamp}_data_level2.csv")
    
    # Find frames where level2 differs from raw (known swaps)
    head_diff = (
        (raw_data['X-Head'] != level2_data['X-Head']) |
        (raw_data['Y-Head'] != level2_data['Y-Head'])
    )
    tail_diff = (
        (raw_data['X-Tail'] != level2_data['X-Tail']) |
        (raw_data['Y-Tail'] != level2_data['Y-Tail'])
    )
    known_swaps = head_diff & tail_diff
    
    # Find continuous segments of swaps
    transitions = np.diff(known_swaps.astype(int))
    swap_starts = np.where(transitions == 1)[0] + 1
    swap_ends = np.where(transitions == -1)[0] + 1
    
    if known_swaps[0]:
        swap_starts = np.concatenate([[0], swap_starts])
    if known_swaps.iloc[-1]:
        swap_ends = np.concatenate([swap_ends, [len(known_swaps)]])
    
    swap_segments = list(zip(swap_starts, swap_ends))
    
    return {
        'raw_data': raw_data,
        'level2_data': level2_data,
        'known_swaps': known_swaps,
        'swap_segments': swap_segments
    }

def test_turn_detection_artificial(initialized_turn_detector, artificial_data):
    """Test turn detector on artificial data."""
    # Run detector
    detected_swaps = initialized_turn_detector.detect(artificial_data)
    
    # Check that swaps are detected
    assert np.any(detected_swaps), "No swaps detected"
    assert len(detected_swaps) == len(artificial_data)

def test_turn_detection_real(initialized_turn_detector, real_data):
    """Test turn detector on real data."""
    raw_data = real_data['raw_data']
    known_swaps = real_data['known_swaps']
    swap_segments = real_data['swap_segments']
    
    # Run detector
    detected_swaps = initialized_turn_detector.detect(raw_data)
    
    # Check that swaps are detected
    assert np.any(detected_swaps), "No swaps detected"
    assert len(detected_swaps) == len(raw_data)

def test_multi_scale_analysis(initialized_turn_detector, artificial_data):
    """Test multi-scale movement analysis."""
    # Run analysis
    scale_metrics = initialized_turn_detector._analyze_movement_scales(artificial_data)
    
    # Check that all scales are present
    assert all(scale in scale_metrics for scale in ['short_term', 'medium_term', 'long_term'])
    
    # Check that all metrics are present for each scale
    expected_metrics = ['curvature', 'angular_velocity']
    for scale in scale_metrics:
        metrics = scale_metrics[scale]
        assert all(metric in metrics for metric in expected_metrics)
        
        # Check that metrics have reasonable values
        assert np.all(np.isfinite(metrics['curvature']))
        assert np.all(np.isfinite(metrics['angular_velocity']))

def test_turn_radius_calculation(initialized_turn_detector, artificial_data):
    """Test turn radius calculation."""
    # Calculate turn radius
    radius = initialized_turn_detector._calculate_turn_radius(artificial_data)
    
    # Check that radius is consistent for circular motion
    mean_radius = np.mean(radius[radius > 0])
    std_radius = np.std(radius[radius > 0])
    
    # The radius should be close to the body length (10 units)
    # Since we're using body length normalization, the radius should be close to 1
    assert abs(mean_radius - 1) < 0.5, "Calculated radius doesn't match expected value"
    assert std_radius < 0.5, "Radius too variable for constant circular motion"

def test_confidence_scoring(initialized_turn_detector, artificial_data):
    """Test confidence scoring."""
    # Get confidence scores
    confidences = initialized_turn_detector.get_confidence(artificial_data)
    
    # Check that all confidences are between 0 and 1
    assert np.all((confidences >= 0) & (confidences <= 1))
    
    # Check that high confidence corresponds to known swaps and rapid turns
    assert confidences[50] > 0.5, "Low confidence for known swap"
    assert confidences[75] > 0.5, "Low confidence for rapid turn"
    
    # Check that average confidence is higher for known swaps and rapid turns
    swap_frames = np.zeros(len(artificial_data), dtype=bool)
    swap_frames[50] = True  # Known swap
    swap_frames[75] = True  # Rapid turn
    
    swap_confidences = confidences[swap_frames]
    non_swap_confidences = confidences[~swap_frames]
    
    assert np.mean(swap_confidences) > np.mean(non_swap_confidences), \
        "Average confidence not higher for known swaps and rapid turns" 
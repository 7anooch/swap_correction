"""Tests for proximity-based swap detector."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from swap_corrector.config import SwapConfig
from swap_corrector.detectors.proximity import ProximityDetector

@pytest.fixture
def proximity_detector(config):
    """Create a proximity detector instance."""
    return ProximityDetector(config)

@pytest.fixture
def initialized_proximity_detector(proximity_detector, sample_data):
    """Create an initialized proximity detector instance."""
    data, fps = sample_data
    proximity_detector.config.fps = fps
    proximity_detector.setup(data)
    return proximity_detector

@pytest.fixture
def real_data():
    """Load real experimental data with known swaps."""
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

def test_proximity_detection(initialized_proximity_detector, real_data):
    """Test that proximity detector identifies real swaps correctly."""
    raw_data = real_data['raw_data']
    known_swaps = real_data['known_swaps']
    swap_segments = real_data['swap_segments']
    
    # Run detector
    detected_swaps = initialized_proximity_detector.detect(raw_data)
    detected_segments = initialized_proximity_detector.get_swap_segments(detected_swaps)
    
    # Calculate detection metrics
    true_positives = 0
    false_positives = 0
    
    for start, end in detected_segments:
        # If detected segment overlaps with any known swap, count as true positive
        is_true_positive = False
        for known_start, known_end in swap_segments:
            if (start <= known_end and end >= known_start):
                true_positives += 1
                is_true_positive = True
                break
        if not is_true_positive:
            false_positives += 1
    
    false_negatives = len(swap_segments) - true_positives
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Assert reasonable performance
    assert precision > 0.7, f"Precision too low: {precision}"
    assert recall > 0.5, f"Recall too low: {recall}"  # Lower recall threshold since proximity is just one detector

def test_proximity_validation(initialized_proximity_detector, real_data):
    """Test swap validation on real data."""
    raw_data = real_data['raw_data']
    swap_segments = real_data['swap_segments']
    
    # Test validation on known swap regions
    valid_count = 0
    for start, end in swap_segments[:5]:  # Test first 5 known swaps
        if initialized_proximity_detector.validate_swap(raw_data, start, end):
            valid_count += 1
    
    # At least 60% of known swaps should be validated
    assert valid_count / min(5, len(swap_segments)) >= 0.6
    
    # Test validation on non-swap regions
    non_swap_valid_count = 0
    for i in range(5):  # Test 5 random non-swap regions
        # Find a region without known swaps
        while True:
            start = np.random.randint(0, len(raw_data) - 10)
            end = start + 5
            is_swap_region = False
            for swap_start, swap_end in swap_segments:
                if start <= swap_end and end >= swap_start:
                    is_swap_region = True
                    break
            if not is_swap_region:
                break
        
        if initialized_proximity_detector.validate_swap(raw_data, start, end):
            non_swap_valid_count += 1
    
    # Less than 20% of non-swap regions should be validated
    assert non_swap_valid_count / 5 <= 0.2

def test_proximity_confidence(initialized_proximity_detector, real_data):
    """Test confidence scoring on real data."""
    raw_data = real_data['raw_data']
    swap_segments = real_data['swap_segments']
    
    # Test confidence on known swap regions
    swap_confidences = []
    for start, end in swap_segments[:5]:  # Test first 5 known swaps
        confidence = initialized_proximity_detector.get_confidence(raw_data, start, end)
        assert 0 <= confidence <= 1
        swap_confidences.append(confidence)
    
    # Average confidence for known swaps should be reasonable
    assert np.mean(swap_confidences) > 0.4  # Lower threshold since proximity is just one signal
    
    # Test confidence on non-swap regions
    non_swap_confidences = []
    for i in range(5):  # Test 5 random non-swap regions
        while True:
            start = np.random.randint(0, len(raw_data) - 10)
            end = start + 5
            is_swap_region = False
            for swap_start, swap_end in swap_segments:
                if start <= swap_end and end >= swap_start:
                    is_swap_region = True
                    break
            if not is_swap_region:
                break
        
        confidence = initialized_proximity_detector.get_confidence(raw_data, start, end)
        assert 0 <= confidence <= 1
        non_swap_confidences.append(confidence)
    
    # Average confidence for non-swap regions should be low
    assert np.mean(non_swap_confidences) < 0.4 
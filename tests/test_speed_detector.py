"""Tests for speed-based swap detector."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from swap_corrector.config import SwapConfig
from swap_corrector.detectors.speed import SpeedDetector

@pytest.fixture
def speed_detector(config):
    """Create a speed detector instance."""
    return SpeedDetector(config)

@pytest.fixture
def initialized_speed_detector(speed_detector, sample_data):
    """Create an initialized speed detector instance."""
    data, fps = sample_data
    speed_detector.config.fps = fps
    speed_detector.setup(data)
    return speed_detector

@pytest.fixture
def artificial_data():
    """Create artificial data with known speed-related swaps."""
    # Create a sequence of frames
    n_frames = 100
    data = pd.DataFrame()
    
    # Create linear movement with constant speed
    t = np.linspace(0, 1, n_frames)
    base_speed = 10  # mm/s
    
    # Create raw data (with swaps)
    x_head_raw = base_speed * t
    y_head_raw = np.zeros_like(t)
    x_tail_raw = x_head_raw - 20  # Tail follows 20mm behind
    y_tail_raw = y_head_raw
    
    # Create level2 data (ground truth, no swaps)
    x_head_l2 = x_head_raw.copy()
    y_head_l2 = y_head_raw.copy()
    x_tail_l2 = x_tail_raw.copy()
    y_tail_l2 = y_tail_raw.copy()
    
    # Add a high-speed swap at frame 50 in raw data
    x_head_raw[50], x_tail_raw[50] = x_tail_raw[50], x_head_raw[50]
    y_head_raw[50], y_tail_raw[50] = y_tail_raw[50], y_head_raw[50]
    
    # Add rapid acceleration at frame 75 in both raw and level2
    x_head_raw[75:] = x_head_raw[74] + base_speed * 3 * (t[75:] - t[74])
    x_tail_raw[75:] = x_tail_raw[74] + base_speed * 3 * (t[75:] - t[74])
    x_head_l2[75:] = x_head_l2[74] + base_speed * 3 * (t[75:] - t[74])
    x_tail_l2[75:] = x_tail_l2[74] + base_speed * 3 * (t[75:] - t[74])
    
    # Create raw DataFrame
    raw_data = pd.DataFrame({
        'X-Head': x_head_raw,
        'Y-Head': y_head_raw,
        'X-Tail': x_tail_raw,
        'Y-Tail': y_tail_raw,
        'X-Midpoint': (x_head_raw + x_tail_raw) / 2,
        'Y-Midpoint': (y_head_raw + y_tail_raw) / 2
    })
    
    # Create level2 DataFrame
    level2_data = pd.DataFrame({
        'X-Head': x_head_l2,
        'Y-Head': y_head_l2,
        'X-Tail': x_tail_l2,
        'Y-Tail': y_tail_l2,
        'X-Midpoint': (x_head_l2 + x_tail_l2) / 2,
        'Y-Midpoint': (y_head_l2 + y_tail_l2) / 2
    })
    
    # Find frames where raw differs from level2 (known swaps)
    head_diff = (
        (raw_data['X-Head'] != level2_data['X-Head']) |
        (raw_data['Y-Head'] != level2_data['Y-Head'])
    )
    tail_diff = (
        (raw_data['X-Tail'] != level2_data['X-Tail']) |
        (raw_data['Y-Tail'] != level2_data['Y-Tail'])
    )
    known_swaps = head_diff & tail_diff
    
    return {
        'raw_data': raw_data,
        'level2_data': level2_data,
        'known_swaps': known_swaps
    }

@pytest.fixture
def real_data_list():
    """Load all real experimental data."""
    data_dir = Path("data")
    exp_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    
    all_data = []
    for exp_dir in exp_dirs:
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
        
        all_data.append({
            'name': exp_dir.name,
            'raw_data': raw_data,
            'level2_data': level2_data,
            'known_swaps': known_swaps,
            'swap_segments': swap_segments
        })
    
    return all_data

def test_speed_detection_artificial(initialized_speed_detector, artificial_data):
    """Test speed detector on artificial data."""
    # Run detector
    detected_swaps = initialized_speed_detector.detect(artificial_data['raw_data'])
    known_swaps = artificial_data['known_swaps']
    
    # Check that swaps are detected
    assert np.any(detected_swaps), "No swaps detected"
    assert len(detected_swaps) == len(artificial_data['raw_data'])
    
    # Calculate metrics
    true_positives = np.sum(detected_swaps & known_swaps)
    false_positives = np.sum(detected_swaps & ~known_swaps)
    false_negatives = np.sum(~detected_swaps & known_swaps)
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    # Check metrics
    assert precision > 0.7, f"Precision too low: {precision:.2f}"
    assert recall > 0.7, f"Recall too low: {recall:.2f}"
    assert f1_score > 0.7, f"F1 score too low: {f1_score:.2f}"
    
    # Check specific frames
    assert detected_swaps[50], "Missed high-speed swap at frame 50"

def test_speed_detection_real(initialized_speed_detector, real_data_list):
    """Test speed detector on real data."""
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for exp_data in real_data_list:
        raw_data = exp_data['raw_data']
        known_swaps = exp_data['known_swaps']
        
        # Run detector
        detected_swaps = initialized_speed_detector.detect(raw_data)
        
        # Check that swaps are detected
        assert np.any(detected_swaps), f"No swaps detected in {exp_data['name']}"
        assert len(detected_swaps) == len(raw_data)
        
        # Calculate metrics
        true_positives = np.sum(detected_swaps & known_swaps)
        false_positives = np.sum(detected_swaps & ~known_swaps)
        false_negatives = np.sum(~detected_swaps & known_swaps)
        
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
        
        # Calculate per-experiment metrics
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        print(f"\nMetrics for {exp_data['name']}:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1_score:.2f}")
    
    # Calculate overall metrics
    precision = total_true_positives / (total_true_positives + total_false_positives + 1e-6)
    recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    print("\nOverall metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    
    # Check against success metrics from implementation plan
    assert precision > 0.7, f"Precision too low: {precision:.2f}"
    assert recall > 0.7, f"Recall too low: {recall:.2f}"
    assert f1_score > 0.7, f"F1 score too low: {f1_score:.2f}"

def test_multi_scale_analysis(initialized_speed_detector, artificial_data, real_data_list):
    """Test multi-scale movement analysis."""
    # Test with artificial data
    scale_metrics = initialized_speed_detector._analyze_movement_scales(artificial_data['raw_data'])
    
    # Check that all scales are present
    assert all(scale in scale_metrics for scale in ['short_term', 'medium_term', 'long_term'])
    
    # Check that all metrics are present for each scale
    for scale, params in initialized_speed_detector.analysis_scales.items():
        metrics = scale_metrics[scale]
        assert all(metric in metrics for metric in params['metrics'])
    
    # Test with real data
    for exp_data in real_data_list:
        scale_metrics = initialized_speed_detector._analyze_movement_scales(exp_data['raw_data'])
        
        # Check that all scales are present
        assert all(scale in scale_metrics for scale in ['short_term', 'medium_term', 'long_term'])
        
        # Check that all metrics are present for each scale
        for scale, params in initialized_speed_detector.analysis_scales.items():
            metrics = scale_metrics[scale]
            assert all(metric in metrics for metric in params['metrics'])

def test_speed_thresholds(initialized_speed_detector, artificial_data, real_data_list):
    """Test speed threshold calculations."""
    # Test with artificial data
    scale_metrics = initialized_speed_detector._analyze_movement_scales(artificial_data['raw_data'])
    speed = scale_metrics['short_term']['speed']
    acceleration = scale_metrics['short_term']['acceleration']
    
    # Check that high speeds are detected
    high_speed_frames = speed > initialized_speed_detector._current_thresholds['speed']
    assert np.any(high_speed_frames), "No high speeds detected"
    
    # Check that high accelerations are detected
    accel_threshold = initialized_speed_detector._current_thresholds['speed'] * initialized_speed_detector._current_thresholds['accel_factor']
    high_accel_frames = np.abs(acceleration) > accel_threshold
    assert np.any(high_accel_frames), "No high accelerations detected"
    
    # Test with real data
    for exp_data in real_data_list[:3]:  # Test first 3 experiments
        raw_data = exp_data['raw_data']
        scale_metrics = initialized_speed_detector._analyze_movement_scales(raw_data)
        speed = scale_metrics['short_term']['speed']
        acceleration = scale_metrics['short_term']['acceleration']
        
        # Check that high speeds are detected
        high_speed_frames = speed > initialized_speed_detector._current_thresholds['speed']
        assert np.any(high_speed_frames), f"No high speeds detected in {exp_data['name']}"
        
        # Check that high accelerations are detected
        accel_threshold = initialized_speed_detector._current_thresholds['speed'] * initialized_speed_detector._current_thresholds['accel_factor']
        high_accel_frames = np.abs(acceleration) > accel_threshold
        assert np.any(high_accel_frames), f"No high accelerations detected in {exp_data['name']}"

def test_confidence_scoring(initialized_speed_detector, artificial_data, real_data_list):
    """Test confidence scoring."""
    # Test with artificial data
    raw_data = artificial_data['raw_data']
    known_swaps = artificial_data['known_swaps']
    
    # Get confidence scores
    confidences = initialized_speed_detector.get_confidence(raw_data)
    
    # Check that all confidences are between 0 and 1
    assert np.all((confidences >= 0) & (confidences <= 1))
    
    # Check that high confidence corresponds to known swaps
    assert confidences[50] > 0.7, "Low confidence for high-speed swap at frame 50"
    
    # Check that average confidence is higher for known swaps
    swap_confidences = confidences[known_swaps]
    non_swap_confidences = confidences[~known_swaps]
    
    assert np.mean(swap_confidences) > np.mean(non_swap_confidences), "Average confidence not higher for known swaps"
    
    # Test with real data
    for exp_data in real_data_list[:3]:  # Test first 3 experiments
        raw_data = exp_data['raw_data']
        known_swaps = exp_data['known_swaps']
        
        # Get confidence scores
        confidences = initialized_speed_detector.get_confidence(raw_data)
        
        # Check that all confidences are between 0 and 1
        assert np.all((confidences >= 0) & (confidences <= 1))
        
        # Check that average confidence is higher for known swaps
        swap_confidences = confidences[known_swaps]
        non_swap_confidences = confidences[~known_swaps]
        
        assert np.mean(swap_confidences) > np.mean(non_swap_confidences), \
            f"Average confidence not higher for known swaps in {exp_data['name']}" 
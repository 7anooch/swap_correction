"""Integration tests for the swap correction pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import os
from typing import List, Tuple

from swap_corrector.config import SwapConfig, SwapCorrectionConfig
from swap_corrector.processor import SwapProcessor

@pytest.fixture
def real_data_paths():
    """Get paths to real experimental data."""
    data_dir = Path("data")
    exp_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    
    # Get paths for raw and level2 (manually corrected) data
    data_paths = []
    for exp_dir in exp_dirs:
        timestamp = exp_dir.name[:19]  # YYYY.MM.DD_HH-MM-SS
        raw_path = exp_dir / f"{timestamp}_data.csv"
        level2_path = exp_dir / f"{timestamp}_data_level2.csv"
        
        if raw_path.exists() and level2_path.exists():
            data_paths.append((raw_path, level2_path))
    
    return data_paths

@pytest.fixture
def processor():
    """Create a processor instance with default settings."""
    config = SwapConfig(fps=30)
    correction_config = SwapCorrectionConfig(
        validate=True,
        debug=True
    )
    return SwapProcessor(config, correction_config)

def test_real_data_processing(processor, real_data_paths):
    """Test processing on real experimental data."""
    total_swaps = 0
    correct_detections = 0
    false_positives = 0
    
    for raw_path, level2_path in real_data_paths:
        # Load data
        raw_data = pd.read_csv(raw_path)
        level2_data = pd.read_csv(level2_path)
        
        # Process raw data
        processed_data = processor.process(raw_data)
        
        # Find actual swaps (differences between raw and level2)
        actual_swaps = _find_actual_swaps(raw_data, level2_data)
        total_swaps += len(actual_swaps)
        
        # Find detected swaps (differences between raw and processed)
        detected_swaps = _find_actual_swaps(raw_data, processed_data)
        
        # Calculate metrics
        for detected_start, detected_end in detected_swaps:
            is_true_positive = False
            for actual_start, actual_end in actual_swaps:
                if _segments_overlap(
                    detected_start, detected_end,
                    actual_start, actual_end
                ):
                    correct_detections += 1
                    is_true_positive = True
                    break
            if not is_true_positive:
                false_positives += 1
    
    # Calculate overall metrics
    precision = correct_detections / (correct_detections + false_positives)
    recall = correct_detections / total_swaps
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Assert reasonable performance
    assert precision > 0.8, f"Precision too low: {precision:.2f}"
    assert recall > 0.8, f"Recall too low: {recall:.2f}"
    assert f1_score > 0.8, f"F1 score too low: {f1_score:.2f}"

def test_complex_scenarios(processor, real_data_paths):
    """Test handling of complex scenarios in real data."""
    for raw_path, _ in real_data_paths:
        raw_data = pd.read_csv(raw_path)
        
        # Test rapid movement sequences
        rapid_segments = _find_rapid_movements(raw_data)
        if rapid_segments:
            processed_data = processor.process(raw_data)
            for start, end in rapid_segments:
                # Check that rapid movements weren't mistaken for swaps
                assert _movement_preserved(
                    raw_data[start:end],
                    processed_data[start:end]
                )
        
        # Test close proximity scenarios
        proximity_segments = _find_close_proximity_segments(raw_data)
        if proximity_segments:
            processed_data = processor.process(raw_data)
            for start, end in proximity_segments:
                # Check that legitimate close movements weren't altered
                assert _movement_preserved(
                    raw_data[start:end],
                    processed_data[start:end]
                )

def test_edge_cases(processor, real_data_paths):
    """Test handling of edge cases."""
    for raw_path, _ in real_data_paths:
        raw_data = pd.read_csv(raw_path)
        
        # Test data with missing frames
        missing_data = _introduce_missing_frames(raw_data)
        processed_missing = processor.process(missing_data)
        assert not processed_missing.isna().any().any()
        
        # Test data with noise
        noisy_data = _add_noise(raw_data)
        processed_noisy = processor.process(noisy_data)
        assert _noise_handled_properly(raw_data, processed_noisy)
        
        # Test boundary cases
        boundary_data = raw_data.iloc[:10]  # Test with few frames
        processed_boundary = processor.process(boundary_data)
        assert processed_boundary.shape == boundary_data.shape

def test_performance(processor, real_data_paths):
    """Test processing performance."""
    import time
    
    processing_times = []
    
    for raw_path, _ in real_data_paths:
        raw_data = pd.read_csv(raw_path)
        
        # Measure processing time
        start_time = time.time()
        processor.process(raw_data)
        end_time = time.time()
        
        processing_times.append(end_time - start_time)
    
    # Calculate performance metrics
    avg_time = np.mean(processing_times)
    max_time = np.max(processing_times)
    
    # Assert reasonable performance
    assert avg_time < 1.0, f"Average processing time too high: {avg_time:.2f}s"
    assert max_time < 2.0, f"Maximum processing time too high: {max_time:.2f}s"

# Helper functions

def _find_actual_swaps(data1: pd.DataFrame, data2: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find frames where head/tail positions differ between datasets."""
    head_diff = (
        (data1['X-Head'] != data2['X-Head']) |
        (data1['Y-Head'] != data2['Y-Head'])
    )
    tail_diff = (
        (data1['X-Tail'] != data2['X-Tail']) |
        (data1['Y-Tail'] != data2['Y-Tail'])
    )
    swaps = head_diff & tail_diff
    
    # Convert to segments
    transitions = np.diff(swaps.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1
    
    if swaps.iloc[0]:
        starts = np.concatenate([[0], starts])
    if swaps.iloc[-1]:
        ends = np.concatenate([ends, [len(swaps)]])
    
    return list(zip(starts, ends))

def _segments_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check if two segments overlap."""
    return start1 <= end2 and end1 >= start2

def _find_rapid_movements(data: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find segments with rapid movement."""
    dx = np.diff(data['X-Midpoint'])
    dy = np.diff(data['Y-Midpoint'])
    speed = np.sqrt(dx*dx + dy*dy)
    
    # Find high-speed segments
    high_speed = speed > np.mean(speed) + 2*np.std(speed)
    
    # Convert to segments
    transitions = np.diff(high_speed.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1
    
    if high_speed[0]:
        starts = np.concatenate([[0], starts])
    if high_speed[-1]:
        ends = np.concatenate([ends, [len(high_speed)]])
    
    return list(zip(starts, ends))

def _find_close_proximity_segments(data: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find segments where head and tail are close."""
    dist = np.sqrt(
        (data['X-Head'] - data['X-Tail'])**2 +
        (data['Y-Head'] - data['Y-Tail'])**2
    )
    
    # Find close proximity segments
    close = dist < np.mean(dist) - np.std(dist)
    
    # Convert to segments
    transitions = np.diff(close.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1
    
    if close[0]:
        starts = np.concatenate([[0], starts])
    if close[-1]:
        ends = np.concatenate([ends, [len(close)]])
    
    return list(zip(starts, ends))

def _movement_preserved(original: pd.DataFrame, processed: pd.DataFrame) -> bool:
    """Check if movement characteristics are preserved."""
    # Calculate movement metrics
    orig_dx = np.diff(original['X-Midpoint'])
    orig_dy = np.diff(original['Y-Midpoint'])
    proc_dx = np.diff(processed['X-Midpoint'])
    proc_dy = np.diff(processed['Y-Midpoint'])
    
    # Compare movement patterns
    dx_corr = np.corrcoef(orig_dx, proc_dx)[0, 1]
    dy_corr = np.corrcoef(orig_dy, proc_dy)[0, 1]
    
    return dx_corr > 0.8 and dy_corr > 0.8

def _introduce_missing_frames(data: pd.DataFrame) -> pd.DataFrame:
    """Introduce missing data frames."""
    missing_data = data.copy()
    
    # Randomly set some frames to NaN
    n_frames = len(data)
    missing_indices = np.random.choice(
        n_frames,
        size=n_frames//20,  # 5% missing
        replace=False
    )
    
    for idx in missing_indices:
        missing_data.iloc[idx, :] = np.nan
    
    return missing_data

def _add_noise(data: pd.DataFrame) -> pd.DataFrame:
    """Add random noise to positions."""
    noisy_data = data.copy()
    
    # Add Gaussian noise
    noise_level = 0.1  # 10% of standard deviation
    for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']:
        std = np.std(data[col])
        noise = np.random.normal(0, noise_level * std, len(data))
        noisy_data[col] = data[col] + noise
    
    # Update midpoints
    noisy_data['X-Midpoint'] = (noisy_data['X-Head'] + noisy_data['X-Tail']) / 2
    noisy_data['Y-Midpoint'] = (noisy_data['Y-Head'] + noisy_data['Y-Tail']) / 2
    
    return noisy_data

def _noise_handled_properly(original: pd.DataFrame, processed: pd.DataFrame) -> bool:
    """Check if noise was handled appropriately."""
    # Calculate position differences
    head_diff = np.sqrt(
        (processed['X-Head'] - original['X-Head'])**2 +
        (processed['Y-Head'] - original['Y-Head'])**2
    )
    tail_diff = np.sqrt(
        (processed['X-Tail'] - original['X-Tail'])**2 +
        (processed['Y-Tail'] - original['Y-Tail'])**2
    )
    
    # Check that differences are reasonable
    max_allowed_diff = np.std(original['X-Head']) * 0.2  # 20% of std
    return (
        np.mean(head_diff) < max_allowed_diff and
        np.mean(tail_diff) < max_allowed_diff
    ) 
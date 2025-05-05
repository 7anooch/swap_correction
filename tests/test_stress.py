"""Stress tests for the swap correction pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import time

from swap_corrector.config import SwapConfig, SwapCorrectionConfig
from swap_corrector.processor import SwapProcessor

@pytest.fixture
def processor():
    """Create a processor instance."""
    config = SwapConfig(fps=30)
    correction_config = SwapCorrectionConfig(
        validate=True,
        debug=True
    )
    return SwapProcessor(config, correction_config)

def test_large_dataset(processor):
    """Test processing of a large dataset."""
    # Create large dataset (1 hour at 30fps = 108,000 frames)
    n_frames = 108000
    data = _create_large_dataset(n_frames)
    
    # Measure processing time
    start_time = time.time()
    processed_data = processor.process(data)
    processing_time = time.time() - start_time
    
    # Check performance
    assert processing_time < 60.0, f"Processing too slow: {processing_time:.2f}s"
    assert processed_data.shape == data.shape
    assert not processed_data.isna().any().any()

def test_rapid_alternating_swaps(processor):
    """Test handling of rapidly alternating swap patterns."""
    # Create data with swaps every few frames
    n_frames = 1000
    data = _create_alternating_swaps(n_frames)
    
    # Process data
    processed_data = processor.process(data)
    
    # Verify corrections
    assert _corrections_are_consistent(processed_data)
    assert not processed_data.isna().any().any()

def test_concurrent_swaps(processor):
    """Test handling of multiple simultaneous swap events."""
    # Create data with overlapping swap patterns
    n_frames = 1000
    data = _create_concurrent_swaps(n_frames)
    
    # Process data
    processed_data = processor.process(data)
    
    # Verify corrections
    assert _corrections_are_consistent(processed_data)
    assert not processed_data.isna().any().any()

def test_extreme_noise(processor):
    """Test handling of extremely noisy data."""
    # Create noisy data
    n_frames = 1000
    data = _create_noisy_data(n_frames, noise_level=0.5)  # 50% noise
    
    # Process data
    processed_data = processor.process(data)
    
    # Verify noise handling
    assert _noise_reduction_effective(data, processed_data)
    assert not processed_data.isna().any().any()

def test_memory_usage(processor):
    """Test memory usage with large datasets."""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Process increasingly large datasets
    for n_frames in [1000, 10000, 100000]:
        data = _create_large_dataset(n_frames)
        processor.process(data)
        
        # Check memory usage
        current_memory = process.memory_info().rss
        memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory usage should scale roughly linearly
        assert memory_increase < n_frames * 0.001, f"Memory usage too high: {memory_increase:.2f}MB"

# Helper functions

def _create_large_dataset(n_frames: int) -> pd.DataFrame:
    """Create a large synthetic dataset."""
    # Create time points
    t = np.linspace(0, 8*np.pi, n_frames)
    radius = 10
    
    # Create complex movement pattern
    x_head = radius * np.cos(t) + radius * np.sin(2*t)
    y_head = radius * np.sin(t) + radius * np.cos(2*t)
    x_tail = radius * np.cos(t + np.pi) + radius * np.sin(2*t + np.pi)
    y_tail = radius * np.sin(t + np.pi) + radius * np.cos(2*t + np.pi)
    
    # Create DataFrame
    data = pd.DataFrame({
        'X-Head': x_head,
        'Y-Head': y_head,
        'X-Tail': x_tail,
        'Y-Tail': y_tail,
        'X-Midpoint': (x_head + x_tail) / 2,
        'Y-Midpoint': (y_head + y_tail) / 2
    })
    
    return data

def _create_alternating_swaps(n_frames: int) -> pd.DataFrame:
    """Create data with rapidly alternating swaps."""
    data = _create_large_dataset(n_frames)
    
    # Add alternating swaps every 10 frames
    for i in range(0, n_frames, 20):
        if i + 10 >= n_frames:
            break
            
        # Swap head and tail
        head_x = data.loc[i:i+10, 'X-Head'].copy()
        head_y = data.loc[i:i+10, 'Y-Head'].copy()
        data.loc[i:i+10, 'X-Head'] = data.loc[i:i+10, 'X-Tail']
        data.loc[i:i+10, 'Y-Head'] = data.loc[i:i+10, 'Y-Tail']
        data.loc[i:i+10, 'X-Tail'] = head_x
        data.loc[i:i+10, 'Y-Tail'] = head_y
    
    return data

def _create_concurrent_swaps(n_frames: int) -> pd.DataFrame:
    """Create data with overlapping swap patterns."""
    data = _create_large_dataset(n_frames)
    
    # Add multiple types of swaps
    for i in range(0, n_frames, 50):
        if i + 30 >= n_frames:
            break
            
        # Proximity swap
        data.loc[i:i+10, 'X-Head'] = data.loc[i:i+10, 'X-Tail']
        data.loc[i:i+10, 'Y-Head'] = data.loc[i:i+10, 'Y-Tail']
        
        # Speed-based swap
        if i + 20 < n_frames:
            speed_factor = 3.0
            # Create array with correct length for the slice
            t = np.linspace(0, 2*np.pi, 21)  # 21 points for 20 intervals
            data.loc[i+10:i+30, 'X-Head'] = 10 * np.cos(speed_factor * t)
            data.loc[i+10:i+30, 'Y-Head'] = 10 * np.sin(speed_factor * t)
    
    return data

def _create_noisy_data(n_frames: int, noise_level: float) -> pd.DataFrame:
    """Create extremely noisy data."""
    data = _create_large_dataset(n_frames)
    
    # Add significant noise
    for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']:
        std = np.std(data[col])
        noise = np.random.normal(0, noise_level * std, n_frames)
        data[col] += noise
    
    # Update midpoints
    data['X-Midpoint'] = (data['X-Head'] + data['X-Tail']) / 2
    data['Y-Midpoint'] = (data['Y-Head'] + data['Y-Tail']) / 2
    
    return data

def _corrections_are_consistent(data: pd.DataFrame) -> bool:
    """Check if corrections maintain consistent movement patterns."""
    # Calculate movement metrics
    dx_head = np.diff(data['X-Head'])
    dy_head = np.diff(data['Y-Head'])
    dx_tail = np.diff(data['X-Tail'])
    dy_tail = np.diff(data['Y-Tail'])
    
    # Calculate velocities
    v_head = np.sqrt(dx_head**2 + dy_head**2)
    v_tail = np.sqrt(dx_tail**2 + dy_tail**2)
    
    # Check for unrealistic velocity changes
    max_velocity_ratio = 5.0
    velocity_consistent = np.all(
        v_head[1:] / v_head[:-1] < max_velocity_ratio
    ) and np.all(
        v_tail[1:] / v_tail[:-1] < max_velocity_ratio
    )
    
    return velocity_consistent

def _noise_reduction_effective(original: pd.DataFrame, processed: pd.DataFrame) -> bool:
    """Check if noise was effectively reduced."""
    # Calculate noise levels
    orig_noise = np.std(np.diff(original['X-Head']))
    proc_noise = np.std(np.diff(processed['X-Head']))
    
    # Check noise reduction
    noise_reduction = (orig_noise - proc_noise) / orig_noise
    return noise_reduction > 0.3  # At least 30% noise reduction 
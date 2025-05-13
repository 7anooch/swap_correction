"""
Pytest configuration and fixtures for PiVR swap correction tests.
"""

import os
import pytest
import numpy as np
import pandas as pd
from typing import Generator, Tuple
from swap_corrector import config, logger
from swap_corrector.config import SwapConfig
from swap_corrector.detectors.proximity import ProximityDetector
from swap_corrector.detectors.speed import SpeedDetector
from swap_corrector.detectors.turn import TurnDetector
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_config(temp_dir):
    """Create a test configuration."""
    config = SwapCorrectionConfig(
        output_dir=temp_dir,
        log_level="INFO",
        fps=30,
        speed_threshold=10.0,
        distance_threshold=5.0,
        angle_threshold=45.0,
        curvature_threshold=0.5,
        velocity_threshold=10.0,
        acceleration_threshold=5.0,
        radius_threshold=10.0
    )
    return config

@pytest.fixture(scope="session")
def config():
    """Create a configuration instance."""
    return SwapConfig(fps=30)

@pytest.fixture(scope="session")
def sample_data():
    """Create sample trajectory data.
    
    Returns:
        Tuple of (DataFrame, fps)
    """
    # Create a DataFrame with sample trajectory data
    data = pd.DataFrame({
        'X-Head': np.random.randn(100),
        'Y-Head': np.random.randn(100),
        'X-Tail': np.random.randn(100),
        'Y-Tail': np.random.randn(100),
        'Speed': np.random.uniform(0, 20, 100),
        'Acceleration': np.random.uniform(-5, 5, 100),
        'Curvature': np.random.uniform(0, 1, 100)
    })
    fps = 30
    return data, fps

@pytest.fixture(scope="session")
def proximity_detector(config, sample_data):
    """Create a proximity detector instance."""
    data, fps = sample_data
    config.fps = fps  # Update config with correct fps
    detector = ProximityDetector(config)
    detector.setup(data)
    return detector

@pytest.fixture(scope="session")
def speed_detector(config, sample_data):
    """Create a speed detector instance."""
    data, fps = sample_data
    config.fps = fps  # Update config with correct fps
    detector = SpeedDetector(config)
    detector.setup(data)
    return detector

@pytest.fixture(scope="session")
def turn_detector(config, sample_data):
    """Create a turn detector instance."""
    data, fps = sample_data
    config.fps = fps  # Update config with correct fps
    detector = TurnDetector(config)
    detector.setup(data)
    return detector

@pytest.fixture(scope="session")
def sample_data_with_swaps(sample_data):
    """Create sample data with known swaps."""
    data, fps = sample_data
    
    # Add known swaps at specific frames
    swap_frames = [10, 30, 50, 70, 90]
    for frame in swap_frames:
        # Swap head and tail positions
        data.loc[frame:frame+5, ['X-Head', 'Y-Head']], data.loc[frame:frame+5, ['X-Tail', 'Y-Tail']] = \
            data.loc[frame:frame+5, ['X-Tail', 'Y-Tail']].values, data.loc[frame:frame+5, ['X-Head', 'Y-Head']].values
    
    return data, fps

@pytest.fixture(scope="session")
def sample_data_with_overlaps() -> Tuple[pd.DataFrame, int]:
    """
    Create sample tracking data with intentional overlaps.
    Returns a tuple of (dataframe, fps).
    """
    data, fps = sample_data()
    
    # Create overlap segments
    overlap_segments = [(30, 35), (70, 75)]
    for start, end in overlap_segments:
        # Make head and tail positions identical
        data.loc[start:end, ['X-Tail', 'Y-Tail']] = data.loc[start:end, ['X-Head', 'Y-Head']].values
    
    return data, fps

@pytest.fixture(scope="session")
def test_logger(test_config):
    """Initialize logger with test configuration."""
    return logger.setup_logger(test_config)

@pytest.fixture(scope="session")
def sample_led_data():
    """Create sample LED data."""
    # Create a DataFrame with sample LED data
    data = pd.DataFrame({
        'LED1': np.random.randint(0, 2, 100),
        'LED2': np.random.randint(0, 2, 100),
        'LED3': np.random.randint(0, 2, 100)
    })
    return data 
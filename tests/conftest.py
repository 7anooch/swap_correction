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

@pytest.fixture(scope="session")
def test_config() -> config.SwapCorrectionConfig:
    """Create a test configuration with debug settings enabled."""
    return config.SwapCorrectionConfig(
        debug=True,
        diagnostic_plots=False,
        show_plots=False,
        log_level="DEBUG"
    )

@pytest.fixture(scope="session")
def config():
    """Create a configuration instance."""
    return SwapConfig(fps=30)

@pytest.fixture(scope="session")
def sample_data():
    """Create sample tracking data."""
    # Create circular motion data
    t = np.linspace(0, 2*np.pi, 100)
    radius = 10
    fps = 30  # Standard frame rate
    
    # Head follows a circle
    head_x = radius * np.cos(t)
    head_y = radius * np.sin(t)
    
    # Tail follows with offset
    tail_x = 0.7071 * radius * np.cos(t - np.pi/4)
    tail_y = -0.7071 * radius * np.sin(t - np.pi/4)
    
    # Calculate midpoints
    mid_x = (head_x + tail_x) / 2
    mid_y = (head_y + tail_y) / 2
    
    # Create DataFrame
    data = pd.DataFrame({
        'X-Head': head_x,
        'Y-Head': head_y,
        'X-Tail': tail_x,
        'Y-Tail': tail_y,
        'X-Midpoint': mid_x,
        'Y-Midpoint': mid_y
    })
    
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
def sample_data_with_swaps(sample_data) -> Tuple[pd.DataFrame, int]:
    """Create sample data with known swaps."""
    data, fps = sample_data
    
    # Add a swap at frame 50
    swap_frame = 50
    data.iloc[swap_frame:, [0, 1, 2, 3]] = data.iloc[swap_frame:, [2, 3, 0, 1]].values
    
    return data, swap_frame

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
def temp_dir(tmp_path_factory) -> str:
    """Create a temporary directory for test outputs."""
    temp_dir = tmp_path_factory.mktemp("test_outputs")
    return str(temp_dir) 
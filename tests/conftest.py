"""
Pytest configuration and fixtures for PiVR swap correction tests.
"""

import os
import pytest
import numpy as np
import pandas as pd
from typing import Generator, Tuple
from swap_corrector import config, logger

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
def sample_data() -> Tuple[pd.DataFrame, int]:
    """
    Create sample tracking data for testing.
    Returns a tuple of (dataframe, fps).
    """
    # Create a simple trajectory with known properties
    n_frames = 100
    fps = 30
    t = np.linspace(0, 2*np.pi, n_frames)
    
    # Create head and tail positions
    head_x = 10 * np.cos(t)
    head_y = 10 * np.sin(t)
    tail_x = -5 * np.cos(t)
    tail_y = 5 * np.sin(t)
    
    # Create center and midpoint positions
    ctr_x = np.full(n_frames, 2.5)
    ctr_y = np.full(n_frames, 2.5)
    mid_x = np.full(n_frames, 2.5)
    mid_y = np.full(n_frames, 2.5)
    
    # Create DataFrame
    data = pd.DataFrame({
        'xhead': head_x,
        'yhead': head_y,
        'xtail': tail_x,
        'ytail': tail_y,
        'xctr': ctr_x,
        'yctr': ctr_y,
        'xmid': mid_x,
        'ymid': mid_y
    })
    
    return data, fps

@pytest.fixture(scope="session")
def sample_data_with_swaps(sample_data) -> Tuple[pd.DataFrame, int]:
    """
    Create sample tracking data with head-tail swaps.
    Returns a tuple of (dataframe, fps).
    """
    data, fps = sample_data
    data_with_swaps = data.copy()
    
    # Introduce swaps at specific frames
    swap_frames = [20, 40, 60, 80]
    for frame in swap_frames:
        # Swap head and tail positions
        data_with_swaps.loc[frame, ['xhead', 'yhead']] = data.loc[frame, ['xtail', 'ytail']].values
        data_with_swaps.loc[frame, ['xtail', 'ytail']] = data.loc[frame, ['xhead', 'yhead']].values
    
    return data_with_swaps, fps

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
        data.loc[start:end, ['xtail', 'ytail']] = data.loc[start:end, ['xhead', 'yhead']].values
    
    return data, fps

@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory) -> str:
    """Create a temporary directory for test outputs."""
    temp_dir = tmp_path_factory.mktemp("test_outputs")
    return str(temp_dir) 
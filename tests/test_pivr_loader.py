"""
Tests for the pivr_loader module.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from swap_corrector.pivr_loader import (
    get_sample_directories,
    load_raw_data,
    import_analysed_data,
    export_to_PiVR,
    get_settings,
    import_distance_data,
    get_fps,
    get_led_data
)

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a sample data directory with necessary files."""
    # Create sample data directory
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()
    
    # Create sample data file
    data = pd.DataFrame({
        'X-Head': [0, 1, 2],
        'Y-Head': [0, 1, 2],
        'X-Tail': [1, 2, 3],
        'Y-Tail': [1, 2, 3],
        'X-Midpoint': [0.5, 1.5, 2.5],
        'Y-Midpoint': [0.5, 1.5, 2.5],
        'X-Centroid': [0.5, 1.5, 2.5],
        'Y-Centroid': [0.5, 1.5, 2.5],
        'Xmin-bbox': [0, 1, 2],
        'Ymin-bbox': [0, 1, 2],
        'Xmax-bbox': [1, 2, 3],
        'Ymax-bbox': [1, 2, 3],
        'stimulation': [0, 1, 0]
    })
    data.to_csv(data_dir / "data.csv", index=False)
    
    # Create settings file
    settings = {
        'Framerate': 30,
        'Pixel per mm': 10,
        'Source x': 0,
        'Source y': 0
    }
    with open(data_dir / "experiment_settings.json", 'w') as f:
        json.dump(settings, f)
    
    return str(data_dir)

def test_get_sample_directories(sample_data_dir):
    """Test that get_sample_directories finds valid data directories."""
    # Create a subdirectory with valid data
    subdir = os.path.join(sample_data_dir, "subdir")
    os.makedirs(subdir)
    pd.DataFrame({'X-Head': [0], 'Y-Head': [0]}).to_csv(os.path.join(subdir, "data.csv"), index=False)
    
    # Create a subdirectory without valid data
    invalid_dir = os.path.join(sample_data_dir, "invalid")
    os.makedirs(invalid_dir)
    
    directories = get_sample_directories(sample_data_dir)
    assert len(directories) == 1
    assert os.path.basename(directories[0]) == "subdir"

def test_load_raw_data(sample_data_dir):
    """Test loading raw data from a PiVR file."""
    data = load_raw_data(sample_data_dir)
    
    assert isinstance(data, pd.DataFrame)
    assert 'xhead' in data.columns
    assert 'yhead' in data.columns
    assert 'stimulus' in data.columns
    assert len(data) == 3

def test_import_analysed_data(sample_data_dir):
    """Test importing analyzed data."""
    # Create analyzed data file
    analyzed_data = pd.DataFrame({
        'xhead': [0, 1, 2],
        'yhead': [0, 1, 2],
        'stimulus': [0, 1, 0]
    })
    analyzed_data.to_csv(os.path.join(sample_data_dir, "analysis.csv"), index=False)
    
    data = import_analysed_data(sample_data_dir)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert 'xhead' in data.columns

def test_get_settings(sample_data_dir):
    """Test getting experiment settings."""
    fps, ppmm, source = get_settings(sample_data_dir)
    
    assert fps == 30
    assert ppmm == 10
    assert isinstance(source, np.ndarray)
    assert len(source) == 2
    assert np.all(source == 0)

def test_import_distance_data(sample_data_dir):
    """Test importing distance data."""
    # Create distance data file
    distance_data = pd.DataFrame({
        'Frame': [0, 1, 2],
        'Distance': [0, 1, 2]
    })
    distance_data.to_csv(os.path.join(sample_data_dir, "distance_to_source.csv"), index=False)
    
    # Create sample data with frame numbers
    data = pd.DataFrame({'Frame': [0, 1, 2]})
    distance = import_distance_data(data, sample_data_dir)
    
    assert isinstance(distance, pd.Series)
    assert len(distance) == 3

def test_get_fps(sample_data_dir):
    """Test getting frame rate."""
    fps = get_fps(sample_data_dir)
    assert fps == 30

def test_get_led_data(sample_data_dir):
    """Test getting LED data."""
    # Create subdirectory with analyzed data
    subdir = os.path.join(sample_data_dir, "subdir")
    os.makedirs(subdir)
    
    # Create analyzed data file with LED data
    analyzed_data = pd.DataFrame({
        'stimulus': [0, 1, 0]
    })
    analyzed_data.to_csv(os.path.join(subdir, "analysis.csv"), index=False)
    
    led_data = get_led_data(sample_data_dir)
    assert isinstance(led_data, np.ndarray)
    assert len(led_data) == 3 
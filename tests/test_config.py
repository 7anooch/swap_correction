"""
Tests for the config module.
"""

import pytest
import numpy as np
import pandas as pd
from swap_corrector.config import SwapCorrectionConfig, SwapConfig, SwapThresholds

def test_swap_correction_config_defaults():
    """Test that SwapCorrectionConfig has correct default values."""
    config = SwapCorrectionConfig()
    
    assert config.filtered_data_filename == "filtered_data.csv"
    assert config.fix_swaps is True
    assert config.validate is True
    assert config.remove_errors is True
    assert config.interpolate is True
    assert config.filter_data is True
    assert config.debug is True
    assert config.diagnostic_plots is True
    assert config.show_plots is False
    assert config.times is None
    assert config.log_level == "DEBUG"
    assert config.log_file == "swap_corrector.log"

def test_swap_correction_config_custom_values():
    """Test that SwapCorrectionConfig can be initialized with custom values."""
    config = SwapCorrectionConfig(
        filtered_data_filename="custom.csv",
        fix_swaps=False,
        validate=True,
        remove_errors=False,
        interpolate=False,
        filter_data=True,
        debug=True,
        diagnostic_plots=False,
        show_plots=True,
        times=(0.0, 10.0),
        log_level="DEBUG",
        log_file="log.txt"
    )
    
    assert config.filtered_data_filename == "custom.csv"
    assert config.fix_swaps is False
    assert config.validate is True
    assert config.remove_errors is False
    assert config.interpolate is False
    assert config.filter_data is True
    assert config.debug is True
    assert config.diagnostic_plots is False
    assert config.show_plots is True
    assert config.times == (0.0, 10.0)
    assert config.log_level == "DEBUG"
    assert config.log_file == "log.txt"

def test_default_config():
    """Test default configuration initialization."""
    config = SwapConfig()
    
    # Check default values
    assert config.fps == 30
    assert isinstance(config.thresholds, SwapThresholds)
    assert isinstance(config.window_sizes, dict)
    
    # Check window sizes
    assert config.window_sizes['speed'] == 5
    assert config.window_sizes['acceleration'] == 3
    assert config.window_sizes['curvature'] == 5
    assert config.window_sizes['outlier'] == 5

def test_custom_config():
    """Test custom configuration initialization."""
    custom_windows = {
        'speed': 3,
        'acceleration': 5,
        'curvature': 3,
        'outlier': 7
    }
    
    config = SwapConfig(
        fps=60,
        window_sizes=custom_windows
    )
    
    assert config.fps == 60
    assert config.window_sizes == custom_windows

def test_threshold_adjustment():
    """Test threshold adjustment based on movement."""
    thresholds = SwapThresholds(
        proximity=2.0,
        speed=10.0,
        angle=np.pi/4,
        curvature=0.1,
        body_length=0.7,
        speed_factor=0.1,
        accel_factor=0.05
    )
    
    # Create test metrics data with multiple values for better statistics
    metrics_data = pd.DataFrame({
        'Speed': [0.0, 0.0, 0.0, 0.0, 0.0],  # No movement
        'Acceleration': [0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    # Test with no movement
    thresholds.adjust_for_movement(metrics_data)
    assert thresholds.proximity == 2.0
    assert thresholds.speed == 10.0  # Should not change with no movement
    assert thresholds.angle == np.pi/4
    assert thresholds.curvature == 0.1
    assert thresholds.body_length == 0.7
    
    # Create test metrics data with movement
    metrics_data = pd.DataFrame({
        'Speed': [80.0, 90.0, 100.0, 110.0, 120.0],  # High speed values
        'Acceleration': [20.0, 30.0, 40.0, 50.0, 60.0]  # High acceleration values
    })
    
    # Test with movement
    thresholds.adjust_for_movement(metrics_data)
    assert thresholds.proximity == 2.0  # Should remain constant
    assert thresholds.speed > 10.0  # Should increase due to high mean and std
    assert thresholds.angle == np.pi/4  # Should remain constant
    assert thresholds.curvature == 0.1  # Should remain constant
    assert thresholds.body_length == 0.7  # Should remain constant

def test_config_from_dict():
    """Test configuration creation from dictionary."""
    config_dict = {
        'fps': 60,
        'thresholds': {
            'proximity': 3.0,
            'speed': 15.0,
            'angle': np.pi/3,
            'curvature': 0.2,
            'body_length': 0.8,
            'speed_factor': 0.15,
            'accel_factor': 0.07
        },
        'window_sizes': {
            'speed': 3,
            'acceleration': 5,
            'curvature': 3,
            'outlier': 7
        }
    }
    
    config = SwapConfig.from_dict(config_dict)
    
    assert config.fps == 60
    assert config.thresholds.proximity == 3.0
    assert config.thresholds.speed == 15.0
    assert config.thresholds.angle == np.pi/3
    assert config.thresholds.curvature == 0.2
    assert config.thresholds.body_length == 0.8
    assert config.thresholds.speed_factor == 0.15
    assert config.thresholds.accel_factor == 0.07
    assert config.window_sizes == config_dict['window_sizes']

def test_get_thresholds():
    """Test getting adjusted thresholds from metrics data."""
    config = SwapConfig()
    
    # Create test metrics data
    metrics_data = pd.DataFrame({
        'Speed': [5.0],
        'Acceleration': [2.0]
    })
    
    thresholds = config.get_thresholds(metrics_data)
    
    # Check that thresholds are returned correctly
    assert 'speed' in thresholds
    assert 'acceleration' in thresholds
    assert 'proximity' in thresholds
    assert 'turn' in thresholds
    assert 'angle' in thresholds
    assert 'curvature' in thresholds
    assert 'body_length' in thresholds
    assert 'confidence' in thresholds
    
    # Test with no metrics data
    thresholds = config.get_thresholds(None)
    assert 'speed' in thresholds
    assert 'acceleration' in thresholds
    assert 'proximity' in thresholds
    assert 'turn' in thresholds
    assert 'angle' in thresholds
    assert 'curvature' in thresholds
    assert 'body_length' in thresholds
    assert 'confidence' in thresholds 
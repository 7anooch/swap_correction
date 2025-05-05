"""
Tests for the config module.
"""

import pytest
import numpy as np
from swap_corrector.config import SwapCorrectionConfig, SwapConfig, SwapThresholds

def test_swap_correction_config_defaults():
    """Test that SwapCorrectionConfig has correct default values."""
    config = SwapCorrectionConfig()
    
    assert config.filtered_data_filename == "filtered_data.csv"
    assert config.fix_swaps is True
    assert config.validate is False
    assert config.remove_errors is True
    assert config.interpolate is True
    assert config.filter_data is False
    assert config.debug is False
    assert config.diagnostic_plots is True
    assert config.show_plots is False
    assert config.times is None
    assert config.log_level == "INFO"
    assert config.log_file is None

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
    assert config.window_sizes['acceleration'] == 7
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
    
    # Test with no movement
    adjusted = thresholds.adjust_for_movement(speed=0.0, acceleration=0.0)
    assert adjusted['proximity'] == 2.0
    assert adjusted['speed'] == 10.0
    assert adjusted['angle'] == np.pi/4
    assert adjusted['curvature'] == 0.1
    assert adjusted['body_length'] == 0.7
    
    # Test with movement
    adjusted = thresholds.adjust_for_movement(speed=5.0, acceleration=2.0)
    assert adjusted['proximity'] == pytest.approx(2.0 * (1 + 0.1 * 5.0))
    assert adjusted['speed'] == pytest.approx(10.0 * (1 + 0.1 * 5.0))
    assert adjusted['angle'] == np.pi/4  # Should remain constant
    assert adjusted['curvature'] == pytest.approx(0.1 * (1 + 0.05 * 2.0))
    assert adjusted['body_length'] == 0.7  # Should remain constant

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
    
    metrics_data = {
        'speed': 5.0,
        'acceleration': 2.0
    }
    
    adjusted = config.get_thresholds(metrics_data)
    
    # Check that thresholds are adjusted correctly
    assert adjusted['proximity'] == pytest.approx(2.0 * (1 + 0.1 * 5.0))
    assert adjusted['speed'] == pytest.approx(10.0 * (1 + 0.1 * 5.0))
    assert adjusted['angle'] == np.pi/4
    assert adjusted['curvature'] == pytest.approx(0.1 * (1 + 0.05 * 2.0))
    assert adjusted['body_length'] == 0.7
    
    # Test with missing metrics
    adjusted = config.get_thresholds({})
    assert adjusted == config.thresholds.adjust_for_movement(0.0, 0.0) 
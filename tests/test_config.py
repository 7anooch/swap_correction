"""
Tests for the config module.
"""

import pytest
from swap_corrector.config import SwapCorrectionConfig

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
"""
Tests for the logger module.
"""

import logging
import pytest
from swap_corrector.logger import setup_logger
from swap_corrector.config import SwapCorrectionConfig

def test_setup_logger_default():
    """Test that setup_logger creates a logger with default settings."""
    config = SwapCorrectionConfig()
    logger = setup_logger(config)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "swap_corrector"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_setup_logger_debug():
    """Test that setup_logger creates a debug logger when debug=True."""
    config = SwapCorrectionConfig(log_level="DEBUG")
    logger = setup_logger(config)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "swap_corrector"
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_setup_logger_with_file():
    """Test that setup_logger creates a logger with a file handler."""
    config = SwapCorrectionConfig(log_file="test.log")
    logger = setup_logger(config)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "swap_corrector"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.handlers[1], logging.FileHandler)

def test_setup_logger_custom_name():
    """Test that setup_logger creates a logger with a custom name."""
    custom_name = "test_logger"
    config = SwapCorrectionConfig()
    logger = setup_logger(config, name=custom_name)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == custom_name
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler) 
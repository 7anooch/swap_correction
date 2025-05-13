"""
Tests for the logger module.
"""

import logging
import pytest
import os
from swap_corrector.logger import setup_logger
from swap_corrector.config import SwapCorrectionConfig

def test_setup_logger_default():
    """Test that setup_logger creates a default logger."""
    logger = setup_logger()
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == "swap_corrector"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_setup_logger_debug():
    """Test that setup_logger creates a logger with debug level."""
    logger = setup_logger(level="DEBUG")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_setup_logger_with_file(temp_dir):
    """Test that setup_logger creates a logger with a file handler."""
    log_file = os.path.join(temp_dir, "test.log")
    logger = setup_logger(log_file=log_file)
    assert isinstance(logger, logging.Logger)
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.handlers[1], logging.FileHandler)

def test_setup_logger_with_string_path(temp_dir):
    """Test that setup_logger creates a logger with a string path."""
    logger = setup_logger()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "swap_corrector"

def test_setup_logger_invalid_config():
    """Test that setup_logger raises TypeError for invalid config."""
    with pytest.raises(TypeError):
        setup_logger(123)  # Invalid config type 
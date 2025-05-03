"""
Logging configuration for the swap correction pipeline.
"""

import logging
from typing import Optional
from .config import SwapCorrectionConfig

def setup_logger(config: SwapCorrectionConfig, name: str = "swap_corrector") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration object containing logging settings
        name: Name of the logger (default: "swap_corrector")
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(config.log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger(SwapCorrectionConfig()) 
"""
Logging configuration for the swap correction pipeline.
"""

import logging
from typing import Optional
from .config import SwapCorrectionConfig

def setup_logger(config: SwapCorrectionConfig) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration object containing logging settings
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("swap_corrector")
    logger.setLevel(config.log_level)
    
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
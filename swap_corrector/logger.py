"""
Logging configuration for swap correction pipeline.
"""

import logging
import os
from datetime import datetime
from typing import Union, Optional
from pathlib import Path
from .config import SwapCorrectionConfig

def setup_logger(name: str = "swap_corrector", level: str = "INFO",
                log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    """Setup logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        
    Returns:
        Configured logger
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), 20)  # Default to INFO (20)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file is not None:
        # Convert string to Path if needed
        if isinstance(log_file, str):
            log_file = Path(log_file)
            
        # Create parent directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize default logger
logger = setup_logger() 
"""
Configuration settings for the swap correction pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SwapCorrectionConfig:
    """Configuration settings for swap correction pipeline."""
    
    # File settings
    filtered_data_filename: str = "filtered_data.csv"
    
    # Processing settings
    fix_swaps: bool = True
    validate: bool = False
    remove_errors: bool = True
    interpolate: bool = True
    filter_data: bool = False
    
    # Debug settings
    debug: bool = False
    diagnostic_plots: bool = True
    show_plots: bool = False
    
    # Plot settings
    times: Optional[Tuple[float, float]] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

# Default configuration
default_config = SwapCorrectionConfig() 
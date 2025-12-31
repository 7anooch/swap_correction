"""
Machine Learning module for swap detection.

This module provides tools for training, evaluating, and using ML models
to detect head-tail swaps in animal tracking data.
"""

__version__ = "1.0.0"

# Import main API components for easy access
from swap_correction.ml.api import (
    SwapPredictor,
    BatchProcessor,
    load_model,
    get_model_info,
    list_available_models
)

__all__ = [
    'SwapPredictor',
    'BatchProcessor',
    'load_model',
    'get_model_info',
    'list_available_models',
]


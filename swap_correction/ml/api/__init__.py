"""
Model API for easy swap detection.

Provides clean interfaces for loading models and making predictions.
"""

from swap_correction.ml.api.model_loader import (
    load_model,
    get_model_info,
    list_available_models
)

from swap_correction.ml.api.predictor import SwapPredictor

from swap_correction.ml.api.batch_processor import BatchProcessor

__all__ = [
    'load_model',
    'get_model_info',
    'list_available_models',
    'SwapPredictor',
    'BatchProcessor',
]


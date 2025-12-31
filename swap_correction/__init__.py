"""
Swap Correction Package

A package for correcting head-tail swaps in animal tracking data.
"""

__version__ = "0.1.0"

# Export ML API for easy access
try:
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
except ImportError:
    # ML module not available
    __all__ = [] 
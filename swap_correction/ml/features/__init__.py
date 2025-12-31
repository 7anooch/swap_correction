"""
Feature extraction for ML swap detection.

This module provides functions to extract frame-level features from tracking data
for training and inference with ML models.

The implementation uses optimized algorithms (400x faster than naive approaches)
with pre-computation and vectorized operations.
"""

from swap_correction.ml.features.features import (
    extract_all_frame_features_optimized
)

__all__ = [
    'extract_all_frame_features_optimized',
]


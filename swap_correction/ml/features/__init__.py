"""
Feature extraction for ML swap detection.

This module provides functions to extract frame-level features from tracking data
for training and inference with ML models.

The implementation uses optimized algorithms (400x faster than naive approaches)
with pre-computation and vectorized operations.

Default implementation: Features V4 (~36 features)
- Phase 1: Removed underperforming features (collapsed_keypoints, raw curvatures, window size 5)
- Phase 2: Added acceleration features and body length normalization
- Best balance of performance and efficiency

Legacy feature versions are available in the legacy/ subdirectory.
"""

from swap_correction.ml.features.features import (
    extract_all_frame_features_optimized
)

__all__ = [
    'extract_all_frame_features_optimized',
]


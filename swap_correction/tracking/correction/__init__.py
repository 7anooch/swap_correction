"""
Core functions for correcting head-tail swaps and tracking errors.
"""

from .correction import (
    tracking_correction,
    remove_edge_frames,
    correct_tracking_errors,
    validate_corrected_data,
    remove_overlaps,
    interpolate_gaps,
    correct_global_swap,
    correct_swapped_segments,
    get_swapped_segments
)

__all__ = [
    'tracking_correction',
    'remove_edge_frames',
    'correct_tracking_errors',
    'validate_corrected_data',
    'remove_overlaps',
    'interpolate_gaps',
    'correct_global_swap',
    'correct_swapped_segments',
    'get_swapped_segments'
] 
"""
Functions for detecting potential head-tail swaps and tracking errors.
"""

from .flags import (
    flag_all_swaps,
    flag_discontinuities,
    flag_delta_mismatches,
    flag_sign_reversals,
    flag_overlaps,
    flag_overlap_sign_reversals,
    get_overlap_edges,
    get_all_overlap_edges,
    get_all_deltas
)

__all__ = [
    'flag_all_swaps',
    'flag_discontinuities',
    'flag_delta_mismatches',
    'flag_sign_reversals',
    'flag_overlaps',
    'flag_overlap_sign_reversals',
    'get_overlap_edges',
    'get_all_overlap_edges',
    'get_all_deltas'
] 
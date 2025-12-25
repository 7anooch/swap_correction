"""
Tracking correction package for handling head-tail swap detection and correction.
"""

from .correction import tracking_correction, remove_edge_frames, interpolate_gaps, correct_global_swap, correct_tracking_errors, validate_corrected_data, remove_overlaps, correct_swapped_segments, get_swapped_segments, correct_no_flags, correct_swaps, correct_both_flags, correct_empty_df
from .flags import flag_all_swaps, flag_overlaps, get_overlap_edges, flag_discontinuities, flag_delta_mismatches, flag_sign_reversals, flag_overlap_sign_reversals, get_all_overlap_edges, flag_overlap_minimum_mismatches, get_all_deltas
from .filters import filter_sgolay, filter_gaussian, filter_meanmed, filter_median, filter_data

__all__ = [
    'tracking_correction', 'remove_edge_frames', 'interpolate_gaps', 'correct_global_swap',
    'correct_tracking_errors', 'validate_corrected_data', 'remove_overlaps',
    'correct_swapped_segments', 'get_swapped_segments', 'correct_no_flags',
    'correct_swaps', 'correct_both_flags', 'correct_empty_df',
    'flag_all_swaps', 'flag_overlaps', 'get_overlap_edges', 'flag_discontinuities',
    'flag_delta_mismatches', 'flag_sign_reversals', 'flag_overlap_sign_reversals',
    'get_all_overlap_edges', 'flag_overlap_minimum_mismatches', 'get_all_deltas',
    'filter_sgolay', 'filter_gaussian', 'filter_meanmed', 'filter_median', 'filter_data'
] 
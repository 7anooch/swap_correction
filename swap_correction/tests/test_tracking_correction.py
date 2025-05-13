import numpy as np
import pandas as pd
import pytest
from unittest import mock
from swap_correction import tracking_correction

@pytest.fixture
def simple_data():
    # Minimal DataFrame with all required columns
    return pd.DataFrame({
        'xhead': [0, 1, 2], 'yhead': [0, 1, 2],
        'xtail': [0, 1, 2], 'ytail': [0, 1, 2],
        'xmid': [0, 1, 2], 'ymid': [0, 1, 2],
        'xctr': [0, 1, 2], 'yctr': [0, 1, 2]
    })

@pytest.fixture
def float_simple_data():
    # Like simple_data, but all columns are float (to avoid NaN assignment warnings)
    return pd.DataFrame({
        'xhead': [0.0, 1.0, 2.0], 'yhead': [0.0, 1.0, 2.0],
        'xtail': [0.0, 1.0, 2.0], 'ytail': [0.0, 1.0, 2.0],
        'xmid': [0.0, 1.0, 2.0], 'ymid': [0.0, 1.0, 2.0],
        'xctr': [0.0, 1.0, 2.0], 'yctr': [0.0, 1.0, 2.0]
    })

def test_tracking_correction_pipeline(simple_data):
    # Patch all sub-functions to just return the data
    with mock.patch('swap_correction.tracking_correction.remove_edge_frames', side_effect=lambda d, **kwargs: d), \
         mock.patch('swap_correction.tracking_correction.correct_tracking_errors', side_effect=lambda d, **kwargs: d), \
         mock.patch('swap_correction.tracking_correction.validate_corrected_data', side_effect=lambda d, *a, **k: d), \
         mock.patch('swap_correction.tracking_correction.remove_overlaps', side_effect=lambda d, *a, **k: d), \
         mock.patch('swap_correction.tracking_correction.interpolate_gaps', side_effect=lambda d, **kwargs: d), \
         mock.patch('swap_correction.tracking_correction.filter_data', side_effect=lambda d: d):
        out = tracking_correction.tracking_correction(simple_data, fps=30)
        assert isinstance(out, pd.DataFrame)
        assert np.allclose(out.values, simple_data.values)

def test_remove_edge_frames(float_simple_data):
    # Patch utils.flatten and metrics.POSDICT
    with mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking_correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}):
        out = tracking_correction.remove_edge_frames(float_simple_data)
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) == set(float_simple_data.columns)

def test_correct_tracking_errors(simple_data):
    # Patch flag_all_swaps, utils.indices_to_segments, correct_swapped_segments, correct_global_swap
    with mock.patch('swap_correction.tracking_correction.flag_all_swaps', return_value=np.array([])), \
         mock.patch('swap_correction.tracking_correction.utils.indices_to_segments', return_value=[]), \
         mock.patch('swap_correction.tracking_correction.correct_swapped_segments', side_effect=lambda d, s, **k: d), \
         mock.patch('swap_correction.tracking_correction.correct_global_swap', side_effect=lambda d, **k: d):
        out = tracking_correction.correct_tracking_errors(simple_data)
        assert isinstance(out, pd.DataFrame)

def test_validate_corrected_data(simple_data):
    # Patch get_swapped_segments, correct_swapped_segments
    with mock.patch('swap_correction.tracking_correction.get_swapped_segments', return_value=[]), \
         mock.patch('swap_correction.tracking_correction.correct_swapped_segments', side_effect=lambda d, s, **k: d):
        out = tracking_correction.validate_corrected_data(simple_data, fps=30)
        assert isinstance(out, pd.DataFrame)

def test_remove_overlaps(simple_data):
    # Patch get_overlap_edges, flag_discontinuities, utils.flatten
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=[(0, 1)]), \
         mock.patch('swap_correction.tracking_correction.flag_discontinuities', return_value=[0]), \
         mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: np.concatenate(x) if len(x) > 0 else np.array([])):
        out = tracking_correction.remove_overlaps(simple_data, fps=30)
        assert isinstance(out, pd.DataFrame)

def test_interpolate_gaps(simple_data):
    # Patch utils.get_value_segments, utils.ranges_to_list
    with mock.patch('swap_correction.tracking_correction.utils.get_value_segments', return_value=np.array([[0, 1]])), \
         mock.patch('swap_correction.tracking_correction.utils.ranges_to_list', return_value=[0, 1]):
        # Insert NaNs to test interpolation
        data = simple_data.copy()
        data.loc[0, 'xhead'] = np.nan
        data.loc[1, 'xhead'] = np.nan
        out = tracking_correction.interpolate_gaps(data, method='linear', maxSegment=2)
        assert isinstance(out, pd.DataFrame)

def test_flag_all_swaps(float_simple_data):
    # Patch all sub-functions and utils.merge/filter_array
    with mock.patch('swap_correction.tracking_correction.flag_overlaps', return_value=np.array([0, 1])), \
         mock.patch('swap_correction.tracking_correction.flag_min_delta_mismatches', return_value=np.array([1])), \
         mock.patch('swap_correction.tracking_correction.flag_overlap_sign_reversals', return_value=np.array([2])), \
         mock.patch('swap_correction.tracking_correction.flag_overlap_minimum_mismatches', return_value=np.array([3])), \
         mock.patch('swap_correction.tracking_correction.utils.merge', side_effect=lambda *a: np.unique(np.concatenate(a))), \
         mock.patch('swap_correction.tracking_correction.utils.filter_array', side_effect=lambda arr, filt: arr):
        result = tracking_correction.flag_all_swaps(float_simple_data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin([1, 2, 3], result))

def test_flag_discontinuities(float_simple_data):
    # Patch metrics.get_delta_between_frames
    with mock.patch('swap_correction.tracking_correction.metrics.get_delta_between_frames', return_value=np.array([0, 25, 10])):
        result = tracking_correction.flag_discontinuities(float_simple_data, key='head', fps=30, threshold=20)
        # Should flag index 1 (value 25 > 20), so returns 2 (index+1)
        assert np.array_equal(result, [2])

def test_flag_delta_mismatches(float_simple_data):
    # Patch get_all_deltas
    with mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=(np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), np.array([1, 2]))):
        result = tracking_correction.flag_delta_mismatches(float_simple_data, tolerance=0.0)
        assert isinstance(result, np.ndarray)

def test_flag_min_delta_mismatches(float_simple_data):
    # Patch get_all_deltas
    with mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=np.array([[1, 2], [1, 2], [1, 2], [1, 2]])):
        result = tracking_correction.flag_min_delta_mismatches(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_flag_sign_reversals(float_simple_data):
    # Patch metrics.get_ht_cross_sign and metrics.get_head_angle
    with mock.patch('swap_correction.tracking_correction.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        result = tracking_correction.flag_sign_reversals(float_simple_data, threshold=np.pi/2)
        assert isinstance(result, np.ndarray)

def test_flag_overlaps(float_simple_data):
    # Patch metrics.perfectly_overlapping
    with mock.patch('swap_correction.tracking_correction.metrics.perfectly_overlapping', return_value=np.array([0, 2])):
        result = tracking_correction.flag_overlaps(float_simple_data, tolerance=0)
        assert np.array_equal(result, [0, 2])

def test_flag_overlap_mismatches(float_simple_data):
    # Patch get_overlap_edges, get_all_deltas
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=(np.array([3, 5]), np.array([3, 5]), np.array([1, 2]), np.array([1, 2]))):
        result = tracking_correction.flag_overlap_mismatches(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_flag_overlap_minimum_mismatches(float_simple_data):
    # Patch get_overlap_edges, get_all_deltas
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=np.array([[1, 2], [1, 2], [3, 4], [5, 6]])):
        result = tracking_correction.flag_overlap_minimum_mismatches(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_flag_overlap_sign_reversals(float_simple_data):
    # Patch get_overlap_edges, metrics.get_ht_cross_sign, metrics.get_head_angle
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [1, 2]])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        result = tracking_correction.flag_overlap_sign_reversals(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_get_overlap_edges(float_simple_data):
    # Patch flag_overlaps and utils.get_consecutive_ranges
    with mock.patch('swap_correction.tracking_correction.flag_overlaps', return_value=np.array([0, 1, 2])), \
         mock.patch('swap_correction.tracking_correction.utils.get_consecutive_ranges', return_value=[[0, 2]]):
        result = tracking_correction.get_overlap_edges(float_simple_data)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 2

def test_get_all_overlap_edges(float_simple_data):
    # Patch flag_overlaps and utils.match_arrays/filter_array/get_consecutive_ranges
    with mock.patch('swap_correction.tracking_correction.flag_overlaps', side_effect=[np.array([0]), np.array([1]), np.array([2])]), \
         mock.patch('swap_correction.tracking_correction.utils.match_arrays', side_effect=lambda a, b: np.intersect1d(a, b)), \
         mock.patch('swap_correction.tracking_correction.utils.filter_array', side_effect=lambda a, b: a), \
         mock.patch('swap_correction.tracking_correction.utils.get_consecutive_ranges', return_value=[[0, 1]]):
        result = tracking_correction.get_all_overlap_edges(float_simple_data)
        assert isinstance(result, tuple)
        assert all(isinstance(arr, np.ndarray) for arr in result)

def test_get_all_deltas(float_simple_data):
    # Patch metrics.get_delta_between_frames and metrics.get_cross_segment_deltas
    with mock.patch('swap_correction.tracking_correction.metrics.get_delta_between_frames', return_value=np.array([1, 2, 3])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_cross_segment_deltas', return_value=np.array([1, 2, 3])):
        # No edges
        result = tracking_correction.get_all_deltas(float_simple_data)
        assert isinstance(result, np.ndarray) or (isinstance(result, tuple) and all(isinstance(x, np.ndarray) for x in result))
        # With edges
        result2 = tracking_correction.get_all_deltas(float_simple_data, edges=np.array([[0, 1]]))
        assert isinstance(result2, np.ndarray) or (isinstance(result2, tuple) and all(isinstance(x, np.ndarray) for x in result2))

def test_correct_global_swap(float_simple_data):
    # Patch filter_data and metrics.get_speed_from_df
    with mock.patch('swap_correction.tracking_correction.filter_data', return_value=float_simple_data), \
         mock.patch('swap_correction.tracking_correction.metrics.get_speed_from_df', side_effect=[np.array([1, 2, 3]), np.array([4, 5, 6])]):
        out = tracking_correction.correct_global_swap(float_simple_data)
        assert isinstance(out, pd.DataFrame)

def test_get_swapped_segments(float_simple_data):
    # Patch remove_overlaps, filter_data, get_overlap_edges, utils.invert_ranges, utils.segment_lengths, _flag_segment_metrics
    with mock.patch('swap_correction.tracking_correction.remove_overlaps', return_value=float_simple_data), \
         mock.patch('swap_correction.tracking_correction.filter_data', return_value=float_simple_data), \
         mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.utils.invert_ranges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.utils.segment_lengths', return_value=np.array([2, 2])), \
         mock.patch('swap_correction.tracking_correction._flag_segment_metrics', return_value=np.array([1, 0])):
        out = tracking_correction.get_swapped_segments(float_simple_data, fps=30)
        assert isinstance(out, np.ndarray)

def test__swap_following_ambiguous_flags():
    flag = np.array([1, 0, 0, -1, 0, 1])
    with mock.patch('swap_correction.tracking_correction.utils.get_value_segments', return_value=[[1, 3], [4, 5]]):
        out = tracking_correction._swap_following_ambiguous_flags(flag)
        assert isinstance(out, np.ndarray)

def test__merge_ambiguous_flags():
    flag = np.array([1, 0, 0, -1, 0, 1])
    with mock.patch('swap_correction.tracking_correction.utils.get_value_segments', return_value=[[1, 3], [4, 5]]):
        out = tracking_correction._merge_ambiguous_flags(flag)
        assert isinstance(out, np.ndarray)

def test__flag_segment_metrics():
    metric = np.array([0.5, 1.2, 0.8, 1.5])
    nvals = np.array([10, 10, 2, 10])
    thresh = (0.9, 1.1)
    minFrames = 5
    out = tracking_correction._flag_segment_metrics(metric, nvals, thresh, minFrames)
    assert isinstance(out, np.ndarray)

def test__get_travel_distance_ratios(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.metrics.get_segment_distance', side_effect=[np.array([2, 4]), np.array([1, 2])]):
        segs = np.array([[0, 1], [1, 2]])
        out = tracking_correction._get_travel_distance_ratios(float_simple_data, segs)
        assert isinstance(out, np.ndarray)

def test__get_speed_ratios(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.metrics.get_speed_from_df', side_effect=[np.array([1, 2, 3]), np.array([2, 2, 2])]):
        segs = np.array([[0, 2], [1, 2]])
        out, nvals = tracking_correction._get_speed_ratios(float_simple_data, segs)
        assert isinstance(out, np.ndarray)
        assert isinstance(nvals, np.ndarray)

def test__get_alignment_angles(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.metrics.get_motion_vector', return_value=np.array([[1, 0], [0, 1], [1, 1]])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_orientation_vectors', return_value=np.array([[1, 0], [0, 1], [1, 1]])), \
         mock.patch('swap_correction.tracking_correction.utils.get_angle', return_value=0.5), \
         mock.patch('swap_correction.tracking_correction.utils.metrics_by_segment', return_value=np.array([[0.5, 0.5, 0.1], [0.6, 0.6, 0.1]])):
        segs = np.array([[0, 1], [1, 2]])
        out = tracking_correction._get_alignment_angles(float_simple_data, segs)
        assert isinstance(out, np.ndarray)

def test__get_alignment_angles_legacy(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.metrics.get_motion_vector', return_value=np.array([[1, 0], [0, 1], [1, 1]])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_orientation_vectors', return_value=np.array([[1, 0], [0, 1], [1, 1]])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_speed_from_df', return_value=np.array([1, 2, 3])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_head_angle', return_value=np.array([0.1, 0.2, 0.3])):
        segs = np.array([[0, 1], [1, 2]])
        out = tracking_correction._get_alignment_angles_legacy(float_simple_data, fps=30, segs=segs, minFrames=1)
        assert isinstance(out, np.ndarray)

def test_correct_swapped_segments(float_simple_data):
    segs = np.array([[0, 1], [2, 2]])
    out = tracking_correction.correct_swapped_segments(float_simple_data, segs)
    assert isinstance(out, pd.DataFrame)

def test_filter_data(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.filter_gaussian', return_value=float_simple_data):
        out = tracking_correction.filter_data(float_simple_data)
        assert isinstance(out, pd.DataFrame)

def test_filter_sgolay(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking_correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}), \
         mock.patch('scipy.signal.savgol_filter', return_value=np.array([1.0, 2.0, 3.0])):
        out = tracking_correction.filter_sgolay(float_simple_data, window=3, order=1)
        assert isinstance(out, pd.DataFrame)

def test_filter_gaussian(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking_correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}), \
         mock.patch('scipy.ndimage.gaussian_filter1d', return_value=np.array([1.0, 2.0, 3.0])):
        out = tracking_correction.filter_gaussian(float_simple_data, sigma=1)
        assert isinstance(out, pd.DataFrame)

def test_filter_meanmed(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking_correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}), \
         mock.patch('scipy.ndimage.median_filter', return_value=np.array([1.0, 2.0, 3.0])), \
         mock.patch('scipy.ndimage.uniform_filter', return_value=np.array([1.0, 2.0, 3.0])):
        out = tracking_correction.filter_meanmed(float_simple_data, medWin=2, meanWin=2)
        assert isinstance(out, pd.DataFrame)

def test_filter_median(float_simple_data):
    with mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking_correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}), \
         mock.patch('scipy.ndimage.median_filter', return_value=np.array([1.0, 2.0, 3.0])):
        out = tracking_correction.filter_median(float_simple_data, win=2)
        assert isinstance(out, pd.DataFrame)

def test_remove_edge_frames_debug(float_simple_data, capsys):
    # Patch utils.flatten and metrics.POSDICT
    with mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking_correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}):
        out = tracking_correction.remove_edge_frames(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert 'Zeroed Frames' in captured.out or 'Edge Segments' in captured.out

def test_remove_overlaps_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=[(0, 1)]), \
         mock.patch('swap_correction.tracking_correction.flag_discontinuities', return_value=[0]), \
         mock.patch('swap_correction.tracking_correction.utils.flatten', side_effect=lambda x: np.concatenate(x) if len(x) > 0 else np.array([])):
        out = tracking_correction.remove_overlaps(float_simple_data, fps=30, debug=True)
        captured = capsys.readouterr()
        assert 'Head Segments Removed' in captured.out or 'Tail Segments Removed' in captured.out

def test_flag_all_swaps_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.flag_overlaps', return_value=np.array([0, 1])), \
         mock.patch('swap_correction.tracking_correction.flag_min_delta_mismatches', return_value=np.array([1])), \
         mock.patch('swap_correction.tracking_correction.flag_overlap_sign_reversals', return_value=np.array([2])), \
         mock.patch('swap_correction.tracking_correction.flag_overlap_minimum_mismatches', return_value=np.array([3])), \
         mock.patch('swap_correction.tracking_correction.utils.merge', side_effect=lambda *a: np.unique(np.concatenate(a))), \
         mock.patch('swap_correction.tracking_correction.utils.filter_array', side_effect=lambda arr, filt: arr):
        tracking_correction.flag_all_swaps(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert 'All Flags' in captured.out

def test_flag_discontinuities_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.metrics.get_delta_between_frames', return_value=np.array([0, 25, 10])):
        tracking_correction.flag_discontinuities(float_simple_data, key='head', fps=30, threshold=20, debug=True)
        captured = capsys.readouterr()
        assert 'Discontinuities' in captured.out

def test_flag_delta_mismatches_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=(np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), np.array([1, 2]))):
        tracking_correction.flag_delta_mismatches(float_simple_data, tolerance=0.0, debug=True)
        captured = capsys.readouterr()
        assert 'Delta Mismatches' in captured.out

def test_flag_min_delta_mismatches_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=np.array([[1, 2], [1, 2], [1, 2], [1, 2]])):
        tracking_correction.flag_min_delta_mismatches(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert 'Minimum-Delta Mismatches' in captured.out

def test_flag_sign_reversals_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        tracking_correction.flag_sign_reversals(float_simple_data, threshold=np.pi/2, debug=True)
        captured = capsys.readouterr()
        assert 'Sign Reversals' in captured.out

def test_flag_overlaps_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.metrics.perfectly_overlapping', return_value=np.array([0, 2])):
        tracking_correction.flag_overlaps(float_simple_data, tolerance=0, debug=True)
        captured = capsys.readouterr()
        assert 'Overlaps' in captured.out

def test_flag_overlap_mismatches_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=(np.array([3, 5]), np.array([3, 5]), np.array([1, 2]), np.array([1, 2]))):
        tracking_correction.flag_overlap_mismatches(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert 'Cross-Overlap Mismatches' in captured.out

def test_flag_overlap_minimum_mismatches_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.get_all_deltas', return_value=np.array([[1, 2], [1, 2], [3, 4], [5, 6]])):
        tracking_correction.flag_overlap_minimum_mismatches(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert 'Cross-Overlap Minimum-Delta Mismatch' in captured.out

def test_flag_overlap_sign_reversals_debug(float_simple_data, capsys):
    with mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [1, 2]])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking_correction.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        tracking_correction.flag_overlap_sign_reversals(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert 'Cross-Overlap Sign Reversals' in captured.out

def test_get_swapped_segments_modes(float_simple_data):
    # Patch all dependencies and test all match/case branches
    with mock.patch('swap_correction.tracking_correction.remove_overlaps', return_value=float_simple_data), \
         mock.patch('swap_correction.tracking_correction.filter_data', return_value=float_simple_data), \
         mock.patch('swap_correction.tracking_correction.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.utils.invert_ranges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking_correction.utils.segment_lengths', return_value=np.array([2, 2])), \
         mock.patch('swap_correction.tracking_correction._flag_segment_metrics', return_value=np.array([1, 0])), \
         mock.patch('swap_correction.tracking_correction._get_alignment_angles', return_value=np.array([0.5, 1.5])), \
         mock.patch('swap_correction.tracking_correction._get_speed_ratios', return_value=(np.array([0.5, 1.5]), np.array([2, 2]))), \
         mock.patch('swap_correction.tracking_correction._get_travel_distance_ratios', return_value=np.array([0.5, 1.5])):
        # alignment mode
        out = tracking_correction.get_swapped_segments(float_simple_data, fps=30, mode='alignment')
        assert isinstance(out, np.ndarray)
        # speed mode
        out = tracking_correction.get_swapped_segments(float_simple_data, fps=30, mode='speed')
        assert isinstance(out, np.ndarray)
        # distance mode
        out = tracking_correction.get_swapped_segments(float_simple_data, fps=30, mode='distance')
        assert isinstance(out, np.ndarray) 
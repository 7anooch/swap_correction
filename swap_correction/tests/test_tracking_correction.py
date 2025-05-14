import numpy as np
import pandas as pd
import pytest
from unittest import mock
from swap_correction.tracking.correction import correction as tracking_correction
from swap_correction.tracking.flagging import flags as flagging
from swap_correction.tracking.filtering import (
    filter_data,
    filter_sgolay,
    filter_gaussian,
    filter_meanmed,
    filter_median
)

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
    with mock.patch('swap_correction.tracking.correction.correction.remove_edge_frames', side_effect=lambda d, **kwargs: d), \
         mock.patch('swap_correction.tracking.correction.correction.correct_tracking_errors', side_effect=lambda d, fps, swapCorrection, validate, debug: d), \
         mock.patch('swap_correction.tracking.correction.correction.validate_corrected_data', side_effect=lambda d, fps, debug: d), \
         mock.patch('swap_correction.tracking.correction.correction.remove_overlaps', side_effect=lambda d, debug: d), \
         mock.patch('swap_correction.tracking.correction.correction.interpolate_gaps', side_effect=lambda d, debug: d), \
         mock.patch('swap_correction.tracking.filtering.filters.filter_data', side_effect=lambda d: d):
        out = tracking_correction.tracking_correction(simple_data, fps=30)
        assert isinstance(out, pd.DataFrame)
        assert np.allclose(out.values, simple_data.values)

def test_remove_edge_frames(float_simple_data):
    # Patch utils.flatten and metrics.POSDICT
    with mock.patch('swap_correction.tracking.correction.correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking.correction.correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}):
        out = tracking_correction.remove_edge_frames(float_simple_data)
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) == set(float_simple_data.columns)

def test_correct_tracking_errors(simple_data):
    # Patch flag_all_swaps, utils.indices_to_segments, correct_swapped_segments, correct_global_swap
    with mock.patch('swap_correction.tracking.flagging.flags.flag_all_swaps', return_value=np.array([])), \
         mock.patch('swap_correction.tracking.correction.correction.utils.get_consecutive_ranges', return_value=[]), \
         mock.patch('swap_correction.tracking.correction.correction.correct_swapped_segments', side_effect=lambda d, s, e, **k: d), \
         mock.patch('swap_correction.tracking.correction.correction.correct_global_swap', side_effect=lambda d, **k: d):
        out = tracking_correction.correct_tracking_errors(simple_data, fps=30)
        assert isinstance(out, pd.DataFrame)

def test_validate_corrected_data(simple_data):
    # Patch get_swapped_segments, correct_swapped_segments
    with mock.patch('swap_correction.tracking.correction.correction.get_swapped_segments', return_value=[]), \
         mock.patch('swap_correction.tracking.correction.correction.correct_swapped_segments', side_effect=lambda d, start, end, debug: d):
        out = tracking_correction.validate_corrected_data(simple_data, fps=30)
        assert isinstance(out, pd.DataFrame)

def test_remove_overlaps(simple_data):
    # Patch get_overlap_edges, flag_discontinuities, utils.flatten
    with mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=pd.DataFrame({'start': [0], 'end': [1]})), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_discontinuities', return_value=np.array([0])), \
         mock.patch('swap_correction.tracking.correction.correction.utils.flatten', side_effect=lambda x: ['xhead', 'yhead', 'xtail', 'ytail', 'xmid', 'ymid', 'xctr', 'yctr']):
        out = tracking_correction.remove_overlaps(simple_data)
        assert isinstance(out, pd.DataFrame)

def test_interpolate_gaps(simple_data):
    # Insert NaNs to test interpolation
    data = simple_data.copy()
    data.loc[0, 'xhead'] = np.nan
    data.loc[1, 'xhead'] = np.nan
    out = tracking_correction.interpolate_gaps(data)
    assert isinstance(out, pd.DataFrame)

def test_flag_all_swaps(float_simple_data):
    # Patch all the individual flagging functions
    with mock.patch('swap_correction.tracking.flagging.flags.flag_discontinuities', return_value=np.array([1])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_delta_mismatches', return_value=np.array([2])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_sign_reversals', return_value=np.array([3])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_overlaps', return_value=np.array([4])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_overlap_sign_reversals', return_value=np.array([5])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_overlap_minimum_mismatches', return_value=np.array([6])):
        result = flagging.flag_all_swaps(float_simple_data, fps=30)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

def test_flag_discontinuities(float_simple_data):
    # Patch metrics.get_delta_between_frames
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.get_delta_between_frames', return_value=np.array([0, 25, 10])):
        result = flagging.flag_discontinuities(float_simple_data, point='head', fps=30)
        assert isinstance(result, np.ndarray)

def test_flag_delta_mismatches(float_simple_data):
    # Patch get_all_deltas
    with mock.patch('swap_correction.tracking.flagging.flags.get_all_deltas', return_value=(np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), np.array([1, 2]))):
        result = flagging.flag_delta_mismatches(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_flag_sign_reversals(float_simple_data):
    # Patch metrics.get_ht_cross_sign and metrics.get_head_angle
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking.flagging.flags.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        result = flagging.flag_sign_reversals(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_flag_overlaps(float_simple_data):
    # Patch metrics.perfectly_overlapping
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.perfectly_overlapping', return_value=np.array([0, 2])):
        result = flagging.flag_overlaps(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_flag_overlap_minimum_mismatches(float_simple_data):
    # Patch get_overlap_edges, get_all_deltas
    with mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking.flagging.flags.get_all_deltas', return_value=np.array([[1, 2], [1, 2], [3, 4], [5, 6]])):
        result = flagging.flag_overlap_minimum_mismatches(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_flag_overlap_sign_reversals(float_simple_data):
    # Patch get_overlap_edges, metrics.get_ht_cross_sign, metrics.get_head_angle
    with mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=np.array([[0, 1], [1, 2]])), \
         mock.patch('swap_correction.tracking.flagging.flags.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking.flagging.flags.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        result = flagging.flag_overlap_sign_reversals(float_simple_data)
        assert isinstance(result, np.ndarray)

def test_get_overlap_edges(float_simple_data):
    # Patch metrics.perfectly_overlapping
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.perfectly_overlapping', return_value=np.array([0, 2])):
        result = flagging.get_overlap_edges(np.array([0, 2]))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, np.ndarray) for x in result)
        assert len(result[0]) == len(result[1])

def test_get_all_overlap_edges(float_simple_data):
    # Patch get_overlap_edges
    with mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=pd.DataFrame({'start': [0], 'end': [1]})):
        result = flagging.get_all_overlap_edges(float_simple_data)
        assert isinstance(result, pd.DataFrame)
        assert 'start' in result.columns
        assert 'end' in result.columns

def test_get_all_deltas(float_simple_data):
    # Patch metrics.get_delta_between_frames and metrics.get_cross_segment_deltas
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.get_delta_between_frames', return_value=np.array([1, 2, 3])), \
         mock.patch('swap_correction.tracking.flagging.flags.metrics.get_cross_segment_deltas', return_value=np.array([1, 2, 3])):
        # No edges
        result = flagging.get_all_deltas(float_simple_data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(float_simple_data), 2)

def test_correct_global_swap(float_simple_data):
    # Patch filter_data and metrics.get_speed_from_df
    with mock.patch('swap_correction.tracking.filtering.filters.filter_data', return_value=float_simple_data), \
         mock.patch('swap_correction.tracking.correction.correction.metrics.get_speed_from_df', side_effect=[np.array([1, 2, 3]), np.array([4, 5, 6])]):
        out = tracking_correction.correct_global_swap(float_simple_data)
        assert isinstance(out, pd.DataFrame)

def test_get_swapped_segments(float_simple_data):
    # Patch flag_all_swaps and utils.get_consecutive_ranges
    with mock.patch('swap_correction.tracking.flagging.flags.flag_all_swaps', return_value=np.array([1, 2])), \
         mock.patch('swap_correction.tracking.correction.correction.utils.get_consecutive_ranges', return_value=[[1, 2]]):
        result = tracking_correction.get_swapped_segments(float_simple_data, fps=30)
        assert isinstance(result, list)
        assert len(result) > 0

def test_filter_data(float_simple_data):
    # Test filter_data directly
    out = filter_data(float_simple_data)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(float_simple_data.columns)

def test_filter_sgolay(float_simple_data):
    # Test filter_sgolay directly
    out = filter_sgolay(float_simple_data, window=3, order=2)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(float_simple_data.columns)

def test_filter_gaussian(float_simple_data):
    # Test filter_gaussian directly
    out = filter_gaussian(float_simple_data, sigma=1)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(float_simple_data.columns)

def test_filter_meanmed(float_simple_data):
    # Test filter_meanmed directly
    out = filter_meanmed(float_simple_data, medWin=3, meanWin=3)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(float_simple_data.columns)

def test_filter_median(float_simple_data):
    # Test filter_median directly
    out = filter_median(float_simple_data, win=3)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(float_simple_data.columns)

def test_remove_edge_frames_debug(float_simple_data, capsys):
    # Patch utils.flatten and metrics.POSDICT
    with mock.patch('swap_correction.tracking.correction.correction.utils.flatten', side_effect=lambda x: sum(x, [])), \
         mock.patch('swap_correction.tracking.correction.correction.metrics.POSDICT', {'head': ['xhead', 'yhead'], 'tail': ['xtail', 'ytail'], 'mid': ['xmid', 'ymid'], 'ctr': ['xctr', 'yctr']}):
        tracking_correction.remove_edge_frames(float_simple_data)
        captured = capsys.readouterr()
        # No debug output in current implementation
        assert captured.out == ''

def test_remove_overlaps_debug(float_simple_data, capsys):
    # Patch get_overlap_edges, flag_discontinuities, utils.flatten
    with mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=(np.array([0]), np.array([1]))), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_discontinuities', return_value=np.array([0])), \
         mock.patch('swap_correction.tracking.correction.correction.utils.flatten', side_effect=lambda x: ['xhead', 'yhead', 'xtail', 'ytail', 'xmid', 'ymid', 'xctr', 'yctr']):
        tracking_correction.remove_overlaps(float_simple_data)
        captured = capsys.readouterr()
        # No debug output in current implementation
        assert captured.out == ''

def test_flag_all_swaps_debug(float_simple_data, capsys):
    # Patch all the individual flagging functions
    with mock.patch('swap_correction.tracking.flagging.flags.flag_discontinuities', return_value=np.array([1])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_delta_mismatches', return_value=np.array([2])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_sign_reversals', return_value=np.array([3])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_overlaps', return_value=np.array([4])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_overlap_sign_reversals', return_value=np.array([5])), \
         mock.patch('swap_correction.tracking.flagging.flags.flag_overlap_minimum_mismatches', return_value=np.array([6])), \
         mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=(np.array([0]), np.array([1]))):
        result = flagging.flag_all_swaps(float_simple_data, fps=30, debug=True)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

def test_flag_discontinuities_debug(float_simple_data, capsys):
    # Patch metrics.get_delta_between_frames
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.get_delta_between_frames', return_value=np.array([0, 25, 10])):
        flagging.flag_discontinuities(float_simple_data, point='head', fps=30, debug=True)
        captured = capsys.readouterr()
        assert "Found" in captured.out

def test_flag_delta_mismatches_debug(float_simple_data, capsys):
    # Patch get_all_deltas
    with mock.patch('swap_correction.tracking.flagging.flags.get_all_deltas', return_value=(np.array([1, 2]), np.array([1, 2]), np.array([1, 2]), np.array([1, 2]))):
        flagging.flag_delta_mismatches(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert "Found" in captured.out

def test_flag_sign_reversals_debug(float_simple_data, capsys):
    # Patch metrics.get_ht_cross_sign and metrics.get_head_angle
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking.flagging.flags.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        flagging.flag_sign_reversals(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert "Found" in captured.out

def test_flag_overlaps_debug(float_simple_data, capsys):
    # Patch metrics.perfectly_overlapping
    with mock.patch('swap_correction.tracking.flagging.flags.metrics.perfectly_overlapping', return_value=np.array([0, 2])):
        flagging.flag_overlaps(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert "Found" in captured.out

def test_flag_overlap_minimum_mismatches_debug(float_simple_data, capsys):
    # Patch get_overlap_edges, get_all_deltas
    with mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=np.array([[0, 1], [2, 3]])), \
         mock.patch('swap_correction.tracking.flagging.flags.get_all_deltas', return_value=np.array([[1, 2], [1, 2], [3, 4], [5, 6]])):
        flagging.flag_overlap_minimum_mismatches(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert "Found" in captured.out

def test_flag_overlap_sign_reversals_debug(float_simple_data, capsys):
    # Patch get_overlap_edges, metrics.get_ht_cross_sign, metrics.get_head_angle
    with mock.patch('swap_correction.tracking.flagging.flags.get_overlap_edges', return_value=np.array([[0, 1], [1, 2]])), \
         mock.patch('swap_correction.tracking.flagging.flags.metrics.get_ht_cross_sign', return_value=np.array([1, -1, 1])), \
         mock.patch('swap_correction.tracking.flagging.flags.metrics.get_head_angle', return_value=np.array([np.pi, np.pi, np.pi])):
        flagging.flag_overlap_sign_reversals(float_simple_data, debug=True)
        captured = capsys.readouterr()
        assert "Found" in captured.out 
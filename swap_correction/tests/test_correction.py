import numpy as np
import pandas as pd
import pytest
from swap_correction.tracking.correction import tracking_correction, remove_edge_frames, interpolate_gaps, correct_global_swap, correct_tracking_errors, validate_corrected_data, remove_overlaps, correct_swapped_segments, get_swapped_segments, correct_no_flags, correct_swaps, correct_both_flags, correct_empty_df
from swap_correction import metrics

# NOTE: The old SwapCorrection class has been replaced by procedural functions in swap_correction.tracking.correction.
# Update test function calls to use the correct procedural API.

def make_full_df(n=3, nan_idx=1):
    cols = [col for pair in metrics.POSDICT.values() for col in pair]
    # Ensure correct PIVRCOLS names
    cols = [
        'X-Head', 'Y-Head', 'X-Tail', 'Y-Tail',
        'X-Midpoint', 'Y-Midpoint', 'X-Centroid', 'Y-Centroid'
    ]
    data = {col: np.arange(n, dtype=float) for col in cols}
    # Add NaNs for interpolation test
    for col in cols:
        data[col][nan_idx] = np.nan
    return pd.DataFrame(data)

def test_remove_edge_frames():
    df = make_full_df()
    # Set edge values for head and tail
    df.loc[0, ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']] = [0, 0, 0, 0]
    result = remove_edge_frames(df.copy(), resolution='100x100')
    assert result.isnull().any().any()

def test_interpolate_gaps():
    df = make_full_df()
    result = interpolate_gaps(df.copy())
    assert not result.isnull().any().any()

def test_correct_global_swap():
    df = make_full_df()
    # Set up so mean head-tail separation is negative
    df['X-Head'] = 0
    df['X-Tail'] = 1
    df['Y-Head'] = 0
    df['Y-Tail'] = 1
    result = correct_global_swap(df.copy())
    assert isinstance(result, pd.DataFrame)

def test_tracking_correction():
    df = make_full_df()
    result = tracking_correction(df, fps=30, resolution='100x100')
    assert isinstance(result, pd.DataFrame)

def test_correct_tracking_errors():
    df = make_full_df()
    result = correct_tracking_errors(df, fps=30)
    assert isinstance(result, pd.DataFrame)

def test_validate_corrected_data():
    df = make_full_df()
    result = validate_corrected_data(df, fps=30)
    assert isinstance(result, pd.DataFrame)

def test_remove_overlaps():
    df = make_full_df()
    result = remove_overlaps(df)
    assert isinstance(result, pd.DataFrame)

def test_correct_swapped_segments():
    df = make_full_df()
    result = correct_swapped_segments(df, start=0, end=1)
    assert isinstance(result, pd.DataFrame)

def test_get_swapped_segments():
    df = make_full_df()
    segments = get_swapped_segments(df, fps=30)
    assert isinstance(segments, list)

def make_simple_df(n=10):
    data = {
        'X-Head': np.arange(n, dtype=float),
        'Y-Head': np.arange(n, dtype=float),
        'X-Tail': np.arange(n, dtype=float),
        'Y-Tail': np.arange(n, dtype=float),
        'X-Midpoint': np.arange(n, dtype=float),
        'Y-Midpoint': np.arange(n, dtype=float),
        'X-Centroid': np.arange(n, dtype=float),
        'Y-Centroid': np.arange(n, dtype=float),
        'angle': np.zeros(n, dtype=float),
    }
    return pd.DataFrame(data)

def test_correct_no_flags():
    df = make_simple_df()
    out = correct_no_flags(df, {})
    pd.testing.assert_frame_equal(df, out)

def test_correct_swaps():
    df = make_simple_df()
    # Swap rows 2-4
    flags = {'swaps': [(2, 4)]}
    out = correct_swaps(df, flags)
    # Head and tail should be swapped in rows 2-4
    assert np.allclose(out.loc[2:4, 'X-Head'], df.loc[2:4, 'X-Tail'])
    assert np.allclose(out.loc[2:4, 'Y-Head'], df.loc[2:4, 'Y-Tail'])
    assert np.allclose(out.loc[2:4, 'X-Tail'], df.loc[2:4, 'X-Head'])
    assert np.allclose(out.loc[2:4, 'Y-Tail'], df.loc[2:4, 'Y-Head'])
    # Angle should be shifted by pi and wrapped
    expected_angle = (df.loc[2:4, 'angle'] + np.pi) % (2 * np.pi)
    assert np.allclose(out.loc[2:4, 'angle'], expected_angle)

def test_correct_tracking_errors():
    df = make_simple_df()
    # Set a region to a wrong value
    df.loc[5:7, 'X-Head'] = 100
    df.loc[5:7, 'Y-Head'] = 100
    df.loc[5:7, 'angle'] = 2
    # Also ensure 'X-Midpoint' and 'Y-Midpoint' are present and valid
    df.loc[5:7, 'X-Midpoint'] = 50
    df.loc[5:7, 'Y-Midpoint'] = 50
    flags = {'tracking_errors': [(5, 7)]}
    out = correct_tracking_errors(df, fps=30)
    # Should interpolate between 4 and 8
    for col in ['X-Head', 'Y-Head', 'angle', 'X-Midpoint', 'Y-Midpoint']:
        expected = np.interp(np.arange(5, 8), [4, 8], [df.loc[4, col], df.loc[8, col]])
        assert np.allclose(out.loc[5:7, col], expected)

def test_correct_both_flags():
    df = make_simple_df()
    # Add a swap and a tracking error
    flags = {'swaps': [(1, 2)], 'tracking_errors': [(5, 6)]}
    out = correct_both_flags(df, flags)
    # Head and tail swapped in 1-2
    assert np.allclose(out.loc[1:2, 'X-Head'], df.loc[1:2, 'X-Tail'])
    assert np.allclose(out.loc[1:2, 'Y-Head'], df.loc[1:2, 'Y-Tail'])
    assert np.allclose(out.loc[1:2, 'X-Tail'], df.loc[1:2, 'X-Head'])
    assert np.allclose(out.loc[1:2, 'Y-Tail'], df.loc[1:2, 'Y-Head'])
    # tracking error interpolated in 5-6
    for col in ['X-Head', 'Y-Head', 'angle']:
        expected = np.interp(np.arange(5, 7), [4, 7], [df.loc[4, col], df.loc[7, col]])
        assert np.allclose(out.loc[5:6, col], expected)

def test_correct_empty_df():
    df = pd.DataFrame({
        'X-Head': [], 'Y-Head': [], 'X-Tail': [], 'Y-Tail': [],
        'X-Midpoint': [], 'Y-Midpoint': [], 'X-Centroid': [], 'Y-Centroid': [], 'angle': []
    })
    out = correct_empty_df(df, {'swaps': [(0, 0)], 'tracking_errors': [(0, 0)]})
    assert out.empty 
import numpy as np
import pandas as pd
from swap_correction.tracking.correction import correction
from swap_correction import metrics

def make_full_df(n=3, nan_idx=1):
    cols = [col for pair in metrics.POSDICT.values() for col in pair]
    data = {col: np.arange(n, dtype=float) for col in cols}
    # Add NaNs for interpolation test
    for col in cols:
        data[col][nan_idx] = np.nan
    return pd.DataFrame(data)

def test_remove_edge_frames():
    df = make_full_df()
    # Set edge values for head and tail
    df.loc[0, ['xhead', 'yhead', 'xtail', 'ytail']] = [0, 0, 0, 0]
    result = correction.remove_edge_frames(df.copy())
    assert result.isnull().any().any()

def test_interpolate_gaps():
    df = make_full_df()
    result = correction.interpolate_gaps(df.copy())
    assert not result.isnull().any().any()

def test_correct_global_swap():
    df = make_full_df()
    # Set up so mean head-tail separation is negative
    df['xhead'] = 0
    df['xtail'] = 1
    df['yhead'] = 0
    df['ytail'] = 1
    result = correction.correct_global_swap(df.copy())
    assert isinstance(result, pd.DataFrame)

def test_tracking_correction():
    df = make_full_df()
    result = correction.tracking_correction(df, fps=30)
    assert isinstance(result, pd.DataFrame)

def test_correct_tracking_errors():
    df = make_full_df()
    result = correction.correct_tracking_errors(df, fps=30)
    assert isinstance(result, pd.DataFrame)

def test_validate_corrected_data():
    df = make_full_df()
    result = correction.validate_corrected_data(df, fps=30)
    assert isinstance(result, pd.DataFrame)

def test_remove_overlaps():
    df = make_full_df()
    result = correction.remove_overlaps(df)
    assert isinstance(result, pd.DataFrame)

def test_correct_swapped_segments():
    df = make_full_df()
    result = correction.correct_swapped_segments(df, start=0, end=1)
    assert isinstance(result, pd.DataFrame)

def test_get_swapped_segments():
    df = make_full_df()
    segments = correction.get_swapped_segments(df, fps=30)
    assert isinstance(segments, list) 
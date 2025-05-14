import numpy as np
import pandas as pd
from swap_correction.tracking.flagging import flags

def test_get_all_deltas():
    df = pd.DataFrame({'xhead': [1,2], 'yhead': [1,2], 'xtail': [0,1], 'ytail': [0,1]})
    deltas = flags.get_all_deltas(df)
    assert np.allclose(deltas, np.array([[1,1],[1,1]]))

def test_flag_overlaps():
    df = pd.DataFrame({'xhead': [0,1], 'yhead': [0,1], 'xtail': [0,1], 'ytail': [0,1]})
    overlaps = flags.flag_overlaps(df)
    assert isinstance(overlaps, np.ndarray)

def test_get_overlap_edges():
    overlaps = np.array([1,2,3,7,8,10])
    starts, ends = flags.get_overlap_edges(overlaps)
    assert np.all(starts == np.array([1,7,10]))
    assert np.all(ends == np.array([3,8,10]))

def test_flag_all_swaps():
    df = pd.DataFrame({'xhead': [0,1,2], 'yhead': [0,1,2], 'xtail': [0,1,2], 'ytail': [0,1,2]})
    all_flags = flags.flag_all_swaps(df, fps=30)
    assert isinstance(all_flags, np.ndarray)

def test_flag_discontinuities():
    df = pd.DataFrame({'xhead': [0,1,2], 'yhead': [0,1,2], 'xtail': [0,1,2], 'ytail': [0,1,2]})
    discontinuities = flags.flag_discontinuities(df, point='head', fps=30)
    assert isinstance(discontinuities, np.ndarray)

def test_flag_delta_mismatches():
    df = pd.DataFrame({'xhead': [0,1,2], 'yhead': [0,1,2], 'xtail': [0,1,2], 'ytail': [0,1,2]})
    mismatches = flags.flag_delta_mismatches(df)
    assert isinstance(mismatches, np.ndarray)

def test_flag_sign_reversals():
    df = pd.DataFrame({'xhead': [0,1,2], 'yhead': [0,1,2], 'xtail': [0,1,2], 'ytail': [0,1,2]})
    reversals = flags.flag_sign_reversals(df)
    assert isinstance(reversals, np.ndarray)

def test_flag_overlap_sign_reversals():
    df = pd.DataFrame({'xhead': [0,1,2], 'yhead': [0,1,2], 'xtail': [0,1,2], 'ytail': [0,1,2]})
    overlap_reversals = flags.flag_overlap_sign_reversals(df)
    assert isinstance(overlap_reversals, np.ndarray)

def test_get_all_overlap_edges():
    df = pd.DataFrame({'xhead': [0,1,2], 'yhead': [0,1,2], 'xtail': [0,1,2], 'ytail': [0,1,2]})
    starts, ends = flags.get_all_overlap_edges(df)
    assert isinstance(starts, np.ndarray)
    assert isinstance(ends, np.ndarray)

def test_flag_overlap_minimum_mismatches():
    df = pd.DataFrame({'xhead': [0,1,2], 'yhead': [0,1,2], 'xtail': [0,1,2], 'ytail': [0,1,2]})
    mismatches = flags.flag_overlap_minimum_mismatches(df)
    assert isinstance(mismatches, np.ndarray) 
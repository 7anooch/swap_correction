"""Tests for the metrics module."""
import numpy as np
import pandas as pd
import pytest
from swap_correction import metrics


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    data = {
        'xhead': [0, 1, 2, 3, 4],
        'yhead': [0, 1, 2, 3, 4],
        'xtail': [0, 0, 0, 0, 0],
        'ytail': [0, 0, 0, 0, 0],
        'xmid': [0, 0.5, 1, 1.5, 2],
        'ymid': [0, 0.5, 1, 1.5, 2],
        'xctr': [0, 0.5, 1, 1.5, 2],
        'yctr': [0, 0.5, 1, 1.5, 2]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_segments():
    """Create sample segments for testing."""
    return np.array([[0, 2], [2, 4]])


def test_vectors_from_key(sample_dataframe):
    """Test extraction of position vectors from dataframe."""
    # Test with transpose
    vec = metrics.vectors_from_key(sample_dataframe, 'head')
    assert vec.shape == (2, 5)
    np.testing.assert_array_equal(vec[0], sample_dataframe['xhead'])
    np.testing.assert_array_equal(vec[1], sample_dataframe['yhead'])
    
    # Test without transpose
    vec = metrics.vectors_from_key(sample_dataframe, 'head', transpose=False)
    assert vec.shape == (5, 2)
    np.testing.assert_array_equal(vec[:, 0], sample_dataframe['xhead'])
    np.testing.assert_array_equal(vec[:, 1], sample_dataframe['yhead'])


def test_get_delta_in_frame(sample_dataframe):
    """Test calculation of distance between points in a frame."""
    dist = metrics.get_delta_in_frame(sample_dataframe, 'head', 'tail')
    expected = np.array([0, np.sqrt(2), 2*np.sqrt(2), 3*np.sqrt(2), 4*np.sqrt(2)])
    np.testing.assert_array_almost_equal(dist, expected)


def test_get_orientation(sample_dataframe):
    """Test calculation of body orientation."""
    ba = metrics.get_orientation(sample_dataframe)
    expected = np.array([np.pi/2, np.pi/4, np.pi/4, np.pi/4, np.pi/4])
    np.testing.assert_array_almost_equal(ba, expected)


def test_get_custom_orientation(sample_dataframe):
    """Test calculation of custom orientation."""
    ba = metrics.get_custom_orientation(sample_dataframe, 'tail', 'head')
    expected = np.array([np.pi/2, np.pi/4, np.pi/4, np.pi/4, np.pi/4])
    np.testing.assert_array_almost_equal(ba, expected)


def test_get_bearing(sample_dataframe):
    """Test calculation of bearing angle."""
    bearing = metrics.get_bearing(sample_dataframe, source=[0, 0])
    assert bearing.shape == (5,)
    assert not np.any(np.isnan(bearing))


def test_get_ht_cross_sign(sample_dataframe):
    """Test calculation of head-tail cross product sign."""
    signs = metrics.get_ht_cross_sign(sample_dataframe)
    assert signs.shape == (5,)
    assert not np.any(np.isnan(signs))


def test_get_vectors_between(sample_dataframe):
    """Test calculation of vectors between points."""
    vecs = metrics.get_vectors_between(sample_dataframe, 'tail', 'head')
    assert vecs.shape == (5, 2)
    np.testing.assert_array_equal(vecs[:, 0], sample_dataframe['xhead'] - sample_dataframe['xtail'])
    np.testing.assert_array_equal(vecs[:, 1], sample_dataframe['yhead'] - sample_dataframe['ytail'])


def test_get_motion_vector(sample_dataframe):
    """Test calculation of motion vector."""
    vecs = metrics.get_motion_vector(sample_dataframe)
    assert vecs.shape == (5, 2)
    assert not np.any(np.isnan(vecs))


def test_get_segment_distance(sample_dataframe, sample_segments):
    """Test calculation of segment distances."""
    dist = metrics.get_segment_distance(sample_dataframe, sample_segments, key='head')
    assert dist.shape == (2,)
    assert not np.any(np.isnan(dist))


def test_get_delta_between_frames(sample_dataframe):
    """Test calculation of deltas between frames."""
    delta = metrics.get_delta_between_frames(sample_dataframe, 'head', fps=30)
    assert delta.shape == (4,)  # One less than number of frames
    assert not np.any(np.isnan(delta))


def test_get_cross_segment_deltas(sample_dataframe, sample_segments):
    """Test calculation of cross-segment deltas."""
    deltas = metrics.get_cross_segment_deltas(sample_dataframe, sample_segments, 'head', 'tail')
    assert deltas.shape == (2,)
    assert not np.any(np.isnan(deltas))


def test_get_df_bounds(sample_dataframe):
    """Test calculation of dataframe bounds."""
    xlim = metrics.get_df_bounds([sample_dataframe], ['xhead', 'xtail'])
    ylim = metrics.get_df_bounds([sample_dataframe], ['yhead', 'ytail'])
    
    # The function adds a 5% buffer to the bounds
    expected_xmin = -0.2  # 0 - (4-0)*0.05
    expected_xmax = 4.2   # 4 + (4-0)*0.05
    expected_ymin = -0.2  # 0 - (4-0)*0.05
    expected_ymax = 4.2   # 4 + (4-0)*0.05
    
    np.testing.assert_almost_equal(xlim[0], expected_xmin, decimal=2)
    np.testing.assert_almost_equal(xlim[1], expected_xmax, decimal=2)
    np.testing.assert_almost_equal(ylim[0], expected_ymin, decimal=2)
    np.testing.assert_almost_equal(ylim[1], expected_ymax, decimal=2) 
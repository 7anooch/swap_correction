import os
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
import matplotlib.pyplot as plt
from unittest.mock import patch

import swap_correction.performance_analysis as pa

def test_load_data():
    raw, gt = pa.load_data()
    assert isinstance(raw, pd.DataFrame)
    assert isinstance(gt, pd.DataFrame)
    assert not raw.empty
    assert not gt.empty
    for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail', 'X-Midpoint', 'Y-Midpoint', 'X-Centroid', 'Y-Centroid', 'angle']:
        assert col in raw.columns
        assert col in gt.columns

def test_run_pipeline():
    raw, _ = pa.load_data()
    # Ensure columns are correct for the pipeline
    for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail', 'X-Midpoint', 'Y-Midpoint', 'X-Centroid', 'Y-Centroid', 'angle']:
        if col not in raw.columns:
            raw[col] = 0.0
    filtered, flags = pa.run_pipeline(raw)
    assert isinstance(filtered, pd.DataFrame)
    assert isinstance(flags, dict)
    assert not filtered.empty
    for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail', 'X-Midpoint', 'Y-Midpoint', 'X-Centroid', 'Y-Centroid', 'angle']:
        assert col in filtered.columns

def test_compute_metrics():
    raw, gt = pa.load_data()
    # Ensure all required columns are present with PIVRCOLS names
    required_cols = ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail', 
                    'X-Midpoint', 'Y-Midpoint', 'X-Centroid', 'Y-Centroid', 'angle']
    for col in required_cols:
        if col not in raw.columns:
            raw[col] = 0.0
        if col not in gt.columns:
            gt[col] = 0.0
    filtered, _ = pa.run_pipeline(raw)
    raw_err, filtered_err = pa.compute_metrics(raw, filtered, gt)
    assert isinstance(raw_err, dict)
    assert isinstance(filtered_err, dict)
    for key in ['position_error', 'angle_error', 'swap_errors']:
        assert key in raw_err
        assert key in filtered_err

def test_plot_trajectories(tmp_path):
    raw, gt = pa.load_data()
    for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail', 'X-Midpoint', 'Y-Midpoint', 'X-Centroid', 'Y-Centroid', 'angle']:
        if col not in raw.columns:
            raw[col] = 0.0
        if col not in gt.columns:
            gt[col] = 0.0
    filtered, _ = pa.run_pipeline(raw)
    traj_path = tmp_path / 'traj.png'
    with patch('matplotlib.pyplot.show'):
        pa.plot_trajectories(raw, filtered, gt, save_path=str(traj_path))
    assert traj_path.exists()

def test_plot_flagged_frames_time_series_and_on_trajectory(tmp_path):
    # Use a small synthetic DataFrame and flags
    n = 20
    df = pd.DataFrame({'X-Tail': np.arange(n), 'Y-Tail': np.arange(n)})
    flags = [(3, 5), (10, 12)]
    # Time series
    ts_path = tmp_path / 'flagged_timeseries.png'
    with patch('matplotlib.pyplot.show'):
        pa.plot_flagged_frames_time_series(flags, n, 'testflag', save_path=str(ts_path))
    assert ts_path.exists()
    # Trajectory
    traj_path = tmp_path / 'flagged_traj.png'
    with patch('matplotlib.pyplot.show'):
        pa.plot_flagged_on_trajectory(df, flags, 'testflag', save_path=str(traj_path))
    assert traj_path.exists()

def test_compute_velocity_ratio():
    n = 10
    df = pd.DataFrame({
        'X-Head': np.arange(n),
        'Y-Head': np.arange(n),
        'X-Tail': np.arange(n),
        'Y-Tail': np.arange(n)
    })
    ratio = pa.compute_velocity_ratio(df)
    assert isinstance(ratio, np.ndarray)
    assert ratio.shape[0] == n - 1

# Remove test_plot_trajectories_and_error_time_series since plot_error_time_series no longer exists. 
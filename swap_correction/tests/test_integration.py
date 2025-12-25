"""Integration tests for the swap correction pipeline using real data sets."""
import os
import numpy as np
import pandas as pd
import pytest
from swap_correction.metrics import Metrics
from swap_correction.tracking.flags import flag_all_swaps
from swap_correction.tracking.correction import tracking_correction
from swap_correction.tracking.filters import filter_data

# Robustly resolve the project root and test data directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'swap_correction', 'tests', 'test_data')
SAMPLE_EXPERIMENT = "2024.11.13_00-48-15_Sussex_e2hex"
# Extract just the timestamp prefix for filenames
SAMPLE_PREFIX = SAMPLE_EXPERIMENT.split('_Sussex')[0] if '_Sussex' in SAMPLE_EXPERIMENT else SAMPLE_EXPERIMENT.split('_')[0]

# Helper to add 'x', 'y', and 'angle' columns to a DataFrame
def add_xy_angle_columns(df):
    df = df.copy()
    # Add required columns if they don't exist
    required_cols = ['xhead', 'yhead', 'xtail', 'ytail', 'xmid', 'ymid', 'xctr', 'yctr']
    for col in required_cols:
        if col not in df.columns:
            if 'X-Centroid' in df.columns:
                df[col] = df['X-Centroid']
            else:
                df[col] = 0.0
    
    # Add angle column
    if 'angle' not in df.columns:
        dx = df['xmid'] - df['xtail']
        dy = df['ymid'] - df['ytail']
        df['angle'] = np.arctan2(dy, dx)
    
    return df

@pytest.fixture
def sample_data_paths():
    """Get paths to sample data files."""
    exp_dir = os.path.join(TEST_DATA_DIR, SAMPLE_EXPERIMENT)
    return {
        'raw_data': os.path.join(exp_dir, f"{SAMPLE_PREFIX}_data.csv"),
        'ground_truth': os.path.join(exp_dir, f"{SAMPLE_PREFIX}_data_level2.csv"),
        'settings': os.path.join(exp_dir, "experiment_settings.json")
    }

@pytest.fixture
def raw_data(sample_data_paths):
    """Load raw tracking data and add x, y, angle columns."""
    df = pd.read_csv(sample_data_paths['raw_data'])
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return add_xy_angle_columns(df)

@pytest.fixture
def ground_truth_data(sample_data_paths):
    """Load ground truth data and add x, y, angle columns."""
    df = pd.read_csv(sample_data_paths['ground_truth'])
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return add_xy_angle_columns(df)

@pytest.fixture
def experiment_settings(sample_data_paths):
    """Load experiment settings."""
    import json
    with open(sample_data_paths['settings'], 'r') as f:
        return json.load(f)

def test_data_loading(sample_data_paths, raw_data, ground_truth_data, experiment_settings):
    """Test that all required data files can be loaded."""
    print("RAW DATA PATH:", sample_data_paths['raw_data'])
    print("GROUND TRUTH PATH:", sample_data_paths['ground_truth'])
    print("SETTINGS PATH:", sample_data_paths['settings'])
    # Check that files exist
    assert os.path.exists(sample_data_paths['raw_data']), f"Raw data file not found: {sample_data_paths['raw_data']}"
    assert os.path.exists(sample_data_paths['ground_truth']), f"Ground truth file not found: {sample_data_paths['ground_truth']}"
    assert os.path.exists(sample_data_paths['settings']), f"Settings file not found: {sample_data_paths['settings']}"
    # Check that data is loaded correctly
    assert not raw_data.empty, "Raw data is empty"
    assert not ground_truth_data.empty, "Ground truth data is empty"
    assert isinstance(experiment_settings, dict), "Settings should be a dictionary"

def test_pipeline_initialization():
    """Test that all pipeline components can be initialized."""
    metrics = Metrics()
    assert metrics is not None

def test_data_integrity(raw_data, ground_truth_data):
    """Test that the pipeline maintains data integrity."""
    fps = 30
    corrected_data = tracking_correction(raw_data.copy(), fps=fps, resolution='100x100')
    filtered_data = filter_data(corrected_data)

    # Debug prints
    print("raw_data shape:", raw_data.shape)
    print("filtered_data shape:", filtered_data.shape)
    print("ground_truth_data shape:", ground_truth_data.shape)

    # Check required columns
    required_cols = ['xhead', 'yhead', 'xtail', 'ytail', 'xmid', 'ymid', 'xctr', 'yctr', 'angle']
    for col in required_cols:
        assert col in filtered_data.columns, f"Required column {col} missing"

    # Allow NaNs only where they were present in the raw data
    for col in required_cols:
        raw_nans = raw_data[col].isna() if col in raw_data.columns else pd.Series([False]*len(filtered_data))
        filtered_nans = filtered_data[col].isna()
        # All NaNs in filtered_data must be present in raw_data
        assert np.all(~filtered_nans | raw_nans), f"Unexpected NaNs in {col}"

    # Check data types
    for col in required_cols:
        assert filtered_data[col].dtype in [np.float64, np.float32], f"{col} should be float"

    # Check shape matches ground truth
    assert filtered_data.shape == ground_truth_data.shape, "Output shape should match ground truth"

def test_pipeline_flags_or_correction_on_real_data(raw_data):
    """Test that the flagging/correction pipeline actually flags or corrects something on real data."""
    from swap_correction import utils
    fps = 30
    swap_frames = flag_all_swaps(raw_data, fps)
    swap_segments = utils.get_consecutive_ranges(swap_frames)
    corrected = tracking_correction(raw_data.copy(), fps=fps, resolution='100x100')
    filtered = filter_data(corrected)
    # Print flags for debugging
    print('Swap segments:', swap_segments)
    # Check if any flags are non-empty
    any_flagged = len(swap_segments) > 0
    # Check if output is different from input
    data_changed = not raw_data.equals(filtered)
    print('Data changed:', data_changed)
    if not any_flagged and not data_changed:
        # Print a summary of differences if any
        diffs = (raw_data != filtered).sum().sum()
        print('Number of differing elements:', diffs)
    assert any_flagged or data_changed, "Pipeline did not flag or correct anything on real data." 
"""
Unit tests for tracking correction module.
"""

import numpy as np
import pandas as pd
from swap_corrector import tracking_correction
from swap_corrector.tracking_correction import FlagDetector

def test_remove_edge_frames(sample_data):
    """Test removal of edge frames with all zero positions."""
    data, _ = sample_data
    
    # Create edge frames with zeros
    edge_data = data.copy()
    edge_data.iloc[0] = 0  # First frame
    edge_data.iloc[-1] = 0  # Last frame
    
    # Apply edge frame removal
    corrected = tracking_correction.remove_edge_frames(edge_data)
    
    # Check that edge frames are set to NaN
    assert corrected.iloc[0].isna().all()
    assert corrected.iloc[-1].isna().all()
    # Check that other frames are unchanged
    assert not corrected.iloc[1:-1].isna().all().all()

def test_correct_tracking_errors(sample_data_with_swaps):
    """Test correction of tracking errors."""
    data, fps = sample_data_with_swaps
    
    # Apply tracking correction
    corrected = tracking_correction.correct_tracking_errors(data, fps)
    
    # Check that corrected data has the expected columns
    expected_cols = list(data.columns) + ['X-Midpoint', 'Y-Midpoint']
    assert all(col in corrected.columns for col in expected_cols)
    
    # Check that corrected data has the same number of rows
    assert len(corrected) == len(data)

def test_validate_corrected_data(sample_data_with_swaps):
    """Test validation of corrected data."""
    data, fps = sample_data_with_swaps
    
    # Apply validation
    validated = tracking_correction.validate_corrected_data(data, fps)
    
    # Check that validated data has the same shape
    assert validated.shape == data.shape
    
    # Check that validation flags are reasonable
    flags = tracking_correction.FlagDetector().detect_all_flags(validated, fps)
    assert isinstance(flags, dict)
    assert all(isinstance(v, np.ndarray) for v in flags.values())

def test_flag_detector(sample_data_with_swaps):
    """Test flag detection."""
    data, fps = sample_data_with_swaps
    
    # Create detector and detect flags
    detector = tracking_correction.FlagDetector()
    flags = detector.detect_all_flags(data, fps)
    
    # Check that flags have the expected structure
    assert isinstance(flags, dict)
    assert all(isinstance(v, np.ndarray) for v in flags.values())
    assert all(len(v) == len(data) for v in flags.values())

def test_interpolate_overlaps(sample_data_with_swaps):
    """Test interpolation of overlaps."""
    data, fps = sample_data_with_swaps
    
    # Apply interpolation
    interpolated = tracking_correction.interpolate_gaps(data)
    
    # Check that interpolated data has the expected columns
    expected_cols = list(data.columns) + ['X-Midpoint', 'Y-Midpoint']
    assert all(col in interpolated.columns for col in expected_cols)
    
    # Check that interpolated data has the same number of rows
    assert len(interpolated) == len(data)
    
    # Check that NaN values were interpolated
    assert not interpolated.isna().any().any()

def test_tracking_correction_pipeline(sample_data_with_swaps):
    """Test the complete tracking correction pipeline."""
    data, fps = sample_data_with_swaps
    
    # Run pipeline
    processed = tracking_correction.tracking_correction(
        data,
        fps=fps,
        swapCorrection=True,
        removeErrors=True,
        interp=True,
        validate=True,
        filterData=True,
        debug=True
    )
    
    # Check that processed data has the expected columns
    expected_cols = list(data.columns) + ['X-Midpoint', 'Y-Midpoint']
    assert all(col in processed.columns for col in expected_cols)
    
    # Check that processed data has the same number of rows
    assert len(processed) == len(data)
    
    # Check that processed data is not identical to input
    assert not processed.equals(data)
    
    # Check that no NaN values remain
    assert not processed.isna().any().any() 
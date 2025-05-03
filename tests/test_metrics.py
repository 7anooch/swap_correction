"""
Unit tests for metrics module.
"""

import numpy as np
import pandas as pd
from swap_corrector import metrics

def test_calculate_velocity(sample_data):
    """Test velocity calculation."""
    data, fps = sample_data
    
    # Calculate velocity
    velocity = metrics.calculate_velocity(data, fps)
    
    # Check that velocity has the expected shape
    assert velocity.shape[0] == data.shape[0]  # Same number of rows
    assert velocity.shape[1] == len(metrics.POSDICT) * 2  # Two components (x, y) per point
    
    # Check that velocity values are reasonable
    assert not velocity.isna().any().any()
    assert not velocity.isin([np.inf, -np.inf]).any().any()

def test_calculate_acceleration(sample_data):
    """Test acceleration calculation."""
    data, fps = sample_data
    
    # Calculate acceleration
    acceleration = metrics.calculate_acceleration(data, fps)
    
    # Check that acceleration has the expected shape
    assert acceleration.shape[0] == data.shape[0]  # Same number of rows
    assert acceleration.shape[1] == len(metrics.POSDICT) * 2  # Two components (x, y) per point
    
    # Check that acceleration values are reasonable
    assert not acceleration.isna().any().any()
    assert not acceleration.isin([np.inf, -np.inf]).any().any()

def test_calculate_curvature(sample_data):
    """Test curvature calculation."""
    data, fps = sample_data
    
    # Calculate curvature
    curvature = metrics.calculate_curvature(data, fps)
    
    # Check that curvature has the expected shape
    assert curvature.shape[0] == len(data)
    
    # Check that curvature values are reasonable
    assert not np.isnan(curvature).any()
    assert not np.isinf(curvature).any()

def test_calculate_angular_velocity(sample_data):
    """Test angular velocity calculation."""
    data, fps = sample_data
    
    # Calculate angular velocity
    angular_velocity = metrics.calculate_angular_velocity(data, fps)
    
    # Check that angular velocity has the expected shape
    assert angular_velocity.shape[0] == len(data)
    
    # Check that angular velocity values are reasonable
    assert not np.isnan(angular_velocity).any()
    assert not np.isinf(angular_velocity).any()

def test_calculate_metrics_with_nan(sample_data):
    """Test metric calculations with NaN values."""
    data, fps = sample_data
    
    # Introduce NaN values
    data_with_nan = data.copy()
    data_with_nan.iloc[0] = pd.NA
    
    # Test that metric calculations handle NaN values gracefully
    velocity = metrics.calculate_velocity(data_with_nan, fps)
    acceleration = metrics.calculate_acceleration(data_with_nan, fps)
    curvature = metrics.calculate_curvature(data_with_nan, fps)
    angular_velocity = metrics.calculate_angular_velocity(data_with_nan, fps)
    
    # Check that NaN values are propagated correctly
    assert velocity.iloc[0].isna().all()
    assert acceleration.iloc[0].isna().all()
    assert curvature[0] == 0  # Curvature is set to 0 for invalid points

def test_compare_metrics(sample_data, sample_data_with_swaps):
    """Test comparison of metrics between raw and processed data."""
    raw_data, fps = sample_data
    processed_data, _ = sample_data_with_swaps
    
    # Calculate metrics for both datasets
    raw_metrics = metrics.calculate_all_metrics(raw_data, fps)
    processed_metrics = metrics.calculate_all_metrics(processed_data, fps)
    
    # Compare metrics
    comparison = metrics.compare_metrics(raw_metrics, processed_metrics)
    
    # Check that comparison results are reasonable
    assert isinstance(comparison, dict)
    for key, value in comparison.items():
        assert isinstance(value, (float, pd.Series))
        if isinstance(value, float):
            assert -1 <= value <= 1  # Correlation coefficient
        else:
            assert all(-1 <= v <= 1 for v in value)  # Series of correlation coefficients 
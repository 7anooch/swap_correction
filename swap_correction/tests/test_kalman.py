"""Tests for the Kalman filter module."""
import numpy as np
import pytest
from swap_correction.kalman_filter import KalmanFilter


@pytest.fixture
def sample_trajectory():
    """Generate a sample trajectory for testing."""
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    y = np.cos(t)
    return np.column_stack((x, y))


@pytest.fixture
def kalman_filter():
    """Create a Kalman filter instance for testing."""
    return KalmanFilter(dt=0.1, ndim=2, derivatives=2)


def test_kalman_initialization():
    """Test Kalman filter initialization."""
    kf = KalmanFilter(dt=0.1, ndim=2, derivatives=2)
    assert kf.dt == 0.1
    assert kf.dim == 2
    assert kf.npts == 6  # 2 dimensions * (1 + 2 derivatives)


def test_kalman_predict(kalman_filter):
    """Test prediction step of Kalman filter."""
    # Set initial state
    kalman_filter.x = np.array([1.0, 2.0, 0.1, 0.2, 0.01, 0.02])
    
    # Perform prediction
    predicted = kalman_filter.predict()
    
    # Check output shape
    assert predicted.shape == (2,)
    assert not np.any(np.isnan(predicted))


def test_kalman_update(kalman_filter):
    """Test update step of Kalman filter."""
    # Set initial state
    kalman_filter.x = np.array([1.0, 2.0, 0.1, 0.2, 0.01, 0.02])
    
    # Perform prediction
    kalman_filter.predict()
    
    # Perform update with measurement
    measurement = np.array([1.1, 2.1])
    updated = kalman_filter.update(measurement)
    
    # Check output shape and values
    assert updated.shape == (2,)
    assert not np.any(np.isnan(updated))


def test_kalman_filter_full(sample_trajectory, kalman_filter):
    """Test full filtering process."""
    # Add some noise to the trajectory
    noisy_trajectory = sample_trajectory + np.random.normal(0, 0.1, sample_trajectory.shape)
    
    # Apply filter
    filtered_trajectory = kalman_filter.filter(noisy_trajectory)
    
    # Check output
    assert filtered_trajectory.shape == noisy_trajectory.shape
    assert not np.any(np.isnan(filtered_trajectory))
    
    # Check that filtered trajectory is smoother than noisy trajectory
    noisy_diff = np.diff(noisy_trajectory, axis=0)
    filtered_diff = np.diff(filtered_trajectory, axis=0)
    assert np.mean(np.abs(filtered_diff)) < np.mean(np.abs(noisy_diff)) 
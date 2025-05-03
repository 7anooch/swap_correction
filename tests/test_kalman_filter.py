"""
Tests for the kalman_filter module.
"""

import numpy as np
import pytest
from swap_corrector.kalman_filter import KalmanFilter

@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    # Create a simple 2D trajectory (straight line)
    t = np.linspace(0, 10, 100)
    x = t
    y = t
    return np.column_stack((x, y))

@pytest.fixture
def noisy_trajectory(sample_trajectory):
    """Create a noisy version of the sample trajectory."""
    noise = np.random.normal(0, 0.1, sample_trajectory.shape)
    return sample_trajectory + noise

def test_kalman_filter_initialization():
    """Test KalmanFilter initialization."""
    kf = KalmanFilter(dt=0.1, ndim=2, derivatives=2)
    
    assert kf.dt == 0.1
    assert kf.dim == 2
    assert kf.npts == 6  # 2 dimensions * (1 + 2 derivatives)
    assert kf.x.shape == (6,)
    assert kf.F.shape == (6, 6)
    assert kf.H.shape == (2, 6)
    assert kf.P.shape == (6, 6)
    assert kf.Q.shape == (6, 6)
    assert kf.R.shape == (2, 2)

def test_kalman_filter_predict():
    """Test KalmanFilter prediction step."""
    kf = KalmanFilter(dt=0.1, ndim=2, derivatives=2)
    
    # Set initial state
    kf.x = np.array([1.0, 1.0, 0.1, 0.1, 0.01, 0.01])
    
    # Perform prediction
    predicted = kf.predict()
    
    assert isinstance(predicted, np.ndarray)
    assert predicted.shape == (2,)
    assert not np.any(np.isnan(predicted))

def test_kalman_filter_update():
    """Test KalmanFilter update step."""
    kf = KalmanFilter(dt=0.1, ndim=2, derivatives=2)
    
    # Set initial state
    kf.x = np.array([1.0, 1.0, 0.1, 0.1, 0.01, 0.01])
    
    # Perform prediction
    kf.predict()
    
    # Perform update with measurement
    measurement = np.array([1.1, 1.1])
    updated = kf.update(measurement)
    
    assert isinstance(updated, np.ndarray)
    assert updated.shape == (2,)
    assert not np.any(np.isnan(updated))

def test_kalman_filter_filter(sample_trajectory, noisy_trajectory):
    """Test KalmanFilter on a complete trajectory."""
    kf = KalmanFilter(dt=0.1, ndim=2, derivatives=2)
    
    # Filter the noisy trajectory
    filtered = kf.filter(noisy_trajectory)
    
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == noisy_trajectory.shape
    assert not np.any(np.isnan(filtered))
    
    # Check that filtered trajectory is smoother than noisy trajectory
    noisy_diff = np.diff(noisy_trajectory, axis=0)
    filtered_diff = np.diff(filtered, axis=0)
    assert np.mean(np.abs(filtered_diff)) < np.mean(np.abs(noisy_diff))

def test_kalman_filter_with_nan():
    """Test KalmanFilter with NaN values in the trajectory."""
    kf = KalmanFilter(dt=0.1, ndim=2, derivatives=2)
    
    # Create trajectory with NaN values
    trajectory = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [np.nan, np.nan],
        [4.0, 4.0],
        [5.0, 5.0]
    ])
    
    filtered = kf.filter(trajectory)
    
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == trajectory.shape
    assert not np.any(np.isnan(filtered))

def test_kalman_filter_initial_estimate():
    """Test KalmanFilter with custom initial estimate."""
    kf = KalmanFilter(dt=0.1, ndim=2, derivatives=2)
    
    # Create sample trajectory
    trajectory = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])
    
    # Set custom initial estimate
    xo = [0.5, 0.5]
    
    filtered = kf.filter(trajectory, xo=xo)
    
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == trajectory.shape
    assert not np.any(np.isnan(filtered))
    # Check that first filtered point is close to initial estimate
    assert np.allclose(filtered[0], xo, atol=0.5)  # Use a larger tolerance 
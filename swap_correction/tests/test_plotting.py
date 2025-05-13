"""Tests for the plotting module."""
import os
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from swap_correction import plotting


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


def test_plot_trajectory(sample_dataframe):
    """Test trajectory plotting."""
    fig, ax = plt.subplots()
    fps = 30
    plotting.plot_trajectory(ax, sample_dataframe, fps)
    
    # Check that the plot was created
    assert len(ax.lines) > 0
    plt.close(fig)


def test_histogram():
    """Test histogram plotting."""
    fig, ax = plt.subplots()
    data = np.random.normal(0, 1, 1000)
    plotting.histogram(ax, data, (-3, 3), 50, True)
    
    # Check that the histogram was created
    assert len(ax.patches) > 0
    plt.close(fig)


def test_save_figure(tmp_path):
    """Test figure saving functionality."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    # Test saving
    output_path = str(tmp_path)
    plotting.save_figure(fig, "test_plot", output_path, show=False)
    
    # Check that file was created
    assert os.path.exists(os.path.join(output_path, "test_plot.png"))
    plt.close(fig)


def test_gmm():
    """Test GMM plotting."""
    fig, ax = plt.subplots()
    data = np.concatenate([
        np.random.normal(-2, 1, 500),
        np.random.normal(2, 1, 500)
    ])
    plotting.gmm(ax, data, 2, 50, (-5, 5))
    
    # Check that the plot was created
    assert len(ax.lines) > 0
    assert len(ax.patches) > 0
    plt.close(fig)


def test_get_fft():
    """Test FFT calculation."""
    # Create a signal with known frequency components
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    freqs, amp = plotting.get_fft(signal, fps=1000)
    
    # Check output shapes
    assert len(freqs) == len(amp)
    assert len(freqs) == 500  # Half of input length
    
    # Check that we can detect the main frequency components
    peak_freqs = freqs[np.argsort(amp)[-2:]]
    assert np.any(np.abs(peak_freqs - 10) < 1)  # 10 Hz component
    assert np.any(np.abs(peak_freqs - 20) < 1)  # 20 Hz component 
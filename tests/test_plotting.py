"""
Unit tests for plotting module.
"""

import os
import pytest
import pandas as pd
from swap_corrector import plotting
import matplotlib.pyplot as plt
from swap_corrector import tracking_correction

def test_generate_trajectory_plot(sample_data, temp_dir):
    """Test generation of trajectory plots."""
    data, fps = sample_data
    
    # Generate plot
    fig = plotting.generate_trajectory_plot(data, fps)
    
    # Check that figure was created
    assert fig is not None
    
    # Save plot
    output_path = os.path.join(temp_dir, "trajectory_plot.png")
    fig.savefig(output_path)
    
    # Check that file was created
    assert os.path.exists(output_path)

def test_generate_distribution_plot(sample_data, temp_dir):
    """Test generation of distribution plots."""
    data, fps = sample_data
    
    # Generate plot
    fig = plotting.generate_distribution_plot(data, fps)
    
    # Check that figure was created
    assert fig is not None
    
    # Save plot
    output_path = os.path.join(temp_dir, "distribution_plot.png")
    fig.savefig(output_path)
    
    # Check that file was created
    assert os.path.exists(output_path)

def test_generate_flag_plot(sample_data_with_swaps):
    """Test generation of flag plots."""
    data, _ = sample_data_with_swaps
    
    # Create flag detector and detect flags
    detector = tracking_correction.FlagDetector()
    flags = detector.detect_all_flags(data, fps=30)
    
    # Generate plot
    fig = plotting.generate_flag_plot(data, flags)
    
    # Check that figure was created
    assert fig is not None
    assert isinstance(fig, plt.Figure)

def test_generate_comparison_plot(sample_data, sample_data_with_swaps):
    """Test generation of comparison plots."""
    raw_data, fps = sample_data
    processed_data, _ = sample_data_with_swaps
    
    # Generate plot
    fig = plotting.generate_comparison_plot(raw_data, processed_data, fps)
    
    # Check that figure was created
    assert fig is not None
    assert isinstance(fig, plt.Figure)

def test_plotting_functions_with_nan(sample_data):
    """Test plotting functions with NaN values."""
    data, fps = sample_data
    
    # Introduce NaN values
    data_with_nan = data.copy()
    data_with_nan.iloc[0] = pd.NA
    
    # Test that plotting functions handle NaN values gracefully
    fig1 = plotting.generate_trajectory_plot(data_with_nan, fps)
    fig2 = plotting.generate_distribution_plot(data_with_nan, fps)
    
    assert fig1 is not None
    assert fig2 is not None 
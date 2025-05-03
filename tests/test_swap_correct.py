"""
Unit tests for the main swap correction script.
"""

import os
import pytest
import pandas as pd
from swap_corrector import tracking_correction
from swap_corrector.config import SwapCorrectionConfig

def test_compare_filtered_trajectories(sample_data, temp_dir):
    """Test comparison of filtered trajectories."""
    data, fps = sample_data
    
    # Create temporary files
    raw_path = os.path.join(temp_dir, "raw_data.csv")
    processed_path = os.path.join(temp_dir, "raw_data_processed.csv")
    data.to_csv(raw_path)
    data.to_csv(processed_path)  # Use same data for testing
    
    # Compare trajectories
    tracking_correction.compare_filtered_trajectories(temp_dir, temp_dir, "raw_data.csv")
    
    # Check that output file was created
    output_file = os.path.join(temp_dir, "raw_data_trajectory_comparison.png")
    assert os.path.exists(output_file)

def test_compare_filtered_distributions(sample_data, temp_dir):
    """Test comparison of filtered distributions."""
    data, fps = sample_data
    
    # Create temporary files
    raw_path = os.path.join(temp_dir, "raw_data.csv")
    processed_path = os.path.join(temp_dir, "raw_data_processed.csv")
    data.to_csv(raw_path)
    data.to_csv(processed_path)  # Use same data for testing
    
    # Compare distributions
    tracking_correction.compare_filtered_distributions(temp_dir, temp_dir, "raw_data.csv")
    
    # Check that output files were created
    assert os.path.exists(os.path.join(temp_dir, "raw_data_raw_distribution.png"))
    assert os.path.exists(os.path.join(temp_dir, "raw_data_processed_distribution.png"))

def test_examine_flags(sample_data_with_swaps, temp_dir):
    """Test examination of flags."""
    data, fps = sample_data_with_swaps
    
    # Create temporary file
    data_path = os.path.join(temp_dir, "data_with_swaps.csv")
    data.to_csv(data_path)
    
    # Examine flags
    tracking_correction.examine_flags(temp_dir, temp_dir, "data_with_swaps.csv")
    
    # Check that output file was created
    output_file = os.path.join(temp_dir, "data_with_swaps_flags.png")
    assert os.path.exists(output_file)

def test_main_pipeline(sample_data, temp_dir):
    """Test the complete main pipeline."""
    data, fps = sample_data
    
    # Create temporary file
    data_path = os.path.join(temp_dir, "test_data.csv")
    data.to_csv(data_path)
    
    # Create test configuration
    cfg = SwapCorrectionConfig(
        debug=True,
        diagnostic_plots=True,
        show_plots=False,
        log_level="DEBUG"
    )
    
    # Run main pipeline
    tracking_correction.main(
        source_dir=temp_dir,
        output_dir=temp_dir,
        config=cfg
    )
    
    # Check that output files were created
    expected_files = [
        "test_data_processed.csv",
        "test_data_trajectory_comparison.png",
        "test_data_raw_distribution.png",
        "test_data_processed_distribution.png",
        "test_data_flags.png"
    ]
    
    for file in expected_files:
        assert os.path.exists(os.path.join(temp_dir, file))

def test_pipeline_with_invalid_data(temp_dir):
    """Test pipeline with invalid data."""
    # Create empty DataFrame
    data = pd.DataFrame()
    
    # Create temporary file
    data_path = os.path.join(temp_dir, "invalid_data.csv")
    data.to_csv(data_path)
    
    # Create test configuration
    cfg = SwapCorrectionConfig(
        debug=True,
        diagnostic_plots=False,
        show_plots=False,
        log_level="DEBUG"
    )
    
    # Run main pipeline and check for appropriate error handling
    with pytest.raises(Exception):
        tracking_correction.main(
            source_dir=temp_dir,
            output_dir=temp_dir,
            config=cfg
        ) 
"""
Tracking correction module for fixing head-tail swaps and other tracking errors.
"""

import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from . import utils
from .metrics.metrics import (
    get_speed_from_df,
    get_delta_in_frame,
    get_head_angle,
    get_ht_cross_sign,
    get_orientation_vectors,
    get_motion_vector,
    get_vectors_between,
    perfectly_overlapping,
    get_cross_segment_deltas,
    get_delta_between_frames,
    get_all_deltas,
    flag_overlap_sign_reversals,
    flag_overlaps,
    get_overlap_edges,
    get_consecutive_ranges,
    flag_overlap_minimum_mismatches
)
from . import plotting
from .kalman_filter import KalmanFilter
from typing import Dict, List, Optional, Tuple, Union
from .logger import logger
from .config import SwapConfig, SwapCorrectionConfig
from .processor import SwapProcessor
from .plotting import (generate_distribution_plot, generate_flag_plot,
                      plot_trajectory, plot_metrics)
import logging
from scipy import signal

# Parameters
OVERLAP_THRESH = 0 # maximum distance between overlapping points

# Column mappings
POSDICT = {
    'head': ('X-Head', 'Y-Head'),
    'tail': ('X-Tail', 'Y-Tail'),
    'ctr': ('X-Centroid', 'Y-Centroid'),
    'mid': ('X-Midpoint', 'Y-Midpoint')
}

# Alternative column names (for backward compatibility)
ALT_POSDICT = {
    'head': ('xhead', 'yhead'),
    'tail': ('xtail', 'ytail'),
    'ctr': ('xctr', 'yctr'),
    'mid': ('xmid', 'ymid')
}

class FlagDetector:
    """
    Class for detecting various types of tracking errors and flags.
    """
    
    def __init__(self, overlap_threshold: float = OVERLAP_THRESH):
        """
        Initialize the flag detector.
        
        Args:
            overlap_threshold: Maximum distance between overlapping points
        """
        self.overlap_threshold = overlap_threshold
        
    def detect_all_flags(self, data: pd.DataFrame, fps: float = None, debug: bool = False) -> dict:
        """
        Detect all types of flags in the data.

        Args:
            data: DataFrame containing tracking data
            fps: Frame rate of the data
            debug: Whether to print debug messages

        Returns:
            Dictionary of flag arrays
        """
        # Normalize column names
        data = normalize_column_names(data)
        
        # Calculate midpoints if not present
        if 'X-Midpoint' not in data.columns or 'Y-Midpoint' not in data.columns:
            data = utils.calculate_midpoints(data)
        
        flags = {}
        n_frames = len(data)

        # Detect each type of flag
        flags['discontinuities_tail'] = np.array(self.detect_discontinuities(data, 'tail', fps, debug=debug))
        flags['overlaps'] = np.array(self.detect_overlaps(data, debug=debug))
        flags['sign_reversals'] = np.array(self.detect_sign_reversals(data, debug=debug))
        flags['delta_mismatches'] = np.array(self.detect_delta_mismatches(data, debug=debug))
        flags['overlap_sign_reversals'] = np.array(self.detect_overlap_sign_reversals(data, debug=debug))
        flags['overlap_minimum_mismatches'] = np.array(self.detect_overlap_minimum_mismatches(data, debug=debug))

        # Ensure all flag arrays have the same length as the input data
        for key in flags:
            if len(flags[key]) != n_frames:
                # Pad with False values if array is too short
                if len(flags[key]) < n_frames:
                    flags[key] = np.pad(flags[key], (0, n_frames - len(flags[key])), mode='constant', constant_values=False)
                # Truncate if array is too long
                else:
                    flags[key] = flags[key][:n_frames]

        if debug:
            for flag_name, flag_data in flags.items():
                logger.debug(f"{flag_name}: {np.sum(flag_data)} frames flagged")

        return flags
    
    def detect_discontinuities(self, data: pd.DataFrame, key: str, fps: int,
                             threshold: float = 24, debug: bool = False) -> np.ndarray:
        """
        Detect discontinuities in tracking data.
        
        Args:
            data: DataFrame containing tracking data
            key: Which point to check ('head' or 'tail')
            fps: Frame rate
            threshold: Maximum allowed speed in mm/s
            debug: Whether to print debug messages
            
        Returns:
            Array of flagged frames
        """
        # Calculate speed
        speed = get_speed_from_df(data, key, fps)
        
        # Flag frames where speed exceeds threshold
        flags = speed > threshold
        
        if debug:
            logger.debug(f"Discontinuities: {np.sum(flags)} frames flagged")
            
        return flags
    
    def detect_overlaps(self, data: pd.DataFrame, tolerance: float = None,
                       pt1: str = 'head', pt2: str = 'tail',
                       debug: bool = False) -> np.ndarray:
        """
        Detect frames where head and tail points overlap.
        
        Args:
            data: DataFrame containing tracking data
            tolerance: Maximum allowed distance between points
            pt1: First point to check
            pt2: Second point to check
            debug: Whether to print debug messages
            
        Returns:
            Array of flagged frames
        """
        if tolerance is None:
            tolerance = self.overlap_threshold
            
        # Calculate distance between points
        dist = get_delta_in_frame(data, pt1, pt2)
        
        # Flag frames where distance is less than tolerance
        flags = dist < tolerance
        
        if debug:
            logger.debug(f"Overlaps: {np.sum(flags)} frames flagged")
            
        return flags
    
    def detect_sign_reversals(self, data: pd.DataFrame, threshold: float = np.pi/2,
                            debug: bool = False) -> np.ndarray:
        """
        Detect sudden reversals in body angle.
        
        Args:
            data: DataFrame containing tracking data
            threshold: Maximum allowed angle change in radians
            debug: Whether to print debug messages
            
        Returns:
            Array of flagged frames
        """
        # Calculate body angles
        angles = get_head_angle(data)
        
        # Calculate angle changes
        angle_changes = np.abs(np.diff(angles, prepend=angles[0]))
        
        # Flag frames where angle change exceeds threshold
        flags = angle_changes > threshold
        
        if debug:
            logger.debug(f"Sign reversals: {np.sum(flags)} frames flagged")
            
        return flags
    
    def detect_delta_mismatches(self, data: pd.DataFrame, tolerance: float = 0.0,
                              debug: bool = False) -> np.ndarray:
        """
        Detect mismatches in movement between head and tail.

        Args:
            data: DataFrame containing tracking data
            tolerance: Allowed tolerance for mismatches
            debug: Whether to print debug messages

        Returns:
            Array of flagged frames
        """
        # Calculate head and tail speeds
        head_speed = get_speed_from_df(data, 'head', fps=30)
        tail_speed = get_speed_from_df(data, 'tail', fps=30)
        
        # Calculate speed differences
        speed_diff = np.abs(head_speed - tail_speed)
        
        # Flag frames with significant differences
        mismatches = speed_diff > tolerance
        
        if debug:
            print(f"Delta Mismatches: {np.where(mismatches)[0].tolist()}")
        
        return mismatches
    
    def detect_overlap_sign_reversals(self, data: pd.DataFrame,
                                    tolerance: float = None,
                                    threshold: float = np.pi/4,
                                    debug: bool = False) -> np.ndarray:
        """
        Detect sign reversals during overlap periods.
        
        Args:
            data: DataFrame containing tracking data
            tolerance: Maximum allowed distance between points
            threshold: Maximum allowed angle change in radians
            debug: Whether to print debug messages
            
        Returns:
            Array of flagged frames
        """
        if tolerance is None:
            tolerance = self.overlap_threshold
            
        # Get overlap flags
        overlap_flags = self.detect_overlaps(data, tolerance)
        
        # Get sign reversal flags
        reversal_flags = self.detect_sign_reversals(data, threshold)
        
        # Flag frames where both overlap and reversal occur
        flags = np.logical_and(overlap_flags, reversal_flags)
        
        if debug:
            logger.debug(f"Overlap sign reversals: {np.sum(flags)} frames flagged")
            
        return flags
    
    def detect_overlap_minimum_mismatches(self, data: pd.DataFrame,
                                        tolerance: float = None,
                                        debug: bool = False) -> np.ndarray:
        """
        Detect minimum mismatches during overlap periods.

        Args:
            data: DataFrame containing tracking data
            tolerance: Maximum allowed distance between points
            debug: Whether to print debug messages

        Returns:
            Array of flagged frames
        """
        if tolerance is None:
            tolerance = self.overlap_threshold

        # Get overlap flags
        overlap_flags = self.detect_overlaps(data, tolerance)

        # Get delta mismatch flags
        mismatch_flags = self.detect_delta_mismatches(data)

        # Ensure arrays have same length
        if len(overlap_flags) != len(mismatch_flags):
            min_len = min(len(overlap_flags), len(mismatch_flags))
            overlap_flags = overlap_flags[:min_len]
            mismatch_flags = mismatch_flags[:min_len]

        # Flag frames where both overlap and mismatch occur
        flags = np.logical_and(overlap_flags, mismatch_flags)
        return flags

    def detect_zeroed_frames(self, data: pd.DataFrame) -> np.ndarray:
        """
        Detect frames where any position data is zero.

        Args:
            data: DataFrame containing tracking data

        Returns:
            Array of flagged frames
        """
        # Check for zeros in position columns
        flags = np.zeros(len(data), dtype=bool)
        for col in ['xhead', 'yhead', 'xtail', 'ytail', 'xmid', 'ymid']:
            flags |= (data[col] == 0)
        return flags

    def detect_edge_frames(self, data: pd.DataFrame) -> np.ndarray:
        """
        Detect frames where any position data is at the edge of the frame.

        Args:
            data: DataFrame containing tracking data

        Returns:
            Array of flagged frames
        """
        # For now, just return an array of False values
        # This can be implemented properly when we have frame size information
        return np.zeros(len(data), dtype=bool)

# ----- Tracking Correction -----

def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to uppercase with hyphen format.
    
    Args:
        data: DataFrame with position columns
        
    Returns:
        DataFrame with normalized column names
    """
    data = data.copy()
    
    # Define column mappings
    column_mapping = {
        'xhead': 'X-Head', 'yhead': 'Y-Head',
        'xtail': 'X-Tail', 'ytail': 'Y-Tail',
        'xctr': 'X-Centroid', 'yctr': 'Y-Centroid',
        'xmid': 'X-Midpoint', 'ymid': 'Y-Midpoint'
    }
    
    # Only rename columns that exist
    existing_cols = {k: v for k, v in column_mapping.items() if k in data.columns}
    if existing_cols:
        data = data.rename(columns=existing_cols)
    
    return data

def setup_logger(config: SwapCorrectionConfig) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Logger object
    """
    # Create logger
    logger = logging.getLogger("swap_corrector")
    logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if config.log_file is not None:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.DEBUG if config.debug else logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def remove_edge_frames(data: pd.DataFrame) -> pd.DataFrame:
    """Remove edge frames with all zero positions.
    
    Args:
        data: DataFrame with trajectory data
        
    Returns:
        DataFrame with edge frames removed
    """
    # Create a copy of the data
    cleaned = data.copy()
    
    # Find frames where all positions are zero
    position_cols = ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']
    zero_frames = np.all(cleaned[position_cols] == 0, axis=1)
    
    # Set edge frames to NaN
    if zero_frames.iloc[0]:
        cleaned.iloc[0] = np.nan
    if zero_frames.iloc[-1]:
        cleaned.iloc[-1] = np.nan
    
    return cleaned

def interpolate_gaps(data: pd.DataFrame) -> pd.DataFrame:
    """Interpolate over gaps in position data.
    
    Args:
        data: DataFrame with trajectory data
        
    Returns:
        DataFrame with gaps interpolated
    """
    # Define position columns
    position_cols = ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']
    
    # Create a copy of the data
    interpolated = data.copy()
    
    # Interpolate each position column
    for col in position_cols:
        interpolated[col] = interpolated[col].interpolate(method='linear')
        
    # Calculate midpoints
    interpolated['X-Midpoint'] = (interpolated['X-Head'] + interpolated['X-Tail']) / 2
    interpolated['Y-Midpoint'] = (interpolated['Y-Head'] + interpolated['Y-Tail']) / 2
    
    return interpolated

def correct_tracking_errors(data: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Correct tracking errors in data.
    
    Args:
        data: DataFrame with trajectory data
        fps: Frames per second
        
    Returns:
        DataFrame with corrected data
    """
    # Initialize processor
    config = SwapCorrectionConfig()
    detector_config = SwapConfig(fps=fps)
    processor = SwapProcessor(config=config, detector_config=detector_config)
    
    # Process data
    corrected = processor.process(data)
    
    return corrected

def validate_corrected_data(data: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Validate corrected data by recalculating metrics.
    
    Args:
        data: DataFrame containing tracking data
        fps: Frame rate
        
    Returns:
        DataFrame with validated metrics
    """
    # Normalize column names
    data = normalize_column_names(data)
    
    # Calculate metrics
    metrics = {}
    
    # Calculate speed
    metrics['speed'] = get_speed_from_df(data, 'head', fps)
    
    # Calculate acceleration
    metrics['acceleration'] = np.gradient(metrics['speed'], 1/fps)
    
    # Calculate curvature
    metrics['curvature'] = get_head_angle(data)
    
    # Update metrics in DataFrame
    data['Speed'] = metrics['speed']
    data['Acceleration'] = metrics['acceleration']
    data['Curvature'] = metrics['curvature']
    
    return data
    
def tracking_correction(data: pd.DataFrame, fps: float = 30,
                      swapCorrection: bool = True,
                      removeErrors: bool = True,
                      interp: bool = True,
                      validate: bool = True,
                      filterData: bool = True,
                      debug: bool = False) -> pd.DataFrame:
    """Apply tracking correction pipeline.
    
    Args:
        data: DataFrame with trajectory data
        fps: Frames per second
        swapCorrection: Whether to apply swap correction
        removeErrors: Whether to remove tracking errors
        interp: Whether to interpolate missing data
        validate: Whether to validate corrections
        filterData: Whether to filter data
        debug: Whether to enable debug mode
        
    Returns:
        DataFrame with corrected data
    """
    # Create configuration
    config = SwapCorrectionConfig(
        debug=debug,
        diagnostic_plots=debug,
        show_plots=False,
        log_level="DEBUG" if debug else "INFO"
    )
    
    # Create detector configuration
    detector_config = SwapConfig(fps=fps)
    
    # Initialize processor
    processor = SwapProcessor(config=config, detector_config=detector_config)
    
    # Process data
    data = processor.process(data)
    
    # Apply additional corrections if needed
    if removeErrors:
        data = remove_edge_frames(data)
        
    if interp:
        data = interpolate_gaps(data)
        
    if validate:
        data = validate_corrected_data(data, fps)
        
    if filterData:
        data = filter_data(data)
        
    return data

def main(source_dir: str, output_dir: str, config: SwapCorrectionConfig) -> None:
    """Main pipeline for tracking correction.
    
    Args:
        source_dir: Directory containing source files
        output_dir: Directory for output files
        config: Configuration object
    """
    # Setup logging
    setup_logger(config)
    
    # Process each file in source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            # Read data
            data_path = os.path.join(source_dir, filename)
            data = pd.read_csv(data_path)
            
            # Apply correction
            corrected = tracking_correction(
                data,
                fps=config.fps,
                swapCorrection=config.fix_swaps,
                removeErrors=config.remove_errors,
                interp=config.interpolate,
                validate=True,
                filterData=config.filter_data,
                debug=config.debug
            )
            
            # Save corrected data
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_processed.csv")
            corrected.to_csv(output_path, index=False)
            
            if config.debug:
                print(f"Processed {filename}")

def compare_filtered_trajectories(source_dir: str, output_dir: str, filename: str) -> None:
    """Compare raw and filtered trajectories.
    
    Args:
        source_dir: Directory containing raw data
        output_dir: Directory to save output
        filename: Name of data file
    """
    # Load data
    raw_path = os.path.join(source_dir, filename)
    processed_path = os.path.join(output_dir, filename.replace('.csv', '_processed.csv'))
    raw_data = pd.read_csv(raw_path)
    processed_data = pd.read_csv(processed_path)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot raw trajectories
    ax1.plot(raw_data['X-Head'], raw_data['Y-Head'], 'b-', label='Head', alpha=0.5)
    ax1.plot(raw_data['X-Tail'], raw_data['Y-Tail'], 'r-', label='Tail', alpha=0.5)
    ax1.set_title('Raw Trajectories')
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot processed trajectories
    ax2.plot(processed_data['X-Head'], processed_data['Y-Head'], 'b-', label='Head', alpha=0.5)
    ax2.plot(processed_data['X-Tail'], processed_data['Y-Tail'], 'r-', label='Tail', alpha=0.5)
    ax2.set_title('Processed Trajectories')
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    output_path = os.path.join(output_dir, filename.replace('.csv', '_trajectory_comparison.png'))
    plt.savefig(output_path)
    plt.close()

def compare_filtered_distributions(source_dir: str, output_dir: str, filename: str) -> None:
    """Compare raw and filtered data distributions.
    
    Args:
        source_dir: Directory containing raw data
        output_dir: Directory to save output
        filename: Name of data file
    """
    # Load data
    raw_path = os.path.join(source_dir, filename)
    processed_path = os.path.join(output_dir, filename.replace('.csv', '_processed.csv'))
    raw_data = pd.read_csv(raw_path)
    processed_data = pd.read_csv(processed_path)
    
    # Create figures for raw and processed distributions
    for data, label in [(raw_data, 'raw'), (processed_data, 'processed')]:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
        
        # Plot head position distributions
        sns.kdeplot(data=data, x='X-Head', ax=ax1, label='X')
        sns.kdeplot(data=data, x='Y-Head', ax=ax1, label='Y')
        ax1.set_title('Head Position Distribution')
        ax1.legend()
        ax1.grid(True)
        
        # Plot tail position distributions
        sns.kdeplot(data=data, x='X-Tail', ax=ax2, label='X')
        sns.kdeplot(data=data, x='Y-Tail', ax=ax2, label='Y')
        ax2.set_title('Tail Position Distribution')
        ax2.legend()
        ax2.grid(True)
        
        # Plot speed distribution
        if 'Speed' in data.columns:
            sns.histplot(data=data, x='Speed', ax=ax3, bins=50)
            ax3.set_title('Speed Distribution')
            ax3.grid(True)
        
        # Plot curvature distribution
        if 'Curvature' in data.columns:
            sns.histplot(data=data, x='Curvature', ax=ax4, bins=50)
            ax4.set_title('Curvature Distribution')
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename.replace('.csv', '')}_{label}_distribution.png")
        plt.savefig(output_path)
        plt.close()

def plot_flags(source_dir: str, output_dir: str, filename: str) -> None:
    """Plot flags for each detector.
    
    Args:
        source_dir: Directory containing raw data
        output_dir: Directory to save output
        filename: Name of data file
    """
    # Load data
    raw_path = os.path.join(source_dir, filename)
    processed_path = os.path.join(output_dir, filename.replace('.csv', '_processed.csv'))
    raw_data = pd.read_csv(raw_path)
    processed_data = pd.read_csv(processed_path)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot speed flags
    if 'speed_flag' in processed_data.columns:
        ax1.plot(processed_data.index, processed_data['speed_flag'], 'b-', label='Speed')
        ax1.set_title('Speed Flags')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Flag')
        ax1.legend()
        ax1.grid(True)
    
    # Plot turn flags
    if 'turn_flag' in processed_data.columns:
        ax2.plot(processed_data.index, processed_data['turn_flag'], 'r-', label='Turn')
        ax2.set_title('Turn Flags')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Flag')
        ax2.legend()
        ax2.grid(True)
    
    # Plot proximity flags
    if 'proximity_flag' in processed_data.columns:
        ax3.plot(processed_data.index, processed_data['proximity_flag'], 'g-', label='Proximity')
        ax3.set_title('Proximity Flags')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Flag')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, filename.replace('.csv', '_flags.png'))
    plt.savefig(output_path)
    plt.close()

def examine_flags(source_dir: str, output_dir: str, filename: str) -> None:
    """Examine flags in data.
    
    Args:
        source_dir: Directory containing source files
        output_dir: Directory for output files
        filename: Name of file to process
    """
    # Load data
    data_path = os.path.join(source_dir, filename)
    data = pd.read_csv(data_path)
    
    # Get flags
    detector = FlagDetector()
    flags = detector.detect_all_flags(data, fps=30)
    
    # Plot flags
    fig = generate_flag_plot(data['Speed'].values, flags, title='Detected Flags')
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_flags.png")
    fig.savefig(plot_path)
    plt.close(fig)

def plot_diagnostics(raw_data: pd.DataFrame, processed_data: pd.DataFrame,
                    output_dir: str, filename: str, show_plots: bool = False) -> None:
    """Generate diagnostic plots.
    
    Args:
        raw_data: Raw data
        processed_data: Processed data
        output_dir: Directory for output files
        filename: Name of source file
        show_plots: Whether to display plots
    """
    base_name = os.path.splitext(filename)[0]
    
    # Plot trajectories
    fig = plot_trajectory(raw_data[['X-Head', 'Y-Head']].values,
                         title='Raw vs Processed Trajectories')
    fig.gca().plot(processed_data[['X-Head', 'Y-Head']].values[:, 0],
                  processed_data[['X-Head', 'Y-Head']].values[:, 1],
                  'r-', alpha=0.5, label='Processed')
    fig.gca().legend()
    
    if show_plots:
        plt.show()
    else:
        plot_path = os.path.join(output_dir, f"{base_name}_trajectories.png")
        fig.savefig(plot_path)
    plt.close(fig)
    
    # Plot distributions
    fig = generate_distribution_plot(raw_data['Speed'].values,
                                   title='Speed Distribution',
                                   xlabel='Speed (mm/s)',
                                   ylabel='Count')
    fig.gca().hist(processed_data['Speed'].values, bins='auto',
                   alpha=0.5, label='Processed')
    fig.gca().legend()
    
    if show_plots:
        plt.show()
    else:
        plot_path = os.path.join(output_dir, f"{base_name}_distributions.png")
        fig.savefig(plot_path)
    plt.close(fig)
    
    # Plot metrics
    metrics = {
        'Speed': processed_data['Speed'].values,
        'Acceleration': processed_data['Acceleration'].values,
        'Curvature': processed_data['Curvature'].values
    }
    fig = plot_metrics(metrics, title='Tracking Metrics')
    
    if show_plots:
        plt.show()
    else:
        plot_path = os.path.join(output_dir, f"{base_name}_metrics.png")
        fig.savefig(plot_path)
    plt.close(fig)

def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter data to remove noise.
    
    Args:
        data: DataFrame with trajectory data
        
    Returns:
        Filtered DataFrame
    """
    # Create copy of data
    filtered = data.copy()
    
    # Apply Savitzky-Golay filter to position data
    window = 11  # Must be odd
    order = 3  # Polynomial order
    
    for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']:
        if col in filtered.columns:
            filtered[col] = signal.savgol_filter(filtered[col], window, order)
            
    # Update midpoint columns
    if 'X-Head' in filtered.columns and 'X-Tail' in filtered.columns:
        filtered['X-Midpoint'] = (filtered['X-Head'] + filtered['X-Tail']) / 2
        
    if 'Y-Head' in filtered.columns and 'Y-Tail' in filtered.columns:
        filtered['Y-Midpoint'] = (filtered['Y-Head'] + filtered['Y-Tail']) / 2
        
    return filtered
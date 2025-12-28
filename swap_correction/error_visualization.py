"""
Visualization functions for error analysis.

This module provides functions to visualize swap errors and compare
different correction levels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Dict, List
from swap_correction import plotting, metrics, error_analysis, pivr_loader, tracking_correction, utils


def plot_trajectory_comparison(raw_data: Optional[pd.DataFrame],
                               level1_data: pd.DataFrame,
                               level2_data: pd.DataFrame,
                               trial_name: str,
                               output_path: Optional[str] = None,
                               show: bool = False) -> plt.Figure:
    """
    Overlay trajectories for all three datasets with error highlighting.
    
    Parameters:
    -----------
    raw_data : pd.DataFrame or None
        Raw data
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
    trial_name : str
        Name of the trial for title
    output_path : str or None
        Path to save figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Identify swap segments
    swap_segments = error_analysis.get_swap_segments(level1_data, level2_data)
    
    # Plot trajectories
    if raw_data is not None:
        ax.plot(raw_data['xctr'], raw_data['yctr'], 'gray', alpha=0.3, linewidth=0.5, label='Raw')
    
    ax.plot(level1_data['xctr'], level1_data['yctr'], 'orange', alpha=0.7, linewidth=1, label='Level 1 (Auto-corrected)')
    ax.plot(level2_data['xctr'], level2_data['yctr'], 'green', alpha=0.7, linewidth=1, label='Level 2 (Ground Truth)')
    
    # Highlight error regions
    for seg in swap_segments:
        start_idx = seg[0]
        end_idx = min(seg[1], len(level1_data) - 1)
        if start_idx < len(level1_data):
            ax.plot(level1_data['xctr'].iloc[start_idx:end_idx+1],
                   level1_data['yctr'].iloc[start_idx:end_idx+1],
                   'red', linewidth=2, alpha=0.8)
    
    # Mark start positions
    if len(level2_data) > 0:
        start_idx = level2_data['xctr'].first_valid_index()
        if start_idx is not None:
            ax.scatter(level2_data['xctr'].iloc[start_idx], 
                      level2_data['yctr'].iloc[start_idx],
                      c='blue', marker='x', s=100, label='Start', zorder=5)
    
    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Y position (mm)')
    ax.set_title(f'Trajectory Comparison: {trial_name}')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_error_timeseries(level1_data: pd.DataFrame,
                         level2_data: pd.DataFrame,
                         trial_name: str,
                         fps: int = 30,
                         output_path: Optional[str] = None,
                         show: bool = False) -> plt.Figure:
    """
    Plot time series of position and motion errors.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
    trial_name : str
        Name of the trial
    fps : int
        Frame rate
    output_path : str or None
        Path to save figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Calculate errors
    pos_errors = error_analysis.calculate_position_errors(level1_data, level2_data)
    motion_errors = error_analysis.calculate_motion_errors(level1_data, level2_data, fps)
    swap_segments = error_analysis.get_swap_segments(level1_data, level2_data)
    
    # Create time axis
    min_len = min(len(level1_data), len(level2_data))
    time = np.arange(min_len) / fps
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Position errors
    axes[0].plot(time, pos_errors['head_position_error'], label='Head', alpha=0.7)
    axes[0].plot(time, pos_errors['tail_position_error'], label='Tail', alpha=0.7)
    axes[0].set_ylabel('Position Error (mm)')
    axes[0].set_title(f'Position Errors: {trial_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Motion errors
    axes[1].plot(time, motion_errors['head_speed_error'], label='Head Speed', alpha=0.7)
    axes[1].plot(time, motion_errors['tail_speed_error'], label='Tail Speed', alpha=0.7)
    axes[1].set_ylabel('Speed Error (mm/s)')
    axes[1].set_title('Motion Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Speed ratio error
    axes[2].plot(time, motion_errors['speed_ratio_error'], label='Speed Ratio', alpha=0.7, color='purple')
    axes[2].set_ylabel('Speed Ratio Error')
    axes[2].set_title('Speed Ratio Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Swap segments indicator
    swap_indicator = np.zeros(min_len)
    for seg in swap_segments:
        start = min(seg[0], min_len - 1)
        end = min(seg[1], min_len - 1)
        swap_indicator[start:end+1] = 1
    
    axes[3].fill_between(time, 0, swap_indicator, alpha=0.5, color='red', label='Swap Segments')
    axes[3].set_ylabel('Swap Status')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Swap Segments (Ground Truth)')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_error_heatmap(level1_data: pd.DataFrame,
                      level2_data: pd.DataFrame,
                      trial_name: str,
                      output_path: Optional[str] = None,
                      show: bool = False) -> plt.Figure:
    """
    Create 2D heatmap showing error magnitude across trajectory.
    
    Parameters:
    -----------
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
    trial_name : str
        Name of the trial
    output_path : str or None
        Path to save figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    pos_errors = error_analysis.calculate_position_errors(level1_data, level2_data)
    min_len = min(len(level1_data), len(level2_data))
    
    # Get trajectory positions
    x = level2_data['xctr'].iloc[:min_len].values
    y = level2_data['yctr'].iloc[:min_len].values
    error_magnitude = pos_errors['head_position_error'].values + pos_errors['tail_position_error'].values
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create scatter plot with error magnitude as color
    scatter = ax.scatter(x, y, c=error_magnitude, cmap='hot', s=10, alpha=0.6)
    
    # Overlay trajectory
    ax.plot(x, y, 'gray', alpha=0.3, linewidth=0.5)
    
    # Mark start
    if len(x) > 0:
        ax.scatter(x[0], y[0], c='blue', marker='x', s=100, zorder=5, label='Start')
    
    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Y position (mm)')
    ax.set_title(f'Error Heatmap: {trial_name}')
    ax.axis('equal')
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error Magnitude (mm)')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_metric_comparison(raw_data: Optional[pd.DataFrame],
                          level1_data: pd.DataFrame,
                          level2_data: pd.DataFrame,
                          trial_name: str,
                          fps: int = 30,
                          output_path: Optional[str] = None,
                          show: bool = False) -> plt.Figure:
    """
    Compare metrics (speed ratios, angles, distances) across all three datasets.
    
    Parameters:
    -----------
    raw_data : pd.DataFrame or None
        Raw data
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
    trial_name : str
        Name of the trial
    fps : int
        Frame rate
    output_path : str or None
        Path to save figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    min_len = min(len(level1_data), len(level2_data))
    time = np.arange(min_len) / fps
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Speed ratios
    h1_speed = metrics.get_speed_from_df(level1_data.iloc[:min_len], 'head', fps=fps)
    t1_speed = metrics.get_speed_from_df(level1_data.iloc[:min_len], 'tail', fps=fps)
    h2_speed = metrics.get_speed_from_df(level2_data.iloc[:min_len], 'head', fps=fps)
    t2_speed = metrics.get_speed_from_df(level2_data.iloc[:min_len], 'tail', fps=fps)
    
    # Use larger epsilon (0.1 mm/s) and cap ratios to avoid extreme values
    min_tail_speed = 0.1  # Minimum tail speed threshold (mm/s)
    max_ratio = 100.0  # Maximum ratio cap
    
    t1_safe = np.maximum(t1_speed, min_tail_speed)
    t2_safe = np.maximum(t2_speed, min_tail_speed)
    
    ratio1 = np.clip(h1_speed / t1_safe, -max_ratio, max_ratio)
    ratio2 = np.clip(h2_speed / t2_safe, -max_ratio, max_ratio)
    
    axes[0].plot(time, ratio1, label='Level 1', alpha=0.7)
    axes[0].plot(time, ratio2, label='Level 2 (GT)', alpha=0.7)
    if raw_data is not None and len(raw_data) >= min_len:
        h0_speed = metrics.get_speed_from_df(raw_data.iloc[:min_len], 'head', fps=fps)
        t0_speed = metrics.get_speed_from_df(raw_data.iloc[:min_len], 'tail', fps=fps)
        t0_safe = np.maximum(t0_speed, min_tail_speed)
        ratio0 = np.clip(h0_speed / t0_safe, -max_ratio, max_ratio)
        axes[0].plot(time, ratio0, label='Raw', alpha=0.5, linestyle='--')
    axes[0].set_ylabel('Head/Tail Speed Ratio')
    axes[0].set_title(f'Metric Comparison: {trial_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    
    # Body orientation
    orient1 = metrics.get_orientation(level1_data.iloc[:min_len])
    orient2 = metrics.get_orientation(level2_data.iloc[:min_len])
    
    axes[1].plot(time, np.rad2deg(orient1), label='Level 1', alpha=0.7)
    axes[1].plot(time, np.rad2deg(orient2), label='Level 2 (GT)', alpha=0.7)
    if raw_data is not None and len(raw_data) >= min_len:
        orient0 = metrics.get_orientation(raw_data.iloc[:min_len])
        axes[1].plot(time, np.rad2deg(orient0), label='Raw', alpha=0.5, linestyle='--')
    axes[1].set_ylabel('Body Orientation (degrees)')
    axes[1].set_title('Body Orientation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Head-tail distance
    dist1 = metrics.get_delta_in_frame(level1_data.iloc[:min_len], 'head', 'tail')
    dist2 = metrics.get_delta_in_frame(level2_data.iloc[:min_len], 'head', 'tail')
    
    axes[2].plot(time, dist1, label='Level 1', alpha=0.7)
    axes[2].plot(time, dist2, label='Level 2 (GT)', alpha=0.7)
    if raw_data is not None and len(raw_data) >= min_len:
        dist0 = metrics.get_delta_in_frame(raw_data.iloc[:min_len], 'head', 'tail')
        axes[2].plot(time, dist0, label='Raw', alpha=0.5, linestyle='--')
    axes[2].set_ylabel('Head-Tail Distance (mm)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Head-Tail Distance')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_detection_performance(raw_data: Optional[pd.DataFrame],
                              level1_data: pd.DataFrame,
                              level2_data: pd.DataFrame,
                              trial_name: str,
                              fps: int = 30,
                              output_path: Optional[str] = None,
                              show: bool = False) -> plt.Figure:
    """
    Visualize detection method performance against ground truth.
    
    Parameters:
    -----------
    raw_data : pd.DataFrame or None
        Raw data
    level1_data : pd.DataFrame
        Auto-corrected data
    level2_data : pd.DataFrame
        Manually corrected ground truth data
    trial_name : str
        Name of the trial
    fps : int
        Frame rate
    output_path : str or None
        Path to save figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    min_len = min(len(level1_data), len(level2_data))
    time = np.arange(min_len) / fps
    
    # Ground truth swaps
    gt_swaps = error_analysis.identify_swapped_frames(level1_data, level2_data)
    gt_indicator = np.zeros(min_len)
    gt_indicator[gt_swaps] = 1
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Ground truth
    axes[0].fill_between(time, 0, gt_indicator, alpha=0.7, color='red', label='Ground Truth Swaps')
    axes[0].set_ylabel('Swap Status')
    axes[0].set_title(f'Detection Performance: {trial_name}')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if raw_data is not None and len(raw_data) >= min_len:
        raw_subset = raw_data.iloc[:min_len]
        
        # Minimum delta mismatches
        try:
            mdm = tracking_correction.flag_min_delta_mismatches(raw_subset, debug=False)
            mdm_indicator = np.zeros(min_len)
            mdm_indicator[mdm[mdm < min_len]] = 1
            
            axes[1].fill_between(time, 0, mdm_indicator, alpha=0.7, color='blue', label='Min Delta Mismatch')
            axes[1].fill_between(time, 0, gt_indicator, alpha=0.3, color='red', label='Ground Truth')
            axes[1].set_ylabel('Detection')
            axes[1].set_title('Minimum Delta Mismatch Detection')
            axes[1].set_ylim(-0.1, 1.1)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error: {e}', transform=axes[1].transAxes, ha='center')
        
        # Overlap sign reversals
        try:
            cosr = tracking_correction.flag_overlap_sign_reversals(raw_subset, debug=False)
            cosr_indicator = np.zeros(min_len)
            cosr_indicator[cosr[cosr < min_len]] = 1
            
            axes[2].fill_between(time, 0, cosr_indicator, alpha=0.7, color='green', label='Overlap Sign Reversal')
            axes[2].fill_between(time, 0, gt_indicator, alpha=0.3, color='red', label='Ground Truth')
            axes[2].set_ylabel('Detection')
            axes[2].set_title('Overlap Sign Reversal Detection')
            axes[2].set_ylim(-0.1, 1.1)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Error: {e}', transform=axes[2].transAxes, ha='center')
        
        # Combined detection
        try:
            mdm = tracking_correction.flag_min_delta_mismatches(raw_subset, debug=False)
            cosr = tracking_correction.flag_overlap_sign_reversals(raw_subset, debug=False)
            comm = tracking_correction.flag_overlap_minimum_mismatches(raw_subset, debug=False)
            all_detected = utils.merge(mdm, cosr, comm)
            all_indicator = np.zeros(min_len)
            all_indicator[all_detected[all_detected < min_len]] = 1
            
            axes[3].fill_between(time, 0, all_indicator, alpha=0.7, color='purple', label='All Methods Combined')
            axes[3].fill_between(time, 0, gt_indicator, alpha=0.3, color='red', label='Ground Truth')
            axes[3].set_ylabel('Detection')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_title('Combined Detection')
            axes[3].set_ylim(-0.1, 1.1)
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        except Exception as e:
            axes[3].text(0.5, 0.5, f'Error: {e}', transform=axes[3].transAxes, ha='center')
    else:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, 'Raw data not available', transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_error_summary(all_trials_data: Dict[str, Dict[str, pd.DataFrame]],
                      output_path: Optional[str] = None,
                      show: bool = False) -> plt.Figure:
    """
    Create aggregate statistics plots across all trials.
    
    Parameters:
    -----------
    all_trials_data : dict
        Dictionary mapping trial names to their data dictionaries
    output_path : str or None
        Path to save figure
    show : bool
        Whether to display the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Collect statistics from all trials
    error_rates = []
    segment_counts = []
    avg_segment_lengths = []
    trial_names = []
    
    for trial_name, trial_data in all_trials_data.items():
        if trial_data['level1'] is not None and trial_data['level2'] is not None:
            stats = error_analysis.calculate_error_statistics(trial_data['level1'], trial_data['level2'])
            error_rates.append(stats['error_rate'])
            segment_counts.append(stats['num_swap_segments'])
            avg_segment_lengths.append(stats['avg_segment_length'])
            trial_names.append(trial_name)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Error rate distribution
    axes[0, 0].hist(error_rates, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Error Rate')
    axes[0, 0].set_ylabel('Number of Trials')
    axes[0, 0].set_title('Error Rate Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Segment count distribution
    axes[0, 1].hist(segment_counts, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Number of Swap Segments')
    axes[0, 1].set_ylabel('Number of Trials')
    axes[0, 1].set_title('Swap Segment Count Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average segment length distribution
    axes[1, 0].hist(avg_segment_lengths, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Average Segment Length (frames)')
    axes[1, 0].set_ylabel('Number of Trials')
    axes[1, 0].set_title('Average Segment Length Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error rate vs segment count
    axes[1, 1].scatter(segment_counts, error_rates, alpha=0.6)
    axes[1, 1].set_xlabel('Number of Swap Segments')
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].set_title('Error Rate vs Segment Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


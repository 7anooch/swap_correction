"""
Report generation module for error analysis.

This module provides functions to generate comprehensive reports
with visualizations and metrics.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
from swap_correction import error_analysis, error_visualization, pivr_loader


def generate_trial_report(trial_dir: str, output_dir: str, fps: Optional[int] = None) -> Dict:
    """
    Generate comprehensive report for a single trial.
    
    Parameters:
    -----------
    trial_dir : str
        Directory containing trial data files
    output_dir : str
        Directory to save report outputs
    fps : int or None
        Frame rate (will be loaded from settings if None)
        
    Returns:
    --------
    dict
        Dictionary containing all calculated metrics
    """
    trial_name = os.path.basename(trial_dir)
    trial_output_dir = os.path.join(output_dir, f"trial_{trial_name}")
    os.makedirs(trial_output_dir, exist_ok=True)
    
    # Load data
    trial_data = error_analysis.load_trial_data(trial_dir)
    
    if trial_data['level1'] is None or trial_data['level2'] is None:
        print(f"Warning: Missing required data files for trial {trial_name}")
        return {}
    
    # Get FPS if not provided
    if fps is None:
        try:
            settings = pivr_loader.get_all_settings(trial_dir)
            fps = settings['Framerate'] if settings else 30
        except Exception:
            fps = 30
    
    # Calculate all metrics
    metrics_dict = {}
    
    # Error statistics
    error_stats = error_analysis.calculate_error_statistics(trial_data['level1'], trial_data['level2'])
    metrics_dict.update(error_stats)
    
    # Position errors
    pos_errors = error_analysis.calculate_position_errors(trial_data['level1'], trial_data['level2'])
    metrics_dict['mean_head_position_error'] = pos_errors['head_position_error'].mean()
    metrics_dict['mean_tail_position_error'] = pos_errors['tail_position_error'].mean()
    metrics_dict['max_head_position_error'] = pos_errors['head_position_error'].max()
    metrics_dict['max_tail_position_error'] = pos_errors['tail_position_error'].max()
    
    # Motion errors
    motion_errors = error_analysis.calculate_motion_errors(trial_data['level1'], trial_data['level2'], fps)
    metrics_dict['mean_head_speed_error'] = motion_errors['head_speed_error'].abs().mean()
    metrics_dict['mean_tail_speed_error'] = motion_errors['tail_speed_error'].abs().mean()
    metrics_dict['mean_speed_ratio_error'] = motion_errors['speed_ratio_error'].abs().mean()
    
    # Geometric errors
    geom_errors = error_analysis.calculate_geometric_errors(trial_data['level1'], trial_data['level2'])
    metrics_dict['mean_orientation_error'] = np.rad2deg(geom_errors['orientation_error'].mean())
    metrics_dict['cross_sign_match_rate'] = geom_errors['cross_sign_match'].mean()
    
    # Detection metrics
    detection_metrics = error_analysis.calculate_detection_metrics(
        trial_data['raw'], trial_data['level1'], trial_data['level2'], fps
    )
    metrics_dict.update(detection_metrics)
    
    # Generate visualizations
    try:
        error_visualization.plot_trajectory_comparison(
            trial_data['raw'], trial_data['level1'], trial_data['level2'],
            trial_name, os.path.join(trial_output_dir, 'trajectory_comparison.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate trajectory comparison: {e}")
    
    try:
        error_visualization.plot_error_timeseries(
            trial_data['level1'], trial_data['level2'], trial_name, fps,
            os.path.join(trial_output_dir, 'error_timeseries.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate error timeseries: {e}")
    
    try:
        error_visualization.plot_error_heatmap(
            trial_data['level1'], trial_data['level2'], trial_name,
            os.path.join(trial_output_dir, 'error_heatmap.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate error heatmap: {e}")
    
    try:
        error_visualization.plot_metric_comparison(
            trial_data['raw'], trial_data['level1'], trial_data['level2'],
            trial_name, fps, os.path.join(trial_output_dir, 'metric_comparison.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate metric comparison: {e}")
    
    try:
        error_visualization.plot_detection_performance(
            trial_data['raw'], trial_data['level1'], trial_data['level2'],
            trial_name, fps, os.path.join(trial_output_dir, 'detection_performance.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate detection performance: {e}")
    
    # Export metrics to CSV
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_csv(os.path.join(trial_output_dir, 'metrics.csv'), index=False)
    
    return metrics_dict


def generate_summary_report(test_data_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Generate summary report across all trials.
    
    Parameters:
    -----------
    test_data_dir : str
        Directory containing trial subdirectories
    output_dir : str
        Directory to save report outputs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing aggregated metrics for all trials
    """
    summary_output_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_output_dir, exist_ok=True)
    
    # Load all trials
    all_trials_data = error_analysis.load_all_trials(test_data_dir)
    
    # Generate reports for each trial
    all_metrics = []
    trial_names = []
    
    for trial_name, trial_data in all_trials_data.items():
        if trial_data['level1'] is not None and trial_data['level2'] is not None:
            trial_dir = os.path.join(test_data_dir, trial_name)
            try:
                metrics = generate_trial_report(trial_dir, output_dir)
                if metrics:
                    all_metrics.append(metrics)
                    trial_names.append(trial_name)
            except Exception as e:
                print(f"Warning: Could not generate report for {trial_name}: {e}")
    
    # Create summary DataFrame
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df.insert(0, 'trial_name', trial_names)
        
        # Save summary CSV
        summary_df.to_csv(os.path.join(summary_output_dir, 'all_trials_metrics.csv'), index=False)
        
        # Generate summary visualizations
        try:
            error_visualization.plot_error_summary(
                all_trials_data, os.path.join(summary_output_dir, 'error_summary.png')
            )
        except Exception as e:
            print(f"Warning: Could not generate error summary plot: {e}")
        
        return summary_df
    else:
        print("Warning: No metrics collected from any trials")
        return pd.DataFrame()


def export_error_metrics(all_trials_data: Dict, output_file: str) -> None:
    """
    Export all error metrics to CSV.
    
    Parameters:
    -----------
    all_trials_data : dict
        Dictionary mapping trial names to their data dictionaries
    output_file : str
        Path to output CSV file
    """
    all_metrics = []
    trial_names = []
    
    for trial_name, trial_data in all_trials_data.items():
        if trial_data['level1'] is not None and trial_data['level2'] is not None:
            try:
                # Get FPS
                try:
                    settings = pivr_loader.get_all_settings(
                        os.path.dirname(trial_data['level1'].attrs.get('source_path', ''))
                        if hasattr(trial_data['level1'], 'attrs') else None
                    )
                    fps = settings['Framerate'] if settings else 30
                except Exception:
                    fps = 30
                
                # Calculate metrics
                error_stats = error_analysis.calculate_error_statistics(trial_data['level1'], trial_data['level2'])
                pos_errors = error_analysis.calculate_position_errors(trial_data['level1'], trial_data['level2'])
                motion_errors = error_analysis.calculate_motion_errors(trial_data['level1'], trial_data['level2'], fps)
                geom_errors = error_analysis.calculate_geometric_errors(trial_data['level1'], trial_data['level2'])
                detection_metrics = error_analysis.calculate_detection_metrics(
                    trial_data['raw'], trial_data['level1'], trial_data['level2'], fps
                )
                
                # Combine all metrics
                metrics = {}
                metrics.update(error_stats)
                metrics['mean_head_position_error'] = pos_errors['head_position_error'].mean()
                metrics['mean_tail_position_error'] = pos_errors['tail_position_error'].mean()
                metrics['mean_head_speed_error'] = motion_errors['head_speed_error'].abs().mean()
                metrics['mean_tail_speed_error'] = motion_errors['tail_speed_error'].abs().mean()
                metrics['mean_orientation_error'] = np.rad2deg(geom_errors['orientation_error'].mean())
                metrics.update(detection_metrics)
                
                all_metrics.append(metrics)
                trial_names.append(trial_name)
            except Exception as e:
                print(f"Warning: Could not calculate metrics for {trial_name}: {e}")
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        df.insert(0, 'trial_name', trial_names)
        df.to_csv(output_file, index=False)
        print(f"Exported metrics to {output_file}")
    else:
        print("Warning: No metrics to export")


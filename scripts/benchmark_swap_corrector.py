"""
Benchmark script for evaluating swap corrector performance.
Compares current and legacy versions against ground truth data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
from tqdm import tqdm
import glob
import logging
import sys
from datetime import datetime

# Import current version
from swap_corrector import tracking_correction as tc
from swap_corrector import pivr_loader as loader
from swap_corrector.metrics import metrics
from swap_corrector import plotting
from swap_corrector.metrics import normalize_column_names

# Import legacy version
from swap_corrector.legacy import tracking_correction as legacy_tc
from swap_corrector.legacy import pivr_loader as legacy_loader

# Setup logging
def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw, level1, and level2 (ground truth) data.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (raw_data, level1_data, level2_data)
    """
    # Check if this is synthetic data
    if os.path.basename(data_dir) == 'synthetic':
        raw_file = os.path.join(data_dir, 'synthetic_data.csv')
        level1_file = None  # No level1 for synthetic data
        level2_file = os.path.join(data_dir, 'synthetic_data_level2.csv')
    else:
        # Real data follows the pattern: *_data.csv, *_data_level1.csv, *_data_level2.csv
        raw_files = glob.glob(os.path.join(data_dir, '*_data.csv'))
        if not raw_files:
            raise FileNotFoundError(f"No raw data file found in {data_dir}")
        
        base_name = os.path.basename(raw_files[0]).replace('_data.csv', '')
        raw_file = os.path.join(data_dir, f"{base_name}_data.csv")
        level1_file = os.path.join(data_dir, f"{base_name}_data_level1.csv")
        level2_file = os.path.join(data_dir, f"{base_name}_data_level2.csv")
    
    # Load data files
    logging.info(f"Loading data from {data_dir}")
    logging.info(f"Raw file: {raw_file}")
    logging.info(f"Level2 file: {level2_file}")
    
    try:
        raw_data = pd.read_csv(raw_file)
        level2_data = pd.read_csv(level2_file)
        
        # Level1 data is optional (not available for synthetic data)
        if level1_file and os.path.exists(level1_file):
            logging.info(f"Level1 file: {level1_file}")
            level1_data = pd.read_csv(level1_file)
        else:
            level1_data = None
            logging.info("No Level1 data available")
        
        # Normalize column names
        raw_data = normalize_column_names(raw_data)
        level2_data = normalize_column_names(level2_data)
        if level1_data is not None:
            level1_data = normalize_column_names(level1_data)
        
        # Validate data
        required_cols = ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']
        for df, name in [(raw_data, 'raw'), (level2_data, 'level2')]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {name} data: {missing_cols}")
        
        return raw_data, level1_data, level2_data
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def calculate_metrics(predicted: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, float]:
    """Calculate performance metrics.
    
    Args:
        predicted: DataFrame with predicted positions
        ground_truth: DataFrame with ground truth positions
        
    Returns:
        Dictionary of metrics
    """
    try:
        # Normalize column names
        predicted = normalize_column_names(predicted)
        ground_truth = normalize_column_names(ground_truth)
        
        # Calculate head-tail orientation difference
        pred_orientation = metrics.get_orientation(predicted)
        true_orientation = metrics.get_orientation(ground_truth)
        orientation_diff = np.abs(pred_orientation - true_orientation)
        
        # Calculate head-tail separation difference
        pred_separation = metrics.get_delta_in_frame(predicted, 'head', 'tail')
        true_separation = metrics.get_delta_in_frame(ground_truth, 'head', 'tail')
        separation_diff = np.abs(pred_separation - true_separation)
        
        # Calculate position differences
        head_pos_diff = np.sqrt(
            (predicted['X-Head'] - ground_truth['X-Head'])**2 +
            (predicted['Y-Head'] - ground_truth['Y-Head'])**2
        )
        tail_pos_diff = np.sqrt(
            (predicted['X-Tail'] - ground_truth['X-Tail'])**2 +
            (predicted['Y-Tail'] - ground_truth['Y-Tail'])**2
        )
        
        # Calculate additional metrics
        head_speed_diff = np.abs(
            metrics.get_speed_from_df(predicted, 'head') -
            metrics.get_speed_from_df(ground_truth, 'head')
        )
        tail_speed_diff = np.abs(
            metrics.get_speed_from_df(predicted, 'tail') -
            metrics.get_speed_from_df(ground_truth, 'tail')
        )
        
        return {
            'orientation_mae': np.nanmean(orientation_diff),
            'separation_mae': np.nanmean(separation_diff),
            'head_pos_mae': np.nanmean(head_pos_diff),
            'tail_pos_mae': np.nanmean(tail_pos_diff),
            'head_speed_mae': np.nanmean(head_speed_diff),
            'tail_speed_mae': np.nanmean(tail_speed_diff),
            'total_mae': np.nanmean([
                np.nanmean(orientation_diff),
                np.nanmean(separation_diff),
                np.nanmean(head_pos_diff),
                np.nanmean(tail_pos_diff),
                np.nanmean(head_speed_diff),
                np.nanmean(tail_speed_diff)
            ])
        }
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        raise

def run_benchmark(data_dir: str, output_dir: str) -> Dict[str, Dict[str, float]]:
    """Run benchmark on a single data directory.
    
    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save output files
        
    Returns:
        Dictionary of metrics for current and legacy versions
    """
    try:
        # Load data
        logging.info(f"\nProcessing directory: {data_dir}")
        raw_data, level1_data, level2_data = load_data(data_dir)
        
        logging.info("Data loaded successfully")
        logging.info(f"Raw data columns: {raw_data.columns.tolist()}")
        
        # Get FPS from settings if available, otherwise use default
        try:
            fps = loader.get_all_settings(data_dir)['Framerate']
        except:
            fps = 30  # Default FPS for synthetic data
            logging.warning(f"Using default FPS: {fps}")
        
        logging.info(f"Using FPS: {fps}")
        
        # Run current version
        logging.info("Running current version...")
        start_time = time.time()
        current_result = tc.tracking_correction(
            raw_data,
            fps=fps,
            filterData=False,
            swapCorrection=True,
            validate=False,
            removeErrors=True,
            interp=True
        )
        current_time = time.time() - start_time
        logging.info("Current version completed successfully")
        
        # Run legacy version
        logging.info("Running legacy version...")
        start_time = time.time()
        legacy_result = legacy_tc.tracking_correction(
            raw_data,
            fps=fps,
            filterData=False,
            swapCorrection=True,
            validate=False,
            removeErrors=True,
            interp=True
        )
        legacy_time = time.time() - start_time
        logging.info("Legacy version completed successfully")
        
        # Calculate metrics
        logging.info("Calculating metrics...")
        current_metrics = calculate_metrics(current_result, level2_data)
        legacy_metrics = calculate_metrics(legacy_result, level2_data)
        
        # Add timing information
        current_metrics['processing_time'] = current_time
        legacy_metrics['processing_time'] = legacy_time
        
        # Generate visualizations
        output_path = Path(output_dir) / Path(data_dir).name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot trajectories
        fig, axs = plt.subplots(1, 4 if level1_data is not None else 3, figsize=(15, 5))
        
        # Raw data
        raw_data = normalize_column_names(raw_data)
        axs[0].plot(raw_data['X-Head'], raw_data['Y-Head'], 'b-', alpha=0.5, label='Head')
        axs[0].plot(raw_data['X-Tail'], raw_data['Y-Tail'], 'r-', alpha=0.5, label='Tail')
        axs[0].set_title('Raw Data')
        axs[0].legend()
        axs[0].axis('equal')
        
        # Current version
        current_result = normalize_column_names(current_result)
        axs[1].plot(current_result['X-Head'], current_result['Y-Head'], 'b-', alpha=0.5, label='Head')
        axs[1].plot(current_result['X-Tail'], current_result['Y-Tail'], 'r-', alpha=0.5, label='Tail')
        axs[1].set_title('Current Version')
        axs[1].legend()
        axs[1].axis('equal')
        
        # Legacy version
        legacy_result = normalize_column_names(legacy_result)
        axs[2].plot(legacy_result['X-Head'], legacy_result['Y-Head'], 'b-', alpha=0.5, label='Head')
        axs[2].plot(legacy_result['X-Tail'], legacy_result['Y-Tail'], 'r-', alpha=0.5, label='Tail')
        axs[2].set_title('Legacy Version')
        axs[2].legend()
        axs[2].axis('equal')
        
        # Level1 data (if available)
        if level1_data is not None:
            level1_data = normalize_column_names(level1_data)
            axs[3].plot(level1_data['X-Head'], level1_data['Y-Head'], 'b-', alpha=0.5, label='Head')
            axs[3].plot(level1_data['X-Tail'], level1_data['Y-Tail'], 'r-', alpha=0.5, label='Tail')
            axs[3].set_title('Level1 Data')
            axs[3].legend()
            axs[3].axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_path / 'trajectory_comparison.png')
        plt.close()
        
        return {
            'current': current_metrics,
            'legacy': legacy_metrics
        }
        
    except Exception as e:
        logging.error(f"Error in benchmark: {e}")
        raise

def plot_benchmark_results(results: Dict[str, Dict[str, float]], output_dir: str):
    """Plot benchmark results."""
    try:
        metrics = [
            'orientation_mae', 'separation_mae', 'head_pos_mae', 'tail_pos_mae',
            'head_speed_mae', 'tail_speed_mae', 'total_mae', 'processing_time'
        ]
        labels = [
            'Orientation MAE', 'Separation MAE', 'Head Position MAE', 'Tail Position MAE',
            'Head Speed MAE', 'Tail Speed MAE', 'Total MAE', 'Processing Time (s)'
        ]
        
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.flatten()
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            current_values = [r['current'][metric] for r in results.values()]
            legacy_values = [r['legacy'][metric] for r in results.values()]
            
            axs[i].boxplot([current_values, legacy_values], tick_labels=['Current', 'Legacy'])
            axs[i].set_title(label)
            axs[i].set_ylabel('Value')
            
            # Add individual data points
            for j, values in enumerate([current_values, legacy_values]):
                x = np.random.normal(j+1, 0.04, size=len(values))
                axs[i].plot(x, values, 'r.', alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_results.png'))
        plt.close()
        
        # Save summary statistics
        summary = {
            'metric': [],
            'version': [],
            'mean': [],
            'std': [],
            'median': [],
            'min': [],
            'max': []
        }
        
        for metric in metrics:
            for version in ['current', 'legacy']:
                values = [r[version][metric] for r in results.values()]
                summary['metric'].append(metric)
                summary['version'].append(version)
                summary['mean'].append(np.mean(values))
                summary['std'].append(np.std(values))
                summary['median'].append(np.median(values))
                summary['min'].append(np.min(values))
                summary['max'].append(np.max(values))
        
        pd.DataFrame(summary).to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
        
    except Exception as e:
        logging.error(f"Error plotting results: {e}")
        raise

def main():
    # Setup
    data_dir = 'data'
    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logging.info("Starting benchmark")
    
    # Get all data directories
    data_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    logging.info(f"Found {len(data_dirs)} data directories")
    
    # Run benchmarks
    results = {}
    for d in tqdm(data_dirs, desc="Running benchmarks"):
        try:
            results[d] = run_benchmark(os.path.join(data_dir, d), output_dir)
            logging.info(f"Successfully processed {d}")
        except Exception as e:
            logging.error(f"Error processing {d}: {e}")
    
    if results:
        # Plot results
        plot_benchmark_results(results, output_dir)
        
        # Save detailed results
        pd.DataFrame(results).to_csv(os.path.join(output_dir, 'detailed_results.csv'))
        logging.info("Benchmark completed successfully")
    else:
        logging.error("No results to plot - all benchmarks failed")

if __name__ == '__main__':
    main() 
"""
Script to analyze swap correction performance by comparing different data files.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

# Import current version
from swap_corrector import tracking_correction as tc
from swap_corrector import pivr_loader as loader
from swap_corrector.metrics import metrics
from swap_corrector.metrics import normalize_column_names

def setup_logging(output_dir: str):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, f'swap_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def detect_swaps_by_comparison(data: pd.DataFrame, ground_truth: pd.DataFrame) -> List[Tuple[int, int]]:
    """Detect frames where head and tail positions are swapped by comparing with ground truth.
    
    Args:
        data: DataFrame with tracking data to check
        ground_truth: DataFrame with ground truth data
        
    Returns:
        List of (start_frame, end_frame) tuples indicating swap regions
    """
    # Check if head and tail positions are swapped
    head_diff = (
        (data['X-Head'] != ground_truth['X-Head']) |
        (data['Y-Head'] != ground_truth['Y-Head'])
    )
    tail_diff = (
        (data['X-Tail'] != ground_truth['X-Tail']) |
        (data['Y-Tail'] != ground_truth['Y-Tail'])
    )
    
    # A swap occurs when both head and tail positions differ
    swaps = head_diff & tail_diff
    
    # Find continuous segments of swaps
    transitions = np.diff(swaps.astype(int))
    swap_starts = np.where(transitions == 1)[0] + 1
    swap_ends = np.where(transitions == -1)[0] + 1
    
    # Handle edge cases
    if swaps.iloc[0]:
        swap_starts = np.concatenate([[0], swap_starts])
    if swaps.iloc[-1]:
        swap_ends = np.concatenate([swap_ends, [len(swaps)]])
    
    return list(zip(swap_starts, swap_ends))

def analyze_dataset(data_dir: str) -> Dict:
    """Analyze swap correction performance for a single dataset.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Get base name from directory
        base_name = os.path.basename(data_dir)
        base_timestamp = base_name.split("_Biarmipes84")[0]
        
        # Find data files
        raw_file = os.path.join(data_dir, f"{base_timestamp}_data.csv")
        level1_file = os.path.join(data_dir, f"{base_timestamp}_data_level1.csv")
        level2_file = os.path.join(data_dir, f"{base_timestamp}_data_level2.csv")
        
        logging.info(f"Analyzing dataset: {base_name}")
        logging.info(f"Raw file: {raw_file}")
        logging.info(f"Level1 file: {level1_file}")
        logging.info(f"Level2 file: {level2_file}")
        
        # Load data files
        raw_data = pd.read_csv(raw_file)
        level1_data = pd.read_csv(level1_file)
        level2_data = pd.read_csv(level2_file)  # Ground truth
        
        # Normalize column names
        raw_data = normalize_column_names(raw_data)
        level1_data = normalize_column_names(level1_data)
        level2_data = normalize_column_names(level2_data)
        
        # Detect swaps by comparing with ground truth
        raw_swaps = detect_swaps_by_comparison(raw_data, level2_data)
        level1_swaps = detect_swaps_by_comparison(level1_data, level2_data)
        
        # Calculate metrics
        raw_metrics = {
            'num_swaps': len(raw_swaps),
            'swap_frames': sum(end - start for start, end in raw_swaps),
            'swap_regions': raw_swaps
        }
        
        level1_metrics = {
            'num_swaps': len(level1_swaps),
            'swap_frames': sum(end - start for start, end in level1_swaps),
            'swap_regions': level1_swaps
        }
        
        return {
            'dataset': base_name,
            'raw_metrics': raw_metrics,
            'level1_metrics': level1_metrics
        }
        
    except Exception as e:
        logging.error(f"Error analyzing dataset {data_dir}: {str(e)}")
        return None

def main():
    """Main function to analyze all datasets."""
    # Setup output directory
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    # Find all data directories
    data_dirs = [d for d in glob.glob("data/*") if os.path.isdir(d)]
    logging.info(f"Found {len(data_dirs)} datasets to analyze")
    
    # Analyze each dataset
    results = []
    for data_dir in data_dirs:
        result = analyze_dataset(data_dir)
        if result:
            results.append(result)
    
    # Aggregate results
    if results:
        # Calculate average metrics across all datasets
        avg_metrics = {
            'raw_num_swaps': np.mean([r['raw_metrics']['num_swaps'] for r in results]),
            'raw_swap_frames': np.mean([r['raw_metrics']['swap_frames'] for r in results]),
            'level1_num_swaps': np.mean([r['level1_metrics']['num_swaps'] for r in results]),
            'level1_swap_frames': np.mean([r['level1_metrics']['swap_frames'] for r in results])
        }
        
        # Save results
        output_file = os.path.join(output_dir, f"swap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(output_file, 'w') as f:
            f.write("Swap Correction Analysis Results\n")
            f.write("==============================\n\n")
            f.write(f"Number of datasets analyzed: {len(results)}\n\n")
            f.write("Average Metrics Across All Datasets:\n")
            for metric, value in avg_metrics.items():
                f.write(f"{metric}: {value:.2f}\n")
            
            f.write("\nDetailed Results by Dataset:\n")
            for result in results:
                f.write(f"\nDataset: {result['dataset']}\n")
                f.write("Raw Data vs Ground Truth:\n")
                f.write(f"  Number of swaps: {result['raw_metrics']['num_swaps']}\n")
                f.write(f"  Total frames with swaps: {result['raw_metrics']['swap_frames']}\n")
                f.write(f"  Swap regions: {result['raw_metrics']['swap_regions']}\n")
                f.write("Level1 Data vs Ground Truth:\n")
                f.write(f"  Number of swaps: {result['level1_metrics']['num_swaps']}\n")
                f.write(f"  Total frames with swaps: {result['level1_metrics']['swap_frames']}\n")
                f.write(f"  Swap regions: {result['level1_metrics']['swap_regions']}\n")
        
        logging.info(f"Analysis results saved to {output_file}")
    else:
        logging.error("No valid results to analyze")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""Script for profiling and optimizing the swap correction pipeline."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from swap_corrector.config import SwapConfig, SwapCorrectionConfig
from swap_corrector.profiling import PerformanceProfiler
from swap_corrector.processor import SwapProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile and optimize swap correction pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing experimental data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_results",
        help="Directory to save profiling results"
    )
    parser.add_argument(
        "--n-experiments",
        type=int,
        default=5,
        help="Number of experiments to profile"
    )
    return parser.parse_args()

def load_experiment_data(data_dir: Path, n_experiments: int) -> list:
    """Load experimental data for profiling.
    
    Args:
        data_dir: Directory containing experiment data
        n_experiments: Number of experiments to load
        
    Returns:
        List of DataFrames containing tracking data
    """
    experiments = []
    
    # Get experiment directories
    exp_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    exp_dirs = exp_dirs[:n_experiments]
    
    for exp_dir in exp_dirs:
        # Load raw data
        timestamp = exp_dir.name[:19]  # YYYY.MM.DD_HH-MM-SS
        data_path = exp_dir / f"{timestamp}_data.csv"
        
        if data_path.exists():
            data = pd.read_csv(data_path)
            experiments.append(data)
    
    return experiments

def plot_timing_results(
    timing_results: dict,
    output_dir: Path,
    show_plots: bool = False
):
    """Plot timing analysis results.
    
    Args:
        timing_results: Dictionary of timing results
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot total time by component
    components = list(timing_results.keys())
    times = [np.mean(timing_results[c]) for c in components]
    
    plt.bar(components, times)
    plt.xticks(rotation=45)
    plt.ylabel('Average Time (seconds)')
    plt.title('Processing Time by Component')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'timing_analysis.png')
    if show_plots:
        plt.show()
    plt.close()

def plot_memory_usage(
    memory_results: dict,
    output_dir: Path,
    show_plots: bool = False
):
    """Plot memory usage analysis results.
    
    Args:
        memory_results: Dictionary of memory usage results
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot peak memory by component
    components = list(memory_results.keys())
    peak_memory = [np.max(memory_results[c]) for c in components]
    
    plt.bar(components, peak_memory)
    plt.xticks(rotation=45)
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage by Component')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_analysis.png')
    if show_plots:
        plt.show()
    plt.close()

def plot_scaling_analysis(
    timing_results: dict,
    data_sizes: list,
    output_dir: Path,
    show_plots: bool = False
):
    """Plot scaling analysis results.
    
    Args:
        timing_results: Dictionary of timing results
        data_sizes: List of data sizes
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot processing time vs data size
    for component in timing_results:
        times = timing_results[component]
        plt.plot(data_sizes, times, 'o-', label=component)
    
    plt.xlabel('Data Size (frames)')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time vs Data Size')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_analysis.png')
    if show_plots:
        plt.show()
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load experimental data
    data_dir = Path(args.data_dir)
    experiments = load_experiment_data(data_dir, args.n_experiments)
    
    if not experiments:
        print("No experimental data found.")
        return
    
    print(f"Loaded {len(experiments)} experiments for profiling")
    
    # Initialize profiler
    profiler = PerformanceProfiler(output_dir)
    
    # Profile each experiment
    timing_results = {}
    memory_results = {}
    data_sizes = []
    
    for i, data in enumerate(experiments):
        print(f"\nProfiling experiment {i+1}/{len(experiments)}")
        print(f"Data size: {len(data)} frames")
        
        # Profile complete pipeline
        processed_data, results = profiler.profile_pipeline(data)
        
        # Store results
        data_sizes.append(len(data))
        for component, metrics in results['timing'].items():
            if component not in timing_results:
                timing_results[component] = []
            timing_results[component].append(metrics['total_time'])
        
        for component, metrics in results['memory'].items():
            if component not in memory_results:
                memory_results[component] = []
            memory_results[component].append(metrics['peak_usage'])
    
    # Analyze bottlenecks
    print("\nAnalyzing bottlenecks...")
    bottleneck_analysis = profiler.analyze_bottlenecks()
    
    # Generate optimization report
    print("Generating optimization report...")
    profiler.generate_optimization_report(bottleneck_analysis)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_timing_results(timing_results, output_dir)
    plot_memory_usage(memory_results, output_dir)
    plot_scaling_analysis(timing_results, data_sizes, output_dir)
    
    print(f"\nProfiling results saved to {output_dir}")

if __name__ == "__main__":
    main() 
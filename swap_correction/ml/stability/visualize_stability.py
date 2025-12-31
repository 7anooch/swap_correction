"""
Visualize stability analysis results.

Generates plots showing performance distributions, trajectories, and comparisons.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List
from swap_correction.ml.stability.aggregate_results import extract_metrics, calculate_stability_metrics


def plot_performance_distributions(results: List[Dict], output_dir: str):
    """
    Plot performance distributions across iterations.
    
    Parameters:
    -----------
    results : list of dict
        List of iteration results
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    level1_metrics = extract_metrics(results, model_type='level1')
    raw_metrics = extract_metrics(results, model_type='raw')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Distributions Across Iterations', fontsize=16)
    
    # F1-Score distributions
    ax = axes[0, 0]
    if len(level1_metrics) > 0:
        ax.boxplot([level1_metrics['test_f1'].dropna()], labels=['Level1'], positions=[1])
    if len(raw_metrics) > 0:
        ax.boxplot([raw_metrics['test_f1'].dropna()], labels=['Raw'], positions=[2])
    ax.set_ylabel('F1-Score')
    ax.set_title('Test F1-Score Distribution')
    ax.grid(True, alpha=0.3)
    
    # Precision distributions
    ax = axes[0, 1]
    if len(level1_metrics) > 0:
        ax.boxplot([level1_metrics['test_precision'].dropna()], labels=['Level1'], positions=[1])
    if len(raw_metrics) > 0:
        ax.boxplot([raw_metrics['test_precision'].dropna()], labels=['Raw'], positions=[2])
    ax.set_ylabel('Precision')
    ax.set_title('Test Precision Distribution')
    ax.grid(True, alpha=0.3)
    
    # Recall distributions
    ax = axes[1, 0]
    if len(level1_metrics) > 0:
        ax.boxplot([level1_metrics['test_recall'].dropna()], labels=['Level1'], positions=[1])
    if len(raw_metrics) > 0:
        ax.boxplot([raw_metrics['test_recall'].dropna()], labels=['Raw'], positions=[2])
    ax.set_ylabel('Recall')
    ax.set_title('Test Recall Distribution')
    ax.grid(True, alpha=0.3)
    
    # AUC distributions
    ax = axes[1, 1]
    if len(level1_metrics) > 0:
        ax.boxplot([level1_metrics['test_auc'].dropna()], labels=['Level1'], positions=[1])
    if len(raw_metrics) > 0:
        ax.boxplot([raw_metrics['test_auc'].dropna()], labels=['Raw'], positions=[2])
    ax.set_ylabel('AUC')
    ax.set_title('Test AUC Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'performance_distributions.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {plot_file}")


def plot_performance_trajectories(results: List[Dict], output_dir: str):
    """
    Plot performance trajectories across iterations.
    
    Parameters:
    -----------
    results : list of dict
        List of iteration results
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    level1_metrics = extract_metrics(results, model_type='level1')
    raw_metrics = extract_metrics(results, model_type='raw')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Trajectories Across Iterations', fontsize=16)
    
    # F1-Score trajectories
    ax = axes[0, 0]
    if len(level1_metrics) > 0:
        ax.plot(level1_metrics['iteration'], level1_metrics['test_f1'], 
               'o-', label='Level1', alpha=0.7)
    if len(raw_metrics) > 0:
        ax.plot(raw_metrics['iteration'], raw_metrics['test_f1'],
               's-', label='Raw', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('F1-Score')
    ax.set_title('Test F1-Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision trajectories
    ax = axes[0, 1]
    if len(level1_metrics) > 0:
        ax.plot(level1_metrics['iteration'], level1_metrics['test_precision'],
               'o-', label='Level1', alpha=0.7)
    if len(raw_metrics) > 0:
        ax.plot(raw_metrics['iteration'], raw_metrics['test_precision'],
               's-', label='Raw', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Precision')
    ax.set_title('Test Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall trajectories
    ax = axes[1, 0]
    if len(level1_metrics) > 0:
        ax.plot(level1_metrics['iteration'], level1_metrics['test_recall'],
               'o-', label='Level1', alpha=0.7)
    if len(raw_metrics) > 0:
        ax.plot(raw_metrics['iteration'], raw_metrics['test_recall'],
               's-', label='Raw', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Recall')
    ax.set_title('Test Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUC trajectories
    ax = axes[1, 1]
    if len(level1_metrics) > 0:
        ax.plot(level1_metrics['iteration'], level1_metrics['test_auc'],
               'o-', label='Level1', alpha=0.7)
    if len(raw_metrics) > 0:
        ax.plot(raw_metrics['iteration'], raw_metrics['test_auc'],
               's-', label='Raw', alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('AUC')
    ax.set_title('Test AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'performance_trajectories.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {plot_file}")


def plot_stability_comparison(results: List[Dict], output_dir: str):
    """
    Plot stability comparison between models.
    
    Parameters:
    -----------
    results : list of dict
        List of iteration results
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    level1_metrics = extract_metrics(results, model_type='level1')
    raw_metrics = extract_metrics(results, model_type='raw')
    
    level1_stability = calculate_stability_metrics(level1_metrics) if len(level1_metrics) > 0 else {}
    raw_stability = calculate_stability_metrics(raw_metrics) if len(raw_metrics) > 0 else {}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Stability Comparison', fontsize=16)
    
    # Coefficient of Variation
    ax = axes[0]
    models = []
    cvs = []
    
    if 'test_f1' in level1_stability and not np.isnan(level1_stability['test_f1']['cv']):
        models.append('Level1')
        cvs.append(level1_stability['test_f1']['cv'])
    
    if 'test_f1' in raw_stability and not np.isnan(raw_stability['test_f1']['cv']):
        models.append('Raw')
        cvs.append(raw_stability['test_f1']['cv'])
    
    if models:
        ax.bar(models, cvs, alpha=0.7)
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('F1-Score Stability (Lower = More Stable)')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Standard Deviation
    ax = axes[1]
    models = []
    stds = []
    
    if 'test_f1' in level1_stability:
        models.append('Level1')
        stds.append(level1_stability['test_f1']['std'])
    
    if 'test_f1' in raw_stability:
        models.append('Raw')
        stds.append(raw_stability['test_f1']['std'])
    
    if models:
        ax.bar(models, stds, alpha=0.7)
        ax.set_ylabel('Standard Deviation')
        ax.set_title('F1-Score Variability')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'stability_comparison.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {plot_file}")


def generate_all_visualizations(results_dir: str, output_dir: str = None):
    """
    Generate all stability visualizations.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing iteration results
    output_dir : str, optional
        Output directory for plots (default: results_dir/figures)
    """
    from swap_correction.ml.stability.aggregate_results import load_iteration_results
    
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'figures')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find iteration directories
    from pathlib import Path
    results_path = Path(results_dir)
    iteration_dirs = [
        str(d) for d in results_path.iterdir()
        if d.is_dir() and d.name.startswith('iteration_')
    ]
    iteration_dirs.sort()
    
    # Load results
    all_results = load_iteration_results(iteration_dirs)
    
    print(f"Generating visualizations for {len(all_results)} iterations...")
    
    # Generate plots
    plot_performance_distributions(all_results, output_dir)
    plot_performance_trajectories(all_results, output_dir)
    plot_stability_comparison(all_results, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate stability visualizations')
    parser.add_argument('results_dir', type=str,
                       help='Directory containing iteration results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: results_dir/figures)')
    
    args = parser.parse_args()
    
    generate_all_visualizations(args.results_dir, args.output_dir)


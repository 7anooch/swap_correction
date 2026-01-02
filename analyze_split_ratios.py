#!/usr/bin/env python3
"""
Analyze optimal train/validation/test split ratios.

This script trains models with different train/val/test split ratios to determine
which ratio provides the best performance. This is useful for:
- Determining if more training data improves performance
- Finding optimal validation set size for early stopping
- Understanding data efficiency

Alternatively, this can perform a learning curve analysis by training on
different amounts of training data while keeping split ratios constant.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from swap_correction.ml.training.train_model import (
    load_training_data, prepare_train_val_test_split, train_xgboost_model, evaluate_model
)
from swap_correction.ml.training.prepare_data import create_train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def train_with_split_ratio(features_df, labels, trial_names, 
                           train_ratio, val_ratio, test_ratio,
                           model_type='level1'):
    """
    Train a model with a specific train/val/test split ratio.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix
    labels : pd.Series
        Binary labels
    trial_names : list
        List of trial names for each sample
    train_ratio : float
        Proportion for training set
    val_ratio : float
        Proportion for validation set
    test_ratio : float
        Proportion for test set
    model_type : str
        Model type identifier (for logging)
        
    Returns:
    --------
    dict
        Dictionary with split sizes, training metrics, and evaluation metrics
    """
    # Create split based on trials (stratified by trial, not by frame)
    unique_trials = pd.Series(trial_names).unique()
    split = create_train_test_split(
        unique_trials.tolist(),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=42  # Fixed seed for reproducibility
    )
    
    # Map trials to splits
    trial_to_split = {}
    for split_name, trials in split.items():
        for trial in trials:
            trial_to_split[trial] = split_name
    
    # Assign each frame to a split based on its trial
    frame_splits = [trial_to_split[trial] for trial in trial_names]
    
    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        features_df, labels, trial_names, {'train': split['train'], 'val': split['val'], 'test': split['test']}
    )
    
    # Preprocess
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model, _ = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate
    results = evaluate_model(model, X_train_scaled, y_train, 
                            X_val_scaled, y_val, 
                            X_test_scaled, y_test)
    
    return {
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'split_sizes': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test),
            'train_swapped': int(y_train.sum()),
            'val_swapped': int(y_val.sum()),
            'test_swapped': int(y_test.sum())
        },
        'results': results
    }


def learning_curve_analysis(features_df, labels, trial_names,
                            base_train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                            train_sizes=None, model_type='level1'):
    """
    Perform learning curve analysis by training on different amounts of training data.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix
    labels : pd.Series
        Binary labels
    trial_names : list
        List of trial names
    base_train_ratio : float
        Base training ratio (used to determine max training size)
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    train_sizes : list of float, optional
        List of training set sizes as fractions (e.g., [0.1, 0.2, ..., 1.0])
        If None, uses [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_type : str
        Model type identifier
        
    Returns:
    --------
    pd.DataFrame
        Results for each training size
    """
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    all_results = []
    
    print("=" * 80)
    print("LEARNING CURVE ANALYSIS")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Training sizes to test: {train_sizes}")
    print()
    
    # First, create the base split (this will be our "full" training set)
    print("Creating base train/val/test split...")
    unique_trials = pd.Series(trial_names).unique()
    base_split = create_train_test_split(
        unique_trials.tolist(),
        train_ratio=base_train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=42
    )
    
    # Map trials to splits
    trial_to_split = {}
    for split_name, trials in base_split.items():
        for trial in trials:
            trial_to_split[trial] = split_name
    
    # Prepare the full splits
    X_train_full, X_val, X_test, y_train_full, y_val, y_test = prepare_train_val_test_split(
        features_df, labels, trial_names, base_split
    )
    
    # Preprocess validation and test sets once (they stay fixed)
    imputer = SimpleImputer(strategy='median')
    X_val_imputed = imputer.fit_transform(X_train_full)  # Fit on full training set
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Calculate class weight ratio from full training set
    n_positive = y_train_full.sum()
    n_negative = len(y_train_full) - n_positive
    class_weight_ratio = n_negative / n_positive if n_positive > 0 else 1.0
    
    print(f"Base split: Train={len(X_train_full)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"Class weight ratio: {class_weight_ratio:.2f}")
    print()
    
    # Now train on progressively larger subsets of the training set
    for train_size_frac in train_sizes:
        # Calculate how many samples to use from training set
        n_train_samples = int(len(X_train_full) * train_size_frac)
        
        print(f"Training with {train_size_frac*100:.0f}% of training data "
              f"({n_train_samples}/{len(X_train_full)} samples)...")
        
        try:
            # Take subset of training data
            X_train_subset = X_train_full[:n_train_samples]
            y_train_subset = y_train_full[:n_train_samples]
            
            # Preprocess training subset (fit imputer and scaler on subset)
            imputer_subset = SimpleImputer(strategy='median')
            X_train_imputed = imputer_subset.fit_transform(X_train_subset)
            X_val_imputed_subset = imputer_subset.transform(X_val)
            X_test_imputed_subset = imputer_subset.transform(X_test)
            
            scaler_subset = StandardScaler()
            X_train_scaled = scaler_subset.fit_transform(X_train_imputed)
            X_val_scaled_subset = scaler_subset.transform(X_val_imputed_subset)
            X_test_scaled_subset = scaler_subset.transform(X_test_imputed_subset)
            
            # Train model
            model, _ = train_xgboost_model(X_train_scaled, y_train_subset, 
                                         X_val_scaled_subset, y_val)
            
            # Evaluate
            results = evaluate_model(model, X_train_scaled, y_train_subset,
                                   X_val_scaled_subset, y_val,
                                   X_test_scaled_subset, y_test)
            
            result = {
                'split_ratios': {
                    'train': base_train_ratio * train_size_frac,
                    'val': val_ratio,
                    'test': test_ratio
                },
                'split_sizes': {
                    'train': len(X_train_subset),
                    'val': len(X_val),
                    'test': len(X_test),
                    'train_swapped': int(y_train_subset.sum()),
                    'val_swapped': int(y_val.sum()),
                    'test_swapped': int(y_test.sum())
                },
                'results': results
            }
            
            # Extract key metrics
            train_metrics = result['results']['train']
            val_metrics = result['results']['val']
            test_metrics = result['results']['test']
            
            all_results.append({
                'train_size_fraction': train_size_frac,
                'train_size': result['split_sizes']['train'],
                'val_size': result['split_sizes']['val'],
                'test_size': result['split_sizes']['test'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'test_f1': test_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'val_precision': val_metrics['precision'],
                'test_precision': test_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_recall': val_metrics['recall'],
                'test_recall': test_metrics['recall'],
                'train_pct_clean_post': ((train_metrics['confusion_matrix'][0,0] + train_metrics['confusion_matrix'][1,1]) / 
                                         train_metrics['n_samples']) * 100,
                'val_pct_clean_post': ((val_metrics['confusion_matrix'][0,0] + val_metrics['confusion_matrix'][1,1]) / 
                                      val_metrics['n_samples']) * 100,
                'test_pct_clean_post': ((test_metrics['confusion_matrix'][0,0] + test_metrics['confusion_matrix'][1,1]) / 
                                        test_metrics['n_samples']) * 100,
            })
            
            print(f"  ✓ Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                  f"Test F1: {test_metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    return pd.DataFrame(all_results)


def split_ratio_optimization(features_df, labels, trial_names,
                            split_configs=None, model_type='level1'):
    """
    Test different train/val/test split ratios to find optimal configuration.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Feature matrix
    labels : pd.Series
        Binary labels
    trial_names : list
        List of trial names
    split_configs : list of dict, optional
        List of split configurations, each with 'train', 'val', 'test' ratios
        If None, uses common configurations
    model_type : str
        Model type identifier
        
    Returns:
    --------
    pd.DataFrame
        Results for each split configuration
    """
    if split_configs is None:
        split_configs = [
            {'train': 0.6, 'val': 0.2, 'test': 0.2},
            {'train': 0.65, 'val': 0.175, 'test': 0.175},
            {'train': 0.7, 'val': 0.15, 'test': 0.15},
            {'train': 0.75, 'val': 0.125, 'test': 0.125},
            {'train': 0.8, 'val': 0.1, 'test': 0.1},
            {'train': 0.85, 'val': 0.075, 'test': 0.075},
        ]
    
    all_results = []
    
    print("=" * 80)
    print("SPLIT RATIO OPTIMIZATION")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Split configurations to test: {len(split_configs)}")
    print()
    
    for i, config in enumerate(split_configs):
        train_ratio = config['train']
        val_ratio = config['val']
        test_ratio = config['test']
        
        print(f"[{i+1}/{len(split_configs)}] Testing split: "
              f"Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}...")
        
        try:
            result = train_with_split_ratio(
                features_df, labels, trial_names,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                model_type=model_type
            )
            
            # Extract key metrics
            train_metrics = result['results']['train']
            val_metrics = result['results']['val']
            test_metrics = result['results']['test']
            
            all_results.append({
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'train_size': result['split_sizes']['train'],
                'val_size': result['split_sizes']['val'],
                'test_size': result['split_sizes']['test'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'test_f1': test_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'val_precision': val_metrics['precision'],
                'test_precision': test_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_recall': val_metrics['recall'],
                'test_recall': test_metrics['recall'],
                'val_pct_clean_post': ((val_metrics['confusion_matrix'][0,0] + val_metrics['confusion_matrix'][1,1]) / 
                                      val_metrics['n_samples']) * 100,
                'test_pct_clean_post': ((test_metrics['confusion_matrix'][0,0] + test_metrics['confusion_matrix'][1,1]) / 
                                        test_metrics['n_samples']) * 100,
            })
            
            print(f"  ✓ Val F1: {val_metrics['f1']:.4f}, Test F1: {test_metrics['f1']:.4f}, "
                  f"Test % Clean: {all_results[-1]['test_pct_clean_post']:.2f}%")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(all_results)


def plot_learning_curves(results_df, output_path):
    """Plot learning curves showing performance vs training size."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learning Curve Analysis', fontsize=16, fontweight='bold')
    
    train_sizes = results_df['train_size']
    
    # F1-Score
    ax1 = axes[0, 0]
    ax1.plot(train_sizes, results_df['train_f1'], 'o-', label='Train', linewidth=2, markersize=6)
    ax1.plot(train_sizes, results_df['val_f1'], 's-', label='Validation', linewidth=2, markersize=6)
    ax1.plot(train_sizes, results_df['test_f1'], '^-', label='Test', linewidth=2, markersize=6)
    ax1.set_xlabel('Training Set Size', fontsize=11)
    ax1.set_ylabel('F1-Score', fontsize=11)
    ax1.set_title('F1-Score vs Training Size', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision
    ax2 = axes[0, 1]
    ax2.plot(train_sizes, results_df['train_precision'], 'o-', label='Train', linewidth=2, markersize=6)
    ax2.plot(train_sizes, results_df['val_precision'], 's-', label='Validation', linewidth=2, markersize=6)
    ax2.plot(train_sizes, results_df['test_precision'], '^-', label='Test', linewidth=2, markersize=6)
    ax2.set_xlabel('Training Set Size', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision vs Training Size', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Recall
    ax3 = axes[1, 0]
    ax3.plot(train_sizes, results_df['train_recall'], 'o-', label='Train', linewidth=2, markersize=6)
    ax3.plot(train_sizes, results_df['val_recall'], 's-', label='Validation', linewidth=2, markersize=6)
    ax3.plot(train_sizes, results_df['test_recall'], '^-', label='Test', linewidth=2, markersize=6)
    ax3.set_xlabel('Training Set Size', fontsize=11)
    ax3.set_ylabel('Recall', fontsize=11)
    ax3.set_title('Recall vs Training Size', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # % Frames Clean Post
    ax4 = axes[1, 1]
    ax4.plot(train_sizes, results_df['train_pct_clean_post'], 'o-', label='Train', linewidth=2, markersize=6)
    ax4.plot(train_sizes, results_df['val_pct_clean_post'], 's-', label='Validation', linewidth=2, markersize=6)
    ax4.plot(train_sizes, results_df['test_pct_clean_post'], '^-', label='Test', linewidth=2, markersize=6)
    ax4.set_xlabel('Training Set Size', fontsize=11)
    ax4.set_ylabel('% Frames Clean Post', fontsize=11)
    ax4.set_title('% Frames Clean Post vs Training Size', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Learning curves saved to: {output_path}")


def generate_summary_report(all_results, iteration, output_dir):
    """Generate a comprehensive summary report comparing all combinations."""
    report_lines = [
        "# Learning Curve Analysis: Comprehensive Summary",
        "",
        f"**Iteration**: {iteration:03d}",
        "",
        "## Overview",
        "",
        "This report compares learning curves across:",
        "- **Feature Versions**: v2 (46 features) vs v3 (39 features)",
        "- **Model Types**: Level1 vs Raw",
        "",
        "## Summary Statistics",
        ""
    ]
    
    # Create comparison table
    comparison_data = []
    
    for feature_version in ['v2', 'v3', 'v4']:
        if feature_version not in all_results:
            continue
        for model_type in ['level1', 'raw']:
            if model_type not in all_results[feature_version]:
                continue
            
            results_df = all_results[feature_version][model_type]
            
            # Find best performance
            best_test_f1_idx = results_df['test_f1'].idxmax()
            best_test_f1 = results_df.loc[best_test_f1_idx]
            
            # Calculate improvement from 50% to 100%
            # Find the row where train_size_fraction == 0.5
            f1_at_50 = None
            if 'train_size_fraction' in results_df.columns:
                row_50 = results_df[results_df['train_size_fraction'] == 0.5]
                if len(row_50) > 0:
                    f1_at_50 = row_50.iloc[0]['test_f1']
                else:
                    # Fallback: use middle index if 0.5 not found
                    mid_idx = len(results_df) // 2
                    f1_at_50 = results_df.iloc[mid_idx]['test_f1']
            else:
                # Fallback: use middle index
                mid_idx = len(results_df) // 2
                f1_at_50 = results_df.iloc[mid_idx]['test_f1']
            
            f1_at_100 = results_df.iloc[-1]['test_f1']
            improvement = f1_at_100 - f1_at_50
            
            # Check if still improving (last 20% = from 80% to 100%)
            # Find the row where train_size_fraction == 0.8
            f1_at_80 = None
            if 'train_size_fraction' in results_df.columns:
                row_80 = results_df[results_df['train_size_fraction'] == 0.8]
                if len(row_80) > 0:
                    f1_at_80 = row_80.iloc[0]['test_f1']
                else:
                    # Fallback: use 80% index
                    last_20_start = int(len(results_df) * 0.8)
                    f1_at_80 = results_df.iloc[last_20_start]['test_f1']
            else:
                # Fallback: use 80% index
                last_20_start = int(len(results_df) * 0.8)
                f1_at_80 = results_df.iloc[last_20_start]['test_f1']
            
            improvement_last_20 = results_df.iloc[-1]['test_f1'] - f1_at_80
            
            comparison_data.append({
                'Feature Version': feature_version.upper(),
                'Model Type': model_type.upper(),
                'Best Test F1': f"{best_test_f1['test_f1']:.4f}",
                'Best Test % Clean': f"{best_test_f1['test_pct_clean_post']:.2f}%",
                'F1 at 50% Data': f"{f1_at_50:.4f}",
                'F1 at 100% Data': f"{f1_at_100:.4f}",
                'Improvement (50→100%)': f"{improvement:.4f}",
                'Improvement (Last 20%)': f"{improvement_last_20:.4f}",
                'More Data Helpful': 'Yes' if improvement_last_20 > 0.005 else 'No'
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    report_lines.append(comparison_df.to_markdown(index=False))
    
    report_lines.extend([
        "",
        "## Detailed Results by Combination",
        ""
    ])
    
    # Add detailed results for each combination
    for feature_version in ['v2', 'v3', 'v4']:
        if feature_version not in all_results:
            continue
        for model_type in ['level1', 'raw']:
            if model_type not in all_results[feature_version]:
                continue
            
            results_df = all_results[feature_version][model_type]
            
            report_lines.extend([
                f"### {feature_version.upper()} Features, {model_type.upper()} Model",
                "",
                "| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |",
                "|---------------|----------|--------|---------|--------------|"
            ])
            
            for _, row in results_df.iterrows():
                report_lines.append(
                    f"| {row['train_size']} | {row['train_f1']:.4f} | {row['val_f1']:.4f} | "
                    f"{row['test_f1']:.4f} | {row['test_pct_clean_post']:.2f}% |"
                )
            
            report_lines.append("")
    
    # Create comparison plots
    create_comparison_plots(all_results, output_dir)
    
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "### Performance Comparison",
        "",
        "Compare the 'Best Test F1' and 'Best Test % Clean' across combinations to identify:",
        "- Which feature version performs better",
        "- Which model type performs better",
        "- Best overall combination",
        "",
        "### Data Efficiency",
        "",
        "Compare 'Improvement (50→100%)' and 'Improvement (Last 20%)' to determine:",
        "- Whether more training data would help",
        "- If performance has plateaued",
        "- Optimal training set size",
        "",
        "### Recommendations",
        "",
        "Based on the analysis:",
        "- If 'More Data Helpful' = Yes: Consider collecting more training data",
        "- If 'More Data Helpful' = No: Current dataset size is sufficient",
        "- Large gaps between train and validation suggest overfitting",
        "- Converging curves suggest model is reaching capacity",
    ])
    
    report_file = os.path.join(output_dir, 'summary_report.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n✓ Summary report saved to: {report_file}")


def create_comparison_plots(all_results, output_dir):
    """Create comparison plots across all combinations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Learning Curve Comparison: All Combinations', fontsize=16, fontweight='bold')
    
    colors = {'v2': '#2E86AB', 'v3': '#A23B72'}
    markers = {'level1': 'o', 'raw': 's'}
    linestyles = {'level1': '-', 'raw': '--'}
    
    # F1-Score comparison
    ax1 = axes[0, 0]
    for feature_version in ['v2', 'v3', 'v4']:
        if feature_version not in all_results:
            continue
        for model_type in ['level1', 'raw']:
            if model_type not in all_results[feature_version]:
                continue
            
            results_df = all_results[feature_version][model_type]
            label = f"{feature_version.upper()} {model_type.upper()}"
            ax1.plot(results_df['train_size'], results_df['test_f1'], 
                    marker=markers[model_type], linestyle=linestyles[model_type],
                    color=colors[feature_version], label=label, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Training Set Size', fontsize=11)
    ax1.set_ylabel('Test F1-Score', fontsize=11)
    ax1.set_title('Test F1-Score Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # % Clean Post comparison
    ax2 = axes[0, 1]
    for feature_version in ['v2', 'v3', 'v4']:
        if feature_version not in all_results:
            continue
        for model_type in ['level1', 'raw']:
            if model_type not in all_results[feature_version]:
                continue
            
            results_df = all_results[feature_version][model_type]
            label = f"{feature_version.upper()} {model_type.upper()}"
            ax2.plot(results_df['train_size'], results_df['test_pct_clean_post'], 
                    marker=markers[model_type], linestyle=linestyles[model_type],
                    color=colors[feature_version], label=label, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Training Set Size', fontsize=11)
    ax2.set_ylabel('Test % Frames Clean Post', fontsize=11)
    ax2.set_title('Test % Clean Post Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision comparison
    ax3 = axes[1, 0]
    for feature_version in ['v2', 'v3', 'v4']:
        if feature_version not in all_results:
            continue
        for model_type in ['level1', 'raw']:
            if model_type not in all_results[feature_version]:
                continue
            
            results_df = all_results[feature_version][model_type]
            label = f"{feature_version.upper()} {model_type.upper()}"
            ax3.plot(results_df['train_size'], results_df['test_precision'], 
                    marker=markers[model_type], linestyle=linestyles[model_type],
                    color=colors[feature_version], label=label, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Training Set Size', fontsize=11)
    ax3.set_ylabel('Test Precision', fontsize=11)
    ax3.set_title('Test Precision Comparison', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Recall comparison
    ax4 = axes[1, 1]
    for feature_version in ['v2', 'v3', 'v4']:
        if feature_version not in all_results:
            continue
        for model_type in ['level1', 'raw']:
            if model_type not in all_results[feature_version]:
                continue
            
            results_df = all_results[feature_version][model_type]
            label = f"{feature_version.upper()} {model_type.upper()}"
            ax4.plot(results_df['train_size'], results_df['test_recall'], 
                    marker=markers[model_type], linestyle=linestyles[model_type],
                    color=colors[feature_version], label=label, linewidth=2, markersize=6)
    
    ax4.set_xlabel('Training Set Size', fontsize=11)
    ax4.set_ylabel('Test Recall', fontsize=11)
    ax4.set_title('Test Recall Comparison', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'comparison_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plots saved to: {plot_file}")


def plot_split_ratio_comparison(results_df, output_path):
    """Plot comparison of different split ratios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Split Ratio Optimization', fontsize=16, fontweight='bold')
    
    train_ratios = results_df['train_ratio']
    x_pos = np.arange(len(results_df))
    
    # F1-Score
    ax1 = axes[0, 0]
    ax1.plot(x_pos, results_df['val_f1'], 's-', label='Validation', linewidth=2, markersize=8)
    ax1.plot(x_pos, results_df['test_f1'], '^-', label='Test', linewidth=2, markersize=8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{r:.0%}" for r in train_ratios], rotation=45, ha='right')
    ax1.set_xlabel('Training Ratio', fontsize=11)
    ax1.set_ylabel('F1-Score', fontsize=11)
    ax1.set_title('F1-Score vs Training Ratio', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # % Frames Clean Post
    ax2 = axes[0, 1]
    ax2.plot(x_pos, results_df['val_pct_clean_post'], 's-', label='Validation', linewidth=2, markersize=8)
    ax2.plot(x_pos, results_df['test_pct_clean_post'], '^-', label='Test', linewidth=2, markersize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{r:.0%}" for r in train_ratios], rotation=45, ha='right')
    ax2.set_xlabel('Training Ratio', fontsize=11)
    ax2.set_ylabel('% Frames Clean Post', fontsize=11)
    ax2.set_title('% Frames Clean Post vs Training Ratio', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Training set size
    ax3 = axes[1, 0]
    ax3.bar(x_pos, results_df['train_size'], alpha=0.7, color='#2E86AB', label='Train')
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x_pos, results_df['val_size'], alpha=0.7, color='#A23B72', label='Val', width=0.6)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{r:.0%}" for r in train_ratios], rotation=45, ha='right')
    ax3.set_xlabel('Training Ratio', fontsize=11)
    ax3.set_ylabel('Training Set Size', fontsize=11, color='#2E86AB')
    ax3_twin.set_ylabel('Validation Set Size', fontsize=11, color='#A23B72')
    ax3.set_title('Set Sizes vs Training Ratio', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Precision vs Recall
    ax4 = axes[1, 1]
    ax4.scatter(results_df['test_precision'], results_df['test_recall'], 
               s=100, alpha=0.6, c=train_ratios, cmap='viridis')
    for i, ratio in enumerate(train_ratios):
        ax4.annotate(f"{ratio:.0%}", 
                    (results_df['test_precision'].iloc[i], results_df['test_recall'].iloc[i]),
                    fontsize=9)
    ax4.set_xlabel('Test Precision', fontsize=11)
    ax4.set_ylabel('Test Recall', fontsize=11)
    ax4.set_title('Precision vs Recall (colored by train ratio)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Split ratio comparison saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze optimal train/val/test split ratios or learning curves')
    parser.add_argument('--analysis-type', type=str, default='learning_curve',
                       choices=['learning_curve', 'split_ratio'],
                       help='Type of analysis to perform')
    parser.add_argument('--iteration', type=int, default=7,
                       help='Iteration to use from stability analysis (default: 7)')
    parser.add_argument('--model-type', type=str, default='all',
                       choices=['level1', 'raw', 'all'],
                       help='Model type to analyze (all = both level1 and raw)')
    parser.add_argument('--feature-version', type=str, default='all',
                       choices=['v2', 'v3', 'v4', 'all'],
                       help='Feature version to use (all = v2, v3, and v4)')
    parser.add_argument('--base-dir', type=str, default='stability_analysis_v3_features_v3',
                       help='Base directory for stability analysis (will be modified for v2)')
    parser.add_argument('--output-dir', type=str, default='learning_curve_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine which combinations to run
    feature_versions = ['v2', 'v3', 'v4'] if args.feature_version == 'all' else [args.feature_version]
    model_types = ['level1', 'raw'] if args.model_type == 'all' else [args.model_type]
    
    print("=" * 80)
    print(f"{args.analysis_type.upper().replace('_', ' ')} ANALYSIS")
    print("=" * 80)
    print(f"Iteration: {args.iteration:03d}")
    print(f"Feature versions: {feature_versions}")
    print(f"Model types: {model_types}")
    print(f"Total combinations: {len(feature_versions) * len(model_types)}")
    print()
    
    # Get parent directory for test data
    main_dataset = '/Users/hind/Documents/UCSB/Neuroscience/KirstenData/new_data/Main_dataset'
    test_data_dir = main_dataset if os.path.exists(main_dataset) else None
    
    all_results = {}
    
    # Run analysis for each combination
    for feature_version in feature_versions:
        all_results[feature_version] = {}
        
        # Determine base directory
        if feature_version == 'v2':
            base_dir = 'stability_analysis_v3_features_v2'
        elif feature_version == 'v3':
            base_dir = 'stability_analysis_v3_features_v3'
        else:  # v4
            base_dir = 'stability_analysis_v3_features_v4'
        
        for model_type in model_types:
            print("\n" + "=" * 80)
            print(f"PROCESSING: Features {feature_version.upper()}, {model_type.upper()} Model")
            print("=" * 80)
            
            # Load data from the specified iteration
            iter_dir = os.path.join(base_dir, f'iteration_{args.iteration:03d}')
            model_dir = os.path.join(iter_dir, f'{model_type}_model')
            ml_data_dir = os.path.join(model_dir, 'ml_data')
            
            if not os.path.exists(ml_data_dir):
                print(f"⚠ Skipping: ML data directory not found: {ml_data_dir}")
                continue
            
            print("Loading training data...")
            
            # Load split info
            split_file = os.path.join(os.path.dirname(model_dir), 'trial_split.json')
            trial_dirs = None
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                trial_dirs = split_data.get('train', []) + split_data.get('val', []) + split_data.get('test', [])
            
            # Patch features (for legacy versions only; default is now v4)
            original_extract = None
            if feature_version == 'v4':
                # V4 is now the default, no patching needed
                pass
            elif feature_version == 'v3':
                import swap_correction.ml.features.legacy.features_v3 as features_module_to_use
                import swap_correction.ml.features as features_module
                import swap_correction.ml.training.train_model as train_model_module
                import sys
                import importlib
                
                original_extract = features_module.extract_all_frame_features_optimized
                features_module.extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
                sys.modules['swap_correction.ml.features'].extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
                importlib.reload(train_model_module)
            elif feature_version == 'v2':
                import swap_correction.ml.features.legacy.features_v2 as features_module_to_use
                import swap_correction.ml.features as features_module
                import swap_correction.ml.training.train_model as train_model_module
                import sys
                import importlib
                
                original_extract = features_module.extract_all_frame_features_optimized
                features_module.extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
                sys.modules['swap_correction.ml.features'].extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
                importlib.reload(train_model_module)
            
            try:
                features_df, labels, trial_names, split = load_training_data(
                    ml_data_dir=ml_data_dir,
                    test_data_dir=test_data_dir,
                    use_raw_data=(model_type == 'raw'),
                    trial_dirs=trial_dirs
                )
                print(f"✓ Loaded {len(features_df)} samples with {len(features_df.columns)} features")
            except Exception as e:
                print(f"✗ Error loading data: {e}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                if original_extract is not None:
                    import swap_correction.ml.features as features_module
                    import sys
                    import importlib
                    features_module.extract_all_frame_features_optimized = original_extract
                    sys.modules['swap_correction.ml.features'].extract_all_frame_features_optimized = original_extract
                    importlib.reload(train_model_module)
            
            # Create output directory for this combination
            combo_output_dir = os.path.join(args.output_dir, f'{feature_version}_{model_type}')
            os.makedirs(combo_output_dir, exist_ok=True)
            
            # Perform analysis
            if args.analysis_type == 'learning_curve':
                print("\nPerforming learning curve analysis...")
                results_df = learning_curve_analysis(
                    features_df, labels, trial_names,
                    base_train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                    model_type=f"{feature_version}_{model_type}"
                )
                
                # Save results
                results_file = os.path.join(combo_output_dir, 'learning_curve_results.csv')
                results_df.to_csv(results_file, index=False)
                print(f"\n✓ Results saved to: {results_file}")
                
                # Plot
                plot_file = os.path.join(combo_output_dir, 'learning_curves.png')
                plot_learning_curves(results_df, plot_file)
                
                # Store for summary
                all_results[feature_version][model_type] = results_df
                
            else:  # split_ratio
                print("\nPerforming split ratio optimization...")
                results_df = split_ratio_optimization(
                    features_df, labels, trial_names,
                    model_type=f"{feature_version}_{model_type}"
                )
                
                # Save results
                results_file = os.path.join(combo_output_dir, 'split_ratio_results.csv')
                results_df.to_csv(results_file, index=False)
                print(f"\n✓ Results saved to: {results_file}")
                
                # Plot
                plot_file = os.path.join(combo_output_dir, 'split_ratio_comparison.png')
                plot_split_ratio_comparison(results_df, plot_file)
                
                # Store for summary
                all_results[feature_version][model_type] = results_df
    
    # Generate comprehensive summary report
    if args.analysis_type == 'learning_curve' and all_results:
        generate_summary_report(all_results, args.iteration, args.output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


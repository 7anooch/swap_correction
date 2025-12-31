#!/usr/bin/env python3
"""
Generate learning curves for ML models.

Trains models on progressively larger subsets of training data to assess
if more data would improve performance or if performance has plateaued.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Import from train_ml_swap_detector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ml_swap_detector import load_training_data, get_test_data_path, prepare_train_val_test_split


def train_model_subset(X_train_subset, y_train_subset, X_val, y_val, 
                       class_weight_ratio: float = None, n_estimators: int = 200):
    """Train XGBoost model on a subset of training data."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': n_estimators,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': class_weight_ratio,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 20
    }
    
    sklearn_model = xgb.XGBClassifier(**params)
    sklearn_model.fit(
        X_train_subset, y_train_subset,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return sklearn_model


def evaluate_model_subset(model, X_val, y_val, X_test, y_test):
    """Evaluate model on validation and test sets."""
    # Validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_proba) if len(np.unique(y_val)) > 1 else 0.0
    
    # Test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) > 1 else 0.0
    
    return {
        'val': {
            'precision': float(val_precision),
            'recall': float(val_recall),
            'f1': float(val_f1),
            'auc': float(val_auc)
        },
        'test': {
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1': float(test_f1),
            'auc': float(test_auc)
        }
    }


def generate_learning_curves(use_raw_data: bool = False, 
                            training_sizes: list = None,
                            output_dir: str = 'ml_analysis'):
    """
    Generate learning curves by training on progressively larger subsets.
    
    Parameters:
    -----------
    use_raw_data : bool
        If True, use raw data model, else use level1 model
    training_sizes : list
        List of training set sizes as fractions (e.g., [0.1, 0.2, ..., 1.0])
    output_dir : str
        Directory to save results
    """
    if training_sizes is None:
        training_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    model_name = 'raw_data' if use_raw_data else 'level1'
    print("=" * 80)
    print(f"GENERATING LEARNING CURVES: {model_name.upper()} MODEL")
    print("=" * 80)
    
    # Load full training data
    print("\nLoading training data...")
    ml_data_dir = 'ml_data'
    test_data_dir = get_test_data_path()
    
    features_df, labels, trial_names, split = load_training_data(
        ml_data_dir, test_data_dir, use_raw_data=use_raw_data
    )
    
    # Prepare train/val/test split
    print("Preparing train/val/test split...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        features_df, labels, trial_names, split
    )
    
    # Handle NaN values
    print("Preprocessing data...")
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Calculate class weights
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    class_weight_ratio = n_negative / n_positive if n_positive > 0 else 1.0
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Class weight ratio: {class_weight_ratio:.2f}")
    print(f"\nTraining on progressively larger subsets...")
    
    # Store results
    results = []
    n_train_total = len(X_train)
    
    for size_frac in training_sizes:
        n_samples = int(n_train_total * size_frac)
        print(f"\n  Training on {n_samples} samples ({size_frac*100:.0f}% of training data)...", end=' ', flush=True)
        
        # Sample subset (stratified to maintain class balance)
        indices = np.arange(len(X_train))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        # Ensure class balance in subset
        positive_indices = indices[y_train[indices] == 1]
        negative_indices = indices[y_train[indices] == 0]
        
        n_positive_subset = int(n_samples * (n_positive / n_train_total))
        n_negative_subset = n_samples - n_positive_subset
        
        n_positive_subset = min(n_positive_subset, len(positive_indices))
        n_negative_subset = min(n_negative_subset, len(negative_indices))
        
        subset_indices = np.concatenate([
            positive_indices[:n_positive_subset],
            negative_indices[:n_negative_subset]
        ])
        np.random.shuffle(subset_indices)
        
        X_train_subset = X_train[subset_indices]
        y_train_subset = y_train[subset_indices]
        
        # Train model
        start_time = time.time()
        model = train_model_subset(X_train_subset, y_train_subset, X_val, y_val, 
                                   class_weight_ratio=class_weight_ratio)
        train_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_model_subset(model, X_val, y_val, X_test, y_test)
        
        result = {
            'training_size': n_samples,
            'training_fraction': size_frac,
            'train_time_seconds': train_time,
            **metrics
        }
        results.append(result)
        
        print(f"Val F1: {metrics['val']['f1']:.4f}, Test F1: {metrics['test']['f1']:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'learning_curves_{model_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    create_learning_curve_plot(results, model_name, output_dir)
    
    return results


def create_learning_curve_plot(results: list, model_name: str, output_dir: str):
    """Create learning curve visualization."""
    training_sizes = [r['training_size'] for r in results]
    training_fractions = [r['training_fraction'] for r in results]
    
    val_f1 = [r['val']['f1'] for r in results]
    test_f1 = [r['test']['f1'] for r in results]
    val_precision = [r['val']['precision'] for r in results]
    test_precision = [r['test']['precision'] for r in results]
    val_recall = [r['val']['recall'] for r in results]
    test_recall = [r['test']['recall'] for r in results]
    val_auc = [r['val']['auc'] for r in results]
    test_auc = [r['test']['auc'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Learning Curves: {model_name.replace("_", " ").title()} Model', 
                 fontsize=16, fontweight='bold')
    
    # F1 Score
    ax = axes[0, 0]
    ax.plot(training_sizes, val_f1, 'o-', label='Validation', linewidth=2, markersize=6)
    ax.plot(training_sizes, test_f1, 's-', label='Test', linewidth=2, markersize=6)
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score Learning Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes[0, 1]
    ax.plot(training_sizes, val_precision, 'o-', label='Validation', linewidth=2, markersize=6)
    ax.plot(training_sizes, test_precision, 's-', label='Test', linewidth=2, markersize=6)
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision Learning Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes[1, 0]
    ax.plot(training_sizes, val_recall, 'o-', label='Validation', linewidth=2, markersize=6)
    ax.plot(training_sizes, test_recall, 's-', label='Test', linewidth=2, markersize=6)
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Recall Learning Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # AUC
    ax = axes[1, 1]
    ax.plot(training_sizes, val_auc, 'o-', label='Validation', linewidth=2, markersize=6)
    ax.plot(training_sizes, test_auc, 's-', label='Test', linewidth=2, markersize=6)
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('ROC-AUC', fontsize=11)
    ax.set_title('ROC-AUC Learning Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'learning_curves_{model_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curve plot saved to: {output_path}")


def analyze_learning_curves(results: list, model_name: str):
    """Analyze learning curves to determine if more data would help."""
    print("\n" + "=" * 80)
    print(f"LEARNING CURVE ANALYSIS: {model_name.upper()} MODEL")
    print("=" * 80)
    
    # Check if performance plateaus
    test_f1 = [r['test']['f1'] for r in results]
    training_sizes = [r['training_size'] for r in results]
    
    # Calculate improvement from 50% to 100%
    mid_idx = len(results) // 2
    f1_at_50 = test_f1[mid_idx]
    f1_at_100 = test_f1[-1]
    improvement = f1_at_100 - f1_at_50
    
    # Check if performance is still improving in last 20% of data
    last_20_start = int(len(results) * 0.8)
    f1_last_20_start = test_f1[last_20_start]
    f1_last_20_end = test_f1[-1]
    improvement_last_20 = f1_last_20_end - f1_last_20_start
    
    print(f"\nPerformance Analysis:")
    print(f"  F1 at 50% data: {f1_at_50:.4f}")
    print(f"  F1 at 100% data: {f1_at_100:.4f}")
    print(f"  Improvement (50% → 100%): {improvement:.4f}")
    print(f"  Improvement (last 20%): {improvement_last_20:.4f}")
    
    # Determine if more data would help
    if improvement_last_20 < 0.001:
        print(f"\nConclusion: Performance has PLATEAUED")
        print(f"  - Minimal improvement in last 20% of data ({improvement_last_20:.4f})")
        print(f"  - More data is UNLIKELY to significantly improve performance")
    elif improvement_last_20 < 0.005:
        print(f"\nConclusion: Performance is SLOWLY IMPROVING")
        print(f"  - Small improvement in last 20% of data ({improvement_last_20:.4f})")
        print(f"  - More data may provide MARGINAL improvements")
    else:
        print(f"\nConclusion: Performance is STILL IMPROVING")
        print(f"  - Significant improvement in last 20% of data ({improvement_last_20:.4f})")
        print(f"  - More data is LIKELY to improve performance")
    
    return {
        'f1_at_50': f1_at_50,
        'f1_at_100': f1_at_100,
        'improvement_50_to_100': improvement,
        'improvement_last_20': improvement_last_20,
        'plateaued': improvement_last_20 < 0.001,
        'more_data_helpful': improvement_last_20 > 0.005
    }


def main():
    """Main function to generate learning curves for both models."""
    output_dir = 'ml_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate learning curves for both models
    all_results = {}
    
    # Level1 model
    print("\n" + "=" * 80)
    level1_results = generate_learning_curves(use_raw_data=False, output_dir=output_dir)
    level1_analysis = analyze_learning_curves(level1_results, 'level1')
    all_results['level1'] = {
        'results': level1_results,
        'analysis': level1_analysis
    }
    
    # Raw data model
    print("\n" + "=" * 80)
    raw_results = generate_learning_curves(use_raw_data=True, output_dir=output_dir)
    raw_analysis = analyze_learning_curves(raw_results, 'raw_data')
    all_results['raw_data'] = {
        'results': raw_results,
        'analysis': raw_analysis
    }
    
    # Save combined results
    combined_file = os.path.join(output_dir, 'learning_curves_data.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nCombined learning curve data saved to: {combined_file}")
    print("\n" + "=" * 80)
    print("LEARNING CURVE GENERATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


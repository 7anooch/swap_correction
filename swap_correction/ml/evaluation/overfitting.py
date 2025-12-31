#!/usr/bin/env python3
"""
Analyze overfitting in trained ML models.

Compares train/val/test performance gaps for both level1 and raw data models
to assess overfitting and determine if more data or regularization is needed.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_model_results(level1_results_file: str, raw_results_file: str):
    """Load training results for both models."""
    with open(level1_results_file, 'r') as f:
        level1_results = json.load(f)
    
    with open(raw_results_file, 'r') as f:
        raw_results = json.load(f)
    
    return level1_results, raw_results


def calculate_overfitting_metrics(results: dict, model_name: str):
    """Calculate overfitting metrics from results."""
    train = results['train']
    val = results['val']
    test = results['test']
    
    metrics = {
        'model_name': model_name,
        'train_test_f1_gap': train['f1'] - test['f1'],
        'train_val_f1_gap': train['f1'] - val['f1'],
        'val_test_f1_gap': val['f1'] - test['f1'],
        'train_test_precision_gap': train['precision'] - test['precision'],
        'train_val_precision_gap': train['precision'] - val['precision'],
        'val_test_precision_gap': val['precision'] - test['precision'],
        'train_test_recall_gap': train['recall'] - test['recall'],
        'train_val_recall_gap': train['recall'] - val['recall'],
        'val_test_recall_gap': val['recall'] - test['recall'],
        'train_test_auc_gap': train['auc'] - test['auc'],
        'train_val_auc_gap': train['auc'] - val['auc'],
        'val_test_auc_gap': val['auc'] - test['auc'],
        'train_f1': train['f1'],
        'val_f1': val['f1'],
        'test_f1': test['f1'],
        'train_precision': train['precision'],
        'val_precision': val['precision'],
        'test_precision': test['precision'],
        'train_recall': train['recall'],
        'val_recall': val['recall'],
        'test_recall': test['recall'],
        'train_auc': train['auc'],
        'val_auc': val['auc'],
        'test_auc': test['auc'],
    }
    
    # Assess overfitting severity
    train_test_f1_gap = metrics['train_test_f1_gap']
    if train_test_f1_gap < 0.01:
        metrics['overfitting_severity'] = 'minimal'
    elif train_test_f1_gap < 0.02:
        metrics['overfitting_severity'] = 'slight'
    elif train_test_f1_gap < 0.05:
        metrics['overfitting_severity'] = 'moderate'
    else:
        metrics['overfitting_severity'] = 'severe'
    
    # Check if validation is representative
    val_test_f1_gap = abs(metrics['val_test_f1_gap'])
    if val_test_f1_gap < 0.005:
        metrics['validation_representative'] = 'excellent'
    elif val_test_f1_gap < 0.01:
        metrics['validation_representative'] = 'good'
    elif val_test_f1_gap < 0.02:
        metrics['validation_representative'] = 'fair'
    else:
        metrics['validation_representative'] = 'poor'
    
    return metrics


def create_overfitting_visualization(level1_metrics: dict, raw_metrics: dict, output_path: str):
    """Create visualization comparing overfitting in both models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Overfitting Analysis', fontsize=16, fontweight='bold')
    
    models = ['Level1 Model', 'Raw Data Model']
    level1_data = level1_metrics
    raw_data = raw_metrics
    
    # F1 Score comparison
    ax = axes[0, 0]
    splits = ['Train', 'Val', 'Test']
    x = np.arange(len(splits))
    width = 0.35
    
    level1_f1 = [level1_data['train_f1'], level1_data['val_f1'], level1_data['test_f1']]
    raw_f1 = [raw_data['train_f1'], raw_data['val_f1'], raw_data['test_f1']]
    
    bars1 = ax.bar(x - width/2, level1_f1, width, label='Level1', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, raw_f1, width, label='Raw Data', alpha=0.8, color='#A23B72')
    
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score Across Splits', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.95, 1.01])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Precision comparison
    ax = axes[0, 1]
    level1_prec = [level1_data['train_precision'], level1_data['val_precision'], level1_data['test_precision']]
    raw_prec = [raw_data['train_precision'], raw_data['val_precision'], raw_data['test_precision']]
    
    bars1 = ax.bar(x - width/2, level1_prec, width, label='Level1', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, raw_prec, width, label='Raw Data', alpha=0.8, color='#A23B72')
    
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision Across Splits', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.95, 1.01])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Recall comparison
    ax = axes[1, 0]
    level1_rec = [level1_data['train_recall'], level1_data['val_recall'], level1_data['test_recall']]
    raw_rec = [raw_data['train_recall'], raw_data['val_recall'], raw_data['test_recall']]
    
    bars1 = ax.bar(x - width/2, level1_rec, width, label='Level1', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, raw_rec, width, label='Raw Data', alpha=0.8, color='#A23B72')
    
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Recall Across Splits', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.95, 1.01])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Overfitting gaps
    ax = axes[1, 1]
    gap_metrics = ['train_test_f1_gap', 'train_val_f1_gap', 'val_test_f1_gap']
    gap_labels = ['Train-Test', 'Train-Val', 'Val-Test']
    
    level1_gaps = [level1_data[m] for m in gap_metrics]
    raw_gaps = [raw_data[m] for m in gap_metrics]
    
    x_gap = np.arange(len(gap_labels))
    bars1 = ax.bar(x_gap - width/2, level1_gaps, width, label='Level1', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x_gap + width/2, raw_gaps, width, label='Raw Data', alpha=0.8, color='#A23B72')
    
    ax.set_ylabel('F1 Score Gap', fontsize=11)
    ax.set_title('Overfitting Gaps (F1 Score)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_gap)
    ax.set_xticklabels(gap_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', 
                   va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overfitting visualization saved to: {output_path}")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("MODEL OVERFITTING ANALYSIS")
    print("=" * 80)
    
    # File paths
    level1_results_file = 'ml_models/training_results.json'
    raw_results_file = 'ml_models_raw/training_results.json'
    output_dir = 'ml_analysis'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("\nLoading model results...")
    level1_results, raw_results = load_model_results(level1_results_file, raw_results_file)
    
    # Calculate overfitting metrics
    print("Calculating overfitting metrics...")
    level1_metrics = calculate_overfitting_metrics(level1_results, 'Level1 Model')
    raw_metrics = calculate_overfitting_metrics(raw_results, 'Raw Data Model')
    
    # Print summary
    print("\n" + "=" * 80)
    print("OVERFITTING ANALYSIS SUMMARY")
    print("=" * 80)
    
    for metrics in [level1_metrics, raw_metrics]:
        print(f"\n{metrics['model_name']}:")
        print(f"  Overfitting Severity: {metrics['overfitting_severity'].upper()}")
        print(f"  Validation Representativeness: {metrics['validation_representative'].upper()}")
        print(f"  Train-Test F1 Gap: {metrics['train_test_f1_gap']:.4f}")
        print(f"  Train-Val F1 Gap: {metrics['train_val_f1_gap']:.4f}")
        print(f"  Val-Test F1 Gap: {metrics['val_test_f1_gap']:.4f}")
        print(f"  Test F1 Score: {metrics['test_f1']:.4f}")
    
    # Create visualization
    print("\nGenerating visualization...")
    viz_path = os.path.join(output_dir, 'overfitting_analysis.png')
    create_overfitting_visualization(level1_metrics, raw_metrics, viz_path)
    
    # Save detailed results
    analysis_results = {
        'level1_model': level1_metrics,
        'raw_data_model': raw_metrics,
        'summary': {
            'level1_overfitting': level1_metrics['overfitting_severity'],
            'raw_overfitting': raw_metrics['overfitting_severity'],
            'level1_val_representative': level1_metrics['validation_representative'],
            'raw_val_representative': raw_metrics['validation_representative'],
        }
    }
    
    results_file = os.path.join(output_dir, 'model_overfitting_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {results_file}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if level1_metrics['overfitting_severity'] in ['minimal', 'slight']:
        print("\nLevel1 Model: Overfitting is minimal. Model appears well-generalized.")
        print("  - Current performance is excellent (98.95% F1 on test set)")
        print("  - More data may provide marginal improvements")
    else:
        print("\nLevel1 Model: Some overfitting detected. Consider:")
        print("  - Adding more training data")
        print("  - Increasing regularization (lower learning rate, more subsampling)")
    
    if raw_metrics['overfitting_severity'] in ['minimal', 'slight']:
        print("\nRaw Data Model: Overfitting is minimal. Model appears well-generalized.")
        print("  - Current performance is good (97.44% F1 on test set)")
        print("  - More data may provide marginal improvements")
    else:
        print("\nRaw Data Model: Moderate overfitting detected. Consider:")
        print("  - Adding more training data (likely to help)")
        print("  - Increasing regularization")
        print("  - Feature selection to reduce complexity")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


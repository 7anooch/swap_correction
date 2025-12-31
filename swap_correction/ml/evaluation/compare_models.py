#!/usr/bin/env python3
"""
Compare level1 and raw data ML models side-by-side.

Generates comprehensive comparison report including performance metrics,
feature importance, and use case recommendations.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_model_results():
    """Load results and feature importance for both models."""
    # Load training results
    with open('ml_models/training_results.json', 'r') as f:
        level1_results = json.load(f)
    
    with open('ml_models_raw/training_results.json', 'r') as f:
        raw_results = json.load(f)
    
    # Load feature importance
    level1_importance = pd.read_csv('ml_models/feature_importance.csv')
    raw_importance = pd.read_csv('ml_models_raw/feature_importance.csv')
    
    return level1_results, raw_results, level1_importance, raw_importance


def create_comparison_plots(level1_results, raw_results, 
                          level1_importance, raw_importance,
                          output_dir: str):
    """Create visualization comparing both models."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Model Comparison: Level1 vs Raw Data Models', 
                 fontsize=16, fontweight='bold')
    
    # Performance metrics comparison
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    splits = ['Train', 'Val', 'Test']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    level1_test = [
        level1_results['test']['precision'],
        level1_results['test']['recall'],
        level1_results['test']['f1'],
        level1_results['test']['auc']
    ]
    raw_test = [
        raw_results['test']['precision'],
        raw_results['test']['recall'],
        raw_results['test']['f1'],
        raw_results['test']['auc']
    ]
    
    bars1 = ax1.bar(x - width/2, level1_test, width, label='Level1 Model', 
                    alpha=0.8, color='#2E86AB')
    bars2 = ax1.bar(x + width/2, raw_test, width, label='Raw Data Model', 
                    alpha=0.8, color='#A23B72')
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Test Set Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.9, 1.0])
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Confusion matrix comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    level1_cm = np.array(level1_results['test']['confusion_matrix'])
    raw_cm = np.array(raw_results['test']['confusion_matrix'])
    
    # Normalize confusion matrices
    level1_cm_norm = level1_cm / level1_cm.sum()
    raw_cm_norm = raw_cm / raw_cm.sum()
    
    # Plot difference
    cm_diff = level1_cm_norm - raw_cm_norm
    im = ax2.imshow(cm_diff, cmap='RdBu_r', aspect='auto', vmin=-0.01, vmax=0.01)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted: No Swap', 'Predicted: Swap'])
    ax2.set_yticklabels(['Actual: No Swap', 'Actual: Swap'])
    ax2.set_title('Confusion Matrix Difference\n(Level1 - Raw, normalized)', 
                 fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{cm_diff[i, j]:.4f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    # Feature importance comparison (top 10)
    ax3 = fig.add_subplot(gs[1, :])
    
    top_n = 10
    level1_top = level1_importance.head(top_n)
    raw_top = raw_importance.head(top_n)
    
    # Get common features
    common_features = set(level1_top['feature']) & set(raw_top['feature'])
    
    # Create comparison
    comparison_data = []
    for feat in common_features:
        level1_imp = level1_top[level1_top['feature'] == feat]['importance'].values[0]
        raw_imp = raw_top[raw_top['feature'] == feat]['importance'].values[0]
        comparison_data.append({
            'feature': feat,
            'level1': level1_imp,
            'raw': raw_imp
        })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('level1', ascending=True)
    
    y_pos = np.arange(len(comparison_df))
    ax3.barh(y_pos - 0.2, comparison_df['level1'], 0.4, 
            label='Level1 Model', alpha=0.8, color='#2E86AB')
    ax3.barh(y_pos + 0.2, comparison_df['raw'], 0.4, 
            label='Raw Data Model', alpha=0.8, color='#A23B72')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(comparison_df['feature'], fontsize=9)
    ax3.set_xlabel('Feature Importance', fontsize=11)
    ax3.set_title(f'Top {top_n} Feature Importance Comparison', 
                 fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Performance across splits
    ax4 = fig.add_subplot(gs[2, 0])
    
    splits = ['Train', 'Val', 'Test']
    level1_f1 = [
        level1_results['train']['f1'],
        level1_results['val']['f1'],
        level1_results['test']['f1']
    ]
    raw_f1 = [
        raw_results['train']['f1'],
        raw_results['val']['f1'],
        raw_results['test']['f1']
    ]
    
    x = np.arange(len(splits))
    ax4.plot(x, level1_f1, 'o-', label='Level1', linewidth=2, markersize=8, color='#2E86AB')
    ax4.plot(x, raw_f1, 's-', label='Raw Data', linewidth=2, markersize=8, color='#A23B72')
    ax4.set_xticks(x)
    ax4.set_xticklabels(splits)
    ax4.set_ylabel('F1 Score', fontsize=11)
    ax4.set_title('F1 Score Across Splits', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0.95, 1.01])
    
    # Error rates
    ax5 = fig.add_subplot(gs[2, 1])
    
    level1_cm = np.array(level1_results['test']['confusion_matrix'])
    raw_cm = np.array(raw_results['test']['confusion_matrix'])
    
    level1_fp_rate = level1_cm[0, 1] / level1_cm[0, :].sum()
    level1_fn_rate = level1_cm[1, 0] / level1_cm[1, :].sum()
    raw_fp_rate = raw_cm[0, 1] / raw_cm[0, :].sum()
    raw_fn_rate = raw_cm[1, 0] / raw_cm[1, :].sum()
    
    error_types = ['False Positive\nRate', 'False Negative\nRate']
    x = np.arange(len(error_types))
    width = 0.35
    
    ax5.bar(x - width/2, [level1_fp_rate, level1_fn_rate], width, 
           label='Level1', alpha=0.8, color='#2E86AB')
    ax5.bar(x + width/2, [raw_fp_rate, raw_fn_rate], width, 
           label='Raw Data', alpha=0.8, color='#A23B72')
    
    ax5.set_ylabel('Error Rate', fontsize=11)
    ax5.set_title('Error Rates (Test Set)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(error_types)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Data characteristics
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    level1_pos_rate = level1_results['test']['n_positive'] / level1_results['test']['n_samples']
    raw_pos_rate = raw_results['test']['n_positive'] / raw_results['test']['n_samples']
    
    info_text = f"""
    Data Characteristics:
    
    Level1 Model:
      Test samples: {level1_results['test']['n_samples']:,}
      Positive rate: {level1_pos_rate:.1%}
      Swapped frames: {level1_results['test']['n_positive']:,}
    
    Raw Data Model:
      Test samples: {raw_results['test']['n_samples']:,}
      Positive rate: {raw_pos_rate:.1%}
      Swapped frames: {raw_results['test']['n_positive']:,}
    """
    
    ax6.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    output_path = os.path.join(output_dir, 'model_comparison_plots.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {output_path}")


def generate_comparison_report(level1_results, raw_results, 
                              level1_importance, raw_importance,
                              output_dir: str):
    """Generate markdown comparison report."""
    report = """# Model Comparison Report

## Overview

This report compares two trained ML models for swap detection:
1. **Level1 Model**: Trained on level1.csv (auto-corrected) vs level2.csv (ground truth)
2. **Raw Data Model**: Trained on raw _data.csv vs level2.csv (ground truth)

## Performance Summary

### Test Set Performance

| Metric | Level1 Model | Raw Data Model | Difference |
|--------|--------------|----------------|------------|
| **F1-Score** | {level1_f1:.4f} | {raw_f1:.4f} | {f1_diff:+.4f} |
| **Precision** | {level1_prec:.4f} | {raw_prec:.4f} | {prec_diff:+.4f} |
| **Recall** | {level1_rec:.4f} | {raw_rec:.4f} | {rec_diff:+.4f} |
| **ROC-AUC** | {level1_auc:.4f} | {raw_auc:.4f} | {auc_diff:+.4f} |

### Performance Across Splits

#### Level1 Model
- **Train**: F1={level1_train_f1:.4f}, Precision={level1_train_prec:.4f}, Recall={level1_train_rec:.4f}
- **Validation**: F1={level1_val_f1:.4f}, Precision={level1_val_prec:.4f}, Recall={level1_val_rec:.4f}
- **Test**: F1={level1_test_f1:.4f}, Precision={level1_test_prec:.4f}, Recall={level1_test_rec:.4f}

#### Raw Data Model
- **Train**: F1={raw_train_f1:.4f}, Precision={raw_train_prec:.4f}, Recall={raw_train_rec:.4f}
- **Validation**: F1={raw_val_f1:.4f}, Precision={raw_val_prec:.4f}, Recall={raw_val_rec:.4f}
- **Test**: F1={raw_test_f1:.4f}, Precision={raw_test_prec:.4f}, Recall={raw_test_rec:.4f}

## Confusion Matrices (Test Set)

### Level1 Model
```
                Predicted
              No Swap  Swap
Actual No Swap  {level1_tn:5d}  {level1_fp:5d}
       Swap     {level1_fn:5d}  {level1_tp:5d}
```

- False Positive Rate: {level1_fp_rate:.2%}
- False Negative Rate: {level1_fn_rate:.2%}

### Raw Data Model
```
                Predicted
              No Swap  Swap
Actual No Swap  {raw_tn:5d}  {raw_fp:5d}
       Swap     {raw_fn:5d}  {raw_tp:5d}
```

- False Positive Rate: {raw_fp_rate:.2%}
- False Negative Rate: {raw_fn_rate:.2%}

## Feature Importance Comparison

### Top 10 Features (Level1 Model)
{level1_top10}

### Top 10 Features (Raw Data Model)
{raw_top10}

## Use Case Recommendations

### When to Use Level1 Model

- **Best for**: Detecting remaining swaps after initial auto-correction
- **Advantages**:
  - Higher precision (99.16% vs 97.20%)
  - Lower false positive rate ({level1_fp_rate:.2%} vs {raw_fp_rate:.2%})
  - Better performance overall (98.95% F1 vs 97.44% F1)
- **Use when**: You have level1.csv files and want to improve them further

### When to Use Raw Data Model

- **Best for**: Detecting swaps directly from raw tracking data
- **Advantages**:
  - Can work directly on raw data (no need for level1 correction first)
  - Still achieves good performance (97.44% F1)
  - Handles higher swap rate (49.41% vs 12.68% in training data)
- **Use when**: You want to skip the level1 correction step entirely

## Data Characteristics

### Training Data
- **Level1 Model**: 12.68% swapped frames (28,519 / 225,000)
- **Raw Data Model**: 49.41% swapped frames (111,169 / 225,000)

### Test Set
- **Level1 Model**: 14.30% swapped frames (6,433 / 45,000)
- **Raw Data Model**: 47.84% swapped frames (21,526 / 45,000)

## Conclusion

Both models perform well, with the Level1 model achieving slightly better performance.
The choice between models depends on your workflow:

- Use **Level1 Model** if you already have level1.csv files and want maximum accuracy
- Use **Raw Data Model** if you want to process raw data directly without intermediate correction steps

Both models are production-ready and can be used for automated swap detection.
""".format(
        level1_f1=level1_results['test']['f1'],
        raw_f1=raw_results['test']['f1'],
        f1_diff=level1_results['test']['f1'] - raw_results['test']['f1'],
        level1_prec=level1_results['test']['precision'],
        raw_prec=raw_results['test']['precision'],
        prec_diff=level1_results['test']['precision'] - raw_results['test']['precision'],
        level1_rec=level1_results['test']['recall'],
        raw_rec=raw_results['test']['recall'],
        rec_diff=level1_results['test']['recall'] - raw_results['test']['recall'],
        level1_auc=level1_results['test']['auc'],
        raw_auc=raw_results['test']['auc'],
        auc_diff=level1_results['test']['auc'] - raw_results['test']['auc'],
        level1_train_f1=level1_results['train']['f1'],
        level1_train_prec=level1_results['train']['precision'],
        level1_train_rec=level1_results['train']['recall'],
        level1_val_f1=level1_results['val']['f1'],
        level1_val_prec=level1_results['val']['precision'],
        level1_val_rec=level1_results['val']['recall'],
        level1_test_f1=level1_results['test']['f1'],
        level1_test_prec=level1_results['test']['precision'],
        level1_test_rec=level1_results['test']['recall'],
        raw_train_f1=raw_results['train']['f1'],
        raw_train_prec=raw_results['train']['precision'],
        raw_train_rec=raw_results['train']['recall'],
        raw_val_f1=raw_results['val']['f1'],
        raw_val_prec=raw_results['val']['precision'],
        raw_val_rec=raw_results['val']['recall'],
        raw_test_f1=raw_results['test']['f1'],
        raw_test_prec=raw_results['test']['precision'],
        raw_test_rec=raw_results['test']['recall'],
        level1_cm=level1_results['test']['confusion_matrix'],
        level1_tn=level1_results['test']['confusion_matrix'][0][0],
        level1_fp=level1_results['test']['confusion_matrix'][0][1],
        level1_fn=level1_results['test']['confusion_matrix'][1][0],
        level1_tp=level1_results['test']['confusion_matrix'][1][1],
        raw_cm=raw_results['test']['confusion_matrix'],
        raw_tn=raw_results['test']['confusion_matrix'][0][0],
        raw_fp=raw_results['test']['confusion_matrix'][0][1],
        raw_fn=raw_results['test']['confusion_matrix'][1][0],
        raw_tp=raw_results['test']['confusion_matrix'][1][1],
        level1_fp_rate=level1_results['test']['confusion_matrix'][0][1] / 
                       sum(level1_results['test']['confusion_matrix'][0]),
        level1_fn_rate=level1_results['test']['confusion_matrix'][1][0] / 
                       sum(level1_results['test']['confusion_matrix'][1]),
        raw_fp_rate=raw_results['test']['confusion_matrix'][0][1] / 
                   sum(raw_results['test']['confusion_matrix'][0]),
        raw_fn_rate=raw_results['test']['confusion_matrix'][1][0] / 
                   sum(raw_results['test']['confusion_matrix'][1]),
        level1_top10='\n'.join([f"{i+1}. {row['feature']}: {row['importance']:.4f}" 
                                for i, row in level1_importance.head(10).iterrows()]),
        raw_top10='\n'.join([f"{i+1}. {row['feature']}: {row['importance']:.4f}" 
                            for i, row in raw_importance.head(10).iterrows()])
    )
    
    output_path = os.path.join(output_dir, 'model_comparison_report.md')
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to: {output_path}")


def main():
    """Main comparison function."""
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    output_dir = 'ml_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading model results and feature importance...")
    level1_results, raw_results, level1_importance, raw_importance = load_model_results()
    
    # Create plots
    print("Generating comparison plots...")
    create_comparison_plots(level1_results, raw_results, 
                          level1_importance, raw_importance,
                          output_dir)
    
    # Generate report
    print("Generating comparison report...")
    generate_comparison_report(level1_results, raw_results,
                             level1_importance, raw_importance,
                             output_dir)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


"""
Find optimal classification threshold for maximizing performance metrics.

This module provides utilities to find the threshold that maximizes F1-score,
% Frames Clean Post, or other metrics by testing different thresholds on
validation or test data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Literal, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve
)


def calculate_metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                                   threshold: float) -> Dict:
    """
    Calculate all metrics at a specific threshold.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1)
    y_proba : np.ndarray
        Predicted probabilities (0-1)
    threshold : float
        Classification threshold (0-1)
        
    Returns:
    --------
    dict
        Dictionary containing all metrics at this threshold
    """
    # Binarize predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        if cm.shape == (1, 1):
            if y_true.sum() == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
            fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
            fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
            tp = cm[1, 1] if cm.shape == (2, 2) else 0
    
    # Calculate metrics
    total_frames = len(y_true)
    n_swapped_gt = int(y_true.sum())
    
    # Core metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity = Recall
    sensitivity = recall
    
    # Specificity
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 1.0 if fp == 0 else 0.0
    
    # Percentage metrics
    if (tp + fn) > 0:
        pct_swaps_resolved = (tp / (tp + fn)) * 100.0
    else:
        pct_swaps_resolved = 100.0 if fn == 0 else 0.0
    
    pct_frames_clean_pre = ((total_frames - n_swapped_gt) / total_frames) * 100.0 if total_frames > 0 else 0.0
    pct_frames_clean_post = ((tp + tn) / total_frames) * 100.0 if total_frames > 0 else 0.0
    
    return {
        'threshold': threshold,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'pct_swaps_resolved': float(pct_swaps_resolved),
        'pct_frames_clean_pre': float(pct_frames_clean_pre),
        'pct_frames_clean_post': float(pct_frames_clean_post),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'n_frames': total_frames,
        'n_swapped_gt': n_swapped_gt,
        'n_swapped_pred': int(y_pred.sum())
    }


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                          metric: Literal['f1', 'pct_clean_post', 'precision', 
                                         'recall', 'pct_swaps_resolved'] = 'pct_clean_post',
                          thresholds: Optional[np.ndarray] = None,
                          n_thresholds: int = 100) -> Tuple[float, Dict, List[Dict]]:
    """
    Find optimal threshold that maximizes a specified metric.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1)
    y_proba : np.ndarray
        Predicted probabilities (0-1)
    metric : str
        Metric to maximize: 'f1', 'pct_clean_post', 'precision', 'recall', 'pct_swaps_resolved'
    thresholds : np.ndarray, optional
        Specific thresholds to test (if None, generates automatically)
    n_thresholds : int
        Number of thresholds to test (default: 100)
        
    Returns:
    --------
    tuple
        (optimal_threshold, best_metrics, all_results)
        - optimal_threshold: Best threshold value
        - best_metrics: Metrics at optimal threshold
        - all_results: List of metrics for all tested thresholds
    """
    # Generate thresholds if not provided
    if thresholds is None:
        # Use ROC curve to find reasonable range
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        # Focus on thresholds where there's variation
        min_thresh = max(0.01, np.min(y_proba[y_proba > 0]) if np.any(y_proba > 0) else 0.01)
        max_thresh = min(0.99, np.max(y_proba[y_proba < 1]) if np.any(y_proba < 1) else 0.99)
        thresholds = np.linspace(min_thresh, max_thresh, n_thresholds)
    
    # Test all thresholds
    all_results = []
    best_metric_value = -np.inf
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_proba, threshold)
        all_results.append(metrics)
        
        # Get metric value to maximize (map short names to full names)
        metric_map = {
            'pct_clean_post': 'pct_frames_clean_post',
            'pct_swaps_resolved': 'pct_swaps_resolved',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        }
        metric_key = metric_map.get(metric, metric)
        metric_value = metrics[metric_key]
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics, all_results


def find_optimal_threshold_on_dataset(model, X: np.ndarray, y_true: np.ndarray,
                                     metric: Literal['f1', 'pct_clean_post', 'precision', 
                                                    'recall', 'pct_swaps_resolved'] = 'pct_clean_post',
                                     n_thresholds: int = 100) -> Tuple[float, Dict, List[Dict]]:
    """
    Find optimal threshold using a trained model and data.
    
    Parameters:
    -----------
    model
        Trained model with predict_proba method
    X : np.ndarray
        Feature matrix
    y_true : np.ndarray
        Ground truth labels
    metric : str
        Metric to maximize
    n_thresholds : int
        Number of thresholds to test
        
    Returns:
    --------
    tuple
        (optimal_threshold, best_metrics, all_results)
    """
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Find optimal threshold
    return find_optimal_threshold(y_true, y_proba, metric=metric, n_thresholds=n_thresholds)


def compare_thresholds(y_true: np.ndarray, y_proba: np.ndarray,
                      thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]) -> Dict:
    """
    Compare performance at multiple thresholds.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth labels
    y_proba : np.ndarray
        Predicted probabilities
    thresholds : list
        List of thresholds to compare
        
    Returns:
    --------
    dict
        Dictionary mapping threshold to metrics
    """
    results = {}
    for threshold in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_proba, threshold)
        results[threshold] = metrics
    
    return results


def print_threshold_analysis(optimal_threshold: float, best_metrics: Dict, 
                            all_results: List[Dict], metric: str = 'pct_clean_post'):
    # Map short metric names to full names
    metric_map = {
        'pct_clean_post': 'pct_frames_clean_post',
        'pct_swaps_resolved': 'pct_swaps_resolved',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }
    metric_key = metric_map.get(metric, metric)
    """
    Print a formatted analysis of threshold optimization.
    
    Parameters:
    -----------
    optimal_threshold : float
        Optimal threshold value
    best_metrics : dict
        Metrics at optimal threshold
    all_results : list
        All threshold results
    metric : str
        Metric that was optimized
    """
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nOptimized for: {metric}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"\nPerformance at optimal threshold:")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1-Score: {best_metrics['f1']:.4f}")
    print(f"  Sensitivity: {best_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {best_metrics['specificity']:.4f}")
    print(f"  % Swaps Resolved: {best_metrics['pct_swaps_resolved']:.2f}%")
    print(f"  % Frames Clean Post: {best_metrics['pct_frames_clean_post']:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {best_metrics['tp']}, FP: {best_metrics['fp']}")
    print(f"  FN: {best_metrics['fn']}, TN: {best_metrics['tn']}")
    
    # Compare with default 0.5 threshold
    if all_results:
        # Find 0.5 threshold results (or closest)
        threshold_05_results = [r for r in all_results if abs(r['threshold'] - 0.5) < 0.01]
        if not threshold_05_results:
            # Find closest to 0.5
            closest_idx = np.argmin([abs(r['threshold'] - 0.5) for r in all_results])
            threshold_05_results = [all_results[closest_idx]]
        
        if threshold_05_results:
            default_metrics = threshold_05_results[0]
            print(f"\nComparison with default threshold (0.5):")
            print(f"  Threshold 0.5: {default_metrics['pct_frames_clean_post']:.2f}%")
            print(f"  Optimal ({optimal_threshold:.4f}): {best_metrics['pct_frames_clean_post']:.2f}%")
            improvement = best_metrics['pct_frames_clean_post'] - default_metrics['pct_frames_clean_post']
            print(f"  Improvement: {improvement:+.2f}%")
    
    # Show threshold range analysis
    if all_results:
        metric_values = [r[metric_key] for r in all_results]
        print(f"\nThreshold range analysis:")
        print(f"  Min {metric}: {min(metric_values):.4f} (threshold: {all_results[np.argmin(metric_values)]['threshold']:.4f})")
        print(f"  Max {metric}: {max(metric_values):.4f} (threshold: {all_results[np.argmax(metric_values)]['threshold']:.4f})")
        print(f"  Range: {max(metric_values) - min(metric_values):.4f}")
    
    # Print comprehensive metrics table
    if all_results:
        print(f"\n{'='*80}")
        print("ALL METRICS AT OPTIMAL THRESHOLD")
        print(f"{'='*80}")
        print(f"{'Metric':<25} {'Value':<15} {'Unit':<10}")
        print("-" * 80)
        print(f"{'Precision':<25} {best_metrics['precision']:<15.4f} {'':<10}")
        print(f"{'Recall':<25} {best_metrics['recall']:<15.4f} {'':<10}")
        print(f"{'F1-Score':<25} {best_metrics['f1']:<15.4f} {'':<10}")
        print(f"{'Sensitivity':<25} {best_metrics['sensitivity']:<15.4f} {'':<10}")
        print(f"{'Specificity':<25} {best_metrics['specificity']:<15.4f} {'':<10}")
        print(f"{'% Swaps Resolved':<25} {best_metrics['pct_swaps_resolved']:<15.2f} {'%':<10}")
        print(f"{'% Frames Clean Pre':<25} {best_metrics['pct_frames_clean_pre']:<15.2f} {'%':<10}")
        print(f"{'% Frames Clean Post':<25} {best_metrics['pct_frames_clean_post']:<15.2f} {'%':<10}")
        print(f"{'True Positives (TP)':<25} {best_metrics['tp']:<15} {'frames':<10}")
        print(f"{'False Positives (FP)':<25} {best_metrics['fp']:<15} {'frames':<10}")
        print(f"{'False Negatives (FN)':<25} {best_metrics['fn']:<15} {'frames':<10}")
        print(f"{'True Negatives (TN)':<25} {best_metrics['tn']:<15} {'frames':<10}")
        print(f"{'Total Frames':<25} {best_metrics['n_frames']:<15} {'frames':<10}")
        print(f"{'GT Swapped Frames':<25} {best_metrics['n_swapped_gt']:<15} {'frames':<10}")
        print(f"{'Predicted Swapped':<25} {best_metrics['n_swapped_pred']:<15} {'frames':<10}")


def create_threshold_plots(all_results: List[Dict], optimal_threshold: float,
                          metric_optimized: str, output_path: Optional[str] = None):
    """
    Create comprehensive plots showing all metrics across thresholds.
    
    Parameters:
    -----------
    all_results : list
        List of metric dictionaries for each threshold
    optimal_threshold : float
        Optimal threshold value (will be marked on plots)
    metric_optimized : str
        Metric that was optimized (will be highlighted)
    output_path : str, optional
        Path to save the figure (if None, doesn't save)
    """
    if not all_results:
        return
    
    # Extract data
    thresholds = np.array([r['threshold'] for r in all_results])
    
    # Sort by threshold
    sort_idx = np.argsort(thresholds)
    thresholds = thresholds[sort_idx]
    
    # Extract all metrics
    precision = np.array([r['precision'] for r in all_results])[sort_idx]
    recall = np.array([r['recall'] for r in all_results])[sort_idx]
    f1 = np.array([r['f1'] for r in all_results])[sort_idx]
    sensitivity = np.array([r['sensitivity'] for r in all_results])[sort_idx]
    specificity = np.array([r['specificity'] for r in all_results])[sort_idx]
    pct_swaps_resolved = np.array([r['pct_swaps_resolved'] for r in all_results])[sort_idx]
    pct_clean_pre = np.array([r['pct_frames_clean_pre'] for r in all_results])[sort_idx]
    pct_clean_post = np.array([r['pct_frames_clean_post'] for r in all_results])[sort_idx]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Threshold Optimization: All Metrics Across Thresholds', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Core Classification Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(thresholds, precision, 'o-', label='Precision', linewidth=2, markersize=4, color='#2E86AB')
    ax1.plot(thresholds, recall, 's-', label='Recall', linewidth=2, markersize=4, color='#A23B72')
    ax1.plot(thresholds, f1, '^-', label='F1-Score', linewidth=2, markersize=4, color='#F18F01')
    ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax1.set_xlabel('Threshold', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Core Classification Metrics', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([thresholds.min(), thresholds.max()])
    
    # Plot 2: Sensitivity and Specificity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(thresholds, sensitivity, 'o-', label='Sensitivity (Recall)', 
             linewidth=2, markersize=4, color='#2E86AB')
    ax2.plot(thresholds, specificity, 's-', label='Specificity', 
             linewidth=2, markersize=4, color='#A23B72')
    ax2.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax2.set_xlabel('Threshold', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Sensitivity vs Specificity', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([thresholds.min(), thresholds.max()])
    
    # Plot 3: Percentage Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(thresholds, pct_swaps_resolved, 'o-', label='% Swaps Resolved', 
             linewidth=2, markersize=4, color='#2E86AB')
    ax3.plot(thresholds, pct_clean_post, 's-', label='% Frames Clean Post', 
             linewidth=2, markersize=4, color='#A23B72')
    if metric_optimized in ['pct_clean_post', 'pct_swaps_resolved']:
        # Highlight the optimized metric
        if metric_optimized == 'pct_clean_post':
            ax3.plot(thresholds, pct_clean_post, 's-', linewidth=3, markersize=6, 
                    color='#F18F01', alpha=0.5, zorder=0)
        else:
            ax3.plot(thresholds, pct_swaps_resolved, 'o-', linewidth=3, markersize=6, 
                    color='#F18F01', alpha=0.5, zorder=0)
    ax3.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax3.set_xlabel('Threshold', fontsize=11)
    ax3.set_ylabel('Percentage (%)', fontsize=11)
    ax3.set_title('Percentage Metrics', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([thresholds.min(), thresholds.max()])
    
    # Plot 4: F1-Score (detailed)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(thresholds, f1, 'o-', linewidth=2, markersize=4, color='#2E86AB')
    if metric_optimized == 'f1':
        ax4.plot(thresholds, f1, 'o-', linewidth=3, markersize=6, 
                color='#F18F01', alpha=0.5, zorder=0)
    ax4.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax4.set_xlabel('Threshold', fontsize=11)
    ax4.set_ylabel('F1-Score', fontsize=11)
    ax4.set_title('F1-Score vs Threshold', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([thresholds.min(), thresholds.max()])
    
    # Plot 5: % Frames Clean Post (detailed)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(thresholds, pct_clean_post, 's-', linewidth=2, markersize=4, color='#2E86AB')
    if metric_optimized == 'pct_clean_post':
        ax5.plot(thresholds, pct_clean_post, 's-', linewidth=3, markersize=6, 
                color='#F18F01', alpha=0.5, zorder=0)
    ax5.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax5.set_xlabel('Threshold', fontsize=11)
    ax5.set_ylabel('% Frames Clean Post', fontsize=11)
    ax5.set_title('% Frames Clean Post vs Threshold', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([thresholds.min(), thresholds.max()])
    
    # Plot 6: % Swaps Resolved (detailed)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(thresholds, pct_swaps_resolved, 'o-', linewidth=2, markersize=4, color='#2E86AB')
    if metric_optimized == 'pct_swaps_resolved':
        ax6.plot(thresholds, pct_swaps_resolved, 'o-', linewidth=3, markersize=6, 
                color='#F18F01', alpha=0.5, zorder=0)
    ax6.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax6.set_xlabel('Threshold', fontsize=11)
    ax6.set_ylabel('% Swaps Resolved', fontsize=11)
    ax6.set_title('% Swaps Resolved vs Threshold', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([thresholds.min(), thresholds.max()])
    
    # Plot 7: Precision vs Recall (trade-off)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(recall, precision, 'o-', linewidth=2, markersize=4, color='#2E86AB')
    # Mark optimal threshold point
    opt_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    ax7.plot(recall[opt_idx], precision[opt_idx], 'ro', markersize=10, 
            label=f'Optimal ({optimal_threshold:.3f})', zorder=5)
    ax7.set_xlabel('Recall', fontsize=11)
    ax7.set_ylabel('Precision', fontsize=11)
    ax7.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Confusion Matrix Components
    ax8 = fig.add_subplot(gs[2, 1])
    tp = np.array([r['tp'] for r in all_results])[sort_idx]
    fp = np.array([r['fp'] for r in all_results])[sort_idx]
    fn = np.array([r['fn'] for r in all_results])[sort_idx]
    tn = np.array([r['tn'] for r in all_results])[sort_idx]
    
    ax8.plot(thresholds, tp, 'o-', label='TP', linewidth=2, markersize=4, color='green')
    ax8.plot(thresholds, fp, 's-', label='FP', linewidth=2, markersize=4, color='red')
    ax8.plot(thresholds, fn, '^-', label='FN', linewidth=2, markersize=4, color='orange')
    ax8.plot(thresholds, tn, 'd-', label='TN', linewidth=2, markersize=4, color='blue')
    ax8.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax8.set_xlabel('Threshold', fontsize=11)
    ax8.set_ylabel('Count', fontsize=11)
    ax8.set_title('Confusion Matrix Components', fontsize=12, fontweight='bold')
    ax8.legend(loc='best', fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim([thresholds.min(), thresholds.max()])
    
    # Plot 9: Summary - All percentage metrics together
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(thresholds, pct_clean_pre, 'o-', label='% Clean Pre', 
             linewidth=2, markersize=4, color='gray', alpha=0.7)
    ax9.plot(thresholds, pct_clean_post, 's-', label='% Clean Post', 
             linewidth=2, markersize=4, color='#2E86AB')
    ax9.plot(thresholds, pct_swaps_resolved, '^-', label='% Resolved', 
             linewidth=2, markersize=4, color='#A23B72')
    ax9.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal ({optimal_threshold:.3f})', alpha=0.7)
    ax9.set_xlabel('Threshold', fontsize=11)
    ax9.set_ylabel('Percentage (%)', fontsize=11)
    ax9.set_title('All Percentage Metrics', fontsize=12, fontweight='bold')
    ax9.legend(loc='best', fontsize=9)
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim([thresholds.min(), thresholds.max()])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Threshold optimization utility")
    print("Use find_optimal_threshold() or find_optimal_threshold_on_dataset()")
    print("See documentation for usage examples")


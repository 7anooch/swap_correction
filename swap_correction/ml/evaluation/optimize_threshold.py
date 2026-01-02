"""
Script to find optimal classification threshold for a trained model.

This script loads a trained model and finds the threshold that maximizes
% Frames Clean Post (or another metric) on validation or test data.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from swap_correction.ml.api.model_loader import load_model
from swap_correction.ml.training.train_model import load_training_data, prepare_train_val_test_split
from swap_correction.ml.evaluation.find_optimal_threshold import (
    find_optimal_threshold_on_dataset,
    print_threshold_analysis,
    compare_thresholds
)


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal classification threshold for maximizing performance metrics'
    )
    parser.add_argument('--model-type', type=str, default='level1',
                       choices=['level1', 'raw', 'raw_data'],
                       help='Model type to use')
    parser.add_argument('--ml-data-dir', type=str, default='ml_data',
                       help='Directory containing ML training data')
    parser.add_argument('--test-data-dir', type=str, default=None,
                       help='Directory containing test data (default: auto-detect)')
    parser.add_argument('--metric', type=str, default='pct_clean_post',
                       choices=['f1', 'pct_clean_post', 'precision', 'recall', 'pct_swaps_resolved'],
                       help='Metric to maximize (default: pct_clean_post)')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Data split to use for optimization (default: val)')
    parser.add_argument('--n-thresholds', type=int, default=100,
                       help='Number of thresholds to test (default: 100)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results (default: print only)')
    parser.add_argument('--compare-thresholds', type=float, nargs='+', default=None,
                       help='Additional thresholds to compare (e.g., 0.3 0.4 0.5 0.6 0.7)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Model type: {args.model_type}")
    print(f"Metric to maximize: {args.metric}")
    print(f"Data split: {args.split}")
    print(f"Number of thresholds: {args.n_thresholds}")
    print()
    
    # Load model
    print("Loading model...")
    model, scaler, imputer, feature_names = load_model(args.model_type)
    print("✓ Model loaded")
    
    # Load data
    print("\nLoading data...")
    test_data_dir = args.test_data_dir or _get_default_test_data_path()
    features_df, labels, trial_names, split = load_training_data(
        ml_data_dir=args.ml_data_dir,
        test_data_dir=test_data_dir,
        use_raw_data=(args.model_type in ['raw', 'raw_data'])
    )
    
    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        features_df, labels, trial_names, split
    )
    
    # Handle NaN
    from sklearn.impute import SimpleImputer
    X_train = imputer.transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    # Scale
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Select split
    if args.split == 'train':
        X, y = X_train, y_train
        split_name = 'Training'
    elif args.split == 'val':
        X, y = X_val, y_val
        split_name = 'Validation'
    else:
        X, y = X_test, y_test
        split_name = 'Test'
    
    print(f"✓ Data loaded: {split_name} set ({len(X)} samples, {y.sum()} positive)")
    print()
    
    # Find optimal threshold
    print(f"Finding optimal threshold to maximize {args.metric}...")
    optimal_threshold, best_metrics, all_results = find_optimal_threshold_on_dataset(
        model, X, y, metric=args.metric, n_thresholds=args.n_thresholds
    )
    
    # Print analysis
    print_threshold_analysis(optimal_threshold, best_metrics, all_results, metric=args.metric)
    
    # Create and save plots
    if args.output_dir:
        from swap_correction.ml.evaluation.find_optimal_threshold import create_threshold_plots
        plot_path = os.path.join(args.output_dir, 'threshold_optimization_plots.png')
        create_threshold_plots(all_results, optimal_threshold, args.metric, plot_path)
    else:
        # Still create plots but don't save
        from swap_correction.ml.evaluation.find_optimal_threshold import create_threshold_plots
        create_threshold_plots(all_results, optimal_threshold, args.metric)
    
    # Compare with additional thresholds if requested
    if args.compare_thresholds:
        print("\n" + "=" * 80)
        print("COMPARISON WITH SPECIFIED THRESHOLDS")
        print("=" * 80)
        y_proba = model.predict_proba(X)[:, 1]
        comparison = compare_thresholds(y, y_proba, thresholds=args.compare_thresholds)
        
        print(f"\n{'Threshold':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Sensitivity':<12} {'Specificity':<12} {'% Clean Post':<15} {'% Resolved':<15}")
        print("-" * 110)
        for threshold in sorted(comparison.keys()):
            m = comparison[threshold]
            print(f"{threshold:<12.4f} {m['precision']:<10.4f} {m['recall']:<10.4f} "
                  f"{m['f1']:<10.4f} {m['sensitivity']:<12.4f} {m['specificity']:<12.4f} "
                  f"{m['pct_frames_clean_post']:<15.2f} {m['pct_swaps_resolved']:<15.2f}")
    
    # Save results if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save optimal threshold and metrics
        results = {
            'optimal_threshold': float(optimal_threshold),
            'metric_optimized': args.metric,
            'split_used': args.split,
            'best_metrics': best_metrics,
            'n_samples': len(X),
            'n_positive': int(y.sum())
        }
        
        results_file = os.path.join(args.output_dir, 'optimal_threshold.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
        
        # Save all threshold results as CSV
        all_results_df = pd.DataFrame(all_results)
        csv_file = os.path.join(args.output_dir, 'threshold_analysis.csv')
        all_results_df.to_csv(csv_file, index=False)
        print(f"✓ Full analysis saved to: {csv_file}")
        
        # Create summary report with all metrics
        report_lines = [
            "# Threshold Optimization Report",
            "",
            f"**Model Type**: {args.model_type}",
            f"**Metric Optimized**: {args.metric}",
            f"**Data Split**: {args.split}",
            f"**Optimal Threshold**: {optimal_threshold:.4f}",
            "",
            "## Performance at Optimal Threshold",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Precision | {best_metrics['precision']:.4f} |",
            f"| Recall | {best_metrics['recall']:.4f} |",
            f"| F1-Score | {best_metrics['f1']:.4f} |",
            f"| Sensitivity | {best_metrics['sensitivity']:.4f} |",
            f"| Specificity | {best_metrics['specificity']:.4f} |",
            f"| % Swaps Resolved | {best_metrics['pct_swaps_resolved']:.2f}% |",
            f"| % Frames Clean Pre | {best_metrics['pct_frames_clean_pre']:.2f}% |",
            f"| % Frames Clean Post | {best_metrics['pct_frames_clean_post']:.2f}% |",
            "",
            "## Confusion Matrix",
            "",
            f"- **TP**: {best_metrics['tp']}",
            f"- **FP**: {best_metrics['fp']}",
            f"- **FN**: {best_metrics['fn']}",
            f"- **TN**: {best_metrics['tn']}",
            "",
            "## Comparison with Default Threshold (0.5)",
            ""
        ]
        
        # Find default threshold metrics
        threshold_05_results = [r for r in all_results if abs(r['threshold'] - 0.5) < 0.01]
        if not threshold_05_results:
            closest_idx = np.argmin([abs(r['threshold'] - 0.5) for r in all_results])
            threshold_05_results = [all_results[closest_idx]]
        
        if threshold_05_results:
            default_metrics = threshold_05_results[0]
            improvement = best_metrics['pct_frames_clean_post'] - default_metrics['pct_frames_clean_post']
            report_lines.extend([
                f"| Metric | Default (0.5) | Optimal ({optimal_threshold:.4f}) | Improvement |",
                "|--------|---------------|-----------------------------------|-------------|",
                f"| % Frames Clean Post | {default_metrics['pct_frames_clean_post']:.2f}% | {best_metrics['pct_frames_clean_post']:.2f}% | {improvement:+.2f}% |",
                f"| F1-Score | {default_metrics['f1']:.4f} | {best_metrics['f1']:.4f} | {best_metrics['f1'] - default_metrics['f1']:+.4f} |",
                f"| Precision | {default_metrics['precision']:.4f} | {best_metrics['precision']:.4f} | {best_metrics['precision'] - default_metrics['precision']:+.4f} |",
                f"| Recall | {default_metrics['recall']:.4f} | {best_metrics['recall']:.4f} | {best_metrics['recall'] - default_metrics['recall']:+.4f} |",
                "",
                "## Usage",
                "",
                f"```python",
                f"from swap_correction.ml.api import SwapPredictor",
                f"",
                f"predictor = SwapPredictor(model_type='{args.model_type}')",
                f"predictions = predictor.predict(trial_data, fps=30, threshold={optimal_threshold:.4f})",
                f"```",
                "",
                "## Visualization",
                "",
                f"See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds."
            ])
        
        report_file = os.path.join(args.output_dir, 'threshold_optimization_report.md')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"✓ Report saved to: {report_file}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"Use threshold: {optimal_threshold:.4f}")
    print(f"This maximizes {args.metric} on the {args.split} set")
    print(f"\nTo use this threshold in predictions:")
    print(f"  predictor = SwapPredictor(model_type='{args.model_type}')")
    print(f"  predictions = predictor.predict(trial_data, fps=30, threshold={optimal_threshold:.4f})")
    print()


def _get_default_test_data_path():
    """Get default test data path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    return os.path.join(package_dir, 'tests', 'test_data')


if __name__ == '__main__':
    main()


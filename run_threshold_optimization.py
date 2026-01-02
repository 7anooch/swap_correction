#!/usr/bin/env python3
"""
Run threshold optimization on stability analysis models.

This script uses the trained models and validation data from the stability analysis
to find optimal thresholds.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from swap_correction.ml.evaluation.find_optimal_threshold import (
    find_optimal_threshold_on_dataset,
    print_threshold_analysis,
    create_threshold_plots,
    compare_thresholds
)
from swap_correction.ml.training.train_model import (
    load_training_data, prepare_train_val_test_split
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_model_from_stability_analysis(base_dir: str, iteration_id: int, model_type: str):
    """Load model from stability analysis directory."""
    iter_dir = os.path.join(base_dir, f'iteration_{iteration_id:03d}')
    model_dir = os.path.join(iter_dir, f'{model_type}_model')
    
    model_file = os.path.join(model_dir, 'swap_detector_xgb.pkl')
    scaler_file = os.path.join(model_dir, 'feature_scaler.pkl')
    imputer_file = os.path.join(model_dir, 'feature_imputer.pkl')
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(imputer_file, 'rb') as f:
        imputer = pickle.load(f)
    
    return model, scaler, imputer, model_dir


def main():
    """Run threshold optimization on stability analysis models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Find optimal threshold for stability analysis models')
    parser.add_argument('--base-dir', type=str, default='stability_analysis_v3_features_v3',
                       help='Base directory for stability analysis')
    parser.add_argument('--iteration', type=int, default=7,
                       help='Iteration to use (default: 7)')
    parser.add_argument('--model-type', type=str, default='level1',
                       choices=['level1', 'raw'],
                       help='Model type to optimize')
    parser.add_argument('--metric', type=str, default='pct_clean_post',
                       choices=['f1', 'pct_clean_post', 'precision', 'recall', 'pct_swaps_resolved'],
                       help='Metric to maximize')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Data split to use')
    parser.add_argument('--n-thresholds', type=int, default=100,
                       help='Number of thresholds to test')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: base_dir/threshold_analysis)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION FOR STABILITY ANALYSIS")
    print("=" * 80)
    print(f"Base directory: {args.base_dir}")
    print(f"Iteration: {args.iteration:03d}")
    print(f"Model type: {args.model_type}")
    print(f"Metric to maximize: {args.metric}")
    print(f"Data split: {args.split}")
    print()
    
    # Load model
    print("Loading model...")
    model, scaler, imputer, model_dir = load_model_from_stability_analysis(
        args.base_dir, args.iteration, args.model_type
    )
    print("✓ Model loaded")
    
    # Load training data (this will use the same data split as training)
    print("\nLoading training data...")
    ml_data_dir = os.path.join(model_dir, 'ml_data')
    
    # Get parent directory for test data
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.base_dir)))
    # Try to find the main dataset
    main_dataset = '/Users/hind/Documents/UCSB/Neuroscience/KirstenData/new_data/Main_dataset'
    if not os.path.exists(main_dataset):
        # Fall back to test data
        test_data_dir = None
    else:
        test_data_dir = main_dataset
    
    # Load the split info
    split_file = os.path.join(os.path.dirname(model_dir), 'trial_split.json')
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        trial_dirs = split_data.get('train', []) + split_data.get('val', []) + split_data.get('test', [])
    else:
        trial_dirs = None
    
    # Patch feature extraction to use features_v2, features_v3, or features_v4 if needed
    original_extract = None
    if 'features_v4' in args.base_dir:
        print("  Patching to use features_v4...")
        import swap_correction.ml.features.features_v4 as features_module_to_use
        import swap_correction.ml.features as features_module
        import swap_correction.ml.training.train_model as train_model_module
        import sys
        import importlib
        
        # Store original
        original_extract = features_module.extract_all_frame_features_optimized
        
        # Patch in features module (this is what train_model imports from)
        features_module.extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
        sys.modules['swap_correction.ml.features'].extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
        
        # Reload train_model to pick up the patched function
        importlib.reload(train_model_module)
    elif 'features_v3' in args.base_dir:
        print("  Patching to use features_v3...")
        import swap_correction.ml.features.features_v3 as features_module_to_use
        import swap_correction.ml.features as features_module
        import swap_correction.ml.training.train_model as train_model_module
        import sys
        import importlib
        
        # Store original
        original_extract = features_module.extract_all_frame_features_optimized
        
        # Patch in features module (this is what train_model imports from)
        features_module.extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
        sys.modules['swap_correction.ml.features'].extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
        
        # Reload train_model to pick up the patched function
        importlib.reload(train_model_module)
    elif 'features_v2' in args.base_dir:
        print("  Patching to use features_v2...")
        import swap_correction.ml.features.features_v2 as features_module_to_use
        import swap_correction.ml.features as features_module
        import swap_correction.ml.training.train_model as train_model_module
        import sys
        import importlib
        
        # Store original
        original_extract = features_module.extract_all_frame_features_optimized
        
        # Patch in features module (this is what train_model imports from)
        features_module.extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
        sys.modules['swap_correction.ml.features'].extract_all_frame_features_optimized = features_module_to_use.extract_all_frame_features_optimized
        
        # Reload train_model to pick up the patched function
        importlib.reload(train_model_module)
    
    try:
        features_df, labels, trial_names, split = load_training_data(
            ml_data_dir=ml_data_dir,
            test_data_dir=test_data_dir,
            use_raw_data=(args.model_type == 'raw'),
            trial_dirs=trial_dirs
        )
        
        # Verify feature count matches model
        if 'features_v4' in args.base_dir:
            expected_features = 40  # Approximate, v4 has ~40-42 features
        elif 'features_v3' in args.base_dir:
            expected_features = 39
        elif 'features_v2' in args.base_dir:
            expected_features = 46
        else:
            expected_features = 56
        
        if len(features_df.columns) != expected_features:
            print(f"WARNING: Feature count mismatch! Expected {expected_features}, got {len(features_df.columns)}")
            print(f"Features: {list(features_df.columns)[:10]}...")
    except Exception as e:
        print(f"Error loading training data: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError("Could not load training data. Please ensure ml_data directory exists with extracted features.")
    finally:
        # Restore original feature extraction
        if original_extract is not None:
            import swap_correction.ml.features as features_module
            features_module.extract_all_frame_features_optimized = original_extract
            import sys
            if 'swap_correction.ml.features' in sys.modules:
                sys.modules['swap_correction.ml.features'].extract_all_frame_features_optimized = original_extract
    
    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        features_df, labels, trial_names, split
    )
    
    # Preprocess
    X_train = imputer.transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
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
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.base_dir, f'threshold_analysis_{args.model_type}')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save plots
    plot_path = os.path.join(output_dir, 'threshold_optimization_plots.png')
    create_threshold_plots(all_results, optimal_threshold, args.metric, plot_path)
    
    # Save results
    results = {
        'optimal_threshold': float(optimal_threshold),
        'metric_optimized': args.metric,
        'split_used': args.split,
        'iteration': args.iteration,
        'model_type': args.model_type,
        'best_metrics': best_metrics,
        'n_samples': len(X),
        'n_positive': int(y.sum())
    }
    
    results_file = os.path.join(output_dir, 'optimal_threshold.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    # Save all threshold results as CSV
    all_results_df = pd.DataFrame(all_results)
    csv_file = os.path.join(output_dir, 'threshold_analysis.csv')
    all_results_df.to_csv(csv_file, index=False)
    print(f"✓ Full analysis saved to: {csv_file}")
    
    # Create summary report
    report_lines = [
        "# Threshold Optimization Report",
        "",
        f"**Model Type**: {args.model_type}",
        f"**Iteration**: {args.iteration:03d}",
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
    
    report_file = os.path.join(output_dir, 'threshold_optimization_report.md')
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


if __name__ == '__main__':
    main()


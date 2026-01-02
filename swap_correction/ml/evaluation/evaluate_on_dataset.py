#!/usr/bin/env python3
"""
Evaluate trained models on a new dataset.

Compares model predictions against ground truth and generates comprehensive
evaluation reports for assessing model performance on diverse data.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Literal
from swap_correction.ml.api import BatchProcessor
from swap_correction import pivr_loader, error_analysis


def evaluate_model_on_dataset(data_dir: str,
                              model_type: Literal['level1', 'raw', 'raw_data'] = 'level1',
                              ground_truth_level: str = 'level2',
                              output_dir: Optional[str] = None,
                              model_dir: Optional[str] = None) -> Dict:
    """
    Evaluate a trained model on a dataset with ground truth.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing trial subdirectories
    model_type : str
        Type of model to evaluate: 'level1' or 'raw'/'raw_data'
    ground_truth_level : str
        Ground truth level to compare against ('level1' or 'level2')
    output_dir : str, optional
        Directory to save evaluation results (default: ml_analysis/evaluations/)
    model_dir : str, optional
        Directory containing the trained model files. If None, loads from default location.
        
    Returns:
    --------
    dict
        Comprehensive evaluation results
    """
    if output_dir is None:
        output_dir = 'ml_analysis/evaluations'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"EVALUATING {model_type.upper()} MODEL ON DATASET")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Ground truth level: {ground_truth_level}")
    if model_dir:
        print(f"Model directory: {model_dir}")
    print()
    
    # Initialize batch processor with custom model directory if provided
    if model_dir:
        # Create a custom BatchProcessor that loads from the specified model directory
        from swap_correction.ml.api.predictor import SwapPredictor
        from swap_correction.ml.api.model_loader import load_model
        
        # Load model from specified directory
        model, scaler, imputer, feature_names = load_model(
            model_type=model_type,
            model_dir=model_dir
        )
        
        # Create custom predictor
        predictor = SwapPredictor.__new__(SwapPredictor)
        predictor.model_type = model_type
        predictor.filter_sigma = 4.6
        predictor.model = model
        predictor.scaler = scaler
        predictor.imputer = imputer
        predictor.feature_names = feature_names
        
        # Create batch processor with custom predictor
        processor = BatchProcessor.__new__(BatchProcessor)
        processor.predictor = predictor
        processor.model_type = model_type
    else:
        # Use default model
        processor = BatchProcessor(model_type=model_type)
    
    # Evaluate
    results = processor.evaluate_on_dataset(
        data_dir, ground_truth_level=ground_truth_level
    )
    
    # Print summary
    summary = results['summary']
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total trials: {summary['n_trials']}")
    print(f"Valid trials: {summary['n_valid']}")
    
    if summary['n_valid'] > 0:
        print(f"\nPerformance Metrics (mean ± std):")
        print(f"  Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
        print(f"  Recall: {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
        print(f"  Sensitivity: {summary['mean_sensitivity']:.4f} ± {summary['std_sensitivity']:.4f}")
        print(f"  Specificity: {summary['mean_specificity']:.4f} ± {summary['std_specificity']:.4f}")
        print(f"  F1-Score: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
        print(f"\nAdditional Metrics (mean ± std):")
        print(f"  % Swaps Resolved: {summary['mean_pct_swaps_resolved']:.2f}% ± {summary['std_pct_swaps_resolved']:.2f}%")
        print(f"  % Frames Clean (Pre): {summary['mean_pct_frames_clean_pre']:.2f}% ± {summary['std_pct_frames_clean_pre']:.2f}%")
        print(f"  % Frames Clean (Post): {summary['mean_pct_frames_clean_post']:.2f}% ± {summary['std_pct_frames_clean_post']:.2f}%")
    
    # Save results
    dataset_name = os.path.basename(data_dir.rstrip('/'))
    results_file = os.path.join(output_dir, f'evaluation_{model_type}_{dataset_name}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create detailed report
    create_evaluation_report(results, model_type, dataset_name, output_dir)
    
    return results


def create_evaluation_report(results: Dict, model_type: str, dataset_name: str, output_dir: str):
    """Create a detailed markdown evaluation report."""
    summary = results['summary']
    trial_results = results['trial_results']
    
    valid_results = [r for r in trial_results if 'error' not in r]
    
    report = f"""# Model Evaluation Report

**Model Type**: {model_type.upper()}
**Dataset**: {dataset_name}
**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

- **Total Trials**: {summary['n_trials']}
- **Valid Trials**: {summary['n_valid']}
- **Failed Trials**: {summary['n_trials'] - summary['n_valid']}

"""
    
    if summary['n_valid'] > 0:
        report += f"""### Performance Metrics

| Metric | Mean | Std Dev |
|--------|------|---------|
| Precision | {summary['mean_precision']:.4f} | {summary['std_precision']:.4f} |
| Recall | {summary['mean_recall']:.4f} | {summary['std_recall']:.4f} |
| Sensitivity | {summary['mean_sensitivity']:.4f} | {summary['std_sensitivity']:.4f} |
| Specificity | {summary['mean_specificity']:.4f} | {summary['std_specificity']:.4f} |
| F1-Score | {summary['mean_f1']:.4f} | {summary['std_f1']:.4f} |

### Additional Metrics

| Metric | Mean | Std Dev |
|--------|------|---------|
| % Swaps Resolved | {summary['mean_pct_swaps_resolved']:.2f}% | {summary['std_pct_swaps_resolved']:.2f}% |
| % Frames Clean (Pre-correction) | {summary['mean_pct_frames_clean_pre']:.2f}% | {summary['std_pct_frames_clean_pre']:.2f}% |
| % Frames Clean (Post-correction) | {summary['mean_pct_frames_clean_post']:.2f}% | {summary['std_pct_frames_clean_post']:.2f}% |

### Per-Trial Results

| Trial | Precision | Recall | Sensitivity | Specificity | F1 | % Swaps Resolved | % Clean Pre | % Clean Post | TP | FP | FN | TN | Frames | GT Swaps | Pred Swaps |
|-------|-----------|--------|-------------|-------------|----|------------------|-------------|--------------|----|----|----|----|--------|----------|------------|
"""
        for r in valid_results:
            report += f"| {r['trial']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | {r['f1']:.4f} | {r['pct_swaps_resolved']:.2f}% | {r['pct_frames_clean_pre']:.2f}% | {r['pct_frames_clean_post']:.2f}% | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['n_frames']} | {r['n_swapped_gt']} | {r['n_swapped_pred']} |\n"
    
    # Failed trials
    failed = [r for r in trial_results if 'error' in r]
    if failed:
        report += f"\n### Failed Trials\n\n"
        for r in failed:
            report += f"- **{r['trial']}**: {r['error']}\n"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, f'evaluation_report_{model_type}_{dataset_name}.md')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Detailed report saved to: {report_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained ML models on a dataset'
    )
    parser.add_argument('data_dir', type=str,
                       help='Directory containing trial subdirectories')
    parser.add_argument('--model-type', type=str, choices=['level1', 'raw', 'raw_data'],
                       default='level1',
                       help='Type of model to evaluate (default: level1)')
    parser.add_argument('--ground-truth', type=str, choices=['level1', 'level2'],
                       default='level2',
                       help='Ground truth level to compare against (default: level2)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: ml_analysis/evaluations/)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    evaluate_model_on_dataset(
        args.data_dir,
        model_type=args.model_type,
        ground_truth_level=args.ground_truth,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()


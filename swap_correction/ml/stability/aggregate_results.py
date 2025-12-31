"""
Aggregate results across all stability analysis iterations.

Loads results from all iterations and calculates stability metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path


def load_iteration_results(iteration_dirs: List[str]) -> List[Dict]:
    """
    Load results from all iteration directories.
    
    Parameters:
    -----------
    iteration_dirs : list of str
        List of iteration directory paths
        
    Returns:
    --------
    list of dict
        List of iteration results
    """
    all_results = []
    
    for iter_dir in iteration_dirs:
        results_file = os.path.join(iter_dir, 'iteration_results.json')
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            all_results.append(results)
        else:
            print(f"Warning: Results file not found: {results_file}")
    
    return all_results


def extract_metrics(results: List[Dict], model_type: str = 'level1') -> pd.DataFrame:
    """
    Extract performance metrics from iteration results.
    
    Parameters:
    -----------
    results : list of dict
        List of iteration results
    model_type : str
        Model type to extract ('level1' or 'raw')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for each iteration
    """
    metrics_list = []
    
    for i, result in enumerate(results, 1):
        model_key = f'{model_type}_model'
        
        if model_key not in result or 'training' not in result[model_key]:
            continue
        
        training = result[model_key]['training']
        
        # Extract sample_size from result or trial_split
        sample_size = result.get('sample_size', None)
        if sample_size is None and 'trial_split' in result:
            sample_size = result['trial_split'].get('sample_size', None)
        
        metrics = {
            'iteration': i,
            'sample_size': sample_size,
            'train_f1': training.get('train', {}).get('f1', np.nan),
            'train_precision': training.get('train', {}).get('precision', np.nan),
            'train_recall': training.get('train', {}).get('recall', np.nan),
            'train_sensitivity': training.get('train', {}).get('sensitivity', np.nan),
            'train_specificity': training.get('train', {}).get('specificity', np.nan),
            'train_auc': training.get('train', {}).get('auc', np.nan),
            'val_f1': training.get('val', {}).get('f1', np.nan),
            'val_precision': training.get('val', {}).get('precision', np.nan),
            'val_recall': training.get('val', {}).get('recall', np.nan),
            'val_sensitivity': training.get('val', {}).get('sensitivity', np.nan),
            'val_specificity': training.get('val', {}).get('specificity', np.nan),
            'val_auc': training.get('val', {}).get('auc', np.nan),
            'test_f1': training.get('test', {}).get('f1', np.nan),
            'test_precision': training.get('test', {}).get('precision', np.nan),
            'test_recall': training.get('test', {}).get('recall', np.nan),
            'test_sensitivity': training.get('test', {}).get('sensitivity', np.nan),
            'test_specificity': training.get('test', {}).get('specificity', np.nan),
            'test_auc': training.get('test', {}).get('auc', np.nan),
            'n_train_samples': training.get('train', {}).get('n_samples', np.nan),
            'n_val_samples': training.get('val', {}).get('n_samples', np.nan),
            'n_test_samples': training.get('test', {}).get('n_samples', np.nan),
        }
        
        # Add evaluation metrics if available
        # Always try loading from evaluation JSON file first (has most up-to-date metrics)
        # Then fall back to iteration results if JSON file doesn't exist
        eval_summary = None
        iteration_dir = result.get('iteration_dir', '')
        
        if iteration_dir:
            # Determine model type for file pattern
            model_name = 'level1' if model_type == 'level1' else 'raw'
            eval_json_path = os.path.join(iteration_dir, f'{model_type}_model', 'evaluation', 
                                         f'evaluation_{model_name}_test_data.json')
            if os.path.exists(eval_json_path):
                try:
                    with open(eval_json_path, 'r') as f:
                        eval_data = json.load(f)
                    if 'summary' in eval_data:
                        eval_summary = eval_data['summary']
                except Exception:
                    pass
        
        # Fall back to iteration results if JSON file not found
        if eval_summary is None:
            if 'evaluation' in result[model_key] and 'summary' in result[model_key]['evaluation']:
                eval_summary = result[model_key]['evaluation']['summary']
        
        if eval_summary:
            metrics.update({
                'eval_mean_f1': eval_summary.get('mean_f1', np.nan),
                'eval_mean_precision': eval_summary.get('mean_precision', np.nan),
                'eval_mean_recall': eval_summary.get('mean_recall', np.nan),
                'eval_mean_sensitivity': eval_summary.get('mean_sensitivity', np.nan),
                'eval_mean_specificity': eval_summary.get('mean_specificity', np.nan),
                'eval_mean_pct_swaps_resolved': eval_summary.get('mean_pct_swaps_resolved', np.nan),
                'eval_mean_pct_frames_clean_pre': eval_summary.get('mean_pct_frames_clean_pre', np.nan),
                'eval_mean_pct_frames_clean_post': eval_summary.get('mean_pct_frames_clean_post', np.nan),
            })
        
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def calculate_stability_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate stability metrics from metrics DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with metrics for each iteration
        
    Returns:
    --------
    dict
        Dictionary with stability metrics
    """
    metrics = {}
    
    # Primary metric: test F1
    test_f1 = df['test_f1'].dropna()
    
    if len(test_f1) > 0:
        metrics['test_f1'] = {
            'mean': float(test_f1.mean()),
            'std': float(test_f1.std()),
            'min': float(test_f1.min()),
            'max': float(test_f1.max()),
            'median': float(test_f1.median()),
            'cv': float(test_f1.std() / test_f1.mean()) if test_f1.mean() > 0 else np.nan,
            'range': float(test_f1.max() - test_f1.min()),
        }
    
    # Other test metrics
    for metric in ['test_precision', 'test_recall', 'test_sensitivity', 'test_specificity', 'test_auc']:
        values = df[metric].dropna()
        if len(values) > 0:
            metrics[metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'cv': float(values.std() / values.mean()) if values.mean() > 0 else np.nan,
            }
    
    # Evaluation metrics (from evaluation on test dataset)
    for metric in ['eval_mean_f1', 'eval_mean_precision', 'eval_mean_recall', 
                   'eval_mean_sensitivity', 'eval_mean_specificity',
                   'eval_mean_pct_swaps_resolved', 'eval_mean_pct_frames_clean_pre', 
                   'eval_mean_pct_frames_clean_post']:
        values = df[metric].dropna()
        if len(values) > 0:
            metrics[metric] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'cv': float(values.std() / values.mean()) if values.mean() > 0 else np.nan,
            }
    
    # Train/val/test gaps (overfitting indicators)
    train_test_gap = (df['train_f1'] - df['test_f1']).dropna()
    if len(train_test_gap) > 0:
        metrics['train_test_f1_gap'] = {
            'mean': float(train_test_gap.mean()),
            'std': float(train_test_gap.std()),
        }
    
    val_test_gap = (df['val_f1'] - df['test_f1']).dropna()
    if len(val_test_gap) > 0:
        metrics['val_test_f1_gap'] = {
            'mean': float(val_test_gap.mean()),
            'std': float(val_test_gap.std()),
        }
    
    return metrics


def generate_stability_report(all_results: List[Dict], output_dir: str):
    """
    Generate comprehensive stability analysis report.
    
    Parameters:
    -----------
    all_results : list of dict
        List of all iteration results
    output_dir : str
        Output directory for report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for both models
    level1_metrics = extract_metrics(all_results, model_type='level1')
    raw_metrics = extract_metrics(all_results, model_type='raw')
    
    # Group by sample size if available
    if 'sample_size' in level1_metrics.columns:
        # Calculate stability metrics per sample size
        level1_by_size = {}
        raw_by_size = {}
        
        for size in level1_metrics['sample_size'].unique():
            if pd.isna(size):
                continue
            size_metrics_l1 = level1_metrics[level1_metrics['sample_size'] == size]
            size_metrics_raw = raw_metrics[raw_metrics['sample_size'] == size]
            
            level1_by_size[int(size)] = calculate_stability_metrics(size_metrics_l1) if len(size_metrics_l1) > 0 else {}
            raw_by_size[int(size)] = calculate_stability_metrics(size_metrics_raw) if len(size_metrics_raw) > 0 else {}
        
        # Also calculate overall metrics
        level1_stability = calculate_stability_metrics(level1_metrics) if len(level1_metrics) > 0 else {}
        raw_stability = calculate_stability_metrics(raw_metrics) if len(raw_metrics) > 0 else {}
    else:
        # No sample size grouping
        level1_stability = calculate_stability_metrics(level1_metrics) if len(level1_metrics) > 0 else {}
        raw_stability = calculate_stability_metrics(raw_metrics) if len(raw_metrics) > 0 else {}
        level1_by_size = {}
        raw_by_size = {}
    
    # Save metrics DataFrames
    level1_metrics.to_csv(os.path.join(output_dir, 'level1_metrics.csv'), index=False)
    raw_metrics.to_csv(os.path.join(output_dir, 'raw_metrics.csv'), index=False)
    
    # Save stability metrics
    stability_data = {
        'level1': level1_stability,
        'raw': raw_stability
    }
    
    with open(os.path.join(output_dir, 'stability_metrics.json'), 'w') as f:
        json.dump(stability_data, f, indent=2)
    
    # Generate markdown report
    report_lines = [
        "# Model Stability Analysis Report",
        "",
        "## Overview",
        "",
        f"**Total Iterations**: {len(all_results)}",
        f"**Level1 Model Iterations**: {len(level1_metrics)}",
        f"**Raw Model Iterations**: {len(raw_metrics)}",
        ""
    ]
    
    # Add sample size information if available
    if 'sample_size' in level1_metrics.columns:
        sample_sizes = sorted(level1_metrics['sample_size'].dropna().unique())
        if len(sample_sizes) > 0:
            report_lines.append(f"**Sample Sizes Tested**: {', '.join(map(str, sample_sizes))}")
            report_lines.append(f"**Iterations per Sample Size**: {len(all_results) // len(sample_sizes) if len(sample_sizes) > 0 else len(all_results)}")
            report_lines.append("")
        
        # Add per-sample-size analysis
        report_lines.extend([
            "## Analysis by Sample Size",
            ""
        ])
        
        for size in sorted(sample_sizes):
            size_metrics_l1 = level1_metrics[level1_metrics['sample_size'] == size]
            size_metrics_raw = raw_metrics[raw_metrics['sample_size'] == size]
            
            if len(size_metrics_l1) > 0 and 'test_f1' in level1_by_size.get(int(size), {}):
                l1_stats = level1_by_size[int(size)]
                raw_stats = raw_by_size.get(int(size), {})
                
                report_lines.extend([
                    f"### Sample Size: {int(size)}",
                    "",
                    f"**Level1 Model (n={len(size_metrics_l1)} iterations):**",
                ])
                
                # Test F1
                if 'test_f1' in l1_stats:
                    f1 = l1_stats['test_f1']
                    report_lines.append(f"- Mean F1: {f1['mean']:.4f} (std: {f1['std']:.4f}, CV: {f1['cv']:.4f})")
                
                # Test Sensitivity and Specificity
                if 'test_sensitivity' in l1_stats:
                    sens = l1_stats['test_sensitivity']
                    report_lines.append(f"- Mean Sensitivity: {sens['mean']:.4f} (std: {sens['std']:.4f})")
                if 'test_specificity' in l1_stats:
                    spec = l1_stats['test_specificity']
                    report_lines.append(f"- Mean Specificity: {spec['mean']:.4f} (std: {spec['std']:.4f})")
                
                # Evaluation metrics (if available)
                if 'eval_mean_sensitivity' in l1_stats:
                    sens = l1_stats['eval_mean_sensitivity']
                    report_lines.append(f"- Mean Sensitivity (Eval): {sens['mean']:.4f} (std: {sens['std']:.4f})")
                if 'eval_mean_specificity' in l1_stats:
                    spec = l1_stats['eval_mean_specificity']
                    report_lines.append(f"- Mean Specificity (Eval): {spec['mean']:.4f} (std: {spec['std']:.4f})")
                if 'eval_mean_pct_swaps_resolved' in l1_stats:
                    pct_swaps = l1_stats['eval_mean_pct_swaps_resolved']
                    report_lines.append(f"- Mean % Swaps Resolved: {pct_swaps['mean']:.2f}% (std: {pct_swaps['std']:.2f}%)")
                if 'eval_mean_pct_frames_clean_post' in l1_stats:
                    pct_clean = l1_stats['eval_mean_pct_frames_clean_post']
                    report_lines.append(f"- Mean % Frames Clean (Post): {pct_clean['mean']:.2f}% (std: {pct_clean['std']:.2f}%)")
                
                report_lines.append("")
                
                if raw_stats and 'test_f1' in raw_stats:
                    report_lines.extend([
                        f"**Raw Model (n={len(size_metrics_raw)} iterations):**",
                    ])
                    
                    # Test F1
                    f1 = raw_stats['test_f1']
                    report_lines.append(f"- Mean F1: {f1['mean']:.4f} (std: {f1['std']:.4f}, CV: {f1['cv']:.4f})")
                    
                    # Test Sensitivity and Specificity
                    if 'test_sensitivity' in raw_stats:
                        sens = raw_stats['test_sensitivity']
                        report_lines.append(f"- Mean Sensitivity: {sens['mean']:.4f} (std: {sens['std']:.4f})")
                    if 'test_specificity' in raw_stats:
                        spec = raw_stats['test_specificity']
                        report_lines.append(f"- Mean Specificity: {spec['mean']:.4f} (std: {spec['std']:.4f})")
                    
                    # Evaluation metrics (if available)
                    if 'eval_mean_sensitivity' in raw_stats:
                        sens = raw_stats['eval_mean_sensitivity']
                        report_lines.append(f"- Mean Sensitivity (Eval): {sens['mean']:.4f} (std: {sens['std']:.4f})")
                    if 'eval_mean_specificity' in raw_stats:
                        spec = raw_stats['eval_mean_specificity']
                        report_lines.append(f"- Mean Specificity (Eval): {spec['mean']:.4f} (std: {spec['std']:.4f})")
                    if 'eval_mean_pct_swaps_resolved' in raw_stats:
                        pct_swaps = raw_stats['eval_mean_pct_swaps_resolved']
                        report_lines.append(f"- Mean % Swaps Resolved: {pct_swaps['mean']:.2f}% (std: {pct_swaps['std']:.2f}%)")
                    if 'eval_mean_pct_frames_clean_post' in raw_stats:
                        pct_clean = raw_stats['eval_mean_pct_frames_clean_post']
                        report_lines.append(f"- Mean % Frames Clean (Post): {pct_clean['mean']:.2f}% (std: {pct_clean['std']:.2f}%)")
                    
                    report_lines.append("")
    
    report_lines.extend([
        "## Level1 Model Stability (Overall)",
        ""
    ])
    
    if 'test_f1' in level1_stability:
        f1_stats = level1_stability['test_f1']
        report_lines.extend([
            "### Test F1-Score",
            f"- **Mean**: {f1_stats['mean']:.4f}",
            f"- **Std Dev**: {f1_stats['std']:.4f}",
            f"- **Min**: {f1_stats['min']:.4f}",
            f"- **Max**: {f1_stats['max']:.4f}",
            f"- **Range**: {f1_stats['range']:.4f}",
            f"- **Coefficient of Variation**: {f1_stats['cv']:.4f}",
            ""
        ])
    
    # Add sensitivity and specificity if available
    if 'test_sensitivity' in level1_stability:
        sens_stats = level1_stability['test_sensitivity']
        report_lines.extend([
            "### Test Sensitivity",
            f"- **Mean**: {sens_stats['mean']:.4f}",
            f"- **Std Dev**: {sens_stats['std']:.4f}",
            f"- **Min**: {sens_stats['min']:.4f}",
            f"- **Max**: {sens_stats['max']:.4f}",
            ""
        ])
    
    if 'test_specificity' in level1_stability:
        spec_stats = level1_stability['test_specificity']
        report_lines.extend([
            "### Test Specificity",
            f"- **Mean**: {spec_stats['mean']:.4f}",
            f"- **Std Dev**: {spec_stats['std']:.4f}",
            f"- **Min**: {spec_stats['min']:.4f}",
            f"- **Max**: {spec_stats['max']:.4f}",
            ""
        ])
    
    # Add evaluation metrics if available
    if 'eval_mean_pct_swaps_resolved' in level1_stability:
        report_lines.extend([
            "### Evaluation Metrics (Test Dataset)",
        ])
        if 'eval_mean_sensitivity' in level1_stability:
            sens = level1_stability['eval_mean_sensitivity']
            report_lines.append(f"- **Mean Sensitivity**: {sens['mean']:.4f} (std: {sens['std']:.4f})")
        if 'eval_mean_specificity' in level1_stability:
            spec = level1_stability['eval_mean_specificity']
            report_lines.append(f"- **Mean Specificity**: {spec['mean']:.4f} (std: {spec['std']:.4f})")
        pct_swaps = level1_stability['eval_mean_pct_swaps_resolved']
        report_lines.append(f"- **Mean % Swaps Resolved**: {pct_swaps['mean']:.2f}% (std: {pct_swaps['std']:.2f}%)")
        if 'eval_mean_pct_frames_clean_pre' in level1_stability:
            pct_pre = level1_stability['eval_mean_pct_frames_clean_pre']
            report_lines.append(f"- **Mean % Frames Clean (Pre-correction)**: {pct_pre['mean']:.2f}% (std: {pct_pre['std']:.2f}%)")
        if 'eval_mean_pct_frames_clean_post' in level1_stability:
            pct_post = level1_stability['eval_mean_pct_frames_clean_post']
            report_lines.append(f"- **Mean % Frames Clean (Post-correction)**: {pct_post['mean']:.2f}% (std: {pct_post['std']:.2f}%)")
        report_lines.append("")
    
    report_lines.extend([
        "## Raw Model Stability",
        ""
    ])
    
    if 'test_f1' in raw_stability:
        f1_stats = raw_stability['test_f1']
        report_lines.extend([
            "### Test F1-Score",
            f"- **Mean**: {f1_stats['mean']:.4f}",
            f"- **Std Dev**: {f1_stats['std']:.4f}",
            f"- **Min**: {f1_stats['min']:.4f}",
            f"- **Max**: {f1_stats['max']:.4f}",
            f"- **Range**: {f1_stats['range']:.4f}",
            f"- **Coefficient of Variation**: {f1_stats['cv']:.4f}",
            ""
        ])
    
    # Add sensitivity and specificity if available
    if 'test_sensitivity' in raw_stability:
        sens_stats = raw_stability['test_sensitivity']
        report_lines.extend([
            "### Test Sensitivity",
            f"- **Mean**: {sens_stats['mean']:.4f}",
            f"- **Std Dev**: {sens_stats['std']:.4f}",
            f"- **Min**: {sens_stats['min']:.4f}",
            f"- **Max**: {sens_stats['max']:.4f}",
            ""
        ])
    
    if 'test_specificity' in raw_stability:
        spec_stats = raw_stability['test_specificity']
        report_lines.extend([
            "### Test Specificity",
            f"- **Mean**: {spec_stats['mean']:.4f}",
            f"- **Std Dev**: {spec_stats['std']:.4f}",
            f"- **Min**: {spec_stats['min']:.4f}",
            f"- **Max**: {spec_stats['max']:.4f}",
            ""
        ])
    
    # Add evaluation metrics if available
    if 'eval_mean_pct_swaps_resolved' in raw_stability:
        report_lines.extend([
            "### Evaluation Metrics (Test Dataset)",
        ])
        if 'eval_mean_sensitivity' in raw_stability:
            sens = raw_stability['eval_mean_sensitivity']
            report_lines.append(f"- **Mean Sensitivity**: {sens['mean']:.4f} (std: {sens['std']:.4f})")
        if 'eval_mean_specificity' in raw_stability:
            spec = raw_stability['eval_mean_specificity']
            report_lines.append(f"- **Mean Specificity**: {spec['mean']:.4f} (std: {spec['std']:.4f})")
        pct_swaps = raw_stability['eval_mean_pct_swaps_resolved']
        report_lines.append(f"- **Mean % Swaps Resolved**: {pct_swaps['mean']:.2f}% (std: {pct_swaps['std']:.2f}%)")
        if 'eval_mean_pct_frames_clean_pre' in raw_stability:
            pct_pre = raw_stability['eval_mean_pct_frames_clean_pre']
            report_lines.append(f"- **Mean % Frames Clean (Pre-correction)**: {pct_pre['mean']:.2f}% (std: {pct_pre['std']:.2f}%)")
        if 'eval_mean_pct_frames_clean_post' in raw_stability:
            pct_post = raw_stability['eval_mean_pct_frames_clean_post']
            report_lines.append(f"- **Mean % Frames Clean (Post-correction)**: {pct_post['mean']:.2f}% (std: {pct_post['std']:.2f}%)")
        report_lines.append("")
    
    # Model comparison
    report_lines.extend([
        "## Model Comparison",
        ""
    ])
    
    if 'test_f1' in level1_stability and 'test_f1' in raw_stability:
        level1_mean = level1_stability['test_f1']['mean']
        raw_mean = raw_stability['test_f1']['mean']
        level1_cv = level1_stability['test_f1']['cv']
        raw_cv = raw_stability['test_f1']['cv']
        
        report_lines.extend([
            "### Performance",
            f"- **Level1 Mean F1**: {level1_mean:.4f}",
            f"- **Raw Mean F1**: {raw_mean:.4f}",
            f"- **Difference**: {level1_mean - raw_mean:.4f}",
            ""
        ])
        
        # Add evaluation sensitivity and specificity comparison
        if 'eval_mean_sensitivity' in level1_stability and 'eval_mean_sensitivity' in raw_stability:
            l1_sens = level1_stability['eval_mean_sensitivity']['mean']
            raw_sens = raw_stability['eval_mean_sensitivity']['mean']
            report_lines.append(f"- **Level1 Mean Sensitivity (Eval)**: {l1_sens:.4f}")
            report_lines.append(f"- **Raw Mean Sensitivity (Eval)**: {raw_sens:.4f}")
            report_lines.append(f"- **Sensitivity Difference**: {l1_sens - raw_sens:.4f}")
            report_lines.append("")
        
        if 'eval_mean_specificity' in level1_stability and 'eval_mean_specificity' in raw_stability:
            l1_spec = level1_stability['eval_mean_specificity']['mean']
            raw_spec = raw_stability['eval_mean_specificity']['mean']
            report_lines.append(f"- **Level1 Mean Specificity (Eval)**: {l1_spec:.4f}")
            report_lines.append(f"- **Raw Mean Specificity (Eval)**: {raw_spec:.4f}")
            report_lines.append(f"- **Specificity Difference**: {l1_spec - raw_spec:.4f}")
            report_lines.append("")
        
        # Add evaluation metrics comparison
        if 'eval_mean_pct_swaps_resolved' in level1_stability and 'eval_mean_pct_swaps_resolved' in raw_stability:
            l1_swaps = level1_stability['eval_mean_pct_swaps_resolved']['mean']
            raw_swaps = raw_stability['eval_mean_pct_swaps_resolved']['mean']
            report_lines.append(f"- **Level1 Mean % Swaps Resolved**: {l1_swaps:.2f}%")
            report_lines.append(f"- **Raw Mean % Swaps Resolved**: {raw_swaps:.2f}%")
            report_lines.append(f"- **Difference**: {l1_swaps - raw_swaps:.2f}%")
            report_lines.append("")
        
        if 'eval_mean_pct_frames_clean_post' in level1_stability and 'eval_mean_pct_frames_clean_post' in raw_stability:
            l1_clean = level1_stability['eval_mean_pct_frames_clean_post']['mean']
            raw_clean = raw_stability['eval_mean_pct_frames_clean_post']['mean']
            report_lines.append(f"- **Level1 Mean % Frames Clean (Post)**: {l1_clean:.2f}%")
            report_lines.append(f"- **Raw Mean % Frames Clean (Post)**: {raw_clean:.2f}%")
            report_lines.append(f"- **Difference**: {l1_clean - raw_clean:.2f}%")
            report_lines.append("")
        
        report_lines.extend([
            "### Stability (Lower CV = More Stable)",
            f"- **Level1 CV**: {level1_cv:.4f}",
            f"- **Raw CV**: {raw_cv:.4f}",
            f"- **More Stable**: {'Level1' if level1_cv < raw_cv else 'Raw'}",
            ""
        ])
    
    report_content = "\n".join(report_lines)
    
    report_file = os.path.join(output_dir, 'stability_report.md')
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"Stability report saved to: {report_file}")
    
    return {
        'level1_metrics': level1_metrics,
        'raw_metrics': raw_metrics,
        'level1_stability': level1_stability,
        'raw_stability': raw_stability
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate stability analysis results')
    parser.add_argument('results_dir', type=str,
                       help='Directory containing iteration subdirectories')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for aggregated results (default: results_dir)')
    
    args = parser.parse_args()
    
    # Find all iteration directories
    results_path = Path(args.results_dir)
    iteration_dirs = [
        str(d) for d in results_path.iterdir()
        if d.is_dir() and d.name.startswith('iteration_')
    ]
    iteration_dirs.sort()
    
    print(f"Found {len(iteration_dirs)} iteration directories")
    
    # Load results
    all_results = load_iteration_results(iteration_dirs)
    
    # Generate report
    output_dir = args.output_dir or args.results_dir
    generate_stability_report(all_results, output_dir)


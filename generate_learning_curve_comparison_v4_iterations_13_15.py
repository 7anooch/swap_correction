#!/usr/bin/env python3
"""
Generate learning curve comparison report for iterations 13-15 (features v4).
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_learning_curve_data(iteration_id, model_type):
    """Load learning curve results for a specific iteration and model type."""
    base_dir = f'learning_curve_analysis_v4_iter{iteration_id:03d}'
    subdir = f'v4_{model_type}'
    
    json_file = os.path.join(base_dir, subdir, 'learning_curve_results.json')
    csv_file = os.path.join(base_dir, subdir, 'learning_curve_results.csv')
    
    # Try JSON first, then CSV
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            # Convert to DataFrame if it's a list of dicts
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and 'results' in data:
                return pd.DataFrame(data['results'])
            else:
                return pd.DataFrame([data])
        except:
            pass
    
    if os.path.exists(csv_file):
        try:
            return pd.read_csv(csv_file)
        except:
            pass
    
    return None


def calculate_metrics(df):
    """Calculate key metrics from learning curve data."""
    if df is None or df.empty:
        return None
    
    # Best test F1
    best_test_f1_idx = df['test_f1'].idxmax()
    best_test_f1 = df.loc[best_test_f1_idx]
    
    # F1 at 50% and 100%
    row_50 = df[df['train_size_fraction'] == 0.5]
    row_100 = df[df['train_size_fraction'] == 1.0]
    
    f1_at_50 = row_50.iloc[0]['test_f1'] if len(row_50) > 0 else None
    f1_at_100 = row_100.iloc[0]['test_f1'] if len(row_100) > 0 else None
    
    # Improvement from 50% to 100%
    improvement_50_100 = (f1_at_100 - f1_at_50) if (f1_at_50 is not None and f1_at_100 is not None) else None
    
    # Improvement in last 20% (80% to 100%)
    row_80 = df[df['train_size_fraction'] == 0.8]
    f1_at_80 = row_80.iloc[0]['test_f1'] if len(row_80) > 0 else None
    improvement_last_20 = (f1_at_100 - f1_at_80) if (f1_at_80 is not None and f1_at_100 is not None) else None
    
    # Best % clean post
    best_pct_clean = best_test_f1.get('test_pct_clean_post', best_test_f1.get('test_pct_frames_clean_post', None))
    
    return {
        'best_test_f1': best_test_f1['test_f1'],
        'best_pct_clean': best_pct_clean,
        'f1_at_50': f1_at_50,
        'f1_at_100': f1_at_100,
        'improvement_50_100': improvement_50_100,
        'improvement_last_20': improvement_last_20,
        'more_data_helpful': improvement_50_100 > 0 if improvement_50_100 is not None else None
    }


def generate_comparison_report():
    """Generate comprehensive learning curve comparison report for iterations 13-15."""
    print("=" * 80)
    print("GENERATING LEARNING CURVE COMPARISON REPORT (Iterations 13-15, Features V4)")
    print("=" * 80)
    print()
    
    iterations = [13, 14, 15]
    model_types = ['level1', 'raw']
    
    all_data = {}
    
    # Load data for all combinations
    for iter_id in iterations:
        all_data[iter_id] = {}
        for model_type in model_types:
            df = load_learning_curve_data(iter_id, model_type)
            if df is not None:
                metrics = calculate_metrics(df)
                all_data[iter_id][model_type] = {
                    'df': df,
                    'metrics': metrics
                }
                print(f"✓ Loaded iteration {iter_id:03d} {model_type}")
            else:
                print(f"⚠ Missing iteration {iter_id:03d} {model_type}")
    
    # Generate report
    report_lines = [
        "# Learning Curve Analysis: Iterations 13-15 (Features V4)",
        "",
        "## Overview",
        "",
        "This report analyzes learning curves for iterations 13-15 using features_v4 (~36 features).",
        "Each iteration used a sample size of 100 trials, randomly selected from the main dataset.",
        "",
        "## Summary Statistics",
        "",
        "### Per-Iteration Results",
        "",
        "| Iteration | Model Type | Best Test F1 | Best Test % Clean | F1 at 50% | F1 at 100% | Improvement (50→100%) | Improvement (Last 20%) | More Data Helpful |",
        "|:----------|:-----------|--------------:|------------------:|----------:|------------:|----------------------:|----------------------:|:------------------|"
    ]
    
    # Per-iteration results
    for iter_id in iterations:
        for model_type in model_types:
            if iter_id in all_data and model_type in all_data[iter_id]:
                metrics = all_data[iter_id][model_type]['metrics']
                if metrics:
                    best_f1 = f"{metrics['best_test_f1']:.4f}"
                    best_clean = f"{metrics['best_pct_clean']:.2f}%" if metrics['best_pct_clean'] is not None else "N/A"
                    f1_50 = f"{metrics['f1_at_50']:.4f}" if metrics['f1_at_50'] is not None else "N/A"
                    f1_100 = f"{metrics['f1_at_100']:.4f}" if metrics['f1_at_100'] is not None else "N/A"
                    imp_50_100 = f"{metrics['improvement_50_100']:+.4f}" if metrics['improvement_50_100'] is not None else "N/A"
                    imp_last_20 = f"{metrics['improvement_last_20']:+.4f}" if metrics['improvement_last_20'] is not None else "N/A"
                    helpful = "Yes" if metrics['more_data_helpful'] else "No" if metrics['more_data_helpful'] is not None else "N/A"
                    
                    report_lines.append(
                        f"| {iter_id:03d} | {model_type.upper()} | {best_f1} | {best_clean} | {f1_50} | {f1_100} | {imp_50_100} | {imp_last_20} | {helpful} |"
                    )
    
    # Calculate averages across iterations
    report_lines.extend([
        "",
        "## Average Across Iterations",
        "",
        "| Model Type | Avg Best Test F1 | Avg F1 at 50% | Avg F1 at 100% | Avg Improvement (50→100%) | Avg Improvement (Last 20%) | More Data Helpful (Count) |",
        "|:-----------|-----------------:|--------------:|---------------:|--------------------------:|---------------------------:|:--------------------------|"
    ])
    
    for model_type in model_types:
        best_f1_values = []
        f1_50_values = []
        f1_100_values = []
        imp_50_100_values = []
        imp_last_20_values = []
        helpful_count = 0
        total_count = 0
        
        for iter_id in iterations:
            if iter_id in all_data and model_type in all_data[iter_id]:
                metrics = all_data[iter_id][model_type]['metrics']
                if metrics:
                    best_f1_values.append(metrics['best_test_f1'])
                    if metrics['f1_at_50'] is not None:
                        f1_50_values.append(metrics['f1_at_50'])
                    if metrics['f1_at_100'] is not None:
                        f1_100_values.append(metrics['f1_at_100'])
                    if metrics['improvement_50_100'] is not None:
                        imp_50_100_values.append(metrics['improvement_50_100'])
                    if metrics['improvement_last_20'] is not None:
                        imp_last_20_values.append(metrics['improvement_last_20'])
                    if metrics['more_data_helpful'] is not None:
                        if metrics['more_data_helpful']:
                            helpful_count += 1
                        total_count += 1
        
        if best_f1_values:
            avg_best_f1 = np.mean(best_f1_values)
            std_best_f1 = np.std(best_f1_values)
            avg_f1_50 = np.mean(f1_50_values) if f1_50_values else None
            avg_f1_100 = np.mean(f1_100_values) if f1_100_values else None
            avg_imp_50_100 = np.mean(imp_50_100_values) if imp_50_100_values else None
            std_imp_50_100 = np.std(imp_50_100_values) if imp_50_100_values else None
            avg_imp_last_20 = np.mean(imp_last_20_values) if imp_last_20_values else None
            std_imp_last_20 = np.std(imp_last_20_values) if imp_last_20_values else None
            
            avg_f1_str = f"{avg_best_f1:.4f} ± {std_best_f1:.4f}"
            avg_f1_50_str = f"{avg_f1_50:.4f}" if avg_f1_50 is not None else "N/A"
            avg_f1_100_str = f"{avg_f1_100:.4f}" if avg_f1_100 is not None else "N/A"
            avg_imp_50_100_str = f"{avg_imp_50_100:+.4f} ± {std_imp_50_100:.4f}" if avg_imp_50_100 is not None else "N/A"
            avg_imp_last_20_str = f"{avg_imp_last_20:+.4f} ± {std_imp_last_20:.4f}" if avg_imp_last_20 is not None else "N/A"
            helpful_str = f"{helpful_count}/{total_count}" if total_count > 0 else "N/A"
            
            report_lines.append(
                f"| {model_type.upper()} | {avg_f1_str} | {avg_f1_50_str} | {avg_f1_100_str} | {avg_imp_50_100_str} | {avg_imp_last_20_str} | {helpful_str} |"
            )
    
    # Key findings
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "### Learning Curve Characteristics:",
        "",
        "1. **Data Efficiency**: Analysis of how performance changes with training data size",
        "2. **Consistency**: Comparison of learning curves across different random samples",
        "3. **Model Comparison**: Level1 vs Raw model performance trends",
        "",
        "### Recommendations:",
        "",
        "- Based on the learning curves, determine if more training data would improve performance",
        "- Assess the stability of learning curves across different random samples",
        "- Compare Level1 and Raw models to understand their data requirements",
        ""
    ])
    
    # Save report
    report_file = 'learning_curve_comparison_v4_iterations_13_15.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✓ Comparison report saved to: {report_file}")
    print()
    
    # Generate comparison plots if data is available
    try:
        generate_comparison_plots(all_data, iterations, model_types)
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")


def generate_comparison_plots(all_data, iterations, model_types):
    """Generate comparison plots for learning curves."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for model_idx, model_type in enumerate(model_types):
        ax = axes[model_idx]
        
        for iter_id in iterations:
            if iter_id in all_data and model_type in all_data[iter_id]:
                df = all_data[iter_id][model_type]['df']
                if df is not None and not df.empty:
                    ax.plot(df['train_size_fraction'], df['test_f1'], 
                           marker='o', label=f'Iter {iter_id:03d}', linewidth=2, markersize=6)
        
        ax.set_xlabel('Training Data Fraction', fontsize=12)
        ax.set_ylabel('Test F1-Score', fontsize=12)
        ax.set_title(f'Learning Curves: {model_type.upper()} Model (Iterations 13-15)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plot_file = 'learning_curve_comparison_v4_iterations_13_15.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_file}")
    plt.close()


if __name__ == '__main__':
    generate_comparison_report()


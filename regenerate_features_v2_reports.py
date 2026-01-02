#!/usr/bin/env python3
"""
Regenerate evaluation markdown reports for features_v2 testing.

This script finds all evaluation_results.json files in stability_analysis_v3_features_v2
and regenerates the corresponding markdown reports.
"""

import os
import json
from pathlib import Path
from swap_correction.ml.evaluation.evaluate_on_dataset import create_evaluation_report


def find_evaluation_json_files(base_dir: str):
    """Find all evaluation_results.json files."""
    json_files = []
    base_path = Path(base_dir)
    
    for json_file in base_path.rglob('evaluation_results.json'):
        json_files.append(str(json_file))
    
    return sorted(json_files)


def regenerate_report(json_file: str):
    """Regenerate markdown report from evaluation_results.json."""
    json_path = Path(json_file)
    
    # Load evaluation results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Determine model type and dataset name from path
    # Path format: .../iteration_XXX/{model_type}_model/evaluation_results.json
    parts = json_path.parts
    model_type = None
    iteration = None
    
    for i, part in enumerate(parts):
        if part.startswith('iteration_'):
            iteration = part
        if part.endswith('_model'):
            model_type = part.replace('_model', '')
    
    if not model_type:
        print(f"  ⚠ Could not determine model type from path: {json_path}")
        return False
    
    # Dataset name is typically 'test_data'
    dataset_name = 'test_data'
    
    # Output directory is the same as where the JSON file is
    output_dir = json_path.parent
    
    try:
        create_evaluation_report(
            results, model_type, dataset_name, str(output_dir)
        )
        print(f"  ✓ Regenerated report for {model_type} model in {iteration}")
        return True
    except Exception as e:
        print(f"  ✗ Error regenerating report: {e}")
        return False


def main():
    """Main function."""
    base_dir = 'stability_analysis_v3_features_v2'
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return
    
    print("=" * 80)
    print("REGENERATING EVALUATION REPORTS FOR features_v2")
    print("=" * 80)
    print(f"Base directory: {base_dir}\n")
    
    # Find all evaluation JSON files
    json_files = find_evaluation_json_files(base_dir)
    
    if not json_files:
        print("No evaluation_results.json files found.")
        return
    
    print(f"Found {len(json_files)} evaluation results files.\n")
    
    # Regenerate reports
    success_count = 0
    for json_file in json_files:
        print(f"Processing: {json_file}")
        if regenerate_report(json_file):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Regenerated {success_count}/{len(json_files)} reports")
    print("=" * 80)


if __name__ == '__main__':
    main()


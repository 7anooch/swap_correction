#!/usr/bin/env python3
"""
Test features_v4.py on the same data splits from stability_analysis_v3 iterations 007-012.

This script:
1. Loads trial splits from iterations 007-012
2. Uses features_v4.py for feature extraction (by patching imports)
3. Trains models using the same splits
4. Stores results in a separate directory for comparison
"""

import os
import sys
import json
import shutil
import importlib
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def patch_features_module():
    """Patch the features module to use features_v4."""
    import swap_correction.ml.features.features_v4 as features_v4_module
    import swap_correction.ml.features as features_module
    
    # Save original
    original_extract = features_module.extract_all_frame_features_optimized
    
    # Replace with v4
    features_module.extract_all_frame_features_optimized = features_v4_module.extract_all_frame_features_optimized
    
    # Also patch in submodules that import it
    try:
        import swap_correction.ml.training.train_model as train_module
        train_module.extract_all_frame_features_optimized = features_v4_module.extract_all_frame_features_optimized
    except:
        pass
    
    try:
        import swap_correction.ml.api.predictor as predictor_module
        predictor_module.extract_all_frame_features_optimized = features_v4_module.extract_all_frame_features_optimized
    except:
        pass
    
    return original_extract


def restore_features_module(original_extract):
    """Restore the original features module."""
    import swap_correction.ml.features as features_module
    features_module.extract_all_frame_features_optimized = original_extract
    
    # Reload modules to clear cache
    importlib.reload(features_module)
    try:
        import swap_correction.ml.training.train_model as train_module
        importlib.reload(train_module)
    except:
        pass
    try:
        import swap_correction.ml.api.predictor as predictor_module
        importlib.reload(predictor_module)
    except:
        pass


def load_iteration_splits(base_dir: str, iteration_ids: list) -> dict:
    """Load trial splits from specified iterations."""
    splits = {}
    for iter_id in iteration_ids:
        iter_dir = os.path.join(base_dir, f'iteration_{iter_id:03d}')
        split_file = os.path.join(iter_dir, 'trial_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits[iter_id] = json.load(f)
        else:
            print(f"Warning: Split file not found for iteration {iter_id:03d}")
    return splits


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.floating, np.float64, np.float32)):
        return float(obj) if isinstance(obj, (np.floating, np.float64, np.float32)) else int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def main():
    """Main function."""
    base_dir = 'stability_analysis_v3'
    output_dir = 'stability_analysis_v3_features_v4'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    print("=" * 80)
    print("TESTING features_v4.py ON STABILITY ANALYSIS SPLITS")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Iterations: {iteration_ids}\n")
    
    # Load trial splits
    print("Loading trial splits...")
    splits = load_iteration_splits(base_dir, iteration_ids)
    print(f"Loaded splits for {len(splits)} iterations\n")
    
    # Patch features module
    print("Patching features module to use features_v4...")
    original_extract = patch_features_module()
    print("✓ Features module patched\n")
    
    try:
        # Process each iteration
        for iter_id in iteration_ids:
            if iter_id not in splits:
                print(f"Skipping iteration {iter_id:03d} - no split data found")
                continue
            
            print("=" * 80)
            print(f"ITERATION {iter_id:03d}")
            print("=" * 80)
            
            split = splits[iter_id]
            train_trials = split.get('train', [])
            val_trials = split.get('val', [])
            test_trials = split.get('test', [])
            
            # Convert to full paths
            parent_dir = split.get('parent_dir', '/Users/hind/Documents/UCSB/Neuroscience/KirstenData/new_data/Main_dataset')
            all_trials = [os.path.join(parent_dir, t) for t in train_trials + val_trials + test_trials]
            
            iter_output_dir = os.path.join(output_dir, f'iteration_{iter_id:03d}')
            os.makedirs(iter_output_dir, exist_ok=True)
            
            # Save split info
            split_file = os.path.join(iter_output_dir, 'trial_split.json')
            with open(split_file, 'w') as f:
                json.dump(split, f, indent=2)
            
            # Process both model types
            for model_type in ['level1', 'raw']:
                print(f"\n[{iter_id}] Processing {model_type} model...")
                model_dir = os.path.join(iter_output_dir, f'{model_type}_model')
                os.makedirs(model_dir, exist_ok=True)
                
                try:
                    # [1/4] Prepare ML data
                    print(f"  [1/4] Preparing ML data with features_v4...")
                    from swap_correction.ml.training.prepare_data import main as prepare_data_main
                    
                    ml_data_dir = os.path.join(model_dir, 'ml_data')
                    prepare_data_main(
                        use_raw_data=(model_type == 'raw'),
                        trial_dirs=all_trials,
                        test_data_dir=parent_dir,
                        output_dir=ml_data_dir
                    )
                    print(f"  ✓ Data preparation complete")
                    
                    # [2/4] Train model
                    print(f"  [2/4] Training model...")
                    from swap_correction.ml.training.train_model import (
                        load_training_data, prepare_train_val_test_split,
                        train_xgboost_model, evaluate_model
                    )
                    from sklearn.impute import SimpleImputer
                    from sklearn.preprocessing import StandardScaler
                    import pickle
                    
                    # Load training data (will use patched features_v4)
                    features_df, labels, trial_names, split = load_training_data(
                        ml_data_dir=ml_data_dir,
                        test_data_dir=parent_dir,
                        use_raw_data=(model_type == 'raw'),
                        trial_dirs=all_trials
                    )
                    
                    # Ensure trial names are basenames for matching
                    trial_names_basename = np.array([os.path.basename(t) if os.path.isabs(str(t)) else str(t) for t in trial_names])
                    
                    # Normalize split dictionary to use basenames
                    split_basename = {
                        'train': [os.path.basename(t) if os.path.isabs(str(t)) else str(t) for t in split['train']],
                        'val': [os.path.basename(t) if os.path.isabs(str(t)) else str(t) for t in split['val']],
                        'test': [os.path.basename(t) if os.path.isabs(str(t)) else str(t) for t in split['test']]
                    }
                    
                    # Prepare splits
                    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
                        features_df, labels, trial_names_basename, split_basename
                    )
                    
                    # Handle NaN
                    imputer = SimpleImputer(strategy='median')
                    X_train = imputer.fit_transform(X_train)
                    X_val = imputer.transform(X_val)
                    X_test = imputer.transform(X_test)
                    
                    # Scale
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train
                    model, xgb_model = train_xgboost_model(
                        X_train_scaled, y_train, X_val_scaled, y_val
                    )
                    
                    # Evaluate
                    results = evaluate_model(
                        model, X_train_scaled, y_train,
                        X_val_scaled, y_val,
                        X_test_scaled, y_test
                    )
                    
                    # Save model and preprocessors
                    model_file = os.path.join(model_dir, 'swap_detector_xgb.pkl')
                    scaler_file = os.path.join(model_dir, 'feature_scaler.pkl')
                    imputer_file = os.path.join(model_dir, 'feature_imputer.pkl')
                    feature_names_file = os.path.join(model_dir, 'feature_names.pkl')
                    
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(scaler, f)
                    with open(imputer_file, 'wb') as f:
                        pickle.dump(imputer, f)
                    with open(feature_names_file, 'wb') as f:
                        pickle.dump(list(features_df.columns), f)
                    
                    # Save training results
                    results_file = os.path.join(model_dir, 'training_results.json')
                    serializable_results = convert_to_serializable(results)
                    with open(results_file, 'w') as f:
                        json.dump(serializable_results, f, indent=2)
                    
                    print(f"  ✓ Training complete")
                    print(f"    Test F1: {results['test']['f1']:.4f}")
                    print(f"    Test Precision: {results['test']['precision']:.4f}")
                    print(f"    Test Recall: {results['test']['recall']:.4f}")
                    
                    # [3/4] Create test data directory with symlinks
                    print(f"  [3/4] Setting up test data...")
                    test_data_dir = os.path.join(model_dir, 'test_data')
                    if os.path.exists(test_data_dir):
                        shutil.rmtree(test_data_dir)
                    os.makedirs(test_data_dir, exist_ok=True)
                    
                    for trial_path in test_trials:
                        trial_name = os.path.basename(trial_path)
                        link_path = os.path.join(test_data_dir, trial_name)
                        if not os.path.exists(link_path):
                            try:
                                os.symlink(trial_path, link_path)
                            except:
                                # Fallback: copy if symlink fails
                                try:
                                    shutil.copytree(trial_path, link_path)
                                except:
                                    pass
                    
                    # [4/4] Evaluate using the models we just trained (with features_v4)
                    print(f"  [4/4] Evaluating model...")
                    from swap_correction.ml.api.batch_processor import BatchProcessor
                    from swap_correction.ml.api.predictor import SwapPredictor
                    from swap_correction.ml.api.model_loader import load_model
                    import pickle
                    
                    # Load the model we just trained (with features_v4)
                    model, scaler, imputer, feature_names = load_model(
                        model_type=model_type,
                        model_dir=model_dir  # Use the model from this iteration
                    )
                    
                    # Create a custom predictor with the trained model
                    # We need to bypass SwapPredictor.__init__ to avoid loading default models
                    custom_predictor = SwapPredictor.__new__(SwapPredictor)  # Create without calling __init__
                    custom_predictor.model_type = model_type
                    custom_predictor.filter_sigma = 4.6
                    # Set the model components directly (these are the models we just trained with features_v4)
                    custom_predictor.model = model
                    custom_predictor.scaler = scaler
                    custom_predictor.imputer = imputer
                    custom_predictor.feature_names = feature_names
                    
                    # Create batch processor with custom predictor
                    # We also need to bypass BatchProcessor.__init__ to avoid creating a default predictor
                    processor = BatchProcessor.__new__(BatchProcessor)  # Create without calling __init__
                    processor.predictor = custom_predictor
                    processor.model_type = model_type
                    
                    # Evaluate
                    eval_results = processor.evaluate_on_dataset(
                        test_data_dir,
                        ground_truth_level='level2'
                    )
                    
                    # Save evaluation results
                    eval_file = os.path.join(model_dir, 'evaluation_results.json')
                    with open(eval_file, 'w') as f:
                        json.dump(convert_to_serializable(eval_results), f, indent=2, default=str)
                    
                    # Create evaluation report
                    from swap_correction.ml.evaluation.evaluate_on_dataset import create_evaluation_report
                    dataset_name = os.path.basename(test_data_dir.rstrip('/'))
                    create_evaluation_report(
                        eval_results, model_type, dataset_name,
                        os.path.join(model_dir, 'evaluation')
                    )
                    
                    print(f"  ✓ Evaluation complete")
                    if 'summary' in eval_results and eval_results['summary']:
                        summary = eval_results['summary']
                        if 'mean_f1' in summary:
                            print(f"    Mean F1: {summary['mean_f1']:.4f}")
                        if 'mean_precision' in summary:
                            print(f"    Mean Precision: {summary['mean_precision']:.4f}")
                        if 'mean_recall' in summary:
                            print(f"    Mean Recall: {summary['mean_recall']:.4f}")
                        if 'n_valid' in summary:
                            print(f"    Valid trials: {summary['n_valid']}/{summary.get('n_trials', '?')}")
                    else:
                        print(f"    (Summary not available)")
                    
                except Exception as e:
                    print(f"  ✗ ERROR processing {model_type} model: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n✓ Iteration {iter_id:03d} complete")
    
    finally:
        # Restore original features module
        print("\nRestoring original features module...")
        restore_features_module(original_extract)
        print("✓ Features module restored")
    
    print("\n" + "=" * 80)
    print("ALL ITERATIONS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()


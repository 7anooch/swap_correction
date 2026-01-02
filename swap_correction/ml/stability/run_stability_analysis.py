"""
Main stability analysis pipeline.

Orchestrates the full pipeline: data preparation, training, evaluation,
and report generation for multiple iterations.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List
from swap_correction.ml.stability.find_valid_trials import find_valid_trial_directories
from swap_correction.ml.stability.sample_trials import create_splits_for_iterations
from swap_correction.ml.evaluation.evaluate_on_dataset import evaluate_model_on_dataset


def run_single_iteration(iteration_id: int, trial_split: Dict[str, List[str]], 
                        parent_dir: str, output_base_dir: str,
                        n_iterations: int = 10) -> Dict:
    """
    Run a single iteration of the stability analysis.
    
    Parameters:
    -----------
    iteration_id : int
        Iteration number (1-indexed)
    trial_split : dict
        Dictionary with 'train', 'val', 'test' keys containing trial names
    parent_dir : str
        Parent directory containing trial subdirectories
    output_base_dir : str
        Base output directory
    n_iterations : int
        Total number of iterations (for progress display)
        
    Returns:
    --------
    dict
        Results dictionary with paths and status
    """
    iteration_name = f"iteration_{iteration_id:03d}"
    iteration_dir = os.path.join(output_base_dir, iteration_name)
    os.makedirs(iteration_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"ITERATION {iteration_id}/{n_iterations}: {iteration_name}")
    print("=" * 80)
    
    results = {
        'iteration_id': iteration_id,
        'iteration_name': iteration_name,
        'iteration_dir': iteration_dir,
        'trial_split': trial_split,
        'raw_model': {},
        'level1_model': {},
        'status': 'in_progress'
    }
    
    # Save split for this iteration
    split_file = os.path.join(iteration_dir, 'trial_split.json')
    with open(split_file, 'w') as f:
        json.dump(trial_split, f, indent=2)
    
    # Get full paths for trial directories
    # Check if split contains full paths or just names
    if trial_split['train'] and os.path.isabs(trial_split['train'][0]):
        # Already full paths
        train_trials = trial_split['train']
        val_trials = trial_split['val']
        test_trials = trial_split['test']
    else:
        # Just names, need to construct paths (for backward compatibility)
        train_trials = [os.path.join(parent_dir, name) for name in trial_split['train']]
        val_trials = [os.path.join(parent_dir, name) for name in trial_split['val']]
        test_trials = [os.path.join(parent_dir, name) for name in trial_split['test']]
    
    all_trials = train_trials + val_trials + test_trials
    
    # Process both raw and level1 models
    for model_type, use_raw_data in [('raw', True), ('level1', False)]:
        print(f"\n--- Processing {model_type.upper()} Model ---")
        
        model_output_dir = os.path.join(iteration_dir, f'{model_type}_model')
        os.makedirs(model_output_dir, exist_ok=True)
        
        ml_data_dir = os.path.join(model_output_dir, 'ml_data')
        os.makedirs(ml_data_dir, exist_ok=True)
        
        try:
            # Step 1: Prepare data
            print(f"\n[1/4] Preparing data for {model_type} model...")
            prepare_cmd = [
                sys.executable, '-m', 'swap_correction.ml.training.prepare_data',
                '--use-raw-data' if use_raw_data else '',
                '--output-dir', ml_data_dir
            ]
            prepare_cmd = [c for c in prepare_cmd if c]  # Remove empty strings
            
            # Create temporary file with trial directories
            trial_dirs_file = os.path.join(model_output_dir, 'trial_dirs.json')
            with open(trial_dirs_file, 'w') as f:
                json.dump(all_trials, f)
            
            prepare_cmd.extend(['--trial-dirs', trial_dirs_file])
            
            result = subprocess.run(prepare_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR preparing data: {result.stderr}")
                results[f'{model_type}_model']['prepare_error'] = result.stderr
                continue
            
            print("  Data preparation complete")
            
            # Step 2: Train model
            print(f"\n[2/4] Training {model_type} model...")
            train_cmd = [
                sys.executable, '-m', 'swap_correction.ml.training.train_model',
                '--ml-data-dir', ml_data_dir,
                '--output-dir', model_output_dir,
                '--use-raw-data' if use_raw_data else '',
                '--trial-dirs', trial_dirs_file
            ]
            train_cmd = [c for c in train_cmd if c]  # Remove empty strings
            
            result = subprocess.run(train_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR training model: {result.stderr}")
                results[f'{model_type}_model']['train_error'] = result.stderr
                continue
            
            print("  Training complete")
            
            # Step 3: Evaluate model
            print(f"\n[3/4] Evaluating {model_type} model...")
            
            # Create test data directory with symlinks to test trials only
            test_data_dir = os.path.join(model_output_dir, 'test_data')
            if os.path.exists(test_data_dir):
                import shutil
                shutil.rmtree(test_data_dir)
            os.makedirs(test_data_dir, exist_ok=True)
            
            # Create symlinks to test trial directories
            for test_trial_path in test_trials:
                test_trial_name = os.path.basename(test_trial_path)
                dst = os.path.join(test_data_dir, test_trial_name)
                if os.path.exists(test_trial_path) and not os.path.exists(dst):
                    try:
                        os.symlink(test_trial_path, dst)
                    except:
                        # If symlink fails, try copying (slower but works)
                        try:
                            import shutil
                            shutil.copytree(test_trial_path, dst)
                        except:
                            pass
            
            # Evaluate on test data directory (or parent_dir if symlinks failed)
            eval_data_dir = test_data_dir if len(os.listdir(test_data_dir)) > 0 else parent_dir
            eval_results = evaluate_model_on_dataset(
                eval_data_dir,
                model_type=model_type,
                ground_truth_level='level2',
                output_dir=os.path.join(model_output_dir, 'evaluation'),
                model_dir=model_output_dir  # Use the model from this iteration
            )
            
            # Save evaluation results
            eval_file = os.path.join(model_output_dir, 'evaluation_results.json')
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            results[f'{model_type}_model']['evaluation'] = eval_results
            results[f'{model_type}_model']['evaluation_file'] = eval_file
            
            # Load training results
            training_results_file = os.path.join(model_output_dir, 'training_results.json')
            if os.path.exists(training_results_file):
                with open(training_results_file, 'r') as f:
                    training_results = json.load(f)
                results[f'{model_type}_model']['training'] = training_results
                results[f'{model_type}_model']['training_file'] = training_results_file
            
            print("  Evaluation complete")
            
            # Step 4: Generate evaluation reports/figures (if needed)
            print(f"\n[4/4] Generating reports for {model_type} model...")
            # Reports are generated by evaluate_on_dataset
            print("  Reports complete")
            
            results[f'{model_type}_model']['status'] = 'success'
            
        except Exception as e:
            print(f"ERROR in {model_type} model pipeline: {e}")
            import traceback
            traceback.print_exc()
            results[f'{model_type}_model']['status'] = 'error'
            results[f'{model_type}_model']['error'] = str(e)
    
    results['status'] = 'complete'
    
    # Save iteration results
    results_file = os.path.join(iteration_dir, 'iteration_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def run_full_stability_analysis(parent_dir: str, n_iterations: int = 10,
                               n_samples: int = 30, output_dir: str = 'stability_analysis',
                               base_seed: int = 42, sample_sizes: List[int] = None):
    """
    Run full stability analysis with multiple iterations.
    
    Parameters:
    -----------
    parent_dir : str
        Parent directory containing trial subdirectories
    n_iterations : int
        Number of iterations to run per sample size (default: 10)
    n_samples : int
        Number of trials to sample per iteration (default: 30)
        Ignored if sample_sizes is provided
    output_dir : str
        Base output directory (default: 'stability_analysis')
    base_seed : int
        Base random seed (default: 42)
    sample_sizes : list of int, optional
        List of sample sizes to test (e.g., [30, 40, 50])
        If provided, runs n_iterations for each sample size
    """
    # Determine sample sizes to use
    if sample_sizes is None:
        sample_sizes = [n_samples]
    
    print("=" * 80)
    print("MODEL STABILITY ANALYSIS")
    print("=" * 80)
    print(f"Parent directory: {parent_dir}")
    print(f"Iterations per sample size: {n_iterations}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Total iterations: {len(sample_sizes) * n_iterations}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Find all valid trials (filtering out identical level1/level2)
    print("Step 1: Finding valid trial directories (filtering identical level1/level2)...")
    valid_trials = find_valid_trial_directories(parent_dir, filter_identical=True)
    print(f"Found {len(valid_trials)} valid trials with swaps\n")
    
    # Save list of valid trials
    valid_trials_file = os.path.join(output_dir, 'valid_trials.json')
    with open(valid_trials_file, 'w') as f:
        json.dump(valid_trials, f, indent=2)
    
    all_results = []
    iteration_counter = 1
    total_start_time = time.time()
    
    # Run analysis for each sample size
    for sample_size in sample_sizes:
        print("\n" + "=" * 80)
        print(f"SAMPLE SIZE: {sample_size} trials per iteration")
        print("=" * 80)
        
        # Step 2: Create splits for all iterations at this sample size
        print(f"\nStep 2: Creating trial splits for {n_iterations} iterations (sample size: {sample_size})...")
        splits = create_splits_for_iterations(
            valid_trials,
            n_iterations=n_iterations,
            n_samples=sample_size,
            base_seed=base_seed
        )
        print()
        
        # Save splits for this sample size
        splits_file = os.path.join(output_dir, f'all_splits_n{sample_size}.json')
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        # Step 3: Run each iteration
        print(f"Step 3: Running {n_iterations} iterations (sample size: {sample_size})...")
        sample_size_start = time.time()
        
        for i, split in enumerate(splits, 1):
            iteration_start = time.time()
            
            # Add sample size metadata to split
            split['sample_size'] = sample_size
            
            results = run_single_iteration(
                iteration_counter, split, parent_dir, output_dir, 
                n_iterations=len(sample_sizes) * n_iterations
            )
            
            # Add sample size to results
            results['sample_size'] = sample_size
            
            all_results.append(results)
            
            iteration_time = time.time() - iteration_start
            sample_elapsed = time.time() - sample_size_start
            total_elapsed = time.time() - total_start_time
            avg_time_per_iter = sample_elapsed / i
            remaining_for_sample = avg_time_per_iter * (n_iterations - i)
            total_remaining = (total_elapsed / iteration_counter) * (len(sample_sizes) * n_iterations - iteration_counter)
            
            print(f"\nIteration {iteration_counter} (sample size {sample_size}, {i}/{n_iterations}) "
                  f"completed in {iteration_time/60:.1f} minutes")
            print(f"Elapsed for this sample size: {sample_elapsed/60:.1f} minutes")
            print(f"Estimated remaining for this sample size: {remaining_for_sample/60:.1f} minutes")
            print(f"Total elapsed: {total_elapsed/60:.1f} minutes")
            print(f"Estimated total remaining: {total_remaining/60:.1f} minutes")
            
            iteration_counter += 1
    
    # Save all results
    all_results_file = os.path.join(output_dir, 'all_iteration_results.json')
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    total_time = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("STABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Total iterations: {len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run model stability analysis')
    parser.add_argument('--parent-dir', type=str, required=True,
                       help='Parent directory containing trial subdirectories')
    parser.add_argument('--n-iterations', type=int, default=10,
                       help='Number of iterations per sample size (default: 10)')
    parser.add_argument('--n-samples', type=int, default=30,
                       help='Number of trials per iteration (default: 30, ignored if --sample-sizes provided)')
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=None,
                       help='List of sample sizes to test (e.g., --sample-sizes 30 40 50)')
    parser.add_argument('--output-dir', type=str, default='stability_analysis',
                       help='Output directory (default: stability_analysis)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    
    args = parser.parse_args()
    
    run_full_stability_analysis(
        args.parent_dir,
        n_iterations=args.n_iterations,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        sample_sizes=args.sample_sizes
    )


#!/usr/bin/env python3
"""
Run stability analysis iterations 13-15 with sample size 100 using features_v4.

This script:
1. Patches the features module to use features_v4
2. Runs stability analysis for iterations 13-15 with sample size 100
3. Uses the same random selection method as previous stability analyses
"""

import os
import sys
import json
import importlib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def patch_features_module():
    """Patch the features module to use features_v4 by modifying __init__.py import."""
    import sys
    import importlib
    
    # Read the current __init__.py
    init_file = 'swap_correction/ml/features/__init__.py'
    with open(init_file, 'r') as f:
        original_init_content = f.read()
    
    # Create patched version that imports from features_v4
    patched_init_content = original_init_content.replace(
        'from swap_correction.ml.features.features import',
        'from swap_correction.ml.features.features_v4 import'
    )
    
    # Write patched version
    with open(init_file, 'w') as f:
        f.write(patched_init_content)
    
    # Reload the module to pick up the change
    if 'swap_correction.ml.features' in sys.modules:
        importlib.reload(sys.modules['swap_correction.ml.features'])
    if 'swap_correction.ml.features.features' in sys.modules:
        importlib.reload(sys.modules['swap_correction.ml.features.features'])
    
    # Also reload any modules that import from it
    modules_to_reload = [
        'swap_correction.ml.training.train_model',
        'swap_correction.ml.training.prepare_data',
        'swap_correction.ml.api.predictor',
    ]
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except:
                pass
    
    # Verify the patch worked
    from swap_correction.ml.features import extract_all_frame_features_optimized
    print(f"  ✓ Patched features module to use features_v4")
    
    return original_init_content


def restore_features_module(original_init_content):
    """Restore the original features module by restoring __init__.py."""
    import sys
    import importlib
    
    # Restore original __init__.py
    init_file = 'swap_correction/ml/features/__init__.py'
    with open(init_file, 'w') as f:
        f.write(original_init_content)
    
    # Reload the module to pick up the change
    if 'swap_correction.ml.features' in sys.modules:
        importlib.reload(sys.modules['swap_correction.ml.features'])
    if 'swap_correction.ml.features.features' in sys.modules:
        importlib.reload(sys.modules['swap_correction.ml.features.features'])
    
    # Also reload any modules that import from it
    modules_to_reload = [
        'swap_correction.ml.training.train_model',
        'swap_correction.ml.training.prepare_data',
        'swap_correction.ml.api.predictor',
    ]
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except:
                pass


def main():
    """Run stability analysis iterations 13-15 with features_v4."""
    from swap_correction.ml.stability.run_stability_analysis import run_full_stability_analysis
    
    parent_dir = '/Users/hind/Documents/UCSB/Neuroscience/KirstenData/new_data/Main_dataset'
    output_dir = 'stability_analysis_v3_features_v4'
    sample_size = 100
    n_iterations = 3  # Iterations 13, 14, 15
    base_seed = 42 + 12  # Offset by 12 to get iterations 13-15
    
    print("=" * 80)
    print("STABILITY ANALYSIS: ITERATIONS 13-15 (Features V4, Sample Size 100)")
    print("=" * 80)
    print(f"Parent directory: {parent_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample size: {sample_size}")
    print(f"Number of iterations: {n_iterations}")
    print(f"Base seed: {base_seed}")
    print()
    
    # Patch features module
    print("Patching features module to use features_v4...")
    original_extract = patch_features_module()
    print("✓ Features module patched\n")
    
    try:
        # Run stability analysis
        # We need to run it in a way that creates iterations 13-15
        # The run_full_stability_analysis function will create iterations starting from 1
        # So we need to either modify it or run individual iterations
        
        # Let's use the underlying functions to create specific iterations
        from swap_correction.ml.stability.find_valid_trials import find_valid_trial_directories
        from swap_correction.ml.stability.sample_trials import create_splits_for_iterations
        from swap_correction.ml.stability.run_stability_analysis import run_single_iteration
        
        print("Finding valid trial directories...")
        valid_trials = find_valid_trial_directories(
            parent_dir,
            filter_identical=True
        )
        print(f"Found {len(valid_trials)} valid trials\n")
        
        # Create splits for iterations 13-15
        # Use base_seed = 42 + 12 to get iteration 13's seed (since iterations are 1-indexed)
        print("Creating trial splits for iterations 13-15...")
        iteration_ids = [13, 14, 15]
        splits = create_splits_for_iterations(
            valid_trials,
            n_iterations=len(iteration_ids),
            n_samples=sample_size,
            base_seed=base_seed  # This will create seeds 54, 55, 56 (for iterations 13, 14, 15)
        )
        print(f"Created {len(splits)} splits\n")
        
        # Add sample_size to each split
        for split in splits:
            split['sample_size'] = sample_size
            split['parent_dir'] = parent_dir
        
        # Run each iteration
        for i, iteration_id in enumerate(iteration_ids):
            if i < len(splits):
                print(f"\n{'='*80}")
                print(f"RUNNING ITERATION {iteration_id}")
                print(f"{'='*80}\n")
                
                result = run_single_iteration(
                    iteration_id=iteration_id,
                    trial_split=splits[i],
                    parent_dir=parent_dir,
                    output_base_dir=output_dir,
                    n_iterations=len(iteration_ids)
                )
                
                # Save iteration results
                iteration_dir = os.path.join(output_dir, f'iteration_{iteration_id:03d}')
                results_file = os.path.join(iteration_dir, 'iteration_results.json')
                with open(results_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                print(f"\n✓ Iteration {iteration_id} complete")
        
        print("\n" + "=" * 80)
        print("ALL ITERATIONS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"Iterations: 013, 014, 015")
        print()
        
    finally:
        # Restore original features module
        print("Restoring original features module...")
        restore_features_module(original_extract)
        print("✓ Features module restored")


if __name__ == '__main__':
    main()


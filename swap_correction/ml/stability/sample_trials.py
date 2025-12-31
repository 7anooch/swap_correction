"""
Random sampling and train/val/test splitting for stability analysis.

Provides functions to randomly sample trial directories and create
train/validation/test splits for multiple iterations.
"""

import random
import os
from typing import List, Dict, Tuple
from pathlib import Path


def sample_trials(valid_trials: List[str], n_samples: int = 30, 
                 random_seed: int = None) -> List[str]:
    """
    Randomly sample N trial directories from a list of valid trials.
    
    Parameters:
    -----------
    valid_trials : list of str
        List of valid trial directory paths
    n_samples : int
        Number of trials to sample (default: 30)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    list of str
        List of sampled trial directory paths
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    if n_samples > len(valid_trials):
        raise ValueError(
            f"Cannot sample {n_samples} trials from {len(valid_trials)} available trials"
        )
    
    sampled = random.sample(valid_trials, n_samples)
    
    return sampled


def split_trials(trial_dirs: List[str], train_ratio: float = 0.7,
                val_ratio: float = 0.15, test_ratio: float = 0.15,
                random_seed: int = None, return_names: bool = False) -> Dict[str, List[str]]:
    """
    Split trial directories into train/validation/test sets.
    
    Parameters:
    -----------
    trial_dirs : list of str
        List of trial directory paths
    train_ratio : float
        Proportion for training set (default: 0.7)
    val_ratio : float
        Proportion for validation set (default: 0.15)
    test_ratio : float
        Proportion for test set (default: 0.15)
    random_seed : int, optional
        Random seed for reproducibility
    return_names : bool
        If True, return trial names (basenames) instead of full paths (default: False)
        
    Returns:
    --------
    dict
        Dictionary with keys 'train', 'val', 'test' containing lists of trial paths
        (or names if return_names=True)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if random_seed is not None:
        random.seed(random_seed)
    
    # Shuffle trials
    shuffled = trial_dirs.copy()
    random.shuffle(shuffled)
    
    n_trials = len(shuffled)
    n_train = int(n_trials * train_ratio)
    n_val = int(n_trials * val_ratio)
    
    train_trials = shuffled[:n_train]
    val_trials = shuffled[n_train:n_train + n_val]
    test_trials = shuffled[n_train + n_val:]
    
    if return_names:
        # Return basenames instead of full paths
        return {
            'train': [os.path.basename(d) for d in train_trials],
            'val': [os.path.basename(d) for d in val_trials],
            'test': [os.path.basename(d) for d in test_trials]
        }
    else:
        # Return full paths
        return {
            'train': train_trials,
            'val': val_trials,
            'test': test_trials
        }


def create_splits_for_iterations(valid_trials: List[str], n_iterations: int = 10,
                                n_samples: int = 30, base_seed: int = 42) -> List[Dict[str, List[str]]]:
    """
    Create train/val/test splits for multiple iterations.
    
    Each iteration uses a different random seed to ensure different samples.
    
    Parameters:
    -----------
    valid_trials : list of str
        List of all valid trial directory paths
    n_iterations : int
        Number of iterations (default: 10)
    n_samples : int
        Number of trials to sample per iteration (default: 30)
    base_seed : int
        Base random seed (each iteration uses base_seed + iteration_id)
        
    Returns:
    --------
    list of dict
        List of split dictionaries, one per iteration
    """
    splits = []
    
    for i in range(n_iterations):
        iteration_seed = base_seed + i
        
        # Sample trials for this iteration
        sampled_trials = sample_trials(valid_trials, n_samples=n_samples, 
                                      random_seed=iteration_seed)
        
        # Create train/val/test split (return full paths, not just names)
        split = split_trials(sampled_trials, random_seed=iteration_seed, return_names=False)
        
        # Also create a version with names for display
        split_with_names = split_trials(sampled_trials, random_seed=iteration_seed, return_names=True)
        
        # Store both full paths and names
        split['train_names'] = split_with_names['train']
        split['val_names'] = split_with_names['val']
        split['test_names'] = split_with_names['test']
        
        splits.append(split)
        
        print(f"Iteration {i+1}: {len(split['train'])} train, "
              f"{len(split['val'])} val, {len(split['test'])} test")
    
    return splits


if __name__ == '__main__':
    import argparse
    from swap_correction.ml.stability.find_valid_trials import find_valid_trial_directories
    
    parser = argparse.ArgumentParser(description='Create trial splits for stability analysis')
    parser.add_argument('parent_dir', type=str, help='Parent directory containing trials')
    parser.add_argument('--n-iterations', type=int, default=10,
                       help='Number of iterations (default: 10)')
    parser.add_argument('--n-samples', type=int, default=30,
                       help='Number of trials per iteration (default: 30)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    parser.add_argument('--output', type=str, default='trial_splits.json',
                       help='Output JSON file for splits')
    
    args = parser.parse_args()
    
    # Find valid trials
    valid_trials = find_valid_trial_directories(args.parent_dir)
    
    # Create splits
    splits = create_splits_for_iterations(
        valid_trials, 
        n_iterations=args.n_iterations,
        n_samples=args.n_samples,
        base_seed=args.base_seed
    )
    
    # Save splits
    import json
    with open(args.output, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSaved {len(splits)} iteration splits to: {args.output}")


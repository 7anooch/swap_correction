#!/usr/bin/env python3
"""
Prepare machine learning training data from ground truth labels.

Extracts frame-level and segment-level labels by comparing level1 (auto-corrected)
vs level2 (manually corrected) data for all trials.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from swap_correction import pivr_loader as loader
from swap_correction import error_analysis
from swap_correction import utils


def get_test_data_path():
    """Get the default test data directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'swap_correction', 'tests', 'test_data')


def extract_labels_for_trial(trial_dir: str) -> tuple:
    """
    Extract frame-level and segment-level labels for a single trial.
    
    Parameters:
    -----------
    trial_dir : str
        Directory containing trial data files
        
    Returns:
    --------
    tuple
        (frame_labels_df, segment_labels_df, stats_dict)
        frame_labels_df: DataFrame with columns [trial, frame_idx, is_swapped]
        segment_labels_df: DataFrame with columns [trial, start_frame, end_frame, is_swapped]
        stats_dict: Dictionary with statistics about the trial
    """
    trial_name = os.path.basename(trial_dir)
    
    # Load level1 and level2 data
    csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level1.csv')]
    if not csv_files:
        return None, None, None
    
    level1_file = csv_files[0]
    level1_data = loader.load_raw_data(trial_dir, level1_file, px2mm=True)
    
    csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level2.csv')]
    if not csv_files:
        return None, None, None
    
    level2_file = csv_files[0]
    level2_data = loader.load_raw_data(trial_dir, level2_file, px2mm=True)
    
    # Get swapped frames
    swapped_frames = error_analysis.identify_swapped_frames(level1_data, level2_data)
    
    # Create frame-level labels
    n_frames = min(len(level1_data), len(level2_data))
    frame_labels = pd.DataFrame({
        'trial': [trial_name] * n_frames,
        'frame_idx': np.arange(n_frames),
        'is_swapped': np.zeros(n_frames, dtype=bool)
    })
    frame_labels.loc[swapped_frames, 'is_swapped'] = True
    
    # Get swap segments
    swap_segments = error_analysis.get_swap_segments(level1_data, level2_data)
    
    # Create segment-level labels
    if len(swap_segments) > 0:
        segment_labels = pd.DataFrame({
            'trial': [trial_name] * len(swap_segments),
            'start_frame': swap_segments[:, 0],
            'end_frame': swap_segments[:, 1],
            'is_swapped': [True] * len(swap_segments)
        })
    else:
        segment_labels = pd.DataFrame({
            'trial': [],
            'start_frame': [],
            'end_frame': [],
            'is_swapped': []
        })
    
    # Calculate statistics
    n_swapped_frames = len(swapped_frames)
    error_rate = (n_swapped_frames / n_frames * 100) if n_frames > 0 else 0.0
    
    stats = {
        'trial': trial_name,
        'total_frames': n_frames,
        'swapped_frames': n_swapped_frames,
        'error_rate': error_rate,
        'num_segments': len(swap_segments),
        'is_perfect': (n_swapped_frames == 0)
    }
    
    if len(swap_segments) > 0:
        segment_lengths = swap_segments[:, 1] - swap_segments[:, 0] + 1
        stats['mean_segment_length'] = float(np.mean(segment_lengths))
        stats['median_segment_length'] = float(np.median(segment_lengths))
        stats['min_segment_length'] = int(np.min(segment_lengths))
        stats['max_segment_length'] = int(np.max(segment_lengths))
    else:
        stats['mean_segment_length'] = 0.0
        stats['median_segment_length'] = 0.0
        stats['min_segment_length'] = 0
        stats['max_segment_length'] = 0
    
    return frame_labels, segment_labels, stats


def create_train_test_split(trial_names: list, train_ratio: float = 0.7, 
                           val_ratio: float = 0.15, test_ratio: float = 0.15,
                           random_seed: int = 42) -> dict:
    """
    Create train/validation/test split stratified by trial.
    
    Parameters:
    -----------
    trial_names : list
        List of all trial names
    train_ratio : float
        Proportion for training set
    val_ratio : float
        Proportion for validation set
    test_ratio : float
        Proportion for test set
        
    Returns:
    --------
    dict
        Dictionary mapping 'train', 'val', 'test' to lists of trial names
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    shuffled = np.random.permutation(trial_names)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    split = {
        'train': shuffled[:n_train].tolist(),
        'val': shuffled[n_train:n_train+n_val].tolist(),
        'test': shuffled[n_train+n_val:].tolist()
    }
    
    return split


def main():
    """Main entry point."""
    test_data_dir = get_test_data_path()
    
    if not os.path.exists(test_data_dir):
        print(f"Error: Test data directory not found: {test_data_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = 'ml_data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PREPARING ML TRAINING DATA")
    print("=" * 80)
    print(f"Test data directory: {test_data_dir}\n")
    
    # Get all trial directories
    trial_dirs = [os.path.join(test_data_dir, d) for d in os.listdir(test_data_dir)
                  if os.path.isdir(os.path.join(test_data_dir, d))]
    trial_dirs.sort()
    
    print(f"Found {len(trial_dirs)} trials\n")
    
    # Extract labels for all trials
    all_frame_labels = []
    all_segment_labels = []
    all_stats = []
    
    for i, trial_dir in enumerate(trial_dirs):
        trial_name = os.path.basename(trial_dir)
        print(f"[{i+1}/{len(trial_dirs)}] Processing: {trial_name}", end=' ... ', flush=True)
        
        try:
            frame_labels, segment_labels, stats = extract_labels_for_trial(trial_dir)
            
            if frame_labels is not None:
                all_frame_labels.append(frame_labels)
                all_segment_labels.append(segment_labels)
                all_stats.append(stats)
                print(f"OK ({stats['swapped_frames']} swapped frames, {stats['num_segments']} segments)")
            else:
                print("SKIPPED (missing level1 or level2 data)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not all_frame_labels:
        print("\nError: No valid trials found!")
        sys.exit(1)
    
    # Combine all labels
    frame_labels_df = pd.concat(all_frame_labels, ignore_index=True)
    segment_labels_df = pd.concat(all_segment_labels, ignore_index=True)
    stats_df = pd.DataFrame(all_stats)
    
    # Save labels
    frame_labels_file = os.path.join(output_dir, 'training_labels.csv')
    segment_labels_file = os.path.join(output_dir, 'segment_labels.csv')
    stats_file = os.path.join(output_dir, 'trial_statistics.csv')
    
    frame_labels_df.to_csv(frame_labels_file, index=False)
    segment_labels_df.to_csv(segment_labels_file, index=False)
    stats_df.to_csv(stats_file, index=False)
    
    print(f"\nLabels saved:")
    print(f"  Frame labels: {frame_labels_file}")
    print(f"  Segment labels: {segment_labels_file}")
    print(f"  Statistics: {stats_file}")
    
    # Create train/test split
    trial_names = stats_df['trial'].tolist()
    split = create_train_test_split(trial_names, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    split_file = os.path.join(output_dir, 'train_test_split.json')
    with open(split_file, 'w') as f:
        json.dump(split, f, indent=2)
    
    print(f"  Train/test split: {split_file}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATA STATISTICS")
    print("=" * 80)
    
    total_frames = len(frame_labels_df)
    swapped_frames = frame_labels_df['is_swapped'].sum()
    perfect_trials = stats_df[stats_df['is_perfect'] == True]
    problematic_trials = stats_df[stats_df['is_perfect'] == False]
    
    print(f"\nTotal frames: {total_frames:,}")
    print(f"Swapped frames: {swapped_frames:,} ({swapped_frames/total_frames*100:.2f}%)")
    print(f"Non-swapped frames: {total_frames - swapped_frames:,} ({(total_frames-swapped_frames)/total_frames*100:.2f}%)")
    
    print(f"\nTotal trials: {len(stats_df)}")
    print(f"Perfect trials (0% error): {len(perfect_trials)} ({len(perfect_trials)/len(stats_df)*100:.1f}%)")
    print(f"Problematic trials (>0% error): {len(problematic_trials)} ({len(problematic_trials)/len(stats_df)*100:.1f}%)")
    
    if len(problematic_trials) > 0:
        print(f"\nProblematic trial error rates:")
        print(f"  Mean: {problematic_trials['error_rate'].mean():.2f}%")
        print(f"  Median: {problematic_trials['error_rate'].median():.2f}%")
        print(f"  Min: {problematic_trials['error_rate'].min():.2f}%")
        print(f"  Max: {problematic_trials['error_rate'].max():.2f}%")
    
    total_segments = len(segment_labels_df)
    if total_segments > 0:
        print(f"\nTotal swap segments: {total_segments}")
        print(f"Mean segment length: {stats_df['mean_segment_length'].mean():.1f} frames")
        print(f"Median segment length: {stats_df['median_segment_length'].median():.1f} frames")
        print(f"Min segment length: {stats_df['min_segment_length'].min()} frames")
        print(f"Max segment length: {stats_df['max_segment_length'].max()} frames")
    
    print(f"\nTrain/test split:")
    print(f"  Train: {len(split['train'])} trials")
    print(f"  Validation: {len(split['val'])} trials")
    print(f"  Test: {len(split['test'])} trials")
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


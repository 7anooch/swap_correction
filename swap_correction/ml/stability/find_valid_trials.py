"""
Find valid trial directories containing all required data files.

Recursively searches a parent directory to find all subdirectories that contain
the three required CSV files: *_data.csv, *_level1.csv, and *_level2.csv.

Also filters out trials where level1 and level2 are identical (no swaps),
to maintain fair comparison with raw model.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from swap_correction import pivr_loader


def find_valid_trial_directories(parent_dir: str, filter_identical: bool = True) -> List[str]:
    """
    Find all valid trial directories in a parent directory.
    
    A valid trial directory must contain:
    - At least one file ending with '_data.csv' (raw data)
    - At least one file ending with '_level1.csv' (auto-corrected)
    - At least one file ending with '_level2.csv' (ground truth)
    
    Optionally filters out trials where level1 and level2 are identical
    (no swaps) to maintain fair comparison with raw model.
    
    Parameters:
    -----------
    parent_dir : str
        Parent directory to search recursively
    filter_identical : bool
        If True, filter out trials where level1 == level2 (default: True)
        
    Returns:
    --------
    list of str
        List of valid trial directory paths (absolute paths)
    """
    parent_path = Path(parent_dir).resolve()
    
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent_dir}")
    
    if not parent_path.is_dir():
        raise ValueError(f"Path is not a directory: {parent_dir}")
    
    valid_trials = []
    identical_count = 0
    checked_count = 0
    error_count = 0
    start_time = time.time()
    
    # Set random seed for reproducible sampling
    random.seed(42)
    np.random.seed(42)
    
    print(f"Searching for valid trial directories in: {parent_dir}")
    print("This may take a few minutes for large directories...")
    if filter_identical:
        print("Filtering out trials where level1 == level2 (using comprehensive frame sampling)...")
    sys.stdout.flush()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_path):
        root_path = Path(root)
        
        # Check if this directory has all required files
        # Raw data: _data.csv (but not _data_level1.csv or _data_level2.csv)
        has_data = any(f.endswith('_data.csv') and '_level' not in f for f in files)
        # Level1: _level1.csv or _data_level1.csv
        has_level1 = any(f.endswith('_level1.csv') for f in files)
        # Level2: _level2.csv or _data_level2.csv
        has_level2 = any(f.endswith('_level2.csv') for f in files)
        
        if has_data and has_level1 and has_level2:
            checked_count += 1
            
            # Show progress every 10 trials checked
            if checked_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = checked_count / elapsed if elapsed > 0 else 0
                remaining_estimate = (len(valid_trials) + identical_count) / rate if rate > 0 else 0
                print(f"  Checked {checked_count} trials: {len(valid_trials)} valid, {identical_count} identical, "
                      f"{error_count} errors | Rate: {rate:.1f} trials/sec | "
                      f"Elapsed: {elapsed/60:.1f}min")
                sys.stdout.flush()
            
            # Check if level1 and level2 are identical (if filtering enabled)
            if filter_identical:
                try:
                    # Find level1 and level2 files
                    level1_files = [f for f in files if f.endswith('_level1.csv')]
                    level2_files = [f for f in files if f.endswith('_level2.csv')]
                    
                    if level1_files and level2_files:
                        level1_file = level1_files[0]
                        level2_file = level2_files[0]
                        
                        # Quick check: compare file sizes first (if different, definitely not identical)
                        level1_path = root_path / level1_file
                        level2_path = root_path / level2_file
                        
                        if level1_path.stat().st_size != level2_path.stat().st_size:
                            # Files are different sizes, definitely not identical
                            pass  # Include this trial
                        else:
                            # Files are same size, do a quick sample check
                            # Load only a sample of frames (first, middle, last) for speed
                            check_start = time.time()
                            
                            # Load full data (unavoidable, but we'll do a faster comparison)
                            level1_data = pivr_loader.load_raw_data(str(root_path), level1_file, px2mm=True)
                            level2_data = pivr_loader.load_raw_data(str(root_path), level2_file, px2mm=True)
                            
                            # Quick comparison: check only a sample of frames instead of all
                            min_len = min(len(level1_data), len(level2_data))
                            
                            # Comprehensive sampling strategy:
                            # - Divide dataset into chunks (every 1500 frames)
                            # - Randomly sample 50 frames from each chunk
                            # - Also include first 50, middle 50, and last 50 for safety
                            # - If any differ, we know there are swaps
                            
                            sample_indices = []
                            
                            # Always check first, middle, and last chunks
                            sample_indices.extend(range(min(50, min_len)))  # First 50
                            if min_len > 100:
                                mid_start = max(0, min_len // 2 - 25)
                                mid_end = min(min_len // 2 + 25, min_len)
                                sample_indices.extend(range(mid_start, mid_end))  # Middle 50
                            if min_len > 50:
                                sample_indices.extend(range(max(0, min_len - 50), min_len))  # Last 50
                            
                            # Additional random sampling: one chunk of 50 frames per 1500 frames
                            chunk_size = 1500
                            n_chunks = max(1, min_len // chunk_size)
                            
                            for chunk_idx in range(n_chunks):
                                chunk_start = chunk_idx * chunk_size
                                chunk_end = min((chunk_idx + 1) * chunk_size, min_len)
                                chunk_length = chunk_end - chunk_start
                                
                                if chunk_length > 50:
                                    # Randomly sample 50 frames from this chunk
                                    chunk_indices = np.random.choice(
                                        range(chunk_start, chunk_end),
                                        size=50,
                                        replace=False
                                    )
                                    sample_indices.extend(chunk_indices.tolist())
                                else:
                                    # If chunk is smaller than 50, include all frames
                                    sample_indices.extend(range(chunk_start, chunk_end))
                            
                            sample_indices = sorted(set(sample_indices))  # Remove duplicates
                            
                            # Limit total samples to reasonable number (max 500 frames)
                            if len(sample_indices) > 500:
                                # Keep first, middle, last, and randomly sample from the rest
                                keep_indices = (
                                    sample_indices[:50] +  # First 50
                                    sample_indices[len(sample_indices)//2-25:len(sample_indices)//2+25] +  # Middle 50
                                    sample_indices[-50:]  # Last 50
                                )
                                remaining = [i for i in sample_indices if i not in keep_indices]
                                if len(remaining) > 400:
                                    remaining = np.random.choice(remaining, size=400, replace=False).tolist()
                                sample_indices = sorted(set(keep_indices + remaining))
                            
                            has_difference = False
                            for i in sample_indices:
                                # Quick head position check
                                h1_valid = not (pd.isna(level1_data.iloc[i]['xhead']) or pd.isna(level1_data.iloc[i]['yhead']))
                                h2_valid = not (pd.isna(level2_data.iloc[i]['xhead']) or pd.isna(level2_data.iloc[i]['yhead']))
                                
                                if h1_valid and h2_valid:
                                    h1_pos = np.array([level1_data.iloc[i]['xhead'], level1_data.iloc[i]['yhead']])
                                    h2_pos = np.array([level2_data.iloc[i]['xhead'], level2_data.iloc[i]['yhead']])
                                    head_dist = np.linalg.norm(h1_pos - h2_pos)
                                    if head_dist > 0.5:  # Same threshold as identify_swapped_frames
                                        has_difference = True
                                        break
                                
                                # Quick tail position check
                                t1_valid = not (pd.isna(level1_data.iloc[i]['xtail']) or pd.isna(level1_data.iloc[i]['ytail']))
                                t2_valid = not (pd.isna(level2_data.iloc[i]['xtail']) or pd.isna(level2_data.iloc[i]['ytail']))
                                
                                if t1_valid and t2_valid:
                                    t1_pos = np.array([level1_data.iloc[i]['xtail'], level1_data.iloc[i]['ytail']])
                                    t2_pos = np.array([level2_data.iloc[i]['xtail'], level2_data.iloc[i]['ytail']])
                                    tail_dist = np.linalg.norm(t1_pos - t2_pos)
                                    if tail_dist > 0.5:  # Same threshold as identify_swapped_frames
                                        has_difference = True
                                        break
                            
                            check_time = time.time() - check_start
                            
                            # If no differences found in sample, assume identical (conservative)
                            if not has_difference:
                                identical_count += 1
                                # Show progress every 10 filtered trials
                                if identical_count % 10 == 0:
                                    print(f"  Filtered {identical_count} identical trials so far...")
                                    sys.stdout.flush()
                                continue  # Skip this trial
                            
                            # Show detailed progress for slow operations
                            if check_time > 1.0:
                                trial_name = os.path.basename(str(root_path))
                                print(f"    Slow trial {trial_name}: check={check_time:.1f}s")
                                sys.stdout.flush()
                except Exception as e:
                    error_count += 1
                    # Show errors but don't stop
                    if error_count <= 5 or error_count % 10 == 0:
                        trial_name = os.path.basename(str(root_path))
                        print(f"  Warning: Error checking {trial_name}: {str(e)[:100]}")
                        sys.stdout.flush()
                    # If we can't check, include it (better to include than exclude)
                    pass
            
            valid_trials.append(str(root_path))
            
            # Show progress every 50 valid trials
            if len(valid_trials) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Found {len(valid_trials)} valid trials so far (checked {checked_count} total)...")
                sys.stdout.flush()
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(f"TRIAL DISCOVERY COMPLETE")
    print(f"=" * 80)
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Trials checked: {checked_count}")
    print(f"Valid trials (with swaps): {len(valid_trials)}")
    if filter_identical:
        print(f"Filtered out (identical level1/level2): {identical_count}")
    if error_count > 0:
        print(f"Errors encountered: {error_count}")
    if checked_count > 0:
        rate = checked_count / total_time
        print(f"Processing rate: {rate:.2f} trials/second")
    print(f"=" * 80)
    sys.stdout.flush()
    
    return valid_trials


def validate_trial_directory(trial_dir: str) -> bool:
    """
    Check if a directory contains all required files.
    
    Parameters:
    -----------
    trial_dir : str
        Directory path to validate
        
    Returns:
    --------
    bool
        True if directory contains all required files, False otherwise
    """
    trial_path = Path(trial_dir)
    
    if not trial_path.exists() or not trial_path.is_dir():
        return False
    
    files = [f.name for f in trial_path.iterdir() if f.is_file()]
    
    # Raw data: _data.csv (but not _data_level1.csv or _data_level2.csv)
    has_data = any(f.endswith('_data.csv') and '_level' not in f for f in files)
    # Level1: _level1.csv or _data_level1.csv
    has_level1 = any(f.endswith('_level1.csv') for f in files)
    # Level2: _level2.csv or _data_level2.csv
    has_level2 = any(f.endswith('_level2.csv') for f in files)
    
    return has_data and has_level1 and has_level2


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Find valid trial directories')
    parser.add_argument('parent_dir', type=str, help='Parent directory to search')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save list of valid directories (optional)')
    
    args = parser.parse_args()
    
    valid_trials = find_valid_trial_directories(args.parent_dir)
    
    print(f"\nTotal valid trials: {len(valid_trials)}")
    
    if args.output:
        with open(args.output, 'w') as f:
            for trial in valid_trials:
                f.write(f"{trial}\n")
        print(f"Saved list to: {args.output}")


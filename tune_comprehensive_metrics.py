#!/usr/bin/env python3
"""
Systematic parameter tuning for comprehensive metrics approach.

Tests parameter combinations to minimize false positives while maintaining
improvements on problematic trials.
"""

import os
import sys
import itertools
import pandas as pd
import numpy as np
from swap_correction import tracking_correction as tc
from swap_correction import pivr_loader as loader
from swap_correction import error_analysis
from typing import Dict, List, Tuple, Optional


def get_test_data_path():
    """Get the default test data directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'swap_correction', 'tests', 'test_data')


def get_baseline_results(test_data_dir: str) -> Dict[str, Dict]:
    """
    Get baseline results for all trials to identify perfect vs problematic.
    
    Returns:
    --------
    dict
        Dictionary mapping trial names to their baseline metrics
    """
    print("Loading baseline results...")
    baseline_results = {}
    
    trial_dirs = [os.path.join(test_data_dir, d) for d in os.listdir(test_data_dir)
                  if os.path.isdir(os.path.join(test_data_dir, d))]
    
    for trial_dir in trial_dirs:
        trial_name = os.path.basename(trial_dir)
        try:
            # Load level1 (baseline) and level2 (ground truth)
            csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level1.csv')]
            if not csv_files:
                continue
            
            level1_file = csv_files[0]
            level1_data = loader.load_raw_data(trial_dir, level1_file)
            
            csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level2.csv')]
            if not csv_files:
                continue
            
            level2_file = csv_files[0]
            level2_data = loader.load_raw_data(trial_dir, level2_file)
            
            # Calculate error rate
            min_len = min(len(level1_data), len(level2_data))
            swapped_frames = error_analysis.identify_swapped_frames(
                level1_data.iloc[:min_len],
                level2_data.iloc[:min_len]
            )
            error_rate = len(swapped_frames) / min_len * 100 if min_len > 0 else 0
            
            baseline_results[trial_name] = {
                'error_rate': error_rate,
                'is_perfect': error_rate == 0,
                'trial_dir': trial_dir
            }
        except Exception as e:
            print(f"Warning: Could not process {trial_name}: {e}")
            continue
    
    return baseline_results


def process_trial_with_params(trial_dir: str, params: Dict, debug: bool = False) -> Optional[Dict]:
    """
    Process a single trial with given parameters.
    
    Parameters:
    -----------
    trial_dir : str
        Directory containing trial data
    params : dict
        Parameter dictionary for comprehensive metrics
    debug : bool
        Print debug messages
        
    Returns:
    --------
    dict or None
        Results dictionary or None if processing failed
    """
    try:
        # Load raw data
        raw_data = loader.load_raw_data(trial_dir)
        fps = loader.get_all_settings(trial_dir)['Framerate']
        
        # Run correction with comprehensive parameters
        corrected_data = tc.tracking_correction(
            raw_data, fps,
            filterData=False,
            swapCorrection=True,
            validate=False,
            removeErrors=True,
            interp=True,
            debug=debug,
            comprehensive_params=params
        )
        
        # Load level2 for comparison
        csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level2.csv')]
        if not csv_files:
            return None
        
        level2_file = csv_files[0]
        level2_data = loader.load_raw_data(trial_dir, level2_file)
        
        # Calculate error metrics
        min_len = min(len(corrected_data), len(level2_data))
        swapped_frames = error_analysis.identify_swapped_frames(
            corrected_data.iloc[:min_len],
            level2_data.iloc[:min_len]
        )
        error_rate = len(swapped_frames) / min_len * 100 if min_len > 0 else 0
        
        swap_segments = error_analysis.get_swap_segments(
            corrected_data.iloc[:min_len],
            level2_data.iloc[:min_len]
        )
        
        return {
            'trial': os.path.basename(trial_dir),
            'error_rate': error_rate,
            'swapped_frames': len(swapped_frames),
            'num_segments': len(swap_segments),
            'total_frames': min_len,
        }
    except Exception as e:
        if debug:
            print(f"Error processing {trial_dir}: {e}")
        return None


def test_parameter_combination(
    perfect_trials: List[str],
    problematic_trials: List[str],
    baseline_results: Dict[str, Dict],
    params: Dict,
    test_data_dir: str,
    known_trial: str = "2024.11.13_00-48-15_Sussex_e2hex"
) -> Dict:
    """
    Test a single parameter combination on all trials.
    
    Returns:
    --------
    dict
        Metrics for this parameter combination
    """
    results = {
        **params,  # Include all parameters in results
        'perfect_preserved': 0,
        'false_positives': 0,
        'problematic_improved': 0,
        'problematic_worsened': 0,
        'problematic_unchanged': 0,
        'avg_error_change': 0.0,
        'known_trial_error': 0.0,
        'perfect_trials': len(perfect_trials),
        'problematic_trials': len(problematic_trials),
    }
    
    error_changes = []
    known_trial_result = None
    
    # Test on perfect trials
    for trial_name in perfect_trials:
        trial_dir = baseline_results[trial_name]['trial_dir']
        result = process_trial_with_params(trial_dir, params, debug=False)
        if result:
            if result['error_rate'] == 0:
                results['perfect_preserved'] += 1
            else:
                results['false_positives'] += 1
    
    # Test on problematic trials
    for trial_name in problematic_trials:
        trial_dir = baseline_results[trial_name]['trial_dir']
        baseline_error = baseline_results[trial_name]['error_rate']
        result = process_trial_with_params(trial_dir, params, debug=False)
        if result:
            error_change = baseline_error - result['error_rate']
            error_changes.append(error_change)
            
            if error_change > 0.1:  # Improved by at least 0.1%
                results['problematic_improved'] += 1
            elif error_change < -0.1:  # Worsened by at least 0.1%
                results['problematic_worsened'] += 1
            else:
                results['problematic_unchanged'] += 1
            
            # Track known trial
            if trial_name == known_trial:
                known_trial_result = result['error_rate']
    
    # Calculate average error change
    if len(error_changes) > 0:
        results['avg_error_change'] = np.mean(error_changes)
    
    if known_trial_result is not None:
        results['known_trial_error'] = known_trial_result
    
    return results


def run_coarse_grid_search(test_data_dir: str, output_file: str = "tuning_results_coarse.csv", 
                          max_combinations: int = None, resume_from: int = 0,
                          phase: str = "coarse"):
    """
    Run grid search over parameter space.
    
    Phase "coarse": Very coarse search with wide parameter ranges
    Phase "fine": Finer search in promising regions identified from coarse search
    
    Parameters:
    -----------
    max_combinations : int, optional
        Limit number of combinations to test (for testing/debugging)
    resume_from : int
        Resume from this combination index (for resuming interrupted runs)
    phase : str
        "coarse" for initial wide search, "fine" for focused search
    """
    print("=" * 80)
    print(f"{phase.upper()} GRID SEARCH - Parameter Tuning")
    print("=" * 80)
    
    # Check if we're resuming
    if resume_from > 0 and os.path.exists(output_file):
        print(f"Resuming from combination {resume_from}...")
        existing_df = pd.read_csv(output_file)
        results = existing_df.to_dict('records')
        start_idx = len(results)
    else:
        results = []
        start_idx = 0
    
    # Get baseline results
    baseline_results = get_baseline_results(test_data_dir)
    
    # Identify perfect and problematic trials
    perfect_trials = [name for name, res in baseline_results.items() if res['is_perfect']]
    problematic_trials = [name for name, res in baseline_results.items() if not res['is_perfect']]
    
    print(f"\nPerfect trials: {len(perfect_trials)}")
    print(f"Problematic trials: {len(problematic_trials)}")
    
    # Define parameter ranges based on phase
    if phase == "coarse":
        # Very coarse search - test wide ranges with fewer values
        param_grid = {
            'min_votes': [3, 4, 5],  # Test all 3
            'window_size': [50, 100],  # Just min and max
            'alignment_angle_threshold': [90.0, 110.0],  # Just min and max
            'speed_ratio_threshold': [0.9, 1.0],  # Just min and max
            'min_segment_size': [0, 50],  # Just min and max
        }
    else:  # fine phase
        # Finer search - need to be called with specific ranges
        # This will be set based on coarse results
        param_grid = {
            'min_votes': [3, 4, 5],
            'window_size': [50, 75, 100],
            'alignment_angle_threshold': [90.0, 100.0, 110.0],
            'speed_ratio_threshold': [0.9, 0.95, 1.0],
            'min_segment_size': [0, 30, 50],
        }
    
    # Generate combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    if max_combinations:
        combinations = combinations[:max_combinations]
    
    total_combinations = len(combinations)
    print(f"\nTesting {total_combinations} parameter combinations ({phase} phase)...")
    if start_idx > 0:
        print(f"Resuming from combination {start_idx + 1}...")
    print("This may take a while...\n")
    
    for i, combo in enumerate(combinations[start_idx:], start=start_idx):
        params = dict(zip(param_names, combo))
        
        # Fill in defaults for other parameters
        params['angular_vel_ratio'] = 1.0
        params['angular_var_ratio'] = 1.0
        params['distance_ratio_threshold'] = 0.9
        params['min_segment_duration'] = 0.0
        
        print(f"[{i+1}/{total_combinations}] Testing: min_votes={params['min_votes']}, "
              f"window={params['window_size']}, angle={params['alignment_angle_threshold']}, "
              f"speed={params['speed_ratio_threshold']}, min_size={params['min_segment_size']}")
        
        try:
            # Test this combination
            result = test_parameter_combination(
                perfect_trials, problematic_trials, baseline_results, params, test_data_dir
            )
            results.append(result)
            
            # Save incrementally every 10 combinations
            if (i + 1) % 10 == 0 or i == total_combinations - 1:
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                print(f"  -> Saved progress ({len(results)}/{total_combinations} complete)")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            # Save what we have so far
            if len(results) > 0:
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
            raise
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


def identify_promising_regions(coarse_results_file: str, top_n: int = 10) -> dict:
    """
    Analyze coarse search results to identify promising parameter regions.
    
    Returns:
    --------
    dict
        Dictionary with promising parameter ranges for fine search
    """
    df = pd.read_csv(coarse_results_file)
    
    # Score each combination: prioritize high perfect_preserved, low false_positives,
    # good known_trial_error, and positive avg_error_change
    df['score'] = (
        df['perfect_preserved'] * 10 -  # Weight perfect preservation highly
        df['false_positives'] * 20 +      # Penalize false positives heavily
        (df['known_trial_error'] < 30).astype(int) * 5 +  # Bonus if known trial improved
        (df['avg_error_change'] < 0).astype(int) * 3  # Bonus if average error decreased
    )
    
    # Get top N combinations
    top_combinations = df.nlargest(top_n, 'score')
    
    print(f"\nTop {top_n} combinations from coarse search:")
    print(top_combinations[['min_votes', 'window_size', 'alignment_angle_threshold',
                            'speed_ratio_threshold', 'min_segment_size',
                            'perfect_preserved', 'false_positives', 'known_trial_error', 'score']])
    
    # Identify promising ranges
    promising = {
        'min_votes': sorted(top_combinations['min_votes'].unique().tolist()),
        'window_size': {
            'min': int(top_combinations['window_size'].min()),
            'max': int(top_combinations['window_size'].max()),
            'median': int(top_combinations['window_size'].median())
        },
        'alignment_angle_threshold': {
            'min': float(top_combinations['alignment_angle_threshold'].min()),
            'max': float(top_combinations['alignment_angle_threshold'].max()),
            'median': float(top_combinations['alignment_angle_threshold'].median())
        },
        'speed_ratio_threshold': {
            'min': float(top_combinations['speed_ratio_threshold'].min()),
            'max': float(top_combinations['speed_ratio_threshold'].max()),
            'median': float(top_combinations['speed_ratio_threshold'].median())
        },
        'min_segment_size': sorted(top_combinations['min_segment_size'].unique().tolist()),
    }
    
    print(f"\nPromising parameter ranges identified:")
    print(f"  min_votes: {promising['min_votes']}")
    print(f"  window_size: {promising['window_size']['min']}-{promising['window_size']['max']} (median: {promising['window_size']['median']})")
    print(f"  alignment_angle_threshold: {promising['alignment_angle_threshold']['min']}-{promising['alignment_angle_threshold']['max']} (median: {promising['alignment_angle_threshold']['median']})")
    print(f"  speed_ratio_threshold: {promising['speed_ratio_threshold']['min']}-{promising['speed_ratio_threshold']['max']} (median: {promising['speed_ratio_threshold']['median']})")
    print(f"  min_segment_size: {promising['min_segment_size']}")
    
    return promising


def run_fine_grid_search(test_data_dir: str, promising_regions: dict,
                         output_file: str = "tuning_results_fine.csv"):
    """
    Run fine grid search in promising regions identified from coarse search.
    
    Parameters:
    -----------
    promising_regions : dict
        Parameter ranges from identify_promising_regions()
    """
    print("=" * 80)
    print("FINE GRID SEARCH - Focused Parameter Tuning")
    print("=" * 80)
    
    # Get baseline results
    baseline_results = get_baseline_results(test_data_dir)
    
    # Identify perfect and problematic trials
    perfect_trials = [name for name, res in baseline_results.items() if res['is_perfect']]
    problematic_trials = [name for name, res in baseline_results.items() if not res['is_perfect']]
    
    print(f"\nPerfect trials: {len(perfect_trials)}")
    print(f"Problematic trials: {len(problematic_trials)}")
    
    # Generate fine-grained parameter ranges around promising regions
    # Create ranges that span the promising region with finer steps
    fine_param_grid = {
        'min_votes': promising_regions['min_votes'],
        'window_size': _generate_fine_range(
            promising_regions['window_size']['min'],
            promising_regions['window_size']['max'],
            promising_regions['window_size']['median']
        ),
        'alignment_angle_threshold': _generate_fine_range(
            promising_regions['alignment_angle_threshold']['min'],
            promising_regions['alignment_angle_threshold']['max'],
            promising_regions['alignment_angle_threshold']['median']
        ),
        'speed_ratio_threshold': _generate_fine_range(
            promising_regions['speed_ratio_threshold']['min'],
            promising_regions['speed_ratio_threshold']['max'],
            promising_regions['speed_ratio_threshold']['median']
        ),
        'min_segment_size': promising_regions['min_segment_size'],
    }
    
    # Generate combinations
    param_names = list(fine_param_grid.keys())
    param_values = list(fine_param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    total_combinations = len(combinations)
    print(f"\nTesting {total_combinations} parameter combinations (fine phase)...")
    print("This may take a while...\n")
    
    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        # Fill in defaults for other parameters
        params['angular_vel_ratio'] = 1.0
        params['angular_var_ratio'] = 1.0
        params['distance_ratio_threshold'] = 0.9
        params['min_segment_duration'] = 0.0
        
        print(f"[{i+1}/{total_combinations}] Testing: min_votes={params['min_votes']}, "
              f"window={params['window_size']}, angle={params['alignment_angle_threshold']}, "
              f"speed={params['speed_ratio_threshold']}, min_size={params['min_segment_size']}")
        
        try:
            # Test this combination
            result = test_parameter_combination(
                perfect_trials, problematic_trials, baseline_results, params, test_data_dir
            )
            results.append(result)
            
            # Save incrementally every 10 combinations
            if (i + 1) % 10 == 0 or i == total_combinations - 1:
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                print(f"  -> Saved progress ({len(results)}/{total_combinations} complete)")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            if len(results) > 0:
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
            raise
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


def _generate_fine_range(min_val, max_val, median_val, num_steps=3):
    """Generate fine-grained range around promising region."""
    if min_val == max_val:
        return [min_val]
    
    # Include min, max, median, and interpolate between
    if isinstance(min_val, float):
        step = (max_val - min_val) / (num_steps - 1) if num_steps > 1 else 0
        values = [min_val + i * step for i in range(num_steps)]
        # Ensure median is included
        if median_val not in values:
            values.append(median_val)
        values = sorted(set(values))
    else:  # int
        step = (max_val - min_val) // (num_steps - 1) if num_steps > 1 else 0
        values = [min_val + i * step for i in range(num_steps)]
        # Ensure median is included
        if median_val not in values:
            values.append(median_val)
        values = sorted(set(values))
    
    return values


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune comprehensive metrics parameters')
    parser.add_argument('--phase', type=str, choices=['coarse', 'fine', 'both'], default='both',
                       help='Which phase to run: coarse, fine, or both')
    parser.add_argument('--max-combinations', type=int, default=None,
                       help='Limit number of combinations to test (for testing)')
    parser.add_argument('--resume-from', type=int, default=0,
                       help='Resume from this combination index')
    parser.add_argument('--coarse-output', type=str, default='tuning_results_coarse.csv',
                       help='Output CSV file for coarse search')
    parser.add_argument('--fine-output', type=str, default='tuning_results_fine.csv',
                       help='Output CSV file for fine search')
    parser.add_argument('--coarse-results', type=str, default='tuning_results_coarse.csv',
                       help='Coarse results file to use for fine search')
    args = parser.parse_args()
    
    test_data_dir = get_test_data_path()
    
    if not os.path.exists(test_data_dir):
        print(f"Error: Test data directory not found: {test_data_dir}")
        sys.exit(1)
    
    # Phase 1: Coarse grid search
    if args.phase in ['coarse', 'both']:
        print("\n" + "=" * 80)
        print("PHASE 1: COARSE GRID SEARCH")
        print("=" * 80)
        coarse_results_df = run_coarse_grid_search(
            test_data_dir, output_file=args.coarse_output,
            max_combinations=args.max_combinations,
            resume_from=args.resume_from,
            phase='coarse'
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("COARSE SEARCH SUMMARY")
        print("=" * 80)
        print(f"\nTotal combinations tested: {len(coarse_results_df)}")
        
        if len(coarse_results_df) > 0:
            # Find best combinations
            best = coarse_results_df.nlargest(10, 'perfect_preserved')
            print("\nTop 10 combinations by perfect trials preserved:")
            print(best[['min_votes', 'window_size', 'alignment_angle_threshold', 
                        'speed_ratio_threshold', 'min_segment_size',
                        'perfect_preserved', 'false_positives', 'known_trial_error']])
    
    # Phase 2: Fine grid search in promising regions
    if args.phase in ['fine', 'both']:
        if not os.path.exists(args.coarse_results):
            print(f"\nError: Coarse results file not found: {args.coarse_results}")
            print("Please run coarse search first or specify --coarse-results")
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("PHASE 2: IDENTIFYING PROMISING REGIONS")
        print("=" * 80)
        promising_regions = identify_promising_regions(args.coarse_results, top_n=10)
        
        print("\n" + "=" * 80)
        print("PHASE 3: FINE GRID SEARCH")
        print("=" * 80)
        fine_results_df = run_fine_grid_search(
            test_data_dir, promising_regions, output_file=args.fine_output
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("FINE SEARCH SUMMARY")
        print("=" * 80)
        print(f"\nTotal combinations tested: {len(fine_results_df)}")
        
        if len(fine_results_df) > 0:
            # Score and rank results
            fine_results_df['score'] = (
                fine_results_df['perfect_preserved'] * 10 -
                fine_results_df['false_positives'] * 20 +
                (fine_results_df['known_trial_error'] < 30).astype(int) * 5 +
                (fine_results_df['avg_error_change'] < 0).astype(int) * 3
            )
            
            best = fine_results_df.nlargest(10, 'score')
            print("\nTop 10 combinations by score:")
            print(best[['min_votes', 'window_size', 'alignment_angle_threshold', 
                        'speed_ratio_threshold', 'min_segment_size',
                        'perfect_preserved', 'false_positives', 'known_trial_error', 'score']])
            
            # Find best overall
            best_overall = fine_results_df.loc[fine_results_df['score'].idxmax()]
            print("\n" + "=" * 80)
            print("BEST PARAMETER COMBINATION")
            print("=" * 80)
            print(f"\nParameters:")
            print(f"  min_votes: {best_overall['min_votes']}")
            print(f"  window_size: {best_overall['window_size']}")
            print(f"  alignment_angle_threshold: {best_overall['alignment_angle_threshold']}")
            print(f"  speed_ratio_threshold: {best_overall['speed_ratio_threshold']}")
            print(f"  min_segment_size: {best_overall['min_segment_size']}")
            print(f"\nPerformance:")
            print(f"  Perfect trials preserved: {best_overall['perfect_preserved']}/14")
            print(f"  False positives: {best_overall['false_positives']}/14")
            print(f"  Problematic trials improved: {best_overall['problematic_improved']}/11")
            print(f"  Known trial error: {best_overall['known_trial_error']:.2f}%")
            print(f"  Average error change: {best_overall['avg_error_change']:.2f}%")


if __name__ == '__main__':
    main()


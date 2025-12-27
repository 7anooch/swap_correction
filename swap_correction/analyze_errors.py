#!/usr/bin/env python3
"""
Command-line interface for error analysis.

Usage:
    python -m swap_correction.analyze_errors --trial <trial_dir>
    python -m swap_correction.analyze_errors --all-trials <test_data_dir>
    python -m swap_correction.analyze_errors --summary <test_data_dir>
"""

import argparse
import os
import sys
from swap_correction import error_analysis, error_report


def _get_default_test_data_path():
    """Get default path to test_data directory."""
    # Get the directory where this script is located (swap_correction/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # test_data is at swap_correction/tests/test_data/
    default_path = os.path.join(script_dir, 'tests', 'test_data')
    return default_path


def main():
    default_test_data = _get_default_test_data_path()
    
    parser = argparse.ArgumentParser(
        description='Analyze swap errors in tracking data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Analyze a single trial
  python -m swap_correction.analyze_errors --trial tests/test_data/2024.11.13_00-06-31_Sussex_e2hex

  # Analyze all trials (uses default: {default_test_data})
  python -m swap_correction.analyze_errors --all-trials

  # Analyze all trials with custom path
  python -m swap_correction.analyze_errors --all-trials /path/to/test_data

  # Generate summary report only
  python -m swap_correction.analyze_errors --summary tests/test_data --output-dir error_reports
        """
    )
    
    # Input options
    parser.add_argument('--trial', type=str, help='Path to single trial directory')
    parser.add_argument('--all-trials', type=str, nargs='?', const=default_test_data, default=None,
                       help=f'Path to directory containing trial subdirectories (default: {default_test_data})')
    parser.add_argument('--summary', type=str, help='Path to directory containing trial subdirectories (summary only)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='error_analysis_output',
                       help='Output directory for reports (default: error_analysis_output)')
    parser.add_argument('--metrics-only', action='store_true',
                       help='Skip visualizations, only calculate metrics')
    parser.add_argument('--plots-only', action='store_true',
                       help='Skip metrics, only generate plots')
    
    args = parser.parse_args()
    
    # Check that exactly one input option is provided
    input_options = [args.trial, args.all_trials, args.summary]
    if sum(1 for opt in input_options if opt is not None) != 1:
        parser.error("Exactly one of --trial, --all-trials, or --summary must be provided")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.trial:
        # Analyze single trial
        if not os.path.isdir(args.trial):
            print(f"Error: Trial directory not found: {args.trial}")
            sys.exit(1)
        
        print(f"Analyzing trial: {args.trial}")
        try:
            metrics = error_report.generate_trial_report(args.trial, args.output_dir)
            if metrics:
                print(f"Report generated in: {os.path.join(args.output_dir, f'trial_{os.path.basename(args.trial)}')}")
                print(f"Error rate: {metrics.get('error_rate', 0):.2%}")
                print(f"Swapped frames: {metrics.get('swapped_frames', 0)}")
                print(f"Swap segments: {metrics.get('num_swap_segments', 0)}")
            else:
                print("Warning: No metrics generated")
        except Exception as e:
            print(f"Error analyzing trial: {e}")
            sys.exit(1)
    
    elif args.all_trials:
        # Analyze all trials
        if not os.path.isdir(args.all_trials):
            print(f"Error: Test data directory not found: {args.all_trials}")
            sys.exit(1)
        
        print(f"Analyzing all trials in: {args.all_trials}")
        try:
            summary_df = error_report.generate_summary_report(args.all_trials, args.output_dir)
            if not summary_df.empty:
                print(f"\nSummary report generated in: {os.path.join(args.output_dir, 'summary')}")
                print(f"\nTotal trials analyzed: {len(summary_df)}")
                print(f"Average error rate: {summary_df['error_rate'].mean():.2%}")
                print(f"Total swapped frames: {summary_df['swapped_frames'].sum()}")
                print(f"Average swap segments per trial: {summary_df['num_swap_segments'].mean():.2f}")
            else:
                print("Warning: No summary data generated")
        except Exception as e:
            print(f"Error analyzing trials: {e}")
            sys.exit(1)
    
    elif args.summary:
        # Generate summary only
        if not os.path.isdir(args.summary):
            print(f"Error: Test data directory not found: {args.summary}")
            sys.exit(1)
        
        print(f"Generating summary report for: {args.summary}")
        try:
            summary_df = error_report.generate_summary_report(args.summary, args.output_dir)
            if not summary_df.empty:
                print(f"\nSummary report generated in: {os.path.join(args.output_dir, 'summary')}")
                print(f"Total trials: {len(summary_df)}")
            else:
                print("Warning: No summary data generated")
        except Exception as e:
            print(f"Error generating summary: {e}")
            sys.exit(1)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()


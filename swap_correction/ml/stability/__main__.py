#!/usr/bin/env python3
"""
CLI entry point for stability analysis.

Provides unified interface to run the complete stability analysis pipeline.
"""

import sys
import argparse
from swap_correction.ml.stability.run_stability_analysis import run_full_stability_analysis
from swap_correction.ml.stability.aggregate_results import generate_stability_report, load_iteration_results
from swap_correction.ml.stability.visualize_stability import generate_all_visualizations


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Model Stability Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full stability analysis
  python -m swap_correction.ml.stability \\
      --parent-dir /path/to/data \\
      --n-iterations 10 \\
      --n-samples 30 \\
      --output-dir stability_analysis

  # Aggregate existing results
  python -m swap_correction.ml.stability aggregate \\
      --results-dir stability_analysis

  # Generate visualizations
  python -m swap_correction.ml.stability visualize \\
      --results-dir stability_analysis
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run analysis
    run_parser = subparsers.add_parser('run', help='Run full stability analysis')
    run_parser.add_argument('--parent-dir', type=str, required=True,
                           help='Parent directory containing trial subdirectories')
    run_parser.add_argument('--n-iterations', type=int, default=10,
                           help='Number of iterations per sample size (default: 10)')
    run_parser.add_argument('--n-samples', type=int, default=30,
                           help='Number of trials per iteration (default: 30, ignored if --sample-sizes provided)')
    run_parser.add_argument('--sample-sizes', type=int, nargs='+', default=None,
                           help='List of sample sizes to test (e.g., --sample-sizes 30 40 50)')
    run_parser.add_argument('--output-dir', type=str, default='stability_analysis',
                           help='Output directory (default: stability_analysis)')
    run_parser.add_argument('--base-seed', type=int, default=42,
                           help='Base random seed (default: 42)')
    
    # Aggregate results
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate results from iterations')
    agg_parser.add_argument('--results-dir', type=str, required=True,
                            help='Directory containing iteration results')
    agg_parser.add_argument('--output-dir', type=str, default=None,
                           help='Output directory (default: results_dir)')
    
    # Visualize
    viz_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    viz_parser.add_argument('--results-dir', type=str, required=True,
                           help='Directory containing iteration results')
    viz_parser.add_argument('--output-dir', type=str, default=None,
                           help='Output directory for plots (default: results_dir/figures)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'run':
        run_full_stability_analysis(
            args.parent_dir,
            n_iterations=args.n_iterations,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            base_seed=args.base_seed,
            sample_sizes=args.sample_sizes
        )
        
        # Automatically aggregate and visualize after running
        print("\n" + "=" * 80)
        print("AGGREGATING RESULTS")
        print("=" * 80)
        sys.stdout.flush()
        
        try:
            from pathlib import Path
            results_path = Path(args.output_dir)
            iteration_dirs = [
                str(d) for d in results_path.iterdir()
                if d.is_dir() and d.name.startswith('iteration_')
            ]
            iteration_dirs.sort()
            
            print(f"Found {len(iteration_dirs)} iteration directories")
            sys.stdout.flush()
            
            all_results = load_iteration_results(iteration_dirs)
            print(f"Loaded {len(all_results)} iteration results")
            sys.stdout.flush()
            
            generate_stability_report(all_results, args.output_dir)
            print("Aggregation complete")
            sys.stdout.flush()
            
            print("\n" + "=" * 80)
            print("GENERATING VISUALIZATIONS")
            print("=" * 80)
            sys.stdout.flush()
            
            generate_all_visualizations(args.output_dir, None)
            print("Visualization complete")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"ERROR during aggregation/visualization: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
    elif args.command == 'aggregate':
        from pathlib import Path
        results_path = Path(args.results_dir)
        iteration_dirs = [
            str(d) for d in results_path.iterdir()
            if d.is_dir() and d.name.startswith('iteration_')
        ]
        iteration_dirs.sort()
        
        all_results = load_iteration_results(iteration_dirs)
        generate_stability_report(all_results, args.output_dir or args.results_dir)
        
    elif args.command == 'visualize':
        generate_all_visualizations(args.results_dir, args.output_dir)


if __name__ == '__main__':
    main()


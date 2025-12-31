#!/usr/bin/env python3
"""
Unified CLI for evaluation suite.

Provides a single entry point for all evaluation analyses.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ML Model Evaluation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all evaluations
  python -m swap_correction.ml.evaluation all

  # Run specific analysis
  python -m swap_correction.ml.evaluation overfitting
  python -m swap_correction.ml.evaluation learning-curves
  python -m swap_correction.ml.evaluation compare
  python -m swap_correction.ml.evaluation validate
  python -m swap_correction.ml.evaluation evaluate --dataset /path/to/data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Evaluation command')
    
    # All evaluations
    subparsers.add_parser('all', help='Run all evaluations')
    
    # Overfitting analysis
    overfitting_parser = subparsers.add_parser('overfitting', help='Analyze model overfitting')
    overfitting_parser.add_argument('--output-dir', type=str, default='ml_analysis',
                                   help='Output directory (default: ml_analysis)')
    
    # Learning curves
    lc_parser = subparsers.add_parser('learning-curves', help='Generate learning curves')
    lc_parser.add_argument('--output-dir', type=str, default='ml_analysis',
                           help='Output directory (default: ml_analysis)')
    
    # Model comparison
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('--output-dir', type=str, default='ml_analysis',
                               help='Output directory (default: ml_analysis)')
    
    # Validation stability
    validate_parser = subparsers.add_parser('validate', help='Validation stability analysis')
    validate_parser.add_argument('--output-dir', type=str, default='ml_analysis',
                                 help='Output directory (default: ml_analysis)')
    
    # Evaluate on dataset
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on new dataset')
    eval_parser.add_argument('dataset', type=str, help='Dataset directory path')
    eval_parser.add_argument('--model-type', type=str, choices=['level1', 'raw', 'raw_data'],
                            default='level1', help='Model type (default: level1)')
    eval_parser.add_argument('--ground-truth', type=str, choices=['level1', 'level2'],
                             default='level2', help='Ground truth level (default: level2)')
    eval_parser.add_argument('--output-dir', type=str, default='ml_analysis/evaluations',
                            help='Output directory (default: ml_analysis/evaluations)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'all':
        print("=" * 80)
        print("RUNNING ALL EVALUATIONS")
        print("=" * 80)
        
        from swap_correction.ml.evaluation.overfitting import main as overfitting_main
        from swap_correction.ml.evaluation.learning_curves import main as lc_main
        from swap_correction.ml.evaluation.compare_models import main as compare_main
        from swap_correction.ml.evaluation.validation_stability import main as validate_main
        
        overfitting_main()
        lc_main()
        compare_main()
        validate_main()
        
    elif args.command == 'overfitting':
        from swap_correction.ml.evaluation.overfitting import main as overfitting_main
        overfitting_main()
        
    elif args.command == 'learning-curves':
        from swap_correction.ml.evaluation.learning_curves import main as lc_main
        lc_main()
        
    elif args.command == 'compare':
        from swap_correction.ml.evaluation.compare_models import main as compare_main
        compare_main()
        
    elif args.command == 'validate':
        from swap_correction.ml.evaluation.validation_stability import main as validate_main
        validate_main()
        
    elif args.command == 'evaluate':
        from swap_correction.ml.evaluation.evaluate_on_dataset import evaluate_model_on_dataset
        evaluate_model_on_dataset(
            args.dataset,
            model_type=args.model_type,
            ground_truth_level=args.ground_truth,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()


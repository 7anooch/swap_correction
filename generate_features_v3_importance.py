#!/usr/bin/env python3
"""
Generate feature importance files for all models in stability_analysis_v3_features_v3.

This script loads trained models and extracts feature importance, saving them
as CSV files similar to other stability analyses.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from swap_correction.ml.api.model_loader import load_model


def extract_feature_importance(model_dir: str, model_type: str = 'level1'):
    """
    Extract feature importance from a trained model.
    
    Parameters:
    -----------
    model_dir : str
        Directory containing the trained model
    model_type : str
        Model type ('level1' or 'raw')
        
    Returns:
    --------
    pd.DataFrame or None
        DataFrame with feature names and importance, or None if model not found
    """
    model_file = os.path.join(model_dir, 'swap_detector_xgb.pkl')
    feature_names_file = os.path.join(model_dir, 'feature_names.pkl')
    
    if not os.path.exists(model_file) or not os.path.exists(feature_names_file):
        return None
    
    try:
        # Load model and feature names
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        with open(feature_names_file, 'rb') as f:
            feature_names = pickle.load(f)
        
        # Get feature importance
        feature_importance = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None


def main():
    """Generate feature importance files for all iterations."""
    base_dir = 'stability_analysis_v3_features_v3'
    iteration_ids = [7, 8, 9, 10, 11, 12]
    
    print("=" * 80)
    print("GENERATING FEATURE IMPORTANCE FILES FOR FEATURES_V3")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Iterations: {iteration_ids}\n")
    
    total_models = 0
    successful = 0
    failed = 0
    
    for iter_id in iteration_ids:
        iter_dir = os.path.join(base_dir, f'iteration_{iter_id:03d}')
        
        if not os.path.exists(iter_dir):
            print(f"Iteration {iter_id:03d}: Directory not found")
            continue
        
        print(f"Iteration {iter_id:03d}:")
        
        # Process both model types
        for model_type in ['level1', 'raw']:
            model_dir = os.path.join(iter_dir, f'{model_type}_model')
            
            if not os.path.exists(model_dir):
                print(f"  {model_type} model: Directory not found")
                continue
            
            importance_df = extract_feature_importance(model_dir, model_type)
            
            if importance_df is not None:
                # Save feature importance
                importance_file = os.path.join(model_dir, 'feature_importance.csv')
                importance_df.to_csv(importance_file, index=False)
                
                print(f"  {model_type} model: ✓ Generated ({len(importance_df)} features)")
                print(f"    Top 5: {', '.join(importance_df.head(5)['feature'].tolist())}")
                successful += 1
            else:
                print(f"  {model_type} model: ✗ Failed")
                failed += 1
            
            total_models += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models: {total_models}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()


if __name__ == '__main__':
    main()


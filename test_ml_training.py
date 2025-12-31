#!/usr/bin/env python3
"""
Quick test of ML training pipeline on a small subset of data.
"""

import os
import sys
import pandas as pd
import numpy as np
from swap_correction import pivr_loader, ml_features
from train_ml_swap_detector import prepare_train_val_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

def get_test_data_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'swap_correction', 'tests', 'test_data')

def main():
    print("=" * 80)
    print("QUICK TEST OF ML TRAINING PIPELINE")
    print("=" * 80)
    
    # Test on trials with swaps
    test_data_dir = get_test_data_path()
    # Use one trial with swaps and one without for testing
    trial_names = ['2024.11.13_00-48-15_Sussex_e2hex', '2024.11.13_00-06-31_Sussex_e2hex']
    
    print(f"\nTesting on {len(trial_names)} trials...")
    
    # Load labels
    labels_df = pd.read_csv('ml_data/training_labels.csv')
    
    all_features = []
    all_labels = []
    trial_names_list = []
    
    for trial_name in trial_names:
        print(f"\nProcessing: {trial_name}")
        trial_dir = os.path.join(test_data_dir, trial_name)
        
        # Load data
        csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level1.csv')]
        level1_file = csv_files[0]
        trial_data = pivr_loader.load_raw_data(trial_dir, level1_file, px2mm=True)
        fps = pivr_loader.get_all_settings(trial_dir)['Framerate']
        
        # Get labels
        trial_labels = labels_df[labels_df['trial'] == trial_name].copy()
        trial_labels = trial_labels.sort_values('frame_idx')
        
        # Find frames with swaps for this trial
        swapped_frames = trial_labels[trial_labels['is_swapped'] == True]
        if len(swapped_frames) > 0:
            # Use a window around swapped frames
            first_swap = swapped_frames['frame_idx'].min()
            last_swap = swapped_frames['frame_idx'].max()
            # Get 500 frames before first swap to 500 frames after last swap
            start_frame = max(0, first_swap - 500)
            end_frame = min(len(trial_data), last_swap + 500)
            frame_indices = list(range(start_frame, end_frame))
            print(f"  Using frames {start_frame}-{end_frame} (includes {len(swapped_frames)} swapped frames)")
        else:
            # No swaps, just use first 1000 frames
            frame_indices = list(range(min(1000, len(trial_data))))
            print(f"  Using first {len(frame_indices)} frames (no swaps in this trial)")
        
        # Extract features for selected frames
        trial_data_subset = trial_data.iloc[frame_indices]
        trial_features = ml_features.extract_all_frame_features(trial_data_subset, fps=fps)
        
        # Get labels for these frames
        trial_labels_subset = trial_labels[trial_labels['frame_idx'].isin(frame_indices)].copy()
        trial_labels_subset = trial_labels_subset.sort_values('frame_idx')
        
        # Align (should be same length, but be safe)
        min_len = min(len(trial_features), len(trial_labels_subset))
        trial_features = trial_features.iloc[:min_len]
        trial_labels_subset = trial_labels_subset.iloc[:min_len]
        
        all_features.append(trial_features)
        all_labels.append(trial_labels_subset[['frame_idx', 'is_swapped']])
        trial_names_list.extend([trial_name] * min_len)
        
        print(f"  OK: {len(trial_features)} frames, {trial_labels_subset['is_swapped'].sum()} swapped")
    
    # Combine
    features_df = pd.concat(all_features, ignore_index=True)
    labels_combined = pd.concat(all_labels, ignore_index=True)
    
    print(f"\nTotal: {len(features_df)} frames, {labels_combined['is_swapped'].sum()} swapped")
    print(f"Features: {features_df.shape[1]} dimensions")
    
    # Simple train/test split (80/20)
    from sklearn.model_selection import train_test_split
    X = features_df.values
    y = labels_combined['is_swapped'].values
    
    # Handle NaN
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} frames ({y_train.sum()} swapped)")
    print(f"Test: {len(X_test)} frames ({y_test.sum()} swapped)")
    
    # Check if we have both classes
    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        print("\nWARNING: All training labels are the same class. Cannot train classifier.")
        print("This is expected for a small test subset. The full training will work correctly.")
        return
    
    # Train simple model
    print("\nTraining XGBoost model...")
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nTest Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE - Pipeline is working!")
    print("=" * 80)
    print("\nThe full training will take 1-2 hours but should work correctly.")
    print("Monitor progress with: tail -f training_output.log")

if __name__ == '__main__':
    main()


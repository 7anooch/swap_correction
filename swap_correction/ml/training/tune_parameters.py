#!/usr/bin/env python3
"""
Tune Gaussian filter sigma parameter for ML swap detection.

Tests different sigma values and compares model performance to find optimal value.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from swap_correction import pivr_loader
from swap_correction.ml.features import extract_all_frame_features_optimized


def get_test_data_path():
    """Get the default test data directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'swap_correction', 'tests', 'test_data')


def load_training_data(test_data_dir: str, sigma: float):
    """
    Load training labels and extract features with specified sigma.
    
    Returns:
    --------
    tuple
        (features_df, labels_df)
    """
    ml_data_dir = 'ml_data'
    
    # Load labels
    labels_file = os.path.join(ml_data_dir, 'training_labels.csv')
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    labels_df = pd.read_csv(labels_file)
    
    # Load train/test split
    split_file = os.path.join(ml_data_dir, 'train_test_split.json')
    with open(split_file, 'r') as f:
        split = json.load(f)
    
    # Extract features for all trials
    all_features = []
    all_labels = []
    
    unique_trials = labels_df['trial'].unique()
    
    for trial_name in unique_trials:
        trial_dir = os.path.join(test_data_dir, trial_name)
        if not os.path.exists(trial_dir):
            continue
        
        try:
            # Load raw data
            csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level1.csv')]
            if not csv_files:
                continue
            
            level1_file = csv_files[0]
            trial_data = pivr_loader.load_raw_data(trial_dir, level1_file, px2mm=True)
            
            try:
                fps = pivr_loader.get_all_settings(trial_dir)['Framerate']
            except:
                fps = 30
            
            # Get labels for this trial
            trial_labels = labels_df[labels_df['trial'] == trial_name].copy()
            trial_labels = trial_labels.sort_values('frame_idx')
            
            # Extract features with specified sigma
            trial_features = extract_all_frame_features_optimized(
                trial_data, fps=fps, apply_filtering=True, filter_sigma=sigma)
            
            # Align features and labels
            min_len = min(len(trial_features), len(trial_labels))
            trial_features = trial_features.iloc[:min_len]
            trial_labels = trial_labels.iloc[:min_len]
            
            all_features.append(trial_features)
            all_labels.append(trial_labels)
            
        except Exception as e:
            print(f"  Error processing {trial_name}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features extracted!")
    
    # Concatenate all features and labels
    features_df = pd.concat(all_features, ignore_index=True)
    labels_df = pd.concat(all_labels, ignore_index=True)
    
    # Align
    min_len = min(len(features_df), len(labels_df))
    features_df = features_df.iloc[:min_len]
    labels_df = labels_df.iloc[:min_len]
    
    return features_df, labels_df


def train_and_evaluate(features_df, labels_df, sigma: float):
    """
    Train XGBoost model and evaluate on test set.
    
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Split data (same split as original training)
    ml_data_dir = 'ml_data'
    split_file = os.path.join(ml_data_dir, 'train_test_split.json')
    with open(split_file, 'r') as f:
        split = json.load(f)
    
    train_trials = set(split['train'])
    val_trials = set(split['val'])
    test_trials = set(split['test'])
    
    # Create train/val/test masks based on trial names
    train_mask = labels_df['trial'].isin(train_trials)
    val_mask = labels_df['trial'].isin(val_trials)
    test_mask = labels_df['trial'].isin(test_trials)
    
    X_train = features_df[train_mask].values
    y_train = labels_df[train_mask]['is_swapped'].values
    X_val = features_df[val_mask].values
    y_val = labels_df[val_mask]['is_swapped'].values
    X_test = features_df[test_mask].values
    y_test = labels_df[test_mask]['is_swapped'].values
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Calculate class weights
    n_positive = y_train.sum()
    n_negative = len(y_train) - n_positive
    if n_positive > 0:
        class_weight_ratio = n_negative / n_positive
    else:
        class_weight_ratio = 1.0
    
    # Train XGBoost
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': class_weight_ratio,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 20
    }
    
    sklearn_model = xgb.XGBClassifier(**params)
    sklearn_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate on test set
    y_pred = sklearn_model.predict(X_test)
    y_pred_proba = sklearn_model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    tp = np.sum((y_test == 1) & (y_pred == 1))
    
    return {
        'sigma': sigma,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'n_test_samples': len(y_test),
        'n_positive': int(y_test.sum()),
        'n_predicted_positive': int(y_pred.sum())
    }


def main():
    """Main function to tune sigma parameter."""
    print("=" * 80)
    print("GAUSSIAN FILTER SIGMA TUNING")
    print("=" * 80)
    
    test_data_dir = get_test_data_path()
    
    # Test different sigma values (fine grid search in optimal region)
    sigma_values = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0]
    
    results = []
    
    for sigma in sigma_values:
        print(f"\n{'='*80}")
        print(f"Testing sigma = {sigma}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Load data and extract features
            print("Loading data and extracting features...")
            features_df, labels_df = load_training_data(test_data_dir, sigma)
            print(f"  Extracted {len(features_df)} frames with {features_df.shape[1]} features")
            
            # Train and evaluate
            print("Training model...")
            metrics = train_and_evaluate(features_df, labels_df, sigma)
            
            elapsed = time.time() - start_time
            
            print(f"\nResults (sigma={sigma}):")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['auc']:.4f}")
            print(f"  Time: {elapsed:.1f}s")
            
            metrics['time_seconds'] = elapsed
            results.append(metrics)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("SIGMA TUNING SUMMARY")
    print("=" * 80)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('f1', ascending=False)
        
        print("\nResults sorted by F1-score:")
        print(results_df[['sigma', 'precision', 'recall', 'f1', 'auc', 'time_seconds']].to_string(index=False))
        
        # Best sigma
        best = results_df.iloc[0]
        print(f"\nBest sigma: {best['sigma']}")
        print(f"  F1-score: {best['f1']:.4f}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Recall: {best['recall']:.4f}")
        print(f"  ROC-AUC: {best['auc']:.4f}")
        
        # Save results
        results_file = 'sigma_tuning_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save summary CSV
        csv_file = 'sigma_tuning_summary.csv'
        results_df.to_csv(csv_file, index=False)
        print(f"Summary saved to: {csv_file}")
    else:
        print("No results to report!")


if __name__ == '__main__':
    main()


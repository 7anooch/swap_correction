#!/usr/bin/env python3
"""
Train machine learning model for swap detection.

Trains a frame-level XGBoost classifier to detect swapped frames.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_auc_score, roc_curve)
import xgboost as xgb
from swap_correction import pivr_loader, ml_features


def get_test_data_path():
    """Get the default test data directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'swap_correction', 'tests', 'test_data')


def load_training_data(ml_data_dir: str = 'ml_data', test_data_dir: str = None):
    """
    Load training labels and extract features for all frames.
    
    Returns:
    --------
    tuple
        (features_df, labels_df, trial_names)
    """
    if test_data_dir is None:
        test_data_dir = get_test_data_path()
    
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
    print("Extracting features for all trials...")
    all_features = []
    all_labels = []
    trial_names_list = []
    
    unique_trials = labels_df['trial'].unique()
    
    for i, trial_name in enumerate(unique_trials):
        print(f"[{i+1}/{len(unique_trials)}] Processing: {trial_name}", end=' ... ', flush=True)
        
        trial_dir = os.path.join(test_data_dir, trial_name)
        if not os.path.exists(trial_dir):
            print("SKIPPED (directory not found)")
            continue
        
        try:
            # Load raw data (we'll use level1 as input since that's what we're trying to improve)
            csv_files = [f for f in os.listdir(trial_dir) if f.endswith('_level1.csv')]
            if not csv_files:
                print("SKIPPED (no level1 data)")
                continue
            
            level1_file = csv_files[0]
            trial_data = pivr_loader.load_raw_data(trial_dir, level1_file, px2mm=True)
            fps = pivr_loader.get_all_settings(trial_dir)['Framerate']
            
            # Get labels for this trial
            trial_labels = labels_df[labels_df['trial'] == trial_name].copy()
            trial_labels = trial_labels.sort_values('frame_idx')
            
            # Extract features
            trial_features = ml_features.extract_all_frame_features(trial_data, fps=fps)
            
            # Align features and labels
            min_len = min(len(trial_features), len(trial_labels))
            trial_features = trial_features.iloc[:min_len]
            trial_labels = trial_labels.iloc[:min_len]
            
            # Add trial name to features
            trial_features['trial'] = trial_name
            
            all_features.append(trial_features)
            all_labels.append(trial_labels[['frame_idx', 'is_swapped']])
            trial_names_list.extend([trial_name] * min_len)
            
            n_swapped = trial_labels['is_swapped'].sum()
            print(f"OK ({len(trial_features)} frames, {n_swapped} swapped)")
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features extracted!")
    
    # Combine all features and labels
    features_df = pd.concat(all_features, ignore_index=True)
    labels_df_combined = pd.concat(all_labels, ignore_index=True)
    
    # Remove trial name from features (keep it separate for splitting)
    trial_names_array = np.array(trial_names_list)
    if 'trial' in features_df.columns:
        features_df = features_df.drop(columns=['trial'])
    
    print(f"\nTotal features extracted: {len(features_df)} frames")
    print(f"Feature dimensions: {features_df.shape[1]} features")
    print(f"Swapped frames: {labels_df_combined['is_swapped'].sum()} ({labels_df_combined['is_swapped'].mean()*100:.2f}%)")
    
    return features_df, labels_df_combined['is_swapped'], trial_names_array, split


def prepare_train_val_test_split(features_df, labels, trial_names, split_dict):
    """
    Split data into train/val/test sets stratified by trial.
    """
    # Get unique trials in each split
    train_trials = set(split_dict['train'])
    val_trials = set(split_dict['val'])
    test_trials = set(split_dict['test'])
    
    # Create masks
    train_mask = np.array([t in train_trials for t in trial_names])
    val_mask = np.array([t in val_trials for t in trial_names])
    test_mask = np.array([t in test_trials for t in trial_names])
    
    X_train = features_df[train_mask].values
    y_train = labels[train_mask].values
    X_val = features_df[val_mask].values
    y_val = labels[val_mask].values
    X_test = features_df[test_mask].values
    y_test = labels[test_mask].values
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} frames ({y_train.sum()} swapped, {y_train.mean()*100:.2f}%)")
    print(f"  Validation: {len(X_val)} frames ({y_val.sum()} swapped, {y_val.mean()*100:.2f}%)")
    print(f"  Test: {len(X_test)} frames ({y_test.sum()} swapped, {y_test.mean()*100:.2f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost_model(X_train, y_train, X_val, y_val, 
                       class_weight_ratio: float = None):
    """
    Train XGBoost classifier with hyperparameter tuning.
    """
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    # Calculate class weights if needed
    if class_weight_ratio is None:
        # Calculate ratio of negative to positive samples
        n_positive = y_train.sum()
        n_negative = len(y_train) - n_positive
        if n_positive > 0:
            class_weight_ratio = n_negative / n_positive
        else:
            class_weight_ratio = 1.0
    
    print(f"Class weight ratio (negative/positive): {class_weight_ratio:.2f}")
    
    # Set up XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': class_weight_ratio,  # Handle class imbalance
        'random_state': 42,
        'n_jobs': -1,
    }
    
    print(f"\nTraining with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10
    )
    
    # Convert to sklearn wrapper for easier use
    sklearn_model = xgb.XGBClassifier(**params)
    sklearn_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=20,
                     verbose=False)
    
    return sklearn_model, model


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluate model performance on all splits.
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    results = {}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('val', X_val, y_val), 
                              ('test', X_test, y_test)]:
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Metrics
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        results[split_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'n_samples': len(y),
            'n_positive': y.sum(),
            'n_predicted_positive': y_pred.sum()
        }
        
        print(f"\n{split_name.upper()} Set:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
        print(f"  Samples: {len(y)} ({y.sum()} positive, {y_pred.sum()} predicted positive)")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML swap detector')
    parser.add_argument('--ml-data-dir', type=str, default='ml_data',
                       help='Directory containing ML training data')
    parser.add_argument('--output-dir', type=str, default='ml_models',
                       help='Directory to save trained models')
    parser.add_argument('--test-data-dir', type=str, default=None,
                       help='Directory containing test data (default: auto-detect)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MACHINE LEARNING SWAP DETECTOR TRAINING")
    print("=" * 80)
    
    # Load training data
    test_data_dir = args.test_data_dir or get_test_data_path()
    features_df, labels, trial_names, split = load_training_data(
        args.ml_data_dir, test_data_dir
    )
    
    # Prepare train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
        features_df, labels, trial_names, split
    )
    
    # Handle NaN values (fill with median)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model, xgb_model = train_xgboost_model(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # Evaluate model
    results = evaluate_model(
        model, X_train_scaled, y_train, 
        X_val_scaled, y_val, 
        X_test_scaled, y_test
    )
    
    # Save model and scaler
    model_file = os.path.join(args.output_dir, 'swap_detector_xgb.pkl')
    scaler_file = os.path.join(args.output_dir, 'feature_scaler.pkl')
    imputer_file = os.path.join(args.output_dir, 'feature_imputer.pkl')
    feature_names_file = os.path.join(args.output_dir, 'feature_names.pkl')
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(imputer_file, 'wb') as f:
        pickle.dump(imputer, f)
    
    with open(feature_names_file, 'wb') as f:
        pickle.dump(list(features_df.columns), f)
    
    print(f"\nModel saved to: {model_file}")
    print(f"Scaler saved to: {scaler_file}")
    print(f"Imputer saved to: {imputer_file}")
    print(f"Feature names saved to: {feature_names_file}")
    
    # Save evaluation results
    results_file = os.path.join(args.output_dir, 'training_results.json')
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for split_name, metrics in results.items():
        results_serializable[split_name] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
        }
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Evaluation results saved to: {results_file}")
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = features_df.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_file = os.path.join(args.output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_file, index=False)
    
    print(f"\nTop 20 most important features:")
    print(importance_df.head(20).to_string(index=False))
    print(f"\nFeature importance saved to: {importance_file}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


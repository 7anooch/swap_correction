#!/usr/bin/env python3
"""
Analyze validation set stability and representativeness.

Assesses whether the validation set is representative of the test set and
evaluates the stability of performance metrics across different validation folds.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Import from train_ml_swap_detector
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ml_swap_detector import load_training_data, get_test_data_path, prepare_train_val_test_split


def analyze_validation_representativeness(level1_results, raw_results):
    """Analyze if validation set is representative of test set."""
    analysis = {}
    
    for model_name, results in [('level1', level1_results), ('raw', raw_results)]:
        val = results['val']
        test = results['test']
        
        # Calculate gaps
        f1_gap = abs(val['f1'] - test['f1'])
        precision_gap = abs(val['precision'] - test['precision'])
        recall_gap = abs(val['recall'] - test['recall'])
        auc_gap = abs(val['auc'] - test['auc'])
        
        # Assess representativeness
        if f1_gap < 0.005:
            representativeness = 'excellent'
        elif f1_gap < 0.01:
            representativeness = 'good'
        elif f1_gap < 0.02:
            representativeness = 'fair'
        else:
            representativeness = 'poor'
        
        analysis[model_name] = {
            'f1_gap': float(f1_gap),
            'precision_gap': float(precision_gap),
            'recall_gap': float(recall_gap),
            'auc_gap': float(auc_gap),
            'representativeness': representativeness,
            'val_f1': float(val['f1']),
            'test_f1': float(test['f1']),
            'val_precision': float(val['precision']),
            'test_precision': float(test['precision']),
            'val_recall': float(val['recall']),
            'test_recall': float(test['recall']),
        }
    
    return analysis


def cross_validate_model(X_train_val, y_train_val, n_splits: int = 5):
    """Perform k-fold cross-validation to assess stability."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
        X_train_fold = X_train_val[train_idx]
        y_train_fold = y_train_val[train_idx]
        X_val_fold = X_train_val[val_idx]
        y_val_fold = y_train_val[val_idx]
        
        # Preprocess
        imputer = SimpleImputer(strategy='median')
        X_train_fold = imputer.fit_transform(X_train_fold)
        X_val_fold = imputer.transform(X_val_fold)
        
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)
        
        # Calculate class weights
        n_positive = y_train_fold.sum()
        n_negative = len(y_train_fold) - n_positive
        class_weight_ratio = n_negative / n_positive if n_positive > 0 else 1.0
        
        # Train model
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
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_val_fold)
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        
        precision = precision_score(y_val_fold, y_pred, zero_division=0)
        recall = recall_score(y_val_fold, y_pred, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, zero_division=0)
        auc = roc_auc_score(y_val_fold, y_proba) if len(np.unique(y_val_fold)) > 1 else 0.0
        
        # Calculate specificity
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_val_fold, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1] if cm.shape[1] > 1 else 0, 
                                                          cm[1,0] if cm.shape[0] > 1 else 0, 
                                                          cm[1,1] if cm.shape == (2,2) else 0)
        sensitivity = recall  # Sensitivity = Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else (1.0 if fp == 0 else 0.0)
        
        cv_results.append({
            'fold': fold + 1,
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'f1': float(f1),
            'auc': float(auc)
        })
    
    return cv_results


def analyze_split_appropriateness(level1_results, raw_results):
    """Analyze if the train/val/test split is appropriate."""
    analysis = {}
    
    for model_name, results in [('level1', level1_results), ('raw', raw_results)]:
        train = results['train']
        val = results['val']
        test = results['test']
        
        # Check class distribution consistency
        train_pos_rate = train['n_positive'] / train['n_samples']
        val_pos_rate = val['n_positive'] / val['n_samples']
        test_pos_rate = test['n_positive'] / test['n_samples']
        
        pos_rate_std = np.std([train_pos_rate, val_pos_rate, test_pos_rate])
        
        # Check set sizes
        total_samples = train['n_samples'] + val['n_samples'] + test['n_samples']
        train_ratio = train['n_samples'] / total_samples
        val_ratio = val['n_samples'] / total_samples
        test_ratio = test['n_samples'] / total_samples
        
        # Assess split quality
        if pos_rate_std < 0.01:
            class_balance = 'excellent'
        elif pos_rate_std < 0.02:
            class_balance = 'good'
        elif pos_rate_std < 0.05:
            class_balance = 'fair'
        else:
            class_balance = 'poor'
        
        analysis[model_name] = {
            'train_pos_rate': float(train_pos_rate),
            'val_pos_rate': float(val_pos_rate),
            'test_pos_rate': float(test_pos_rate),
            'pos_rate_std': float(pos_rate_std),
            'class_balance_quality': class_balance,
            'train_ratio': float(train_ratio),
            'val_ratio': float(val_ratio),
            'test_ratio': float(test_ratio),
            'total_samples': int(total_samples)
        }
    
    return analysis


def create_stability_visualization(representativeness, split_analysis, cv_results_level1, cv_results_raw, output_dir):
    """Create visualization of validation stability analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Validation Set Stability Analysis', fontsize=16, fontweight='bold')
    
    # Representativeness comparison
    ax = axes[0, 0]
    models = ['Level1', 'Raw Data']
    f1_gaps = [representativeness['level1']['f1_gap'], representativeness['raw']['f1_gap']]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax.bar(models, f1_gaps, color=colors, alpha=0.8)
    ax.set_ylabel('F1 Score Gap (|Val - Test|)', fontsize=11)
    ax.set_title('Validation Representativeness\n(Smaller is Better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, gap in zip(bars, f1_gaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{gap:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Class distribution consistency
    ax = axes[0, 1]
    level1_rates = [
        split_analysis['level1']['train_pos_rate'],
        split_analysis['level1']['val_pos_rate'],
        split_analysis['level1']['test_pos_rate']
    ]
    raw_rates = [
        split_analysis['raw']['train_pos_rate'],
        split_analysis['raw']['val_pos_rate'],
        split_analysis['raw']['test_pos_rate']
    ]
    
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, level1_rates, width, label='Level1', alpha=0.8, color='#2E86AB')
    ax.bar(x + width/2, raw_rates, width, label='Raw Data', alpha=0.8, color='#A23B72')
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Val', 'Test'])
    ax.set_ylabel('Positive Class Rate', fontsize=11)
    ax.set_title('Class Distribution Across Splits', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cross-validation stability (Level1)
    ax = axes[1, 0]
    if cv_results_level1:
        folds = [r['fold'] for r in cv_results_level1]
        f1_scores = [r['f1'] for r in cv_results_level1]
        ax.plot(folds, f1_scores, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Level1')
        ax.axhline(y=np.mean(f1_scores), color='#2E86AB', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(f1_scores):.4f}')
        ax.fill_between(folds, np.mean(f1_scores) - np.std(f1_scores), 
                        np.mean(f1_scores) + np.std(f1_scores), alpha=0.2, color='#2E86AB')
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Cross-Validation Stability (Level1)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cross-validation stability (Raw)
    ax = axes[1, 1]
    if cv_results_raw:
        folds = [r['fold'] for r in cv_results_raw]
        f1_scores = [r['f1'] for r in cv_results_raw]
        ax.plot(folds, f1_scores, 's-', linewidth=2, markersize=8, color='#A23B72', label='Raw Data')
        ax.axhline(y=np.mean(f1_scores), color='#A23B72', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(f1_scores):.4f}')
        ax.fill_between(folds, np.mean(f1_scores) - np.std(f1_scores), 
                        np.mean(f1_scores) + np.std(f1_scores), alpha=0.2, color='#A23B72')
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('Cross-Validation Stability (Raw Data)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'validation_stability_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stability visualization saved to: {output_path}")


def main():
    """Main analysis function."""
    print("=" * 80)
    print("VALIDATION STABILITY ANALYSIS")
    print("=" * 80)
    
    output_dir = 'ml_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("\nLoading model results...")
    with open('ml_models/training_results.json', 'r') as f:
        level1_results = json.load(f)
    with open('ml_models_raw/training_results.json', 'r') as f:
        raw_results = json.load(f)
    
    # Analyze representativeness
    print("Analyzing validation set representativeness...")
    representativeness = analyze_validation_representativeness(level1_results, raw_results)
    
    # Analyze split appropriateness
    print("Analyzing train/val/test split appropriateness...")
    split_analysis = analyze_split_appropriateness(level1_results, raw_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION STABILITY SUMMARY")
    print("=" * 80)
    
    for model_name in ['level1', 'raw']:
        print(f"\n{model_name.upper()} Model:")
        print(f"  Validation Representativeness: {representativeness[model_name]['representativeness'].upper()}")
        print(f"  F1 Gap (|Val - Test|): {representativeness[model_name]['f1_gap']:.4f}")
        print(f"  Class Balance Quality: {split_analysis[model_name]['class_balance_quality'].upper()}")
        print(f"  Positive Rate Std: {split_analysis[model_name]['pos_rate_std']:.4f}")
    
    # Cross-validation (optional, can be slow)
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 80)
    print("Performing 5-fold cross-validation to assess stability...")
    print("(This may take a few minutes)")
    
    cv_results_level1 = None
    cv_results_raw = None
    
    try:
        # Level1 model
        print("\nLevel1 Model:")
        features_df, labels, trial_names, split = load_training_data(
            'ml_data', get_test_data_path(), use_raw_data=False
        )
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
            features_df, labels, trial_names, split
        )
        
        # Combine train and val for CV
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        cv_results_level1 = cross_validate_model(X_train_val, y_train_val)
        print(f"  CV F1 Mean: {np.mean([r['f1'] for r in cv_results_level1]):.4f}")
        print(f"  CV F1 Std: {np.std([r['f1'] for r in cv_results_level1]):.4f}")
        
        # Raw data model
        print("\nRaw Data Model:")
        features_df, labels, trial_names, split = load_training_data(
            'ml_data', get_test_data_path(), use_raw_data=True
        )
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
            features_df, labels, trial_names, split
        )
        
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        
        cv_results_raw = cross_validate_model(X_train_val, y_train_val)
        print(f"  CV F1 Mean: {np.mean([r['f1'] for r in cv_results_raw]):.4f}")
        print(f"  CV F1 Std: {np.std([r['f1'] for r in cv_results_raw]):.4f}")
        
    except Exception as e:
        print(f"  Warning: Cross-validation failed: {e}")
        print("  Skipping cross-validation (this is optional)")
    
    # Create visualization
    print("\nGenerating visualization...")
    create_stability_visualization(representativeness, split_analysis, 
                                  cv_results_level1, cv_results_raw, output_dir)
    
    # Save results
    results = {
        'representativeness': representativeness,
        'split_analysis': split_analysis,
        'cross_validation': {
            'level1': cv_results_level1,
            'raw': cv_results_raw
        }
    }
    
    results_file = os.path.join(output_dir, 'validation_stability_analysis.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {results_file}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    for model_name in ['level1', 'raw']:
        rep = representativeness[model_name]['representativeness']
        if rep in ['excellent', 'good']:
            print(f"\n{model_name.upper()} Model: Validation set is {rep}. Split is appropriate.")
        else:
            print(f"\n{model_name.upper()} Model: Validation set representativeness is {rep}.")
            print("  Consider: Re-splitting data or using cross-validation for more reliable estimates.")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


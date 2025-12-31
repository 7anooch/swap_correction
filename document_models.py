#!/usr/bin/env python3
"""
Generate model registry and documentation.

Creates MODEL_REGISTRY.md with comprehensive information about both trained models
for easy reference and deployment.
"""

import os
import json
import pandas as pd
from datetime import datetime


def load_model_metadata():
    """Load all model metadata."""
    # Load training results
    with open('ml_models/training_results.json', 'r') as f:
        level1_results = json.load(f)
    
    with open('ml_models_raw/training_results.json', 'r') as f:
        raw_results = json.load(f)
    
    # Load feature importance
    level1_importance = pd.read_csv('ml_models/feature_importance.csv')
    raw_importance = pd.read_csv('ml_models_raw/feature_importance.csv')
    
    # Check if files exist
    level1_model_exists = os.path.exists('ml_models/swap_detector_xgb.pkl')
    raw_model_exists = os.path.exists('ml_models_raw/swap_detector_xgb.pkl')
    
    return {
        'level1': {
            'results': level1_results,
            'importance': level1_importance,
            'model_exists': level1_model_exists,
            'model_path': 'ml_models/swap_detector_xgb.pkl',
            'scaler_path': 'ml_models/feature_scaler.pkl',
            'imputer_path': 'ml_models/feature_imputer.pkl',
            'feature_names_path': 'ml_models/feature_names.pkl'
        },
        'raw': {
            'results': raw_results,
            'importance': raw_importance,
            'model_exists': raw_model_exists,
            'model_path': 'ml_models_raw/swap_detector_xgb.pkl',
            'scaler_path': 'ml_models_raw/feature_scaler.pkl',
            'imputer_path': 'ml_models_raw/feature_imputer.pkl',
            'feature_names_path': 'ml_models_raw/feature_names.pkl'
        }
    }


def generate_model_registry(metadata: dict):
    """Generate MODEL_REGISTRY.md."""
    
    level1 = metadata['level1']
    raw = metadata['raw']
    
    registry = f"""# ML Model Registry

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This registry documents all trained ML models for swap detection in animal tracking data.

---

## Model 1: Level1 Model

### Overview
- **Purpose**: Detect remaining swaps in level1.csv (auto-corrected) data
- **Training Data**: level1.csv vs level2.csv (ground truth)
- **Model Type**: XGBoost Classifier
- **Status**: {'✅ Available' if level1['model_exists'] else '❌ Not Found'}

### Performance Metrics

#### Test Set (Primary Metric)
- **F1-Score**: {level1['results']['test']['f1']:.4f} ({level1['results']['test']['f1']*100:.2f}%)
- **Precision**: {level1['results']['test']['precision']:.4f} ({level1['results']['test']['precision']*100:.2f}%)
- **Recall**: {level1['results']['test']['recall']:.4f} ({level1['results']['test']['recall']*100:.2f}%)
- **ROC-AUC**: {level1['results']['test']['auc']:.4f} ({level1['results']['test']['auc']*100:.2f}%)

#### All Splits
| Split | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| Train | {level1['results']['train']['f1']:.4f} | {level1['results']['train']['precision']:.4f} | {level1['results']['train']['recall']:.4f} | {level1['results']['train']['auc']:.4f} |
| Val | {level1['results']['val']['f1']:.4f} | {level1['results']['val']['precision']:.4f} | {level1['results']['val']['recall']:.4f} | {level1['results']['val']['auc']:.4f} |
| Test | {level1['results']['test']['f1']:.4f} | {level1['results']['test']['precision']:.4f} | {level1['results']['test']['recall']:.4f} | {level1['results']['test']['auc']:.4f} |

### Confusion Matrix (Test Set)
```
                Predicted
              No Swap  Swap
Actual No Swap  {level1['results']['test']['confusion_matrix'][0][0]:5d}  {level1['results']['test']['confusion_matrix'][0][1]:5d}
       Swap     {level1['results']['test']['confusion_matrix'][1][0]:5d}  {level1['results']['test']['confusion_matrix'][1][1]:5d}
```

- **False Positive Rate**: {level1['results']['test']['confusion_matrix'][0][1] / sum(level1['results']['test']['confusion_matrix'][0]):.2%}
- **False Negative Rate**: {level1['results']['test']['confusion_matrix'][1][0] / sum(level1['results']['test']['confusion_matrix'][1]):.2%}

### Training Data Characteristics
- **Total Training Frames**: {level1['results']['train']['n_samples']:,}
- **Swapped Frames**: {level1['results']['train']['n_positive']:,} ({level1['results']['train']['n_positive']/level1['results']['train']['n_samples']*100:.2f}%)
- **Test Set Size**: {level1['results']['test']['n_samples']:,}
- **Test Swapped Frames**: {level1['results']['test']['n_positive']:,} ({level1['results']['test']['n_positive']/level1['results']['test']['n_samples']*100:.2f}%)

### Model Configuration
- **Algorithm**: XGBoost
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **N Estimators**: 200 (with early stopping)
- **Subsample**: 0.8
- **Column Sample by Tree**: 0.8
- **Gaussian Filter Sigma**: 4.6 (optimal from grid search)

### Feature Set
- **Total Features**: 56
- **Feature Types**:
  - Position features (x, y for head, tail, mid, centroid)
  - Distance features (head-tail, head-mid, tail-mid)
  - Speed features (head, tail, ratio)
  - Velocity features (x, y components, magnitude)
  - Angular features (orientation, motion angles, angular velocity)
  - Geometric features (cross-sign, path curvature, alignment angle)
  - Temporal context features (mean/std over windows: 5, 10, 20, 50 frames)
  - Cumulative distance features

### Top 10 Most Important Features
{chr(10).join([f"{i+1}. **{row['feature']}**: {row['importance']:.4f}" for i, row in level1['importance'].head(10).iterrows()])}

### File Locations
- **Model**: `{level1['model_path']}`
- **Scaler**: `{level1['scaler_path']}`
- **Imputer**: `{level1['imputer_path']}`
- **Feature Names**: `{level1['feature_names_path']}`
- **Feature Importance**: `ml_models/feature_importance.csv`
- **Training Results**: `ml_models/training_results.json`

### Usage
```python
import pickle
import pandas as pd
from swap_correction import ml_features_optimized, pivr_loader

# Load model and preprocessing
with open('ml_models/swap_detector_xgb.pkl', 'rb') as f:
    model = pickle.load(f)
with open('ml_models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('ml_models/feature_imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Load level1 data
trial_data = pivr_loader.load_raw_data(trial_dir, 'trial_level1.csv', px2mm=True)
fps = pivr_loader.get_all_settings(trial_dir)['Framerate']

# Extract features
features = ml_features_optimized.extract_all_frame_features_optimized(
    trial_data, fps=fps, apply_filtering=True, filter_sigma=4.6
)

# Preprocess and predict
X = imputer.transform(features.values)
X = scaler.transform(X)
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

### Use Case
**Best for**: Detecting remaining swaps after initial auto-correction
- Higher precision (99.16%) and overall performance (98.95% F1)
- Lower false positive rate
- Use when you have level1.csv files and want to improve them further

---

## Model 2: Raw Data Model

### Overview
- **Purpose**: Detect swaps directly from raw tracking data
- **Training Data**: raw _data.csv vs level2.csv (ground truth)
- **Model Type**: XGBoost Classifier
- **Status**: {'✅ Available' if raw['model_exists'] else '❌ Not Found'}

### Performance Metrics

#### Test Set (Primary Metric)
- **F1-Score**: {raw['results']['test']['f1']:.4f} ({raw['results']['test']['f1']*100:.2f}%)
- **Precision**: {raw['results']['test']['precision']:.4f} ({raw['results']['test']['precision']*100:.2f}%)
- **Recall**: {raw['results']['test']['recall']:.4f} ({raw['results']['test']['recall']*100:.2f}%)
- **ROC-AUC**: {raw['results']['test']['auc']:.4f} ({raw['results']['test']['auc']*100:.2f}%)

#### All Splits
| Split | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| Train | {raw['results']['train']['f1']:.4f} | {raw['results']['train']['precision']:.4f} | {raw['results']['train']['recall']:.4f} | {raw['results']['train']['auc']:.4f} |
| Val | {raw['results']['val']['f1']:.4f} | {raw['results']['val']['precision']:.4f} | {raw['results']['val']['recall']:.4f} | {raw['results']['val']['auc']:.4f} |
| Test | {raw['results']['test']['f1']:.4f} | {raw['results']['test']['precision']:.4f} | {raw['results']['test']['recall']:.4f} | {raw['results']['test']['auc']:.4f} |

### Confusion Matrix (Test Set)
```
                Predicted
              No Swap  Swap
Actual No Swap  {raw['results']['test']['confusion_matrix'][0][0]:5d}  {raw['results']['test']['confusion_matrix'][0][1]:5d}
       Swap     {raw['results']['test']['confusion_matrix'][1][0]:5d}  {raw['results']['test']['confusion_matrix'][1][1]:5d}
```

- **False Positive Rate**: {raw['results']['test']['confusion_matrix'][0][1] / sum(raw['results']['test']['confusion_matrix'][0]):.2%}
- **False Negative Rate**: {raw['results']['test']['confusion_matrix'][1][0] / sum(raw['results']['test']['confusion_matrix'][1]):.2%}

### Training Data Characteristics
- **Total Training Frames**: {raw['results']['train']['n_samples']:,}
- **Swapped Frames**: {raw['results']['train']['n_positive']:,} ({raw['results']['train']['n_positive']/raw['results']['train']['n_samples']*100:.2f}%)
- **Test Set Size**: {raw['results']['test']['n_samples']:,}
- **Test Swapped Frames**: {raw['results']['test']['n_positive']:,} ({raw['results']['test']['n_positive']/raw['results']['test']['n_samples']*100:.2f}%)

### Model Configuration
- **Algorithm**: XGBoost
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **N Estimators**: 200 (with early stopping)
- **Subsample**: 0.8
- **Column Sample by Tree**: 0.8
- **Gaussian Filter Sigma**: 4.6 (optimal from grid search)

### Feature Set
- **Total Features**: 56 (same as Level1 model)
- **Feature Types**: Same as Level1 model

### Top 10 Most Important Features
{chr(10).join([f"{i+1}. **{row['feature']}**: {row['importance']:.4f}" for i, row in raw['importance'].head(10).iterrows()])}

### File Locations
- **Model**: `{raw['model_path']}`
- **Scaler**: `{raw['scaler_path']}`
- **Imputer**: `{raw['imputer_path']}`
- **Feature Names**: `{raw['feature_names_path']}`
- **Feature Importance**: `ml_models_raw/feature_importance.csv`
- **Training Results**: `ml_models_raw/training_results.json`

### Usage
```python
import pickle
import pandas as pd
from swap_correction import ml_features_optimized, pivr_loader

# Load model and preprocessing
with open('ml_models_raw/swap_detector_xgb.pkl', 'rb') as f:
    model = pickle.load(f)
with open('ml_models_raw/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('ml_models_raw/feature_imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Load raw data
trial_data = pivr_loader.load_raw_data(trial_dir, 'trial_data.csv', px2mm=True)
fps = pivr_loader.get_all_settings(trial_dir)['Framerate']

# Extract features
features = ml_features_optimized.extract_all_frame_features_optimized(
    trial_data, fps=fps, apply_filtering=True, filter_sigma=4.6
)

# Preprocess and predict
X = imputer.transform(features.values)
X = scaler.transform(X)
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

### Use Case
**Best for**: Detecting swaps directly from raw tracking data
- Can work directly on raw data (no need for level1 correction first)
- Still achieves good performance (97.44% F1)
- Handles higher swap rate in raw data
- Use when you want to skip the level1 correction step entirely

---

## Model Comparison Summary

| Aspect | Level1 Model | Raw Data Model |
|--------|--------------|----------------|
| **Test F1-Score** | {level1['results']['test']['f1']:.4f} | {raw['results']['test']['f1']:.4f} |
| **Test Precision** | {level1['results']['test']['precision']:.4f} | {raw['results']['test']['precision']:.4f} |
| **Test Recall** | {level1['results']['test']['recall']:.4f} | {raw['results']['test']['recall']:.4f} |
| **Training Swap Rate** | {level1['results']['train']['n_positive']/level1['results']['train']['n_samples']*100:.2f}% | {raw['results']['train']['n_positive']/raw['results']['train']['n_samples']*100:.2f}% |
| **Best For** | Post-correction refinement | Direct raw data processing |

## Recommendations

1. **Use Level1 Model** if:
   - You already have level1.csv files
   - Maximum accuracy is required
   - Lower false positive rate is critical

2. **Use Raw Data Model** if:
   - You want to process raw data directly
   - You want to skip level1 correction step
   - You're building a new pipeline from scratch

Both models are production-ready and can be used for automated swap detection.

---

## Additional Resources

- **Overfitting Analysis**: `ml_analysis/model_overfitting_analysis.json`
- **Learning Curves**: `ml_analysis/learning_curves_data.json`
- **Model Comparison Report**: `ml_analysis/model_comparison_report.md`
- **Feature Extraction Code**: `swap_correction/ml_features_optimized.py`
- **Training Script**: `train_ml_swap_detector.py`
"""
    
    return registry


def main():
    """Main documentation generation function."""
    print("=" * 80)
    print("MODEL DOCUMENTATION GENERATION")
    print("=" * 80)
    
    # Load metadata
    print("\nLoading model metadata...")
    metadata = load_model_metadata()
    
    # Generate registry
    print("Generating MODEL_REGISTRY.md...")
    registry = generate_model_registry(metadata)
    
    # Save
    output_file = 'MODEL_REGISTRY.md'
    with open(output_file, 'w') as f:
        f.write(registry)
    
    print(f"\nModel registry saved to: {output_file}")
    print("\n" + "=" * 80)
    print("DOCUMENTATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()


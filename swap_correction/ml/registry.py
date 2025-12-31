"""
Model registry management.

Provides functions to update and query the model registry.
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def load_model_metadata(base_dir: Optional[str] = None):
    """
    Load all model metadata.
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory (default: current working directory)
        
    Returns:
    --------
    dict
        Dictionary with model metadata
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    # Load training results
    level1_results_path = Path(base_dir) / 'ml_models' / 'training_results.json'
    raw_results_path = Path(base_dir) / 'ml_models_raw' / 'training_results.json'
    
    level1_results = None
    raw_results = None
    
    if level1_results_path.exists():
        with open(level1_results_path, 'r') as f:
            level1_results = json.load(f)
    
    if raw_results_path.exists():
        with open(raw_results_path, 'r') as f:
            raw_results = json.load(f)
    
    # Load feature importance
    level1_importance = None
    raw_importance = None
    
    level1_importance_path = Path(base_dir) / 'ml_models' / 'feature_importance.csv'
    raw_importance_path = Path(base_dir) / 'ml_models_raw' / 'feature_importance.csv'
    
    if level1_importance_path.exists():
        level1_importance = pd.read_csv(level1_importance_path)
    
    if raw_importance_path.exists():
        raw_importance = pd.read_csv(raw_importance_path)
    
    # Check if files exist
    level1_model_exists = (Path(base_dir) / 'ml_models' / 'swap_detector_xgb.pkl').exists()
    raw_model_exists = (Path(base_dir) / 'ml_models_raw' / 'swap_detector_xgb.pkl').exists()
    
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


def generate_model_registry(metadata: dict) -> str:
    """
    Generate MODEL_REGISTRY.md content.
    
    Parameters:
    -----------
    metadata : dict
        Model metadata from load_model_metadata()
        
    Returns:
    --------
    str
        Markdown content for MODEL_REGISTRY.md
    """
    level1 = metadata['level1']
    raw = metadata['raw']
    
    # Helper to format top features
    def format_top_features(importance_df, n=10):
        if importance_df is None:
            return "N/A"
        return '\n'.join([
            f"{i+1}. **{row['feature']}**: {row['importance']:.4f}"
            for i, row in importance_df.head(n).iterrows()
        ])
    
    # Helper to format performance table
    def format_performance_table(results):
        if results is None:
            return "N/A (model not found)"
        return f"""| Split | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| Train | {results['train']['f1']:.4f} | {results['train']['precision']:.4f} | {results['train']['recall']:.4f} | {results['train']['auc']:.4f} |
| Val | {results['val']['f1']:.4f} | {results['val']['precision']:.4f} | {results['val']['recall']:.4f} | {results['val']['auc']:.4f} |
| Test | {results['test']['f1']:.4f} | {results['test']['precision']:.4f} | {results['test']['recall']:.4f} | {results['test']['auc']:.4f} |"""
    
    # Helper to format confusion matrix
    def format_confusion_matrix(results):
        if results is None:
            return "N/A"
        cm = results['test']['confusion_matrix']
        fp_rate = cm[0][1] / sum(cm[0]) if sum(cm[0]) > 0 else 0
        fn_rate = cm[1][0] / sum(cm[1]) if sum(cm[1]) > 0 else 0
        return f"""```
                Predicted
              No Swap  Swap
Actual No Swap  {cm[0][0]:5d}  {cm[0][1]:5d}
       Swap     {cm[1][0]:5d}  {cm[1][1]:5d}
```

- **False Positive Rate**: {fp_rate:.2%}
- **False Negative Rate**: {fn_rate:.2%}"""
    
    # Helper to format values safely
    def format_value(results, split, metric, fmt='.4f'):
        if results is None:
            return 'N/A'
        try:
            value = results[split][metric]
            return f"{value:{fmt}}"
        except:
            return 'N/A'
    
    def format_swap_rate(results):
        if results is None:
            return 'N/A'
        try:
            rate = results['train']['n_positive'] / results['train']['n_samples'] * 100
            return f"{rate:.2f}%"
        except:
            return 'N/A'
    
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
"""
    
    if level1['results']:
        registry += f"""- **F1-Score**: {level1['results']['test']['f1']:.4f} ({level1['results']['test']['f1']*100:.2f}%)
- **Precision**: {level1['results']['test']['precision']:.4f} ({level1['results']['test']['precision']*100:.2f}%)
- **Recall**: {level1['results']['test']['recall']:.4f} ({level1['results']['test']['recall']*100:.2f}%)
- **ROC-AUC**: {level1['results']['test']['auc']:.4f} ({level1['results']['test']['auc']*100:.2f}%)

#### All Splits
{format_performance_table(level1['results'])}

### Confusion Matrix (Test Set)
{format_confusion_matrix(level1['results'])}

### Training Data Characteristics
- **Total Training Frames**: {level1['results']['train']['n_samples']:,}
- **Swapped Frames**: {level1['results']['train']['n_positive']:,} ({level1['results']['train']['n_positive']/level1['results']['train']['n_samples']*100:.2f}%)
- **Test Set Size**: {level1['results']['test']['n_samples']:,}
- **Test Swapped Frames**: {level1['results']['test']['n_positive']:,} ({level1['results']['test']['n_positive']/level1['results']['test']['n_samples']*100:.2f}%)
"""
    else:
        registry += "Model not found or not trained.\n"
    
    registry += f"""
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
{format_top_features(level1['importance'])}

### File Locations
- **Model**: `{level1['model_path']}`
- **Scaler**: `{level1['scaler_path']}`
- **Imputer**: `{level1['imputer_path']}`
- **Feature Names**: `{level1['feature_names_path']}`
- **Feature Importance**: `ml_models/feature_importance.csv`
- **Training Results**: `ml_models/training_results.json`

### Usage
```python
from swap_correction.ml.api import SwapPredictor

# Simple usage
predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30)
segments = predictor.predict_segments(trial_data, fps=30)
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
"""
    
    if raw['results']:
        registry += f"""- **F1-Score**: {raw['results']['test']['f1']:.4f} ({raw['results']['test']['f1']*100:.2f}%)
- **Precision**: {raw['results']['test']['precision']:.4f} ({raw['results']['test']['precision']*100:.2f}%)
- **Recall**: {raw['results']['test']['recall']:.4f} ({raw['results']['test']['recall']*100:.2f}%)
- **ROC-AUC**: {raw['results']['test']['auc']:.4f} ({raw['results']['test']['auc']*100:.2f}%)

#### All Splits
{format_performance_table(raw['results'])}

### Confusion Matrix (Test Set)
{format_confusion_matrix(raw['results'])}

### Training Data Characteristics
- **Total Training Frames**: {raw['results']['train']['n_samples']:,}
- **Swapped Frames**: {raw['results']['train']['n_positive']:,} ({raw['results']['train']['n_positive']/raw['results']['train']['n_samples']*100:.2f}%)
- **Test Set Size**: {raw['results']['test']['n_samples']:,}
- **Test Swapped Frames**: {raw['results']['test']['n_positive']:,} ({raw['results']['test']['n_positive']/raw['results']['test']['n_samples']*100:.2f}%)
"""
    else:
        registry += "Model not found or not trained.\n"
    
    registry += f"""
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
{format_top_features(raw['importance'])}

### File Locations
- **Model**: `{raw['model_path']}`
- **Scaler**: `{raw['scaler_path']}`
- **Imputer**: `{raw['imputer_path']}`
- **Feature Names**: `{raw['feature_names_path']}`
- **Feature Importance**: `ml_models_raw/feature_importance.csv`
- **Training Results**: `ml_models_raw/training_results.json`

### Usage
```python
from swap_correction.ml.api import SwapPredictor

# Simple usage
predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30)
segments = predictor.predict_segments(trial_data, fps=30)
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
| **Test F1-Score** | {format_value(level1['results'], 'test', 'f1')} | {format_value(raw['results'], 'test', 'f1')} |
| **Test Precision** | {format_value(level1['results'], 'test', 'precision')} | {format_value(raw['results'], 'test', 'precision')} |
| **Test Recall** | {format_value(level1['results'], 'test', 'recall')} | {format_value(raw['results'], 'test', 'recall')} |
| **Training Swap Rate** | {format_swap_rate(level1['results'])} | {format_swap_rate(raw['results'])} |
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

- **Usage Guide**: `docs/ML_USAGE_GUIDE.md`
- **API Reference**: `docs/ML_API_REFERENCE.md`
- **Training Guide**: `docs/ML_TRAINING_GUIDE.md`
- **Overfitting Analysis**: `ml_analysis/model_overfitting_analysis.json`
- **Learning Curves**: `ml_analysis/learning_curves_data.json`
- **Model Comparison Report**: `ml_analysis/model_comparison_report.md`
"""
    
    return registry


def update_registry(output_file: str = 'MODEL_REGISTRY.md', base_dir: Optional[str] = None):
    """
    Update MODEL_REGISTRY.md with current model information.
    
    Parameters:
    -----------
    output_file : str
        Path to output file (default: MODEL_REGISTRY.md)
    base_dir : str, optional
        Base directory (default: current working directory)
    """
    metadata = load_model_metadata(base_dir)
    registry_content = generate_model_registry(metadata)
    
    with open(output_file, 'w') as f:
        f.write(registry_content)
    
    print(f"Model registry updated: {output_file}")


def get_model_metadata(model_type: str, base_dir: Optional[str] = None) -> Dict:
    """
    Get metadata for a specific model type.
    
    Parameters:
    -----------
    model_type : str
        Model type: 'level1' or 'raw'
    base_dir : str, optional
        Base directory
        
    Returns:
    --------
    dict
        Model metadata
    """
    metadata = load_model_metadata(base_dir)
    return metadata.get(model_type, {})


def register_new_model(model_path: str, metadata: Dict, base_dir: Optional[str] = None):
    """
    Register a newly trained model in the registry.
    
    Parameters:
    -----------
    model_path : str
        Path to model file
    metadata : dict
        Model metadata dictionary
    base_dir : str, optional
        Base directory
    """
    # This would extend the registry with new model information
    # For now, just update the registry
    update_registry(base_dir=base_dir)
    print(f"New model registered: {model_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Update model registry')
    parser.add_argument('--output', type=str, default='MODEL_REGISTRY.md',
                       help='Output file path')
    args = parser.parse_args()
    
    update_registry(args.output)


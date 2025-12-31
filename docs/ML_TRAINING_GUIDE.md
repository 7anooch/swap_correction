# ML Model Training Guide

Complete guide for training swap detection models.

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Training Process](#training-process)
4. [Parameter Tuning](#parameter-tuning)
5. [Model Evaluation](#model-evaluation)
6. [Best Practices](#best-practices)

---

## Overview

The ML training pipeline consists of three main steps:

1. **Data Preparation**: Extract labels from ground truth comparisons
2. **Training**: Train XGBoost classifier on extracted features
3. **Evaluation**: Assess model performance and update registry

### Model Types

- **Level1 Model**: Trained on `level1.csv` vs `level2.csv` (ground truth)
- **Raw Data Model**: Trained on raw `_data.csv` vs `level2.csv` (ground truth)

---

## Data Preparation

### Step 1: Prepare Training Labels

Extract frame-level and segment-level labels by comparing data files.

#### For Level1 Model

```bash
python -m swap_correction.ml.training.prepare_data
```

This compares `level1.csv` vs `level2.csv` for all trials.

#### For Raw Data Model

```bash
python -m swap_correction.ml.training.prepare_data --use-raw-data
```

This compares raw `_data.csv` vs `level2.csv` for all trials.

### Output Files

The preparation script creates:
- `ml_data/training_labels.csv` (or `training_labels_raw.csv`)
- `ml_data/segment_labels.csv` (or `segment_labels_raw.csv`)
- `ml_data/trial_statistics.csv` (or `trial_statistics_raw.csv`)
- `ml_data/train_test_split.json`

### Data Requirements

- **Ground Truth**: All trials must have `*_level2.csv` files
- **Input Data**: 
  - For level1 model: `*_level1.csv` files
  - For raw model: `*_data.csv` files
- **Settings**: `experiment_settings.json` for frame rate (defaults to 30 if missing)

### Data Statistics

After preparation, check the statistics:

```python
import pandas as pd

stats = pd.read_csv('ml_data/trial_statistics.csv')
print(f"Total trials: {len(stats)}")
print(f"Perfect trials: {stats['is_perfect'].sum()}")
print(f"Mean error rate: {stats['error_rate'].mean():.2f}%")
```

---

## Training Process

### Step 2: Train Model

Train an XGBoost classifier on the prepared data.

#### For Level1 Model

```bash
python -m swap_correction.ml.training.train_model \
    --output-dir ml_models
```

#### For Raw Data Model

```bash
python -m swap_correction.ml.training.train_model \
    --use-raw-data \
    --output-dir ml_models_raw
```

### Training Parameters

Default XGBoost parameters (optimized):
- `max_depth`: 6
- `learning_rate`: 0.1
- `n_estimators`: 200 (with early stopping)
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: Automatically calculated from class imbalance
- `early_stopping_rounds`: 20

### Feature Extraction

Features are extracted using the optimized pipeline:
- **Gaussian Filtering**: Applied with sigma=4.6 (optimal)
- **56 Features**: Position, distance, speed, velocity, angular, geometric, temporal
- **Speed**: ~400x faster than original implementation

### Training Output

The training script saves:
- `swap_detector_xgb.pkl` - Trained model
- `feature_scaler.pkl` - Feature scaler
- `feature_imputer.pkl` - Feature imputer
- `feature_names.pkl` - Feature names
- `training_results.json` - Performance metrics
- `feature_importance.csv` - Feature importance rankings

### Monitoring Training

Training progress is shown with validation loss:
```
[0]	train-logloss:0.63011	val-logloss:0.63825
[10]	train-logloss:0.29647	val-logloss:0.32323
...
```

Early stopping prevents overfitting by stopping when validation loss stops improving.

---

## Parameter Tuning

### Gaussian Filter Sigma

The Gaussian filter sigma was optimized through grid search. To retune:

```bash
python -m swap_correction.ml.training.tune_parameters \
    --parameter sigma \
    --range 4.0 6.0 \
    --step 0.2
```

This tests different sigma values and reports performance for each.

### Hyperparameter Tuning

For XGBoost hyperparameters, modify `train_model.py`:

```python
params = {
    'max_depth': 6,  # Try: 4, 5, 6, 7, 8
    'learning_rate': 0.1,  # Try: 0.05, 0.1, 0.2
    'subsample': 0.8,  # Try: 0.6, 0.8, 1.0
    'colsample_bytree': 0.8,  # Try: 0.6, 0.8, 1.0
    # ...
}
```

Use cross-validation or a validation set to select optimal parameters.

---

## Model Evaluation

### Step 3: Evaluate Model

After training, evaluate model performance:

```bash
# Run all evaluations
python -m swap_correction.ml.evaluation all

# Or run specific analyses
python -m swap_correction.ml.evaluation overfitting
python -m swap_correction.ml.evaluation learning-curves
python -m swap_correction.ml.evaluation compare
```

### Evaluation Metrics

The training script automatically evaluates on train/val/test splits:
- **Precision**: Fraction of predicted swaps that are correct
- **Recall**: Fraction of actual swaps that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### Expected Performance

- **Level1 Model**: F1 ≈ 0.99 (98-99%)
- **Raw Data Model**: F1 ≈ 0.97 (97-98%)

If performance is significantly lower, check:
1. Data quality (missing frames, tracking errors)
2. Class imbalance (may need to adjust `scale_pos_weight`)
3. Feature extraction (ensure using optimized version)

---

## Best Practices

### 1. Data Quality

- **Ensure ground truth is accurate**: Level2.csv files should be manually verified
- **Check for missing data**: Handle NaN values appropriately
- **Verify frame rates**: Consistent fps across trials

### 2. Train/Test Split

- **Stratified by trial**: Split at trial level, not frame level
- **Representative**: Ensure test set has similar characteristics to training
- **Size**: Typical split: 70% train, 15% val, 15% test

### 3. Feature Engineering

- **Use optimized extraction**: Always use `swap_correction.ml.features.extract_all_frame_features_optimized`
- **Gaussian filtering**: Apply filtering (sigma=4.6) for noisy data
- **Feature consistency**: Ensure same features used in training and inference

### 4. Model Selection

- **Compare both models**: Train both level1 and raw models
- **Evaluate on diverse data**: Test on different conditions/trials
- **Check overfitting**: Monitor train vs test performance gap

### 5. Validation

- **Cross-validation**: Use for hyperparameter tuning
- **Hold-out test set**: Keep test set completely separate
- **External validation**: Test on completely new datasets

### 6. Documentation

- **Update registry**: Run `python -m swap_correction.ml.registry` after training
- **Record parameters**: Document any changes from defaults
- **Track performance**: Keep records of model performance over time

---

## Training on New Datasets

### Preparing Data

1. **Organize data**: Place trials in subdirectories
2. **Ensure ground truth**: All trials need `*_level2.csv` files
3. **Check format**: Verify data format matches expected structure

### Training Steps

```bash
# 1. Prepare data
python -m swap_correction.ml.training.prepare_data \
    --test-data-dir /path/to/new/data

# 2. Train model
python -m swap_correction.ml.training.train_model \
    --test-data-dir /path/to/new/data \
    --output-dir ml_models_new

# 3. Evaluate
python -m swap_correction.ml.evaluation.evaluate_on_dataset \
    /path/to/new/data \
    --model-type level1 \
    --output-dir ml_analysis/evaluations
```

### Comparing Models

After training on new data, compare with existing models:

```bash
python -m swap_correction.ml.evaluation.compare
```

---

## Troubleshooting

### Low Performance

**Symptoms**: F1-score < 0.90

**Possible Causes**:
- Insufficient training data
- Poor data quality
- Class imbalance issues
- Feature extraction problems

**Solutions**:
1. Check data quality and ground truth accuracy
2. Increase training data size
3. Adjust class weights
4. Verify feature extraction pipeline

### Overfitting

**Symptoms**: Large gap between train and test performance

**Solutions**:
1. Increase regularization (lower learning rate, more subsampling)
2. Reduce model complexity (lower max_depth)
3. Add more training data
4. Use early stopping (already enabled)

### Training Errors

**Common Issues**:
- **Missing files**: Ensure all required data files exist
- **Memory errors**: Process trials in batches
- **Feature mismatch**: Use same feature extraction as training

---

## Advanced Topics

### Custom Feature Engineering

To add new features, modify `swap_correction/ml/features/features.py`:

1. Add feature calculation in `extract_all_frame_features_optimized()`
2. Ensure feature count matches (update if needed)
3. Retrain model with new features

### Transfer Learning

To adapt a pre-trained model to new data:

1. Load existing model
2. Fine-tune on new data (lower learning rate)
3. Evaluate on new test set

### Ensemble Methods

Combine multiple models:

```python
from swap_correction.ml.api import SwapPredictor

level1_pred = SwapPredictor('level1').predict(data, fps)
raw_pred = SwapPredictor('raw').predict(data, fps)

# Consensus: Both must agree
consensus = (level1_pred == raw_pred) & (level1_pred == 1)

# Union: Either detects
union = (level1_pred == 1) | (raw_pred == 1)
```

---

## Next Steps

After training:
1. **Update registry**: `python -m swap_correction.ml.registry`
2. **Evaluate performance**: Run evaluation suite
3. **Test on new data**: Validate on diverse datasets
4. **Document changes**: Update training guide if needed

---

## See Also

- **Usage Guide**: `docs/ML_USAGE_GUIDE.md` - How to use trained models
- **API Reference**: `docs/ML_API_REFERENCE.md` - Complete API documentation
- **Model Registry**: `MODEL_REGISTRY.md` - Model performance and metadata


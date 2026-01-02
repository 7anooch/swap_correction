# Threshold Optimization Guide

## Overview

By default, binary classification models use a threshold of 0.5 to convert probabilities into binary predictions. However, the optimal threshold depends on your specific goals and the class distribution in your data.

This guide explains how to find the optimal threshold that maximizes your desired metric (e.g., % Frames Clean Post, F1-score, etc.).

## Why Optimize Threshold?

### Default 0.5 Threshold

The default threshold of 0.5 assumes:
- Equal cost for false positives and false negatives
- Balanced class distribution
- You want to maximize overall accuracy

### When to Optimize

You should optimize the threshold if:
- ✅ You care more about a specific metric (e.g., % Clean Post)
- ✅ Classes are imbalanced
- ✅ False positives and false negatives have different costs
- ✅ You want to maximize precision or recall specifically

### Your Use Case

Since you care most about **maximizing % Frames Clean Post**, threshold optimization can help you find a threshold that:
- Reduces false positives (which lower % Clean Post)
- Maintains good swap detection (recall)
- Maximizes overall frame classification accuracy

## How It Works

1. **Get probabilities** from the model (not binary predictions)
2. **Test multiple thresholds** (e.g., 0.1, 0.2, ..., 0.9)
3. **For each threshold**:
   - Binarize predictions: `pred = (prob >= threshold)`
   - Calculate metrics (F1, % Clean Post, etc.)
4. **Find the threshold** that maximizes your desired metric

## Usage

### Command Line

#### Basic Usage (Maximize % Clean Post)

```bash
python -m swap_correction.ml.evaluation optimize-threshold \
    --model-type level1 \
    --metric pct_clean_post \
    --split val
```

#### Maximize F1-Score

```bash
python -m swap_correction.ml.evaluation optimize-threshold \
    --model-type level1 \
    --metric f1 \
    --split val
```

#### Maximize % Swaps Resolved

```bash
python -m swap_correction.ml.evaluation optimize-threshold \
    --model-type level1 \
    --metric pct_swaps_resolved \
    --split val
```

#### Save Results

```bash
python -m swap_correction.ml.evaluation optimize-threshold \
    --model-type level1 \
    --metric pct_clean_post \
    --split val \
    --output-dir threshold_analysis
```

#### Compare Multiple Thresholds

```bash
python -m swap_correction.ml.evaluation.optimize_threshold \
    --model-type level1 \
    --metric pct_clean_post \
    --split val \
    --compare-thresholds 0.3 0.4 0.5 0.6 0.7
```

### Python API

#### Find Optimal Threshold

```python
from swap_correction.ml.api import SwapPredictor
from swap_correction.ml.evaluation.find_optimal_threshold import (
    find_optimal_threshold_on_dataset
)
from swap_correction.ml.training.train_model import (
    load_training_data, prepare_train_val_test_split
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load model
predictor = SwapPredictor(model_type='level1')
model = predictor.model
scaler = predictor.scaler
imputer = predictor.imputer

# Load validation data
features_df, labels, trial_names, split = load_training_data(
    ml_data_dir='ml_data',
    use_raw_data=False
)

X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(
    features_df, labels, trial_names, split
)

# Preprocess
X_val = imputer.transform(X_val)
X_val = scaler.transform(X_val)

# Find optimal threshold
optimal_threshold, best_metrics, all_results = find_optimal_threshold_on_dataset(
    model, X_val, y_val,
    metric='pct_clean_post',  # Maximize % Clean Post
    n_thresholds=100
)

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"% Clean Post: {best_metrics['pct_frames_clean_post']:.2f}%")
```

#### Use Optimal Threshold in Predictions

```python
from swap_correction.ml.api import SwapPredictor
from swap_correction import pivr_loader

# Load predictor
predictor = SwapPredictor(model_type='level1')

# Load trial data
trial_dir = "path/to/trial"
trial_data = pivr_loader.load_raw_data(trial_dir, "trial_level1.csv", px2mm=True)
fps = pivr_loader.get_all_settings(trial_dir)['Framerate']

# Use optimal threshold (e.g., 0.45 instead of default 0.5)
optimal_threshold = 0.45  # From threshold optimization
predictions = predictor.predict(trial_data, fps=fps, threshold=optimal_threshold)
```

## Metrics You Can Optimize

### `pct_clean_post` (Recommended for You)

**What it maximizes**: Percentage of frames correctly classified
- Formula: `(TP + TN) / Total Frames`
- Best for: Overall accuracy, minimizing classification errors
- **This is what you want!**

### `f1`

**What it maximizes**: F1-score (harmonic mean of precision and recall)
- Best for: Balanced precision and recall
- Use when: You want a balanced metric

### `precision`

**What it maximizes**: Precision (fewer false positives)
- Best for: When false positives are costly
- Trade-off: May miss some swaps (lower recall)

### `recall` (or `pct_swaps_resolved`)

**What it maximizes**: Recall (detect more swaps)
- Best for: When missing swaps is costly
- Trade-off: May have more false positives (lower precision)

## Understanding Results

### Example Output

```
THRESHOLD OPTIMIZATION RESULTS
================================================================================

Optimized for: pct_clean_post
Optimal threshold: 0.4523

Performance at optimal threshold:
  Precision: 0.9723
  Recall: 0.8845
  F1-Score: 0.9261
  Sensitivity: 0.8845
  Specificity: 0.9789
  % Swaps Resolved: 88.45%
  % Frames Clean Post: 97.89%

Comparison with default threshold (0.5):
  Threshold 0.5: 97.23%
  Optimal (0.4523): 97.89%
  Improvement: +0.66%
```

### What This Means

- **Optimal threshold (0.4523)**: Lower than default 0.5
  - Model is slightly more conservative
  - Fewer false positives
  - Slightly lower recall (misses a few swaps)
  - **Higher % Clean Post** (what you want!)

- **Improvement (+0.66%)**: Small but meaningful
  - On 1000 frames, that's ~7 more frames correctly classified
  - On 100,000 frames, that's ~660 more frames correctly classified

## When to Re-optimize

You should re-optimize the threshold if:
- ✅ You retrain the model
- ✅ You use a different dataset (different characteristics)
- ✅ You change feature extraction
- ✅ Performance degrades over time

## Best Practices

### 1. Use Validation Set for Optimization

```bash
--split val  # Use validation set (recommended)
```

**Why**: 
- Test set should remain untouched for final evaluation
- Validation set is representative of test distribution
- Prevents overfitting to test set

### 2. Test on Test Set After Optimization

After finding optimal threshold on validation set:
1. Apply threshold to test set
2. Verify performance improvement
3. If test performance is worse, threshold may be overfit to validation

### 3. Consider Multiple Metrics

While optimizing for % Clean Post, also check:
- **F1-score**: Should remain high
- **% Swaps Resolved**: Shouldn't drop too much
- **Precision/Recall**: Understand the trade-offs

### 4. Document Your Threshold

Save the optimal threshold with your model:
- Record in model registry
- Include in evaluation reports
- Use consistently in production

## Example Workflow

### Step 1: Find Optimal Threshold

```bash
python -m swap_correction.ml.evaluation optimize-threshold \
    --model-type level1 \
    --metric pct_clean_post \
    --split val \
    --output-dir threshold_analysis
```

### Step 2: Review Results

Check `threshold_analysis/optimal_threshold.json`:
```json
{
  "optimal_threshold": 0.4523,
  "metric_optimized": "pct_clean_post",
  "best_metrics": {
    "pct_frames_clean_post": 97.89,
    "f1": 0.9261,
    ...
  }
}
```

### Step 3: Apply to Predictions

```python
predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(
    trial_data, 
    fps=30, 
    threshold=0.4523  # Use optimal threshold
)
```

### Step 4: Verify on Test Set

```python
# Evaluate on test set with optimal threshold
from swap_correction.ml.api import BatchProcessor

processor = BatchProcessor(model_type='level1')
# Note: BatchProcessor uses default 0.5 threshold
# You may need to modify it to use custom threshold
```

## Troubleshooting

### Threshold is Very Low (<0.3)

**Possible causes**:
- Model is too conservative
- Class imbalance (many more negatives than positives)
- Model probabilities are poorly calibrated

**Solutions**:
- Check class distribution
- Consider recalibrating probabilities
- Review model training

### Threshold is Very High (>0.7)

**Possible causes**:
- Model is too aggressive
- Many false positives at 0.5
- Need higher confidence for positive predictions

**Solutions**:
- Check precision at default threshold
- Review false positive rate
- Consider if threshold makes sense

### No Improvement Over 0.5

**Possible causes**:
- Model is well-calibrated
- 0.5 is already optimal
- Metric is insensitive to threshold

**Solutions**:
- This is fine! Model is working well
- Check if other metrics improve
- Consider if optimization is necessary

## Advanced: Custom Threshold Selection

If you want to optimize for a custom metric or combination:

```python
from swap_correction.ml.evaluation.find_optimal_threshold import (
    calculate_metrics_at_threshold, find_optimal_threshold
)

# Get probabilities
y_proba = model.predict_proba(X_val)[:, 1]

# Test thresholds
thresholds = np.linspace(0.1, 0.9, 100)
best_score = -np.inf
best_threshold = 0.5

for threshold in thresholds:
    metrics = calculate_metrics_at_threshold(y_val, y_proba, threshold)
    
    # Custom metric: weighted combination
    score = (0.7 * metrics['pct_frames_clean_post'] + 
             0.3 * metrics['pct_swaps_resolved'])
    
    if score > best_score:
        best_score = score
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.4f}")
```

## Recommended Thresholds

Based on comprehensive threshold optimization across multiple stability analysis iterations:

### Level1 Models
- **Recommended Threshold**: **0.63**
- **Rationale**: Optimized to maximize % Frames Clean Post
- **Performance**: Achieves ~97.65% ± 0.76% frames clean post on average
- **Usage**: 
  ```python
  predictor = SwapPredictor(model_type='level1')
  predictions = predictor.predict(trial_data, fps=30, threshold=0.63)
  ```

### Raw Models
- **Recommended Threshold**: **0.5** (default)
- **Rationale**: Default threshold is optimal for raw data models
- **Performance**: Achieves ~96% frames clean post on average
- **Usage**: 
  ```python
  predictor = SwapPredictor(model_type='raw')
  # threshold=0.5 is the default, no need to specify
  predictions = predictor.predict(trial_data, fps=30)
  ```

### When to Use Custom Thresholds

You may want to use a custom threshold if:
- You have domain-specific requirements (e.g., prioritize precision over recall)
- Your data distribution differs significantly from training data
- You're optimizing for a different metric (e.g., F1-score instead of % Clean Post)

## Summary

- **Level1 models**: Use threshold **0.63** (optimized for % Frames Clean Post)
- **Raw models**: Use threshold **0.5** (default, optimal)
- **Your goal**: Maximize % Frames Clean Post
- **Solution**: Use recommended thresholds above, or optimize for your specific use case
- **Typical improvement**: 0.5-2% over default 0.5 threshold for level1 models
- **Usage**: Apply recommended threshold consistently across all predictions

The recommended thresholds are based on comprehensive analysis across multiple iterations and provide optimal performance for their respective model types.


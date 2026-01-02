# ML Swap Detection Usage Guide

**Version**: 1.1  
**Last Updated**: 2026-01-02  
**Note**: Updated to reflect Features V4 as the default implementation (~36 features)

This guide provides comprehensive instructions for using the ML-based swap detection system for animal tracking data.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Selection](#model-selection)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Batch Processing](#batch-processing)
6. [Evaluating on New Datasets](#evaluating-on-new-datasets)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Quick Start

### Installation

The ML models are part of the `swap_correction` package. Ensure all dependencies are installed:

```bash
pip install -e .
```

Required packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`

### Simple Example

```python
from swap_correction.ml.api import SwapPredictor
from swap_correction import pivr_loader

# Load your data
trial_dir = "path/to/trial"
trial_data = pivr_loader.load_raw_data(trial_dir, "trial_level1.csv", px2mm=True)
fps = pivr_loader.get_all_settings(trial_dir)['Framerate']

# Create predictor
predictor = SwapPredictor(model_type='level1')

# Get predictions (uses recommended threshold automatically)
predictions = predictor.predict(trial_data, fps=fps)
segments = predictor.predict_segments(trial_data, fps=fps)

print(f"Detected {len(segments)} swap segments")
```

---

## Model Selection

### Level1 Model

**Use when:**
- You have `level1.csv` files (auto-corrected data)
- You want maximum accuracy (98.95% F1-score)
- Lower false positive rate is critical
- You're refining already-corrected data

**Performance:**
- F1-Score: 98.95%
- Precision: 99.16%
- Recall: 98.74%

**Recommended Threshold:** 0.63 (optimized for % Frames Clean Post)

### Raw Data Model

**Use when:**
- You want to process raw `_data.csv` files directly
- You want to skip the level1 correction step
- You're building a new pipeline from scratch
- You can accept slightly lower precision (97.44% F1-score)

**Performance:**
- F1-Score: 97.44%
- Precision: 97.20%
- Recall: 97.69%

**Recommended Threshold:** 0.5 (default, optimal for this model)

### Decision Tree

```
Do you have level1.csv files?
├─ YES → Use Level1 Model (better performance)
└─ NO → Use Raw Data Model (works on raw data)
```

---

## Basic Usage

### 1. Loading a Model

```python
from swap_correction.ml.api import SwapPredictor

# Load level1 model
predictor = SwapPredictor(model_type='level1')

# Or load raw data model
predictor = SwapPredictor(model_type='raw')
```

### 2. Making Predictions

#### Binary Predictions

```python
# Level1 model: Use recommended threshold 0.63
predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.63)
print(f"Swapped frames: {predictions.sum()} / {len(predictions)}")

# Raw model: Uses default threshold 0.5 (optimal)
predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30)  # threshold=0.5 is default
print(f"Swapped frames: {predictions.sum()} / {len(predictions)}")
```

#### Probabilities

```python
# Get swap probabilities (0-1)
probabilities = predictor.predict_proba(trial_data, fps=30)
print(f"Average swap probability: {probabilities.mean():.3f}")
```

#### Swap Segments

```python
# Get contiguous swap segments
segments = predictor.predict_segments(trial_data, fps=30, min_segment_length=5)
for start, end in segments:
    print(f"Swap segment: frames {start} to {end} (length: {end-start+1})")
```

### 3. Loading Data from Files

```python
# Auto-detect data file and fps
predictions, trial_data = predictor.predict_from_file(
    trial_dir="path/to/trial",
    data_file=None,  # Auto-detect based on model_type
    fps=None  # Auto-load from experiment_settings.json
)
```

---

## Advanced Usage

### Custom Filtering

```python
# Use different Gaussian filter sigma
predictor = SwapPredictor(model_type='level1', filter_sigma=5.0)
```

**Note**: The default sigma (4.6) was optimized through grid search. Only change if you have specific requirements.

### Getting Model Information

```python
# Get model metadata
info = predictor.get_model_info()
print(f"Model F1-Score: {info['performance']['test']['f1']:.4f}")
print(f"Feature count: {info['feature_count']}")
```

### Direct Model Access

```python
from swap_correction.ml.api import load_model

# Load model components directly
model, scaler, imputer, feature_names = load_model('level1')

# Use for custom processing
# ... (see API reference for details)
```

---

## Batch Processing

### Process Multiple Trials

```python
from swap_correction.ml.api import BatchProcessor

# Initialize processor
processor = BatchProcessor(model_type='level1')

# Process multiple trials
trial_dirs = [
    "path/to/trial1",
    "path/to/trial2",
    "path/to/trial3"
]

results = processor.process_trials(trial_dirs)

# Access results
for trial_name, result in results.items():
    if 'error' not in result:
        print(f"{trial_name}: {result['n_swapped']} swapped frames")
        print(f"  Segments: {len(result['segments'])}")
```

### Batch Processing Output

Each trial result contains:
- `predictions`: Binary predictions array
- `probabilities`: Probability array
- `segments`: List of (start, end) tuples
- `n_frames`: Total number of frames
- `n_swapped`: Number of swapped frames
- `swap_rate`: Fraction of frames that are swapped

---

## Evaluating on New Datasets

### Evaluate Model Performance

```python
from swap_correction.ml.api import BatchProcessor

processor = BatchProcessor(model_type='level1')

# Evaluate on dataset with ground truth
results = processor.evaluate_on_dataset(
    data_dir="path/to/dataset",
    ground_truth_level='level2'  # Compare against level2.csv
)

# Summary statistics
summary = results['summary']
print(f"Mean F1-Score: {summary['mean_f1']:.4f}")
print(f"Mean Precision: {summary['mean_precision']:.4f}")
print(f"Mean Recall: {summary['mean_recall']:.4f}")
print(f"Mean % Swaps Resolved: {summary['mean_pct_swaps_resolved']:.2f}%")
print(f"Mean % Frames Clean Post: {summary['mean_pct_frames_clean_post']:.2f}%")

# Per-trial results
for trial_result in results['trial_results']:
    if 'error' not in trial_result:
        print(f"{trial_result['trial']}: F1={trial_result['f1']:.4f}, "
              f"% Resolved={trial_result['pct_swaps_resolved']:.2f}%, "
              f"% Clean Post={trial_result['pct_frames_clean_post']:.2f}%")

# Note: See docs/METRIC_CALCULATIONS.md for detailed explanations of these metrics
```

### Using the Evaluation Script

```bash
# Evaluate level1 model on new dataset
python -m swap_correction.ml.evaluation.evaluate_on_dataset \
    /path/to/dataset \
    --model-type level1 \
    --ground-truth level2 \
    --output-dir ml_analysis/evaluations
```

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `FileNotFoundError: Model file not found`

**Solution**: Ensure the model has been trained. Check that `ml_models/` or `ml_models_raw/` directories exist and contain model files.

```python
from swap_correction.ml.api import list_available_models

available = list_available_models()
print(available)  # Shows which models are available
```

#### 2. Feature Mismatch

**Error**: `ValueError: Feature count mismatch`

**Solution**: Ensure you're using the same feature extraction pipeline. Current models expect ~36 features (Features V4) extracted with `swap_correction.ml.features.extract_all_frame_features_optimized()`. Legacy models may use different feature counts.

#### 3. Missing Data Files

**Error**: `FileNotFoundError: No appropriate data file found`

**Solution**: 
- For level1 model: Ensure `*_level1.csv` file exists
- For raw model: Ensure `*_data.csv` file exists
- Check that the trial directory path is correct

#### 4. Low Performance on New Data

**Possible Causes:**
- Data characteristics differ from training data
- Different tracking system or conditions
- Data quality issues (noise, missing frames)

**Solutions:**
- Evaluate model performance on a subset first
- Check data quality (missing frames, tracking errors)
- Consider retraining on diverse data
- Try the other model type

### Getting Help

1. Check model information:
```python
from swap_correction.ml.api import get_model_info
info = get_model_info('level1')
```

2. Verify data format:
```python
from swap_correction import pivr_loader
data = pivr_loader.load_raw_data(trial_dir, "trial_data.csv")
print(data.columns)  # Should include: xhead, yhead, xtail, ytail, etc.
```

3. Check feature extraction:
```python
from swap_correction.ml.features import extract_all_frame_features_optimized
features = extract_all_frame_features_optimized(data, fps=30)
print(features.shape)  # Should be (n_frames, ~36) for Features V4 (current default)
```

---

## Examples

### Example 1: Detect Swaps in a Single Trial

```python
from swap_correction.ml.api import SwapPredictor
from swap_correction import pivr_loader

# Setup
trial_dir = "data/trial_001"
predictor = SwapPredictor(model_type='level1')

# Load data
trial_data = pivr_loader.load_raw_data(
    trial_dir, 
    "trial_001_level1.csv", 
    px2mm=True
)
fps = pivr_loader.get_all_settings(trial_dir)['Framerate']

# Predict
predictions = predictor.predict(trial_data, fps=fps)
segments = predictor.predict_segments(trial_data, fps=fps)

# Results
print(f"Total frames: {len(predictions)}")
print(f"Swapped frames: {predictions.sum()}")
print(f"Swap segments: {len(segments)}")
for start, end in segments:
    print(f"  Frames {start}-{end} (length: {end-start+1})")
```

### Example 2: Batch Process All Trials

```python
from swap_correction.ml.api import BatchProcessor
import os

# Setup
data_dir = "data/all_trials"
processor = BatchProcessor(model_type='level1')

# Get all trial directories
trial_dirs = [
    os.path.join(data_dir, d) 
    for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
]

# Process
results = processor.process_trials(trial_dirs)

# Summary
total_swapped = 0
total_frames = 0
for trial_name, result in results.items():
    if 'error' not in result:
        total_swapped += result['n_swapped']
        total_frames += result['n_frames']
        print(f"{trial_name}: {result['swap_rate']:.2%} swapped")

print(f"\nOverall: {total_swapped}/{total_frames} frames swapped ({total_swapped/total_frames:.2%})")
```

### Example 3: Evaluate Model on New Dataset

```python
from swap_correction.ml.api import BatchProcessor
import json

# Setup
processor = BatchProcessor(model_type='level1')

# Evaluate
results = processor.evaluate_on_dataset(
    data_dir="data/new_dataset",
    ground_truth_level='level2'
)

# Save results
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
summary = results['summary']
print("Evaluation Summary:")
print(f"  Trials: {summary['n_valid']}/{summary['n_trials']}")
print(f"  Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
print(f"  Mean Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
print(f"  Mean Recall: {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
```

### Example 4: Apply Corrections Based on Predictions

```python
from swap_correction.ml.api import SwapPredictor
from swap_correction import pivr_loader, tracking_correction
import numpy as np

# Load data and predict
trial_dir = "data/trial_001"
predictor = SwapPredictor(model_type='level1')
predictions, trial_data = predictor.predict_from_file(trial_dir)

# Get swap segments
segments = predictor.predict_segments(trial_data, fps=30)

# Apply corrections
corrected_data = trial_data.copy()
for start, end in segments:
    # Swap head and tail for this segment
    corrected_data.loc[start:end, ['xhead', 'yhead', 'xtail', 'ytail']] = \
        corrected_data.loc[start:end, ['xtail', 'ytail', 'xhead', 'yhead']].values

# Save corrected data
output_file = os.path.join(trial_dir, "trial_001_level1_ml_corrected.csv")
corrected_data.to_csv(output_file, index=False)
```

### Example 5: Compare Both Models

```python
from swap_correction.ml.api import SwapPredictor
from swap_correction import pivr_loader

trial_dir = "data/trial_001"
trial_data = pivr_loader.load_raw_data(trial_dir, "trial_001_data.csv", px2mm=True)
fps = 30

# Predict with both models
level1_predictor = SwapPredictor(model_type='level1')
raw_predictor = SwapPredictor(model_type='raw')

level1_pred = level1_predictor.predict(trial_data, fps=fps)
raw_pred = raw_predictor.predict(trial_data, fps=fps)

# Compare
agreement = (level1_pred == raw_pred).mean()
print(f"Model agreement: {agreement:.2%}")

# Find disagreements
disagreements = np.where(level1_pred != raw_pred)[0]
print(f"Disagreements: {len(disagreements)} frames")
```

---

## Performance Tips

1. **Use optimized feature extraction**: Always use `swap_correction.ml.features.extract_all_frame_features_optimized()` (400x faster)

2. **Batch processing**: Use `BatchProcessor` for multiple trials to avoid reloading models

3. **Filter sigma**: Default (4.6) is optimal, but you can experiment if needed

4. **Memory**: For very large datasets, process trials in batches

---

## Next Steps

- **Training Guide**: See `docs/ML_TRAINING_GUIDE.md` for training new models
- **API Reference**: See `docs/ML_API_REFERENCE.md` for detailed API documentation
- **Model Registry**: See `MODEL_REGISTRY.md` for model performance and metadata
- **Metric Calculations**: See `docs/METRIC_CALCULATIONS.md` for detailed explanations of all performance metrics

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the API reference documentation
3. Check model registry for model-specific information
4. Examine evaluation results to understand model behavior


# ML API Reference

Complete reference documentation for the ML swap detection API.

## Table of Contents

1. [Model Loader API](#model-loader-api)
2. [Predictor API](#predictor-api)
3. [Batch Processor API](#batch-processor-api)
4. [Registry API](#registry-api)

---

## Model Loader API

### `load_model()`

Load a trained swap detection model and its preprocessors.

```python
from swap_correction.ml.api import load_model

model, scaler, imputer, feature_names = load_model(
    model_type='level1',  # or 'raw'/'raw_data'
    base_dir=None  # Optional: base directory for model files
)
```

**Parameters:**
- `model_type` (str): Type of model to load
  - `'level1'`: Model trained on level1.csv vs level2.csv
  - `'raw'` or `'raw_data'`: Model trained on raw _data.csv vs level2.csv
- `base_dir` (str, optional): Base directory for model files (default: current working directory)

**Returns:**
- `model` (XGBoostClassifier): Trained XGBoost classifier
- `scaler` (StandardScaler): Feature scaler
- `imputer` (SimpleImputer): Feature imputer for missing values
- `feature_names` (list): List of feature names in order

**Raises:**
- `ValueError`: If model_type is invalid
- `FileNotFoundError`: If model files are not found

**Example:**
```python
model, scaler, imputer, feature_names = load_model('level1')
print(f"Model loaded with {len(feature_names)} features")
```

---

### `get_model_info()`

Get metadata and performance information for a model.

```python
from swap_correction.ml.api import get_model_info

info = get_model_info(
    model_type='level1',
    base_dir=None
)
```

**Parameters:**
- `model_type` (str): Type of model
- `base_dir` (str, optional): Base directory

**Returns:**
- `dict`: Model metadata containing:
  - `'model_type'`: Model type
  - `'model_dir'`: Path to model directory
  - `'performance'`: Performance metrics (train/val/test)
  - `'training_data'`: Training data characteristics
  - `'feature_count'`: Number of features
  - `'configuration'`: Model configuration

**Example:**
```python
info = get_model_info('level1')
print(f"Test F1: {info['performance']['test']['f1']:.4f}")
print(f"Features: {info['feature_count']}")
```

---

### `list_available_models()`

List all available models and their availability status.

```python
from swap_correction.ml.api import list_available_models

available = list_available_models(base_dir=None)
```

**Parameters:**
- `base_dir` (str, optional): Base directory

**Returns:**
- `dict`: Dictionary mapping model types to availability (True/False)

**Example:**
```python
available = list_available_models()
for model_type, is_available in available.items():
    status = "Available" if is_available else "Not found"
    print(f"{model_type}: {status}")
```

---

## Predictor API

### `SwapPredictor`

High-level interface for swap detection predictions.

#### Constructor

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(
    model_type='level1',  # or 'raw'/'raw_data'
    filter_sigma=4.6  # Gaussian filter sigma
)
```

**Parameters:**
- `model_type` (str): Type of model to use
- `filter_sigma` (float): Gaussian filter sigma for feature extraction (default: 4.6)

---

#### `predict()`

Predict swapped frames for a trial.

```python
predictions = predictor.predict(
    trial_data,  # pd.DataFrame: Tracking data
    fps=30,  # int: Frame rate
    return_probabilities=False  # bool: Return probabilities instead of binary
)
```

**Parameters:**
- `trial_data` (pd.DataFrame): Tracking data (level1.csv or raw _data.csv)
- `fps` (int): Frame rate (default: 30)
- `return_probabilities` (bool): If True, return probabilities instead of binary predictions

**Returns:**
- `np.ndarray`: Binary predictions (0=no swap, 1=swap) or probabilities if `return_probabilities=True`

**Example:**
```python
predictions = predictor.predict(trial_data, fps=30)
print(f"Swapped frames: {predictions.sum()}")
```

---

#### `predict_proba()`

Get swap probabilities for each frame.

```python
probabilities = predictor.predict_proba(
    trial_data,  # pd.DataFrame
    fps=30  # int
)
```

**Parameters:**
- `trial_data` (pd.DataFrame): Tracking data
- `fps` (int): Frame rate

**Returns:**
- `np.ndarray`: Probability of swap for each frame (0-1)

**Example:**
```python
proba = predictor.predict_proba(trial_data, fps=30)
high_confidence_swaps = np.where(proba > 0.9)[0]
```

---

#### `predict_segments()`

Predict swap segments (contiguous regions of swapped frames).

```python
segments = predictor.predict_segments(
    trial_data,  # pd.DataFrame
    fps=30,  # int
    min_segment_length=1  # int: Minimum segment length
)
```

**Parameters:**
- `trial_data` (pd.DataFrame): Tracking data
- `fps` (int): Frame rate
- `min_segment_length` (int): Minimum length of a segment to include (default: 1)

**Returns:**
- `list of tuples`: List of (start_frame, end_frame) tuples for each swap segment

**Example:**
```python
segments = predictor.predict_segments(trial_data, fps=30, min_segment_length=5)
for start, end in segments:
    print(f"Swap: frames {start}-{end}")
```

---

#### `predict_from_file()`

Predict swaps from a trial directory.

```python
predictions, trial_data = predictor.predict_from_file(
    trial_dir,  # str: Trial directory path
    data_file=None,  # str, optional: Specific data file
    fps=None  # int, optional: Frame rate (auto-load if None)
)
```

**Parameters:**
- `trial_dir` (str): Directory containing trial data
- `data_file` (str, optional): Specific data file (auto-detect if None)
- `fps` (int, optional): Frame rate (load from settings if None)

**Returns:**
- `tuple`: (predictions, trial_data)
  - `predictions`: Binary predictions array
  - `trial_data`: Loaded tracking data

**Example:**
```python
predictions, data = predictor.predict_from_file("data/trial_001")
```

---

#### `get_model_info()`

Get information about the loaded model.

```python
info = predictor.get_model_info()
```

**Returns:**
- `dict`: Model metadata (same format as `get_model_info()`)

---

## Batch Processor API

### `BatchProcessor`

Process multiple trials in batch.

#### Constructor

```python
from swap_correction.ml.api import BatchProcessor

processor = BatchProcessor(
    model_type='level1',
    filter_sigma=4.6
)
```

**Parameters:**
- `model_type` (str): Type of model to use
- `filter_sigma` (float): Gaussian filter sigma

---

#### `process_trials()`

Process multiple trials and return predictions.

```python
results = processor.process_trials(
    trial_dirs,  # list of str: Trial directory paths
    data_file=None,  # str, optional: Specific data file
    fps=None  # int, optional: Frame rate
)
```

**Parameters:**
- `trial_dirs` (list of str): List of trial directory paths
- `data_file` (str, optional): Specific data file (auto-detect if None)
- `fps` (int, optional): Frame rate (load from settings if None)

**Returns:**
- `dict`: Dictionary mapping trial names to results:
  ```python
  {
      'trial_name': {
          'predictions': np.ndarray,
          'probabilities': np.ndarray,
          'segments': list of tuples,
          'n_frames': int,
          'n_swapped': int,
          'swap_rate': float,
          'trial_data': pd.DataFrame
      }
  }
  ```

**Example:**
```python
trial_dirs = ["data/trial1", "data/trial2"]
results = processor.process_trials(trial_dirs)
for name, result in results.items():
    print(f"{name}: {result['n_swapped']} swapped")
```

---

#### `evaluate_on_dataset()`

Evaluate model performance on a dataset with ground truth.

```python
results = processor.evaluate_on_dataset(
    data_dir,  # str: Directory containing trial subdirectories
    ground_truth_level='level2',  # str: 'level1' or 'level2'
    data_file_pattern=None  # str, optional: Pattern for data files
)
```

**Parameters:**
- `data_dir` (str): Directory containing trial subdirectories
- `ground_truth_level` (str): Ground truth level ('level1' or 'level2')
- `data_file_pattern` (str, optional): Pattern for data files

**Returns:**
- `dict`: Evaluation results:
  ```python
  {
      'summary': {
          'n_trials': int,
          'n_valid': int,
          'mean_precision': float,
          'mean_recall': float,
          'mean_f1': float,
          'std_precision': float,
          'std_recall': float,
          'std_f1': float
      },
      'trial_results': [
          {
              'trial': str,
              'precision': float,
              'recall': float,
              'f1': float,
              'tp': int,
              'fp': int,
              'fn': int,
              'tn': int,
              'n_frames': int,
              'n_swapped_gt': int,
              'n_swapped_pred': int
          }
      ]
  }
  ```

**Example:**
```python
results = processor.evaluate_on_dataset("data/new_dataset", ground_truth_level='level2')
print(f"Mean F1: {results['summary']['mean_f1']:.4f}")
```

---

## Registry API

### `update_registry()`

Update MODEL_REGISTRY.md with current model information.

```python
from swap_correction.ml.registry import update_registry

update_registry(
    output_file='MODEL_REGISTRY.md',
    base_dir=None
)
```

**Parameters:**
- `output_file` (str): Path to output file (default: 'MODEL_REGISTRY.md')
- `base_dir` (str, optional): Base directory

---

### `get_model_metadata()`

Get metadata for a specific model type.

```python
from swap_correction.ml.registry import get_model_metadata

metadata = get_model_metadata(
    model_type='level1',
    base_dir=None
)
```

**Parameters:**
- `model_type` (str): Model type ('level1' or 'raw')
- `base_dir` (str, optional): Base directory

**Returns:**
- `dict`: Model metadata dictionary

---

### `load_model_metadata()`

Load all model metadata.

```python
from swap_correction.ml.registry import load_model_metadata

metadata = load_model_metadata(base_dir=None)
```

**Returns:**
- `dict`: Dictionary with 'level1' and 'raw' model metadata

---

## Module-Level Imports

### Quick Import

```python
# Import main components
from swap_correction.ml.api import (
    SwapPredictor,
    BatchProcessor,
    load_model,
    get_model_info,
    list_available_models
)

# Or import from ml module
from swap_correction.ml import SwapPredictor, BatchProcessor
```

---

## Error Handling

All API functions raise appropriate exceptions:

- `ValueError`: Invalid parameters
- `FileNotFoundError`: Missing model files or data files
- `KeyError`: Missing required data columns
- `TypeError`: Incorrect data types

Always wrap API calls in try-except blocks for production code:

```python
try:
    predictor = SwapPredictor(model_type='level1')
    predictions = predictor.predict(trial_data, fps=30)
except FileNotFoundError as e:
    print(f"Model not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
```

---

## Performance Considerations

1. **Feature Extraction**: Use optimized version (`ml_features_optimized`) - 400x faster
2. **Batch Processing**: Reuse `BatchProcessor` instance to avoid reloading models
3. **Memory**: For very large datasets, process in chunks
4. **Filtering**: Default sigma (4.6) is optimal; only change if necessary

---

## See Also

- **Usage Guide**: `docs/ML_USAGE_GUIDE.md` - Comprehensive usage examples
- **Training Guide**: `docs/ML_TRAINING_GUIDE.md` - How to train models
- **Model Registry**: `MODEL_REGISTRY.md` - Model performance and metadata


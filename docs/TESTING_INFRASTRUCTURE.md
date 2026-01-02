# Testing Infrastructure Documentation

**Last Updated**: 2026-01-02  
**Purpose**: Guide to using the ML testing and evaluation infrastructure.

---

## Overview

The testing infrastructure provides tools for:
- **Stability Analysis**: Assessing model robustness across different random samples
- **Learning Curve Analysis**: Determining if more training data improves performance
- **Threshold Optimization**: Finding optimal classification thresholds
- **Model Evaluation**: Comprehensive evaluation on new datasets
- **Feature Comparison**: Comparing different feature sets

---

## Module Structure

### `swap_correction/ml/evaluation/`

Core evaluation and analysis tools.

#### Key Modules

- **`evaluate_on_dataset.py`**: Evaluate trained models on datasets with ground truth
- **`learning_curves.py`**: Generate learning curves to assess data needs
- **`find_optimal_threshold.py`**: Find optimal classification thresholds
- **`optimize_threshold.py`**: CLI for threshold optimization
- **`compare_models.py`**: Compare Level1 and Raw models side-by-side
- **`overfitting.py`**: Analyze overfitting in trained models
- **`validation_stability.py`**: Assess validation set representativeness

#### CLI Usage

```bash
# Evaluate a model on a dataset
python -m swap_correction.ml.evaluation evaluate \
    --data-dir /path/to/trials \
    --model-type level1 \
    --ground-truth-level level2 \
    --output-dir evaluation_results

# Optimize threshold
python -m swap_correction.ml.evaluation optimize-threshold \
    --model-type level1 \
    --metric pct_clean_post \
    --split val \
    --n-thresholds 100

# Generate learning curves
python -m swap_correction.ml.evaluation learning-curves \
    --ml-data-dir ml_data \
    --output-dir learning_curves
```

### `swap_correction/ml/stability/`

Stability analysis tools for assessing model robustness.

#### Key Modules

- **`run_stability_analysis.py`**: Main stability analysis pipeline
- **`find_valid_trials.py`**: Find valid trial directories in nested structures
- **`sample_trials.py`**: Randomly sample trials and create train/val/test splits
- **`aggregate_results.py`**: Aggregate results from multiple iterations
- **`visualize_stability.py`**: Generate visualizations for stability analysis

#### CLI Usage

```bash
# Run stability analysis
python -m swap_correction.ml.stability run \
    --parent-dir /path/to/main/dataset \
    --n-iterations 10 \
    --sample-sizes 30 40 50 \
    --output-dir stability_analysis \
    --base-seed 42

# Aggregate results
python -m swap_correction.ml.stability aggregate \
    --input-dir stability_analysis \
    --output-dir stability_analysis/aggregated
```

---

## Common Workflows

### 1. Stability Analysis

**Purpose**: Assess model robustness across different random samples.

**Steps**:

1. **Find valid trials**:
   ```python
   from swap_correction.ml.stability.find_valid_trials import find_valid_trial_directories
   
   valid_trials = find_valid_trial_directories(
       parent_dir='/path/to/main/dataset',
       filter_identical=True  # Filter out trials where level1 == level2
   )
   ```

2. **Run stability analysis**:
   ```bash
   python -m swap_correction.ml.stability run \
       --parent-dir /path/to/main/dataset \
       --n-iterations 10 \
       --sample-sizes 80 \
       --output-dir stability_analysis_v4
   ```

3. **Aggregate results**:
   ```bash
   python -m swap_correction.ml.stability aggregate \
       --input-dir stability_analysis_v4 \
       --output-dir stability_analysis_v4/aggregated
   ```

**Output**: 
- `stability_report.md`: Comprehensive stability report
- `stability_metrics.json`: Aggregated metrics
- Per-iteration results in `iteration_XXX/` directories

### 2. Learning Curve Analysis

**Purpose**: Determine if more training data improves performance.

**Steps**:

1. **Using the CLI script**:
   ```bash
   python analyze_split_ratios.py \
       --analysis-type learning_curve \
       --iteration 13 \
       --model-type all \
       --feature-version v4 \
       --output-dir learning_curve_analysis_v4_iter013
   ```

2. **Using the evaluation module**:
   ```python
   from swap_correction.ml.evaluation.learning_curves import generate_learning_curves
   
   results = generate_learning_curves(
       ml_data_dir='ml_data',
       output_dir='learning_curves',
       train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   )
   ```

**Output**:
- `learning_curve_results.json`: Detailed results
- `learning_curve_summary.md`: Summary report
- `learning_curves.png`: Visualization plots

### 3. Threshold Optimization

**Purpose**: Find optimal classification threshold for a metric.

**Steps**:

1. **Using the CLI**:
   ```bash
   python -m swap_correction.ml.evaluation optimize-threshold \
       --model-type level1 \
       --metric pct_clean_post \
       --split val \
       --n-thresholds 100 \
       --output-dir threshold_analysis
   ```

2. **Using the Python API**:
   ```python
   from swap_correction.ml.evaluation.find_optimal_threshold import find_optimal_threshold_on_dataset
   
   optimal_threshold, metrics = find_optimal_threshold_on_dataset(
       model_type='level1',
       data_dir='ml_data',
       metric='pct_frames_clean_post',
       split='val',
       n_thresholds=100
   )
   ```

**Output**:
- `optimal_threshold.json`: Optimal threshold and metrics
- `threshold_analysis.csv`: Metrics at all thresholds
- `threshold_plots.png`: Visualization of metrics across thresholds
- `threshold_report.md`: Detailed report

### 4. Model Evaluation

**Purpose**: Evaluate trained models on new datasets.

**Steps**:

1. **Using the CLI**:
   ```bash
   python -m swap_correction.ml.evaluation evaluate \
       --data-dir /path/to/test/trials \
       --model-type level1 \
       --ground-truth-level level2 \
       --output-dir evaluation_results
   ```

2. **Using the Python API**:
   ```python
   from swap_correction.ml.evaluation.evaluate_on_dataset import evaluate_model_on_dataset
   
   results = evaluate_model_on_dataset(
       data_dir='/path/to/test/trials',
       model_type='level1',
       ground_truth_level='level2',
       output_dir='evaluation_results'
   )
   ```

**Output**:
- `evaluation_level1_test_data.json`: Detailed results
- `evaluation_report_level1_test_data.md`: Markdown report
- Per-trial metrics and summary statistics

### 5. Feature Comparison

**Purpose**: Compare different feature sets.

**Steps**:

1. **Train models with different feature sets** (using feature version patching)
2. **Run stability analysis for each feature version**
3. **Generate comparison report**:
   ```bash
   python generate_features_comparison_report.py
   ```

**Output**:
- `features_comparison_report.md`: Comprehensive comparison
- Performance metrics for each feature version
- Feature importance comparisons

---

## Advanced Usage

### Custom Feature Extraction

To test a new feature set:

1. Create new feature file: `swap_correction/ml/features/features_v5.py`
2. Patch the features module:
   ```python
   import swap_correction.ml.features.features_v5 as features_v5_module
   import swap_correction.ml.features as features_module
   
   features_module.extract_all_frame_features_optimized = \
       features_v5_module.extract_all_frame_features_optimized
   ```
3. Run training/evaluation as normal

### Custom Model Evaluation

To evaluate with custom metrics:

```python
from swap_correction.ml.api import BatchProcessor

processor = BatchProcessor(model_type='level1')
results = processor.evaluate_on_dataset(
    data_dir='/path/to/trials',
    ground_truth_level='level2'
)

# Access custom metrics
for trial_result in results['trials']:
    # Add custom calculations
    pass
```

---

## Best Practices

### 1. Stability Analysis

- **Sample Size**: Use 80-100 samples for stable results
- **Iterations**: Run at least 6-10 iterations for statistical significance
- **Filter Identical**: Always filter out trials where level1 == level2 (no swaps)

### 2. Learning Curve Analysis

- **Fixed Splits**: Use fixed train/val/test splits, vary only training data size
- **Multiple Iterations**: Run on multiple stability analysis iterations for robustness
- **Monitor Overfitting**: Watch for declining performance with more data

### 3. Threshold Optimization

- **Validation Set**: Optimize on validation set, not test set
- **Primary Metric**: Use % Frames Clean Post as primary metric for Level1 models
- **Multiple Metrics**: Consider all metrics, not just one

### 4. Model Evaluation

- **Ground Truth**: Always use level2.csv as ground truth
- **Comprehensive Metrics**: Report all metrics (precision, recall, F1, sensitivity, specificity, % swaps resolved, % frames clean)
- **Per-Trial Analysis**: Include per-trial results for detailed analysis

---

## Troubleshooting

### Common Issues

1. **Feature Mismatch**: Ensure feature extraction matches model's expected features
   - Check `feature_names.pkl` in model directory
   - Verify feature count matches

2. **Missing Ground Truth**: Some trials may not have level2.csv
   - Filter these out before evaluation
   - Use `find_valid_trial_directories` with appropriate filters

3. **Memory Issues**: Large datasets may cause memory problems
   - Process in batches
   - Use incremental feature extraction

4. **Slow Performance**: Feature extraction can be slow
   - Ensure using optimized version (`extract_all_frame_features_optimized`)
   - Consider using Gaussian filtering to reduce noise

---

## References

- **ML Usage Guide**: `docs/ML_USAGE_GUIDE.md`
- **ML API Reference**: `docs/ML_API_REFERENCE.md`
- **ML Training Guide**: `docs/ML_TRAINING_GUIDE.md`
- **Experimental History**: `docs/EXPERIMENTAL_HISTORY.md`

---

**End of Testing Infrastructure Documentation**


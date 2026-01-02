# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 008
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5346

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9369 |
| Recall | 0.8722 |
| F1-Score | 0.9034 |
| Sensitivity | 0.8722 |
| Specificity | 0.9631 |
| % Swaps Resolved | 87.22% |
| % Frames Clean Pre | 61.38% |
| % Frames Clean Post | 92.80% |

## Confusion Matrix

- **TP**: 36376
- **FP**: 2448
- **FN**: 5332
- **TN**: 63844

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5346) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 92.75% | 92.80% | +0.05% |
| F1-Score | 0.9037 | 0.9034 | -0.0003 |
| Precision | 0.9274 | 0.9369 | +0.0096 |
| Recall | 0.8812 | 0.8722 | -0.0091 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5346)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
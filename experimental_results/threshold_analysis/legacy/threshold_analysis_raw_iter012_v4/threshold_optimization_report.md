# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 012
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4852

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9489 |
| Recall | 0.8891 |
| F1-Score | 0.9180 |
| Sensitivity | 0.8891 |
| Specificity | 0.9604 |
| % Swaps Resolved | 88.91% |
| % Frames Clean Pre | 54.70% |
| % Frames Clean Post | 92.81% |

## Confusion Matrix

- **TP**: 43501
- **FP**: 2342
- **FN**: 5425
- **TN**: 56732

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4852) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 92.80% | 92.81% | +0.01% |
| F1-Score | 0.9177 | 0.9180 | +0.0004 |
| Precision | 0.9513 | 0.9489 | -0.0024 |
| Recall | 0.8863 | 0.8891 | +0.0028 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4852)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
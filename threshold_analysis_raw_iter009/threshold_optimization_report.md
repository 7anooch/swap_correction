# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 009
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.6138

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9776 |
| Recall | 0.9592 |
| F1-Score | 0.9683 |
| Sensitivity | 0.9592 |
| Specificity | 0.9808 |
| % Swaps Resolved | 95.92% |
| % Frames Clean Pre | 53.35% |
| % Frames Clean Post | 97.07% |

## Confusion Matrix

- **TP**: 48322
- **FP**: 1108
- **FN**: 2055
- **TN**: 56515

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.6138) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.00% | 97.07% | +0.07% |
| F1-Score | 0.9680 | 0.9683 | +0.0003 |
| Precision | 0.9647 | 0.9776 | +0.0129 |
| Recall | 0.9713 | 0.9592 | -0.0121 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.6138)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
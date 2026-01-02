# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 007
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5544

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9773 |
| Recall | 0.9694 |
| F1-Score | 0.9733 |
| Sensitivity | 0.9694 |
| Specificity | 0.9818 |
| % Swaps Resolved | 96.94% |
| % Frames Clean Pre | 55.27% |
| % Frames Clean Post | 97.62% |

## Confusion Matrix

- **TP**: 46825
- **FP**: 1089
- **FN**: 1479
- **TN**: 58607

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5544) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.57% | 97.62% | +0.05% |
| F1-Score | 0.9729 | 0.9733 | +0.0004 |
| Precision | 0.9719 | 0.9773 | +0.0054 |
| Recall | 0.9739 | 0.9694 | -0.0046 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5544)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
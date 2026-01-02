# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 009
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5544

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9719 |
| Recall | 0.9664 |
| F1-Score | 0.9691 |
| Sensitivity | 0.9664 |
| Specificity | 0.9756 |
| % Swaps Resolved | 96.64% |
| % Frames Clean Pre | 53.35% |
| % Frames Clean Post | 97.13% |

## Confusion Matrix

- **TP**: 48682
- **FP**: 1406
- **FN**: 1695
- **TN**: 56217

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5544) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.07% | 97.13% | +0.06% |
| F1-Score | 0.9687 | 0.9691 | +0.0004 |
| Precision | 0.9652 | 0.9719 | +0.0067 |
| Recall | 0.9722 | 0.9664 | -0.0059 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5544)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
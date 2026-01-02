# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 011
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4456

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9631 |
| Recall | 0.9539 |
| F1-Score | 0.9585 |
| Sensitivity | 0.9539 |
| Specificity | 0.9714 |
| % Swaps Resolved | 95.39% |
| % Frames Clean Pre | 56.06% |
| % Frames Clean Post | 96.37% |

## Confusion Matrix

- **TP**: 45267
- **FP**: 1734
- **FN**: 2188
- **TN**: 58811

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4456) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 96.35% | 96.37% | +0.01% |
| F1-Score | 0.9581 | 0.9585 | +0.0004 |
| Precision | 0.9687 | 0.9631 | -0.0056 |
| Recall | 0.9476 | 0.9539 | +0.0063 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4456)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
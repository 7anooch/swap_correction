# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 012
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5643

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9062 |
| Recall | 0.8685 |
| F1-Score | 0.8870 |
| Sensitivity | 0.8685 |
| Specificity | 0.9699 |
| % Swaps Resolved | 86.85% |
| % Frames Clean Pre | 74.94% |
| % Frames Clean Post | 94.45% |

## Confusion Matrix

- **TP**: 23509
- **FP**: 2432
- **FN**: 3560
- **TN**: 78499

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5643) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 94.35% | 94.45% | +0.10% |
| F1-Score | 0.8864 | 0.8870 | +0.0006 |
| Precision | 0.8938 | 0.9062 | +0.0124 |
| Recall | 0.8790 | 0.8685 | -0.0106 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5643)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 007
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.6138

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9582 |
| Recall | 0.9619 |
| F1-Score | 0.9601 |
| Sensitivity | 0.9619 |
| Specificity | 0.9770 |
| % Swaps Resolved | 96.19% |
| % Frames Clean Pre | 64.59% |
| % Frames Clean Post | 97.17% |

## Confusion Matrix

- **TP**: 36783
- **FP**: 1603
- **FN**: 1458
- **TN**: 68156

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.6138) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.05% | 97.17% | +0.11% |
| F1-Score | 0.9590 | 0.9601 | +0.0011 |
| Precision | 0.9454 | 0.9582 | +0.0128 |
| Recall | 0.9729 | 0.9619 | -0.0110 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.6138)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 009
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.6336

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9793 |
| Recall | 0.9602 |
| F1-Score | 0.9697 |
| Sensitivity | 0.9602 |
| Specificity | 0.9822 |
| % Swaps Resolved | 96.02% |
| % Frames Clean Pre | 53.35% |
| % Frames Clean Post | 97.20% |

## Confusion Matrix

- **TP**: 48374
- **FP**: 1025
- **FN**: 2003
- **TN**: 56598

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.6336) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.11% | 97.20% | +0.09% |
| F1-Score | 0.9691 | 0.9697 | +0.0005 |
| Precision | 0.9652 | 0.9793 | +0.0140 |
| Recall | 0.9730 | 0.9602 | -0.0128 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.6336)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
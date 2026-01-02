# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 011
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4852

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9708 |
| Recall | 0.9545 |
| F1-Score | 0.9626 |
| Sensitivity | 0.9545 |
| Specificity | 0.9912 |
| % Swaps Resolved | 95.45% |
| % Frames Clean Pre | 76.44% |
| % Frames Clean Post | 98.25% |

## Confusion Matrix

- **TP**: 24286
- **FP**: 730
- **FN**: 1159
- **TN**: 81825

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4852) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 98.24% | 98.25% | +0.01% |
| F1-Score | 0.9623 | 0.9626 | +0.0002 |
| Precision | 0.9717 | 0.9708 | -0.0009 |
| Recall | 0.9531 | 0.9545 | +0.0013 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4852)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
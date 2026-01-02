# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 008
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4852

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9257 |
| Recall | 0.8902 |
| F1-Score | 0.9076 |
| Sensitivity | 0.8902 |
| Specificity | 0.9550 |
| % Swaps Resolved | 89.02% |
| % Frames Clean Pre | 61.38% |
| % Frames Clean Post | 93.00% |

## Confusion Matrix

- **TP**: 37129
- **FP**: 2982
- **FN**: 4579
- **TN**: 63310

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4852) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 92.98% | 93.00% | +0.02% |
| F1-Score | 0.9071 | 0.9076 | +0.0005 |
| Precision | 0.9279 | 0.9257 | -0.0023 |
| Recall | 0.8872 | 0.8902 | +0.0030 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4852)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 008
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.9108

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9397 |
| Recall | 0.9223 |
| F1-Score | 0.9309 |
| Sensitivity | 0.9223 |
| Specificity | 0.9798 |
| % Swaps Resolved | 92.23% |
| % Frames Clean Pre | 74.60% |
| % Frames Clean Post | 96.52% |

## Confusion Matrix

- **TP**: 25298
- **FP**: 1624
- **FN**: 2130
- **TN**: 78948

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.9108) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 95.17% | 96.52% | +1.35% |
| F1-Score | 0.9110 | 0.9309 | +0.0199 |
| Precision | 0.8562 | 0.9397 | +0.0835 |
| Recall | 0.9732 | 0.9223 | -0.0509 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.9108)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
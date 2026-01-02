# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 008
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.9108

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9488 |
| Recall | 0.9207 |
| F1-Score | 0.9345 |
| Sensitivity | 0.9207 |
| Specificity | 0.9831 |
| % Swaps Resolved | 92.07% |
| % Frames Clean Pre | 74.60% |
| % Frames Clean Post | 96.72% |

## Confusion Matrix

- **TP**: 25253
- **FP**: 1364
- **FN**: 2175
- **TN**: 79208

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.9108) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 94.63% | 96.72% | +2.09% |
| F1-Score | 0.9020 | 0.9345 | +0.0325 |
| Precision | 0.8408 | 0.9488 | +0.1079 |
| Recall | 0.9728 | 0.9207 | -0.0521 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.9108)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
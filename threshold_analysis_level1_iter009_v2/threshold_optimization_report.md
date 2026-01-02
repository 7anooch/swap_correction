# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 009
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.3862

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9745 |
| Recall | 0.9790 |
| F1-Score | 0.9768 |
| Sensitivity | 0.9790 |
| Specificity | 0.9862 |
| % Swaps Resolved | 97.90% |
| % Frames Clean Pre | 64.93% |
| % Frames Clean Post | 98.37% |

## Confusion Matrix

- **TP**: 37078
- **FP**: 970
- **FN**: 795
- **TN**: 69157

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.3862) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 98.34% | 98.37% | +0.03% |
| F1-Score | 0.9762 | 0.9768 | +0.0006 |
| Precision | 0.9806 | 0.9745 | -0.0060 |
| Recall | 0.9718 | 0.9790 | +0.0072 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.3862)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
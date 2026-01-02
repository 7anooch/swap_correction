# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 011
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4357

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9675 |
| Recall | 0.9549 |
| F1-Score | 0.9612 |
| Sensitivity | 0.9549 |
| Specificity | 0.9901 |
| % Swaps Resolved | 95.49% |
| % Frames Clean Pre | 76.44% |
| % Frames Clean Post | 98.18% |

## Confusion Matrix

- **TP**: 24298
- **FP**: 815
- **FN**: 1147
- **TN**: 81740

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4357) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 98.13% | 98.18% | +0.05% |
| F1-Score | 0.9599 | 0.9612 | +0.0013 |
| Precision | 0.9720 | 0.9675 | -0.0044 |
| Recall | 0.9480 | 0.9549 | +0.0069 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4357)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
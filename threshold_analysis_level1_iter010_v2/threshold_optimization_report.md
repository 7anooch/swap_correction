# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 010
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.7722

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9741 |
| Recall | 0.9539 |
| F1-Score | 0.9639 |
| Sensitivity | 0.9539 |
| Specificity | 0.9906 |
| % Swaps Resolved | 95.39% |
| % Frames Clean Pre | 73.01% |
| % Frames Clean Post | 98.07% |

## Confusion Matrix

- **TP**: 27804
- **FP**: 740
- **FN**: 1343
- **TN**: 78113

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.7722) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.97% | 98.07% | +0.10% |
| F1-Score | 0.9629 | 0.9639 | +0.0010 |
| Precision | 0.9509 | 0.9741 | +0.0232 |
| Recall | 0.9751 | 0.9539 | -0.0212 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.7722)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
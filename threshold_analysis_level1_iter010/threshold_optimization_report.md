# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 010
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.7227

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9689 |
| Recall | 0.9589 |
| F1-Score | 0.9639 |
| Sensitivity | 0.9589 |
| Specificity | 0.9886 |
| % Swaps Resolved | 95.89% |
| % Frames Clean Pre | 73.01% |
| % Frames Clean Post | 98.06% |

## Confusion Matrix

- **TP**: 27950
- **FP**: 898
- **FN**: 1197
- **TN**: 77955

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.7227) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.99% | 98.06% | +0.08% |
| F1-Score | 0.9632 | 0.9639 | +0.0007 |
| Precision | 0.9507 | 0.9689 | +0.0182 |
| Recall | 0.9759 | 0.9589 | -0.0170 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.7227)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
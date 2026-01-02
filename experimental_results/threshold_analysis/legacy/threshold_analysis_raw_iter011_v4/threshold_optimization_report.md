# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 011
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4852

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9666 |
| Recall | 0.9571 |
| F1-Score | 0.9618 |
| Sensitivity | 0.9571 |
| Specificity | 0.9741 |
| % Swaps Resolved | 95.71% |
| % Frames Clean Pre | 56.06% |
| % Frames Clean Post | 96.66% |

## Confusion Matrix

- **TP**: 45417
- **FP**: 1569
- **FN**: 2038
- **TN**: 58976

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4852) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 96.66% | 96.66% | +0.00% |
| F1-Score | 0.9618 | 0.9618 | +0.0001 |
| Precision | 0.9679 | 0.9666 | -0.0013 |
| Recall | 0.9557 | 0.9571 | +0.0014 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4852)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
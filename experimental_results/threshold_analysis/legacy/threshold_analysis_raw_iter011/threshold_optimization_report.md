# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 011
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4951

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9682 |
| Recall | 0.9485 |
| F1-Score | 0.9582 |
| Sensitivity | 0.9485 |
| Specificity | 0.9756 |
| % Swaps Resolved | 94.85% |
| % Frames Clean Pre | 56.06% |
| % Frames Clean Post | 96.37% |

## Confusion Matrix

- **TP**: 45009
- **FP**: 1479
- **FN**: 2446
- **TN**: 59066

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4951) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 96.37% | 96.37% | +0.00% |
| F1-Score | 0.9582 | 0.9582 | +0.0000 |
| Precision | 0.9682 | 0.9682 | +0.0000 |
| Recall | 0.9485 | 0.9485 | +0.0000 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4951)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 010
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5049

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9679 |
| Recall | 0.9438 |
| F1-Score | 0.9557 |
| Sensitivity | 0.9438 |
| Specificity | 0.9762 |
| % Swaps Resolved | 94.38% |
| % Frames Clean Pre | 56.87% |
| % Frames Clean Post | 96.23% |

## Confusion Matrix

- **TP**: 43962
- **FP**: 1460
- **FN**: 2616
- **TN**: 59962

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5049) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 96.20% | 96.23% | +0.03% |
| F1-Score | 0.9554 | 0.9557 | +0.0002 |
| Precision | 0.9661 | 0.9679 | +0.0017 |
| Recall | 0.9450 | 0.9438 | -0.0012 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5049)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
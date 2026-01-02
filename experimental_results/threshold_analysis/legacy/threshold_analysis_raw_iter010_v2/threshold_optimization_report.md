# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 010
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5049

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9670 |
| Recall | 0.9430 |
| F1-Score | 0.9548 |
| Sensitivity | 0.9430 |
| Specificity | 0.9756 |
| % Swaps Resolved | 94.30% |
| % Frames Clean Pre | 56.87% |
| % Frames Clean Post | 96.15% |

## Confusion Matrix

- **TP**: 43923
- **FP**: 1499
- **FN**: 2655
- **TN**: 59923

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5049) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 96.14% | 96.15% | +0.02% |
| F1-Score | 0.9547 | 0.9548 | +0.0001 |
| Precision | 0.9651 | 0.9670 | +0.0019 |
| Recall | 0.9446 | 0.9430 | -0.0016 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5049)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
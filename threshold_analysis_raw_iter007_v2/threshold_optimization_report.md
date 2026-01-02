# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 007
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5346

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9747 |
| Recall | 0.9715 |
| F1-Score | 0.9731 |
| Sensitivity | 0.9715 |
| Specificity | 0.9796 |
| % Swaps Resolved | 97.15% |
| % Frames Clean Pre | 55.27% |
| % Frames Clean Post | 97.60% |

## Confusion Matrix

- **TP**: 46927
- **FP**: 1218
- **FN**: 1377
- **TN**: 58478

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5346) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.54% | 97.60% | +0.05% |
| F1-Score | 0.9726 | 0.9731 | +0.0005 |
| Precision | 0.9710 | 0.9747 | +0.0037 |
| Recall | 0.9741 | 0.9715 | -0.0026 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5346)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
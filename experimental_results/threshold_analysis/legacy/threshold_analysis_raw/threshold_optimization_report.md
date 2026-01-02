# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 007
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5445

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9748 |
| Recall | 0.9695 |
| F1-Score | 0.9721 |
| Sensitivity | 0.9695 |
| Specificity | 0.9797 |
| % Swaps Resolved | 96.95% |
| % Frames Clean Pre | 55.27% |
| % Frames Clean Post | 97.51% |

## Confusion Matrix

- **TP**: 46830
- **FP**: 1213
- **FN**: 1474
- **TN**: 58483

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5445) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.47% | 97.51% | +0.04% |
| F1-Score | 0.9718 | 0.9721 | +0.0003 |
| Precision | 0.9706 | 0.9748 | +0.0042 |
| Recall | 0.9730 | 0.9695 | -0.0035 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5445)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
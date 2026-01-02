# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 007
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.6435

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9629 |
| Recall | 0.9592 |
| F1-Score | 0.9610 |
| Sensitivity | 0.9592 |
| Specificity | 0.9797 |
| % Swaps Resolved | 95.92% |
| % Frames Clean Pre | 64.59% |
| % Frames Clean Post | 97.24% |

## Confusion Matrix

- **TP**: 36679
- **FP**: 1415
- **FN**: 1562
- **TN**: 68344

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.6435) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 97.14% | 97.24% | +0.10% |
| F1-Score | 0.9602 | 0.9610 | +0.0008 |
| Precision | 0.9466 | 0.9629 | +0.0162 |
| Recall | 0.9742 | 0.9592 | -0.0150 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.6435)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
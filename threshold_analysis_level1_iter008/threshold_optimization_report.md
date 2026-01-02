# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 008
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.9306

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9519 |
| Recall | 0.9071 |
| F1-Score | 0.9290 |
| Sensitivity | 0.9071 |
| Specificity | 0.9844 |
| % Swaps Resolved | 90.71% |
| % Frames Clean Pre | 74.60% |
| % Frames Clean Post | 96.48% |

## Confusion Matrix

- **TP**: 24881
- **FP**: 1258
- **FN**: 2547
- **TN**: 79314

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.9306) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 94.67% | 96.48% | +1.81% |
| F1-Score | 0.9026 | 0.9290 | +0.0264 |
| Precision | 0.8426 | 0.9519 | +0.1093 |
| Recall | 0.9717 | 0.9071 | -0.0646 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.9306)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
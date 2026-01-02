# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 007
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.6633

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9610 |
| Recall | 0.9547 |
| F1-Score | 0.9579 |
| Sensitivity | 0.9547 |
| Specificity | 0.9788 |
| % Swaps Resolved | 95.47% |
| % Frames Clean Pre | 64.59% |
| % Frames Clean Post | 97.03% |

## Confusion Matrix

- **TP**: 36510
- **FP**: 1480
- **FN**: 1731
- **TN**: 68279

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.6633) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 96.84% | 97.03% | +0.19% |
| F1-Score | 0.9561 | 0.9579 | +0.0018 |
| Precision | 0.9416 | 0.9610 | +0.0195 |
| Recall | 0.9711 | 0.9547 | -0.0163 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.6633)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
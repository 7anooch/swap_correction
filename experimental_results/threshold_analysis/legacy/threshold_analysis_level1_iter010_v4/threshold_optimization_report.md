# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 010
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5643

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9605 |
| Recall | 0.9717 |
| F1-Score | 0.9661 |
| Sensitivity | 0.9717 |
| Specificity | 0.9852 |
| % Swaps Resolved | 97.17% |
| % Frames Clean Pre | 73.01% |
| % Frames Clean Post | 98.16% |

## Confusion Matrix

- **TP**: 28323
- **FP**: 1165
- **FN**: 824
- **TN**: 77688

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5643) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 98.15% | 98.16% | +0.01% |
| F1-Score | 0.9660 | 0.9661 | +0.0000 |
| Precision | 0.9550 | 0.9605 | +0.0055 |
| Recall | 0.9773 | 0.9717 | -0.0056 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5643)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
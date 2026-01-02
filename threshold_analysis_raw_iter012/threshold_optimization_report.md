# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 012
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4456

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9382 |
| Recall | 0.9025 |
| F1-Score | 0.9200 |
| Sensitivity | 0.9025 |
| Specificity | 0.9507 |
| % Swaps Resolved | 90.25% |
| % Frames Clean Pre | 54.70% |
| % Frames Clean Post | 92.89% |

## Confusion Matrix

- **TP**: 44155
- **FP**: 2911
- **FN**: 4771
- **TN**: 56163

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4456) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 92.86% | 92.89% | +0.03% |
| F1-Score | 0.9183 | 0.9200 | +0.0016 |
| Precision | 0.9525 | 0.9382 | -0.0143 |
| Recall | 0.8866 | 0.9025 | +0.0159 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4456)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
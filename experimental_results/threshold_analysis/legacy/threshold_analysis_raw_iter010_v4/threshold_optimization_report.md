# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 010
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.5247

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9709 |
| Recall | 0.9394 |
| F1-Score | 0.9549 |
| Sensitivity | 0.9394 |
| Specificity | 0.9786 |
| % Swaps Resolved | 93.94% |
| % Frames Clean Pre | 56.87% |
| % Frames Clean Post | 96.17% |

## Confusion Matrix

- **TP**: 43756
- **FP**: 1313
- **FN**: 2822
- **TN**: 60109

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.5247) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 96.15% | 96.17% | +0.02% |
| F1-Score | 0.9549 | 0.9549 | -0.0000 |
| Precision | 0.9665 | 0.9709 | +0.0044 |
| Recall | 0.9436 | 0.9394 | -0.0042 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.5247)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
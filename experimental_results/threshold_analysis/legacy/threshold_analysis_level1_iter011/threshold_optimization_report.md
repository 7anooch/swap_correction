# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 011
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4654

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9690 |
| Recall | 0.9565 |
| F1-Score | 0.9627 |
| Sensitivity | 0.9565 |
| Specificity | 0.9906 |
| % Swaps Resolved | 95.65% |
| % Frames Clean Pre | 76.44% |
| % Frames Clean Post | 98.25% |

## Confusion Matrix

- **TP**: 24338
- **FP**: 779
- **FN**: 1107
- **TN**: 81776

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4654) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 98.23% | 98.25% | +0.02% |
| F1-Score | 0.9621 | 0.9627 | +0.0006 |
| Precision | 0.9717 | 0.9690 | -0.0027 |
| Recall | 0.9528 | 0.9565 | +0.0037 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4654)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
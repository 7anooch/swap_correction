# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 009
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4357

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9789 |
| Recall | 0.9782 |
| F1-Score | 0.9786 |
| Sensitivity | 0.9782 |
| Specificity | 0.9886 |
| % Swaps Resolved | 97.82% |
| % Frames Clean Pre | 64.93% |
| % Frames Clean Post | 98.50% |

## Confusion Matrix

- **TP**: 37047
- **FP**: 797
- **FN**: 826
- **TN**: 69330

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4357) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 98.48% | 98.50% | +0.02% |
| F1-Score | 0.9782 | 0.9786 | +0.0004 |
| Precision | 0.9816 | 0.9789 | -0.0027 |
| Recall | 0.9748 | 0.9782 | +0.0034 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4357)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
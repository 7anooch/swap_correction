# Threshold Optimization Report

**Model Type**: level1
**Iteration**: 009
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4159

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9752 |
| Recall | 0.9797 |
| F1-Score | 0.9774 |
| Sensitivity | 0.9797 |
| Specificity | 0.9866 |
| % Swaps Resolved | 97.97% |
| % Frames Clean Pre | 64.93% |
| % Frames Clean Post | 98.41% |

## Confusion Matrix

- **TP**: 37104
- **FP**: 943
- **FN**: 769
- **TN**: 69184

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4159) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 98.37% | 98.41% | +0.05% |
| F1-Score | 0.9767 | 0.9774 | +0.0008 |
| Precision | 0.9799 | 0.9752 | -0.0047 |
| Recall | 0.9735 | 0.9797 | +0.0062 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4159)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
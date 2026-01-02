# Threshold Optimization Report

**Model Type**: raw
**Iteration**: 012
**Metric Optimized**: pct_clean_post
**Data Split**: val
**Optimal Threshold**: 0.4258

## Performance at Optimal Threshold

| Metric | Value |
|--------|-------|
| Precision | 0.9305 |
| Recall | 0.9096 |
| F1-Score | 0.9199 |
| Sensitivity | 0.9096 |
| Specificity | 0.9437 |
| % Swaps Resolved | 90.96% |
| % Frames Clean Pre | 54.70% |
| % Frames Clean Post | 92.83% |

## Confusion Matrix

- **TP**: 44505
- **FP**: 3325
- **FN**: 4421
- **TN**: 55749

## Comparison with Default Threshold (0.5)

| Metric | Default (0.5) | Optimal (0.4258) | Improvement |
|--------|---------------|-----------------------------------|-------------|
| % Frames Clean Post | 92.75% | 92.83% | +0.08% |
| F1-Score | 0.9171 | 0.9199 | +0.0028 |
| Precision | 0.9509 | 0.9305 | -0.0204 |
| Recall | 0.8857 | 0.9096 | +0.0240 |

## Usage

```python
from swap_correction.ml.api import SwapPredictor

predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30, threshold=0.4258)
```

## Visualization

See `threshold_optimization_plots.png` for comprehensive plots showing all metrics across thresholds.
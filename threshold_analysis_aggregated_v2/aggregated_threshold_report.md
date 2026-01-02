# Threshold Optimization: Aggregated Results (Features V2)

This report aggregates threshold optimization results across 6 iterations
(007-012) from the stability analysis.

## Summary Statistics

### Level1 Model

**Optimal Threshold**: 0.6336 ± 0.1896
- Range: 0.3862 - 0.9108
- Median: 0.6138

**% Frames Clean Post**: 97.72% ± 0.65%
- Range: 96.72% - 98.37%

**Precision**: 0.9653 ± 0.0102
**Recall**: 0.9540 ± 0.0190
**F1-Score**: 0.9596 ± 0.0138
**% Swaps Resolved**: 95.40% ± 1.90%

### Raw Model

**Optimal Threshold**: 0.5000 ± 0.0481
- Range: 0.4258 - 0.5544
- Median: 0.5198

**% Frames Clean Post**: 95.48% ± 1.94%
- Range: 92.80% - 97.60%

**Precision**: 0.9574 ± 0.0172
**Recall**: 0.9361 ± 0.0349
**F1-Score**: 0.9465 ± 0.0258
**% Swaps Resolved**: 93.61% ± 3.49%

## Per-Iteration Results

### Level1 Model

| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |
|-----------|-------------------|--------------|-----------|--------|----|------------------|
| 007 | 0.6138 | 97.17% | 0.9582 | 0.9619 | 0.9601 | 96.19% |
| 008 | 0.9108 | 96.72% | 0.9488 | 0.9207 | 0.9345 | 92.07% |
| 009 | 0.3862 | 98.37% | 0.9745 | 0.9790 | 0.9768 | 97.90% |
| 010 | 0.7722 | 98.07% | 0.9741 | 0.9539 | 0.9639 | 95.39% |
| 011 | 0.4852 | 98.25% | 0.9708 | 0.9545 | 0.9626 | 95.45% |

### Raw Model

| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |
|-----------|-------------------|--------------|-----------|--------|----|------------------|
| 007 | 0.5346 | 97.60% | 0.9747 | 0.9715 | 0.9731 | 97.15% |
| 008 | 0.5346 | 92.80% | 0.9369 | 0.8722 | 0.9034 | 87.22% |
| 009 | 0.5544 | 97.13% | 0.9719 | 0.9664 | 0.9691 | 96.64% |
| 010 | 0.5049 | 96.15% | 0.9670 | 0.9430 | 0.9548 | 94.30% |
| 011 | 0.4456 | 96.37% | 0.9631 | 0.9539 | 0.9585 | 95.39% |
| 012 | 0.4258 | 92.83% | 0.9305 | 0.9096 | 0.9199 | 90.96% |

## Recommendations

### Level1 Model
- **Recommended Threshold**: 0.6138 (median across iterations)
- **Expected Performance**: 97.72% ± 0.65% frames clean post

### Raw Model
- **Recommended Threshold**: 0.5198 (median across iterations)
- **Expected Performance**: 95.48% ± 1.94% frames clean post

## Usage

```python
from swap_correction.ml.api import SwapPredictor

# Level1 model
predictor_level1 = SwapPredictor(model_type='level1')
predictions = predictor_level1.predict(trial_data, fps=30, threshold=0.6138)

# Raw model
predictor_raw = SwapPredictor(model_type='raw')
predictions = predictor_raw.predict(trial_data, fps=30, threshold=0.5198)
```
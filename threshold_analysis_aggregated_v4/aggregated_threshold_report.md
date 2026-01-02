# Threshold Optimization: Aggregated Results (Features V4)

This report aggregates threshold optimization results across 6 iterations
(007-012) from the stability analysis.

## Summary Statistics

### Level1 Model

**Optimal Threshold**: 0.5924 ± 0.1606
- Range: 0.4357 - 0.9108
- Median: 0.5643

**% Frames Clean Post**: 97.18% ± 1.39%
- Range: 94.45% - 98.50%

**Precision**: 0.9526 ± 0.0238
**Recall**: 0.9425 ± 0.0375
**F1-Score**: 0.9475 ± 0.0306
**% Swaps Resolved**: 94.25% ± 3.75%

### Raw Model

**Optimal Threshold**: 0.5280 ± 0.0538
- Range: 0.4852 - 0.6336
- Median: 0.5049

**% Frames Clean Post**: 95.58% ± 1.94%
- Range: 92.81% - 97.62%

**Precision**: 0.9614 ± 0.0188
**Recall**: 0.9342 ± 0.0327
**F1-Score**: 0.9475 ± 0.0254
**% Swaps Resolved**: 93.42% ± 3.27%

## Per-Iteration Results

### Level1 Model

| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |
|-----------|-------------------|--------------|-----------|--------|----|------------------|
| 007 | 0.6435 | 97.24% | 0.9629 | 0.9592 | 0.9610 | 95.92% |
| 008 | 0.9108 | 96.52% | 0.9397 | 0.9223 | 0.9309 | 92.23% |
| 009 | 0.4357 | 98.50% | 0.9789 | 0.9782 | 0.9786 | 97.82% |
| 010 | 0.5643 | 98.16% | 0.9605 | 0.9717 | 0.9661 | 97.17% |
| 011 | 0.4357 | 98.18% | 0.9675 | 0.9549 | 0.9612 | 95.49% |
| 012 | 0.5643 | 94.45% | 0.9062 | 0.8685 | 0.8870 | 86.85% |

### Raw Model

| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |
|-----------|-------------------|--------------|-----------|--------|----|------------------|
| 007 | 0.5544 | 97.62% | 0.9773 | 0.9694 | 0.9733 | 96.94% |
| 008 | 0.4852 | 93.00% | 0.9257 | 0.8902 | 0.9076 | 89.02% |
| 009 | 0.6336 | 97.20% | 0.9793 | 0.9602 | 0.9697 | 96.02% |
| 010 | 0.5247 | 96.17% | 0.9709 | 0.9394 | 0.9549 | 93.94% |
| 011 | 0.4852 | 96.66% | 0.9666 | 0.9571 | 0.9618 | 95.71% |
| 012 | 0.4852 | 92.81% | 0.9489 | 0.8891 | 0.9180 | 88.91% |

## Recommendations

### Level1 Model
- **Recommended Threshold**: 0.5643 (median across iterations)
- **Expected Performance**: 97.18% ± 1.39% frames clean post

### Raw Model
- **Recommended Threshold**: 0.5049 (median across iterations)
- **Expected Performance**: 95.58% ± 1.94% frames clean post

## Usage

```python
from swap_correction.ml.api import SwapPredictor

# Level1 model
predictor_level1 = SwapPredictor(model_type='level1')
predictions = predictor_level1.predict(trial_data, fps=30, threshold=0.5643)

# Raw model
predictor_raw = SwapPredictor(model_type='raw')
predictions = predictor_raw.predict(trial_data, fps=30, threshold=0.5049)
```
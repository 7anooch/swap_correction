# Threshold Optimization: Aggregated Results

This report aggregates threshold optimization results across 6 iterations
(007-012) from the stability analysis.

## Summary Statistics

### Level1 Model

**Optimal Threshold**: 0.6396 ± 0.1858
- Range: 0.4159 - 0.9306
- Median: 0.6633

**% Frames Clean Post**: 97.65% ± 0.76%
- Range: 96.48% - 98.41%

**Precision**: 0.9652 ± 0.0080
**Recall**: 0.9514 ± 0.0239
**F1-Score**: 0.9582 ± 0.0160
**% Swaps Resolved**: 95.14% ± 2.39%

### Raw Model

**Optimal Threshold**: 0.5208 ± 0.0562
- Range: 0.4456 - 0.6138
- Median: 0.5049

**% Frames Clean Post**: 96.01% ± 1.63%
- Range: 92.89% - 97.51%

**Precision**: 0.9653 ± 0.0141
**Recall**: 0.9447 ± 0.0229
**F1-Score**: 0.9549 ± 0.0185
**% Swaps Resolved**: 94.47% ± 2.29%

## Per-Iteration Results

### Level1 Model

| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |
|-----------|-------------------|--------------|-----------|--------|----|------------------|
| 007 | 0.6633 | 97.03% | 0.9610 | 0.9547 | 0.9579 | 95.47% |
| 008 | 0.9306 | 96.48% | 0.9519 | 0.9071 | 0.9290 | 90.71% |
| 009 | 0.4159 | 98.41% | 0.9752 | 0.9797 | 0.9774 | 97.97% |
| 010 | 0.7227 | 98.06% | 0.9689 | 0.9589 | 0.9639 | 95.89% |
| 011 | 0.4654 | 98.25% | 0.9690 | 0.9565 | 0.9627 | 95.65% |

### Raw Model

| Iteration | Optimal Threshold | % Clean Post | Precision | Recall | F1 | % Swaps Resolved |
|-----------|-------------------|--------------|-----------|--------|----|------------------|
| 007 | 0.5445 | 97.51% | 0.9748 | 0.9695 | 0.9721 | 96.95% |
| 009 | 0.6138 | 97.07% | 0.9776 | 0.9592 | 0.9683 | 95.92% |
| 010 | 0.5049 | 96.23% | 0.9679 | 0.9438 | 0.9557 | 94.38% |
| 011 | 0.4951 | 96.37% | 0.9682 | 0.9485 | 0.9582 | 94.85% |
| 012 | 0.4456 | 92.89% | 0.9382 | 0.9025 | 0.9200 | 90.25% |

## Recommendations

### Level1 Model
- **Recommended Threshold**: 0.6633 (median across iterations)
- **Expected Performance**: 97.65% ± 0.76% frames clean post

### Raw Model
- **Recommended Threshold**: 0.5049 (median across iterations)
- **Expected Performance**: 96.01% ± 1.63% frames clean post

## Usage

```python
from swap_correction.ml.api import SwapPredictor

# Level1 model
predictor_level1 = SwapPredictor(model_type='level1')
predictions = predictor_level1.predict(trial_data, fps=30, threshold=0.6633)

# Raw model
predictor_raw = SwapPredictor(model_type='raw')
predictions = predictor_raw.predict(trial_data, fps=30, threshold=0.5049)
```
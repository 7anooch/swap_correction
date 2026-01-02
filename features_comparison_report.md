# Feature Extraction Comparison Report

## Overview

This report compares four feature extraction approaches:

1. **Original Features** (56 features): Baseline feature set
2. **Features V2** (46 features): Removed redundant features (8 positions, 2 velocity magnitudes)
3. **Features V3** (39 features): Improved calculations + new features
   - 3-point derivative for angular velocity
   - Tail path curvature
   - Head/tail curvature ratio
   - Collapsed keypoints binary flag
   - Removed small-window std features and velocity components
4. **Features V4** (~40-42 features): Phase 1 & 2 improvements
   - Removed underperforming v3 features (collapsed_keypoints, raw curvatures)
   - Removed window size 5 features
   - Added acceleration features (head, tail, relative)
   - Added body length normalization to distance features

## Comparison: Level1 Model

### Performance Metrics

| Metric | Original | V2 | V3 | V4 | V2 vs Orig | V3 vs Orig | V4 vs Orig | V3 vs V2 | V4 vs V2 | V4 vs V3 |
|--------|----------|----|----|----|------------|------------|------------|----------|----------|----------|
| F1-Score | 0.8585 ± 0.0555 | 0.8920 ± 0.0671 | 0.8900 ± 0.0636 | 0.8965 ± 0.0591 | +0.0335 | +0.0315 | +0.0380 | -0.0019 | +0.0045 | +0.0065 |
| Precision | 0.8772 ± 0.0653 | 0.8667 ± 0.0787 | 0.8646 ± 0.0731 | 0.8737 ± 0.0671 | -0.0105 | -0.0125 | -0.0035 | -0.0021 | +0.0069 | +0.0090 |
| Recall | 0.9000 ± 0.0429 | 0.9794 ± 0.0146 | 0.9768 ± 0.0167 | 0.9789 ± 0.0160 | +0.0794 | +0.0768 | +0.0790 | -0.0026 | -0.0004 | +0.0022 |
| Sensitivity | 0.9000 ± 0.0429 | 0.9794 ± 0.0146 | 0.9768 ± 0.0167 | 0.9789 ± 0.0160 | +0.0794 | +0.0768 | +0.0790 | -0.0026 | -0.0004 | +0.0022 |
| Specificity | 0.9737 ± 0.0186 | 0.9155 ± 0.0519 | 0.9144 ± 0.0493 | 0.9189 ± 0.0470 | -0.0582 | -0.0593 | -0.0548 | -0.0011 | +0.0034 | +0.0045 |
| % Swaps Resolved | 89.9952 ± 4.2890 | 97.9382 ± 1.4613 | 97.6753 ± 1.6704 | 97.8944 ± 1.5981 | +7.9430 | +7.6801 | +7.8992 | -0.2628 | -0.0437 | +0.2191 |
| % Frames Clean Post | 94.9665 ± 2.0566 | 96.2475 ± 1.8221 | 96.1398 ± 1.6657 | 96.5466 ± 1.5865 | +1.2810 | +1.1733 | +1.5801 | -0.1077 | +0.2991 | +0.4068 |

## Comparison: Raw Model

### Performance Metrics

| Metric | Original | V2 | V3 | V4 | V2 vs Orig | V3 vs Orig | V4 vs Orig | V3 vs V2 | V4 vs V2 | V4 vs V3 |
|--------|----------|----|----|----|------------|------------|------------|----------|----------|----------|
| F1-Score | 0.9042 ± 0.0372 | 0.9381 ± 0.0327 | 0.9391 ± 0.0297 | 0.9397 ± 0.0306 | +0.0339 | +0.0349 | +0.0354 | +0.0010 | +0.0016 | +0.0005 |
| Precision | 0.9199 ± 0.0404 | 0.9521 ± 0.0216 | 0.9528 ± 0.0211 | 0.9533 ± 0.0212 | +0.0322 | +0.0329 | +0.0334 | +0.0007 | +0.0012 | +0.0005 |
| Recall | 0.8978 ± 0.0350 | 0.9321 ± 0.0384 | 0.9314 ± 0.0370 | 0.9331 ± 0.0356 | +0.0343 | +0.0336 | +0.0354 | -0.0007 | +0.0011 | +0.0017 |
| Sensitivity | 0.8978 ± 0.0350 | 0.9321 ± 0.0384 | 0.9314 ± 0.0370 | 0.9331 ± 0.0356 | +0.0343 | +0.0336 | +0.0354 | -0.0007 | +0.0011 | +0.0017 |
| Specificity | 0.9519 ± 0.0247 | 0.9652 ± 0.0189 | 0.9661 ± 0.0179 | 0.9658 ± 0.0189 | +0.0133 | +0.0143 | +0.0140 | +0.0010 | +0.0007 | -0.0003 |
| % Swaps Resolved | 89.7775 ± 3.5003 | 93.2086 ± 3.8379 | 93.1407 ± 3.6965 | 93.3146 ± 3.5629 | +3.4311 | +3.3632 | +3.5372 | -0.0679 | +0.1061 | +0.1740 |
| % Frames Clean Post | 92.9993 ± 2.6996 | 95.2394 ± 2.5461 | 95.2239 ± 2.5290 | 95.3605 ± 2.4462 | +2.2400 | +2.2246 | +2.3612 | -0.0154 | +0.1211 | +0.1366 |

## Summary

### Key Findings

1. **Feature Count Reduction**:
   - Original: 56 features
   - V2: 46 features (-10 redundant)
   - V3: 39 features (-17 from original, -7 from V2)
   - V4: ~40-42 features (-3 from V3, +3-6 new)

2. **Performance Comparison**:
   - See tables above for detailed metrics
   - Positive differences indicate improvement
   - Negative differences indicate regression

3. **Recommendations**:
   - Use the feature set with best performance for your use case
   - Consider trade-offs between feature count and performance
   - V3 includes improved calculations that may benefit from more data

## Notes

- All comparisons use the same data splits (iterations 007-012)
- Metrics are mean ± std across iterations
- Differences are calculated as: New - Original

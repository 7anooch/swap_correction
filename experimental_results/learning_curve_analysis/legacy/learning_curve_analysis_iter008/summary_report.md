# Learning Curve Analysis: Comprehensive Summary

**Iteration**: 008

## Overview

This report compares learning curves across:
- **Feature Versions**: v2 (46 features) vs v3 (39 features)
- **Model Types**: Level1 vs Raw

## Summary Statistics

| Feature Version   | Model Type   |   Best Test F1 | Best Test % Clean   |   F1 at 50% Data |   F1 at 100% Data |   Improvement (50→100%) |   Improvement (Last 20%) | More Data Helpful   |
|:------------------|:-------------|---------------:|:--------------------|-----------------:|------------------:|------------------------:|-------------------------:|:--------------------|
| V2                | LEVEL1       |         0.9578 | 96.76%              |           0.9299 |            0.9475 |                  0.0176 |                   0.0007 | No                  |
| V2                | RAW          |         0.9483 | 95.19%              |           0.9466 |            0.946  |                 -0.0006 |                  -0.0007 | No                  |
| V3                | LEVEL1       |         0.9577 | 96.75%              |           0.9297 |            0.9464 |                  0.0168 |                   0.0011 | No                  |
| V3                | RAW          |         0.9468 | 95.06%              |           0.9456 |            0.9434 |                 -0.0021 |                  -0.0007 | No                  |

## Detailed Results by Combination

### V2 Features, LEVEL1 Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9929 | 0.8066 | 0.8456 | 89.38% |
| 100800.0 | 0.9872 | 0.8724 | 0.9049 | 92.83% |
| 151200.0 | 0.9932 | 0.8701 | 0.9144 | 93.53% |
| 201600.0 | 0.9775 | 0.9031 | 0.9302 | 94.70% |
| 252000.0 | 0.9912 | 0.9009 | 0.9248 | 94.26% |
| 302400.0 | 0.9856 | 0.9136 | 0.9299 | 94.62% |
| 352800.0 | 0.9715 | 0.9105 | 0.9578 | 96.76% |
| 403200.0 | 0.9838 | 0.9049 | 0.9479 | 96.01% |
| 453600.0 | 0.9922 | 0.9162 | 0.9469 | 95.94% |
| 504000.0 | 0.9635 | 0.9024 | 0.9475 | 95.91% |

### V2 Features, RAW Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9857 | 0.8870 | 0.9234 | 92.93% |
| 100800.0 | 0.9710 | 0.8875 | 0.9425 | 94.63% |
| 151200.0 | 0.9674 | 0.8921 | 0.9454 | 94.90% |
| 201600.0 | 0.9696 | 0.8964 | 0.9458 | 94.95% |
| 252000.0 | 0.9706 | 0.9012 | 0.9476 | 95.12% |
| 302400.0 | 0.9683 | 0.9045 | 0.9466 | 95.04% |
| 352800.0 | 0.9653 | 0.9040 | 0.9483 | 95.19% |
| 403200.0 | 0.9607 | 0.9018 | 0.9463 | 94.99% |
| 453600.0 | 0.9639 | 0.9025 | 0.9467 | 95.06% |
| 504000.0 | 0.9612 | 0.9037 | 0.9460 | 94.98% |

### V3 Features, LEVEL1 Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9919 | 0.8021 | 0.8486 | 89.54% |
| 100800.0 | 0.9930 | 0.8732 | 0.8992 | 92.41% |
| 151200.0 | 0.9895 | 0.8529 | 0.9038 | 92.82% |
| 201600.0 | 0.9771 | 0.9036 | 0.9267 | 94.44% |
| 252000.0 | 0.9882 | 0.8945 | 0.9224 | 94.07% |
| 302400.0 | 0.9779 | 0.9124 | 0.9297 | 94.66% |
| 352800.0 | 0.9821 | 0.9129 | 0.9577 | 96.75% |
| 403200.0 | 0.9817 | 0.9075 | 0.9496 | 96.12% |
| 453600.0 | 0.9729 | 0.9031 | 0.9453 | 95.79% |
| 504000.0 | 0.9653 | 0.9029 | 0.9464 | 95.85% |

### V3 Features, RAW Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9916 | 0.8855 | 0.9204 | 92.69% |
| 100800.0 | 0.9782 | 0.8885 | 0.9395 | 94.33% |
| 151200.0 | 0.9708 | 0.8942 | 0.9461 | 94.96% |
| 201600.0 | 0.9554 | 0.8888 | 0.9436 | 94.74% |
| 252000.0 | 0.9596 | 0.8974 | 0.9437 | 94.78% |
| 302400.0 | 0.9677 | 0.9072 | 0.9456 | 94.95% |
| 352800.0 | 0.9555 | 0.9018 | 0.9468 | 95.06% |
| 403200.0 | 0.9603 | 0.9003 | 0.9459 | 94.97% |
| 453600.0 | 0.9531 | 0.8997 | 0.9442 | 94.83% |
| 504000.0 | 0.9544 | 0.9051 | 0.9434 | 94.77% |


## Key Findings

### Performance Comparison

Compare the 'Best Test F1' and 'Best Test % Clean' across combinations to identify:
- Which feature version performs better
- Which model type performs better
- Best overall combination

### Data Efficiency

Compare 'Improvement (50→100%)' and 'Improvement (Last 20%)' to determine:
- Whether more training data would help
- If performance has plateaued
- Optimal training set size

### Recommendations

Based on the analysis:
- If 'More Data Helpful' = Yes: Consider collecting more training data
- If 'More Data Helpful' = No: Current dataset size is sufficient
- Large gaps between train and validation suggest overfitting
- Converging curves suggest model is reaching capacity
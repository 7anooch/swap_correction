# Learning Curve Analysis: Comprehensive Summary

**Iteration**: 007

## Overview

This report compares learning curves across:
- **Feature Versions**: v2 (46 features) vs v3 (39 features)
- **Model Types**: Level1 vs Raw

## Summary Statistics

| Feature Version   | Model Type   |   Best Test F1 | Best Test % Clean   |   F1 at 50% Data |   F1 at 100% Data |   Improvement (50→100%) |   Improvement (Last 20%) | More Data Helpful   |
|:------------------|:-------------|---------------:|:--------------------|-----------------:|------------------:|------------------------:|-------------------------:|:--------------------|
| V2                | LEVEL1       |         0.9566 | 97.51%              |           0.9528 |            0.9566 |                  0.0038 |                   0.014  | Yes                 |
| V2                | RAW          |         0.9503 | 95.71%              |           0.9153 |            0.9503 |                  0.035  |                   0.0061 | Yes                 |
| V3                | LEVEL1       |         0.9542 | 97.38%              |           0.9542 |            0.9317 |                 -0.0225 |                  -0.0133 | No                  |
| V3                | RAW          |         0.9447 | 95.27%              |           0.9131 |            0.9437 |                  0.0306 |                  -0.001  | No                  |

## Detailed Results by Combination

### V2 Features, LEVEL1 Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9984 | 0.9283 | 0.9178 | 95.36% |
| 100800.0 | 0.9986 | 0.9551 | 0.9437 | 96.81% |
| 151200.0 | 0.9974 | 0.9583 | 0.9526 | 97.31% |
| 201600.0 | 0.9979 | 0.9552 | 0.9513 | 97.23% |
| 252000.0 | 0.9959 | 0.9535 | 0.9528 | 97.30% |
| 302400.0 | 0.9918 | 0.9548 | 0.9522 | 97.26% |
| 352800.0 | 0.9884 | 0.9468 | 0.9444 | 96.79% |
| 403200.0 | 0.9880 | 0.9517 | 0.9425 | 96.68% |
| 453600.0 | 0.9729 | 0.9499 | 0.9461 | 96.89% |
| 504000.0 | 0.9858 | 0.9590 | 0.9566 | 97.51% |

### V2 Features, RAW Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9965 | 0.9308 | 0.8683 | 88.50% |
| 100800.0 | 0.9909 | 0.9543 | 0.9136 | 92.45% |
| 151200.0 | 0.9957 | 0.9665 | 0.9150 | 92.53% |
| 201600.0 | 0.9945 | 0.9717 | 0.9068 | 91.63% |
| 252000.0 | 0.9913 | 0.9741 | 0.9153 | 92.39% |
| 302400.0 | 0.9902 | 0.9749 | 0.9183 | 92.70% |
| 352800.0 | 0.9894 | 0.9741 | 0.9139 | 92.22% |
| 403200.0 | 0.9878 | 0.9725 | 0.9441 | 95.23% |
| 453600.0 | 0.9858 | 0.9729 | 0.9382 | 94.67% |
| 504000.0 | 0.9846 | 0.9727 | 0.9503 | 95.71% |

### V3 Features, LEVEL1 Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9987 | 0.9367 | 0.9237 | 95.67% |
| 100800.0 | 0.9993 | 0.9543 | 0.9421 | 96.73% |
| 151200.0 | 0.9971 | 0.9610 | 0.9526 | 97.30% |
| 201600.0 | 0.9987 | 0.9568 | 0.9533 | 97.35% |
| 252000.0 | 0.9953 | 0.9521 | 0.9542 | 97.38% |
| 302400.0 | 0.9930 | 0.9544 | 0.9539 | 97.35% |
| 352800.0 | 0.9896 | 0.9493 | 0.9520 | 97.25% |
| 403200.0 | 0.9860 | 0.9513 | 0.9450 | 96.83% |
| 453600.0 | 0.9834 | 0.9524 | 0.9453 | 96.84% |
| 504000.0 | 0.9778 | 0.9562 | 0.9317 | 96.00% |

### V3 Features, RAW Model

| Training Size | Train F1 | Val F1 | Test F1 | Test % Clean |
|---------------|----------|--------|---------|--------------|
| 50400.0 | 0.9955 | 0.9202 | 0.8651 | 88.23% |
| 100800.0 | 0.9931 | 0.9522 | 0.8917 | 90.43% |
| 151200.0 | 0.9936 | 0.9540 | 0.8970 | 90.96% |
| 201600.0 | 0.9944 | 0.9655 | 0.8913 | 90.39% |
| 252000.0 | 0.9907 | 0.9741 | 0.9131 | 92.17% |
| 302400.0 | 0.9898 | 0.9751 | 0.9219 | 93.03% |
| 352800.0 | 0.9894 | 0.9741 | 0.9142 | 92.26% |
| 403200.0 | 0.9866 | 0.9728 | 0.9447 | 95.27% |
| 453600.0 | 0.9860 | 0.9726 | 0.9414 | 94.97% |
| 504000.0 | 0.9841 | 0.9717 | 0.9437 | 95.12% |


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
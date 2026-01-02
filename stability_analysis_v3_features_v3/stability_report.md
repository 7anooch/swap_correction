# Model Stability Analysis Report

## Overview

**Total Iterations**: 6
**Level1 Model Iterations**: 6
**Raw Model Iterations**: 6

**Sample Sizes Tested**: 80
**Iterations per Sample Size**: 6

## Analysis by Sample Size

### Sample Size: 80

**Level1 Model (n=6 iterations):**
- Mean F1: 0.9332 (std: 0.0364, CV: 0.0390)
- Mean Sensitivity: 0.9522 (std: 0.0268)
- Mean Specificity: 0.9567 (std: 0.0223)
- Mean Sensitivity (Eval): 0.9768 (std: 0.0167)
- Mean Specificity (Eval): 0.9144 (std: 0.0493)
- Mean % Swaps Resolved: 97.68% (std: 1.67%)
- Mean % Frames Clean (Post): 96.14% (std: 1.67%)

**Raw Model (n=6 iterations):**
- Mean F1: 0.9345 (std: 0.0343, CV: 0.0367)
- Mean Sensitivity: 0.9269 (std: 0.0321)
- Mean Specificity: 0.9546 (std: 0.0288)
- Mean Sensitivity (Eval): 0.9314 (std: 0.0370)
- Mean Specificity (Eval): 0.9661 (std: 0.0179)
- Mean % Swaps Resolved: 93.14% (std: 3.70%)
- Mean % Frames Clean (Post): 95.22% (std: 2.53%)

## Level1 Model Stability (Overall)

### Test F1-Score
- **Mean**: 0.9332
- **Std Dev**: 0.0364
- **Min**: 0.8741
- **Max**: 0.9695
- **Range**: 0.0954
- **Coefficient of Variation**: 0.0390

### Test Sensitivity
- **Mean**: 0.9522
- **Std Dev**: 0.0268
- **Min**: 0.9048
- **Max**: 0.9799

### Test Specificity
- **Mean**: 0.9567
- **Std Dev**: 0.0223
- **Min**: 0.9280
- **Max**: 0.9821

### Evaluation Metrics (Test Dataset)
- **Mean Sensitivity**: 0.9768 (std: 0.0167)
- **Mean Specificity**: 0.9144 (std: 0.0493)
- **Mean % Swaps Resolved**: 97.68% (std: 1.67%)
- **Mean % Frames Clean (Pre-correction)**: 70.24% (std: 7.08%)
- **Mean % Frames Clean (Post-correction)**: 96.14% (std: 1.67%)

## Raw Model Stability

### Test F1-Score
- **Mean**: 0.9345
- **Std Dev**: 0.0343
- **Min**: 0.8918
- **Max**: 0.9748
- **Range**: 0.0830
- **Coefficient of Variation**: 0.0367

### Test Sensitivity
- **Mean**: 0.9269
- **Std Dev**: 0.0321
- **Min**: 0.8885
- **Max**: 0.9721

### Test Specificity
- **Mean**: 0.9546
- **Std Dev**: 0.0288
- **Min**: 0.9141
- **Max**: 0.9781

### Evaluation Metrics (Test Dataset)
- **Mean Sensitivity**: 0.9314 (std: 0.0370)
- **Mean Specificity**: 0.9661 (std: 0.0179)
- **Mean % Swaps Resolved**: 93.14% (std: 3.70%)
- **Mean % Frames Clean (Pre-correction)**: 56.64% (std: 3.14%)
- **Mean % Frames Clean (Post-correction)**: 95.22% (std: 2.53%)

## Model Comparison

### Performance
- **Level1 Mean F1**: 0.9332
- **Raw Mean F1**: 0.9345
- **Difference**: -0.0012

- **Level1 Mean Sensitivity (Eval)**: 0.9768
- **Raw Mean Sensitivity (Eval)**: 0.9314
- **Sensitivity Difference**: 0.0453

- **Level1 Mean Specificity (Eval)**: 0.9144
- **Raw Mean Specificity (Eval)**: 0.9661
- **Specificity Difference**: -0.0517

- **Level1 Mean % Swaps Resolved**: 97.68%
- **Raw Mean % Swaps Resolved**: 93.14%
- **Difference**: 4.53%

- **Level1 Mean % Frames Clean (Post)**: 96.14%
- **Raw Mean % Frames Clean (Post)**: 95.22%
- **Difference**: 0.92%

### Stability (Lower CV = More Stable)
- **Level1 CV**: 0.0390
- **Raw CV**: 0.0367
- **More Stable**: Raw

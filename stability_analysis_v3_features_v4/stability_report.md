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
- Mean F1: 0.9372 (std: 0.0380, CV: 0.0405)
- Mean Sensitivity: 0.9487 (std: 0.0288)
- Mean Specificity: 0.9624 (std: 0.0231)
- Mean Sensitivity (Eval): 0.9789 (std: 0.0160)
- Mean Specificity (Eval): 0.9189 (std: 0.0470)
- Mean % Swaps Resolved: 97.89% (std: 1.60%)
- Mean % Frames Clean (Post): 96.55% (std: 1.59%)

**Raw Model (n=6 iterations):**
- Mean F1: 0.9345 (std: 0.0355, CV: 0.0379)
- Mean Sensitivity: 0.9250 (std: 0.0382)
- Mean Specificity: 0.9565 (std: 0.0234)
- Mean Sensitivity (Eval): 0.9331 (std: 0.0356)
- Mean Specificity (Eval): 0.9658 (std: 0.0189)
- Mean % Swaps Resolved: 93.31% (std: 3.56%)
- Mean % Frames Clean (Post): 95.36% (std: 2.45%)

## Level1 Model Stability (Overall)

### Test F1-Score
- **Mean**: 0.9372
- **Std Dev**: 0.0380
- **Min**: 0.8734
- **Max**: 0.9683
- **Range**: 0.0949
- **Coefficient of Variation**: 0.0405

### Test Sensitivity
- **Mean**: 0.9487
- **Std Dev**: 0.0288
- **Min**: 0.9008
- **Max**: 0.9782

### Test Specificity
- **Mean**: 0.9624
- **Std Dev**: 0.0231
- **Min**: 0.9295
- **Max**: 0.9833

### Evaluation Metrics (Test Dataset)
- **Mean Sensitivity**: 0.9789 (std: 0.0160)
- **Mean Specificity**: 0.9189 (std: 0.0470)
- **Mean % Swaps Resolved**: 97.89% (std: 1.60%)
- **Mean % Frames Clean (Pre-correction)**: 70.24% (std: 7.08%)
- **Mean % Frames Clean (Post-correction)**: 96.55% (std: 1.59%)

## Raw Model Stability

### Test F1-Score
- **Mean**: 0.9345
- **Std Dev**: 0.0355
- **Min**: 0.8867
- **Max**: 0.9756
- **Range**: 0.0888
- **Coefficient of Variation**: 0.0379

### Test Sensitivity
- **Mean**: 0.9250
- **Std Dev**: 0.0382
- **Min**: 0.8720
- **Max**: 0.9743

### Test Specificity
- **Mean**: 0.9565
- **Std Dev**: 0.0234
- **Min**: 0.9244
- **Max**: 0.9762

### Evaluation Metrics (Test Dataset)
- **Mean Sensitivity**: 0.9331 (std: 0.0356)
- **Mean Specificity**: 0.9658 (std: 0.0189)
- **Mean % Swaps Resolved**: 93.31% (std: 3.56%)
- **Mean % Frames Clean (Pre-correction)**: 56.64% (std: 3.14%)
- **Mean % Frames Clean (Post-correction)**: 95.36% (std: 2.45%)

## Model Comparison

### Performance
- **Level1 Mean F1**: 0.9372
- **Raw Mean F1**: 0.9345
- **Difference**: 0.0027

- **Level1 Mean Sensitivity (Eval)**: 0.9789
- **Raw Mean Sensitivity (Eval)**: 0.9331
- **Sensitivity Difference**: 0.0458

- **Level1 Mean Specificity (Eval)**: 0.9189
- **Raw Mean Specificity (Eval)**: 0.9658
- **Specificity Difference**: -0.0469

- **Level1 Mean % Swaps Resolved**: 97.89%
- **Raw Mean % Swaps Resolved**: 93.31%
- **Difference**: 4.58%

- **Level1 Mean % Frames Clean (Post)**: 96.55%
- **Raw Mean % Frames Clean (Post)**: 95.36%
- **Difference**: 1.19%

### Stability (Lower CV = More Stable)
- **Level1 CV**: 0.0405
- **Raw CV**: 0.0379
- **More Stable**: Raw

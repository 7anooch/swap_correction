# Model Comparison Report

## Overview

This report compares two trained ML models for swap detection:
1. **Level1 Model**: Trained on level1.csv (auto-corrected) vs level2.csv (ground truth)
2. **Raw Data Model**: Trained on raw _data.csv vs level2.csv (ground truth)

## Performance Summary

### Test Set Performance

| Metric | Level1 Model | Raw Data Model | Difference |
|--------|--------------|----------------|------------|
| **F1-Score** | 0.9895 | 0.9744 | +0.0150 |
| **Precision** | 0.9916 | 0.9720 | +0.0195 |
| **Recall** | 0.9874 | 0.9769 | +0.0105 |
| **ROC-AUC** | 0.9998 | 0.9981 | +0.0016 |

### Performance Across Splits

#### Level1 Model
- **Train**: F1=1.0000, Precision=1.0000, Recall=1.0000
- **Validation**: F1=0.9901, Precision=0.9974, Recall=0.9830
- **Test**: F1=0.9895, Precision=0.9916, Recall=0.9874

#### Raw Data Model
- **Train**: F1=0.9988, Precision=0.9986, Recall=0.9990
- **Validation**: F1=0.9837, Precision=0.9762, Recall=0.9913
- **Test**: F1=0.9744, Precision=0.9720, Recall=0.9769

## Confusion Matrices (Test Set)

### Level1 Model
```
                Predicted
              No Swap  Swap
Actual No Swap  38513     54
       Swap        81   6352
```

- False Positive Rate: 0.14%
- False Negative Rate: 1.26%

### Raw Data Model
```
                Predicted
              No Swap  Swap
Actual No Swap  22869    605
       Swap       498  21028
```

- False Positive Rate: 2.58%
- False Negative Rate: 2.31%

## Feature Importance Comparison

### Top 10 Features (Level1 Model)
1. alignment_angle_mean_50: 0.3994
2. alignment_angle_mean_20: 0.2232
3. head_speed_mean_50: 0.0420
4. tail_speed_mean_50: 0.0329
5. alignment_angle: 0.0319
6. tail_speed_mean_20: 0.0269
7. head_speed_std_50: 0.0245
8. tail_velocity_magnitude: 0.0160
9. tail_speed_std_50: 0.0158
10. tail_speed: 0.0154

### Top 10 Features (Raw Data Model)
1. alignment_angle_mean_50: 0.3486
2. alignment_angle_mean_20: 0.1506
3. head_speed_mean_50: 0.1026
4. head_speed_mean_20: 0.0421
5. head_speed_std_50: 0.0327
6. speed_ratio: 0.0313
7. tail_speed_mean_50: 0.0311
8. alignment_angle: 0.0275
9. tail_speed_mean_20: 0.0275
10. tail_speed_std_50: 0.0161

## Use Case Recommendations

### When to Use Level1 Model

- **Best for**: Detecting remaining swaps after initial auto-correction
- **Advantages**:
  - Higher precision (99.16% vs 97.20%)
  - Lower false positive rate (0.14% vs 2.58%)
  - Better performance overall (98.95% F1 vs 97.44% F1)
- **Use when**: You have level1.csv files and want to improve them further

### When to Use Raw Data Model

- **Best for**: Detecting swaps directly from raw tracking data
- **Advantages**:
  - Can work directly on raw data (no need for level1 correction first)
  - Still achieves good performance (97.44% F1)
  - Handles higher swap rate (49.41% vs 12.68% in training data)
- **Use when**: You want to skip the level1 correction step entirely

## Data Characteristics

### Training Data
- **Level1 Model**: 12.68% swapped frames (28,519 / 225,000)
- **Raw Data Model**: 49.41% swapped frames (111,169 / 225,000)

### Test Set
- **Level1 Model**: 14.30% swapped frames (6,433 / 45,000)
- **Raw Data Model**: 47.84% swapped frames (21,526 / 45,000)

## Conclusion

Both models perform well, with the Level1 model achieving slightly better performance.
The choice between models depends on your workflow:

- Use **Level1 Model** if you already have level1.csv files and want maximum accuracy
- Use **Raw Data Model** if you want to process raw data directly without intermediate correction steps

Both models are production-ready and can be used for automated swap detection.

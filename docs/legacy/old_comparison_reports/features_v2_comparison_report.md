# Feature Extraction Comparison Report

## Overview

This report compares two feature extraction approaches:

- **stability_analysis_v3**: Original feature extraction (56 features)
- **stability_analysis_v3_features_v2**: Reduced feature set (46 features)
  - Removed: 8 position features (head_x, head_y, tail_x, tail_y, mid_x, mid_y, centroid_x, centroid_y)
  - Removed: 2 velocity magnitude features (head_velocity_magnitude, tail_velocity_magnitude)

## V2 Training Status

- **Complete Iterations**: 6/6
- **Partial Iterations**: 0/6

## V3 Results Summary (56 Features)

### Level1 Model

#### Test Set Performance (Mean ± Std across 6 iterations)

| Metric | Mean | Std Dev | Min | Max | CV |
|--------|------|---------|-----|-----|-----|
| Test F1 | 0.9110 | 0.0421 | 0.8327 | 0.9713 | 0.0462 |
| Test Precision | 0.9074 | 0.0591 | 0.7474 | 0.9719 | 0.0651 |
| Test Recall | 0.9176 | 0.0509 | 0.8375 | 0.9793 | 0.0555 |
| Test Sensitivity | 0.9074 | 0.0497 | 0.8375 | 0.9762 | 0.0548 |
| Test Specificity | 0.9494 | 0.0361 | 0.8518 | 0.9856 | 0.0380 |
| Test Auc | 0.9839 | 0.0141 | 0.9464 | 0.9980 | 0.0143 |

#### Evaluation on Test Dataset (Mean ± Std across 6 iterations)

| Metric | Mean | Std Dev | Min | Max | CV |
|--------|------|---------|-----|-----|-----|
| F1 | 0.8585 | 0.0555 | 0.7750 | 0.9359 | 0.0646 |
| Precision | 0.8772 | 0.0653 | 0.7824 | 0.9707 | 0.0744 |
| Recall | 0.9000 | 0.0429 | 0.7806 | 0.9607 | 0.0477 |
| Sensitivity | 0.9000 | 0.0429 | 0.7806 | 0.9607 | 0.0477 |
| Specificity | 0.9737 | 0.0186 | 0.9334 | 0.9956 | 0.0191 |
| Pct Swaps Resolved | 89.9952 | 4.2890 | 78.0615 | 96.0743 | 0.0477 |
| Pct Frames Clean Pre | 68.5654 | 6.1924 | 57.4993 | 80.6407 | 0.0903 |
| Pct Frames Clean Post | 94.9665 | 2.0566 | 90.2222 | 98.4259 | 0.0217 |

### Raw Model

#### Test Set Performance (Mean ± Std across 6 iterations)

| Metric | Mean | Std Dev | Min | Max | CV |
|--------|------|---------|-----|-----|-----|
| Test F1 | 0.9263 | 0.0362 | 0.8380 | 0.9755 | 0.0391 |
| Test Precision | 0.9310 | 0.0417 | 0.8013 | 0.9734 | 0.0448 |
| Test Recall | 0.9221 | 0.0359 | 0.8741 | 0.9775 | 0.0389 |
| Test Sensitivity | 0.9145 | 0.0340 | 0.8741 | 0.9775 | 0.0372 |
| Test Specificity | 0.9422 | 0.0302 | 0.8613 | 0.9742 | 0.0321 |
| Test Auc | 0.9828 | 0.0145 | 0.9406 | 0.9980 | 0.0148 |

#### Evaluation on Test Dataset (Mean ± Std across 6 iterations)

| Metric | Mean | Std Dev | Min | Max | CV |
|--------|------|---------|-----|-----|-----|
| F1 | 0.9042 | 0.0372 | 0.8164 | 0.9771 | 0.0411 |
| Precision | 0.9199 | 0.0404 | 0.7998 | 0.9754 | 0.0440 |
| Recall | 0.8978 | 0.0350 | 0.8381 | 0.9790 | 0.0390 |
| Sensitivity | 0.8978 | 0.0350 | 0.8381 | 0.9790 | 0.0390 |
| Specificity | 0.9519 | 0.0247 | 0.8781 | 0.9757 | 0.0259 |
| Pct Swaps Resolved | 89.7775 | 3.5003 | 83.8062 | 97.9032 | 0.0390 |
| Pct Frames Clean Pre | 57.1868 | 2.8936 | 51.1176 | 63.2432 | 0.0506 |
| Pct Frames Clean Post | 92.9993 | 2.6996 | 86.5840 | 97.6926 | 0.0290 |

## V3 Feature Importance Analysis

### Top Features (Averaged across iterations)

#### Level1 Model - Top 15 Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | alignment_angle_mean_50 | 0.362465 |
| 2 | alignment_angle_mean_20 | 0.270602 |
| 3 | alignment_angle | 0.034315 |
| 4 | tail_speed_std_50 | 0.021979 |
| 5 | head_speed_mean_50 | 0.019502 |
| 6 | tail_speed | 0.019055 |
| 7 | head_speed_std_50 | 0.018669 |
| 8 | tail_velocity_magnitude | 0.017573 |
| 9 | head_speed | 0.016351 |
| 10 | tail_speed_mean_20 | 0.016242 |
| 11 | tail_speed_mean_50 | 0.015817 |
| 12 | speed_ratio | 0.014929 |
| 13 | head_velocity_magnitude | 0.013114 |
| 14 | tail_x | 0.009599 |
| 15 | alignment_angle_mean_5 | 0.009321 |

#### Raw Model - Top 15 Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | alignment_angle_mean_50 | 0.286677 |
| 2 | alignment_angle_mean_20 | 0.183173 |
| 3 | head_speed_mean_20 | 0.090135 |
| 4 | alignment_angle | 0.042074 |
| 5 | tail_speed_std_50 | 0.037388 |
| 6 | head_speed_std_50 | 0.032025 |
| 7 | head_speed_mean_50 | 0.030205 |
| 8 | tail_speed_mean_20 | 0.026485 |
| 9 | head_speed_mean_10 | 0.022247 |
| 10 | speed_ratio | 0.021330 |
| 11 | tail_speed_mean_50 | 0.018027 |
| 12 | centroid_x | 0.010741 |
| 13 | centroid_y | 0.010534 |
| 14 | mid_x | 0.009726 |
| 15 | tail_speed | 0.009610 |

## Features Removed in V2

The following 10 features were removed in V2:

### Position Features (8 features)
- head_x, head_y
- tail_x, tail_y
- mid_x, mid_y
- centroid_x, centroid_y

### Velocity Magnitude Features (2 features)
- head_velocity_magnitude (redundant with head_speed)
- tail_velocity_magnitude (redundant with tail_speed)

## Expected Impact of Feature Reduction

### Potential Benefits
1. **Reduced Model Complexity**: 18% fewer features (56 → 46)
2. **Faster Training**: Fewer features to process
3. **Faster Inference**: Smaller feature vectors
4. **Reduced Overfitting Risk**: Fewer parameters to learn
5. **Clearer Feature Importance**: Removal of redundant features

### Potential Risks
1. **Information Loss**: Position features may capture spatial context
2. **Performance Degradation**: If removed features were informative
3. **Reduced Robustness**: Fewer features may reduce model flexibility

## V2 Results Summary (46 Features)

### Level1 Model

#### Evaluation on Test Dataset (Mean ± Std across 6 iterations)

| Metric | Mean | Std Dev | Min | Max | CV |
|--------|------|---------|-----|-----|-----|
| F1 | 0.8920 | 0.0671 | 0.8084 | 0.9592 | 0.0753 |
| Precision | 0.8667 | 0.0787 | 0.7595 | 0.9346 | 0.0908 |
| Recall | 0.9794 | 0.0146 | 0.9573 | 0.9947 | 0.0149 |
| Sensitivity | 0.9794 | 0.0146 | 0.9573 | 0.9947 | 0.0149 |
| Specificity | 0.9155 | 0.0519 | 0.8439 | 0.9816 | 0.0567 |
| Pct Swaps Resolved | 97.9382 | 1.4613 | 95.7339 | 99.4679 | 0.0149 |
| Pct Frames Clean Pre | 70.2360 | 7.0750 | 60.5954 | 80.6407 | 0.1007 |
| Pct Frames Clean Post | 96.2475 | 1.8221 | 94.1130 | 98.8852 | 0.0189 |

### Raw Model

#### Evaluation on Test Dataset (Mean ± Std across 6 iterations)

| Metric | Mean | Std Dev | Min | Max | CV |
|--------|------|---------|-----|-----|-----|
| F1 | 0.9381 | 0.0327 | 0.8864 | 0.9744 | 0.0349 |
| Precision | 0.9521 | 0.0216 | 0.9189 | 0.9756 | 0.0227 |
| Recall | 0.9321 | 0.0384 | 0.8697 | 0.9732 | 0.0412 |
| Sensitivity | 0.9321 | 0.0384 | 0.8697 | 0.9732 | 0.0412 |
| Specificity | 0.9652 | 0.0189 | 0.9403 | 0.9864 | 0.0195 |
| Pct Swaps Resolved | 93.2086 | 3.8379 | 86.9734 | 97.3202 | 0.0412 |
| Pct Frames Clean Pre | 56.6355 | 3.1387 | 51.1176 | 60.6157 | 0.0554 |
| Pct Frames Clean Post | 95.2394 | 2.5461 | 91.6917 | 98.3815 | 0.0267 |

## Direct Comparison: V3 (56 features) vs V2 (46 features)

### Level1 Model Comparison

| Metric | V3 (56 features) | V2 (46 features) | Difference |
|--------|------------------|-------------------|------------|
| F1 | 0.8585 | 0.8920 | +0.0335 |
| Precision | 0.8772 | 0.8667 | -0.0105 |
| Recall | 0.9000 | 0.9794 | +0.0794 |
| Sensitivity | 0.9000 | 0.9794 | +0.0794 |
| Specificity | 0.9737 | 0.9155 | -0.0582 |
| Pct Swaps Resolved | 89.9952 | 97.9382 | +7.9430 |
| Pct Frames Clean Post | 94.9665 | 96.2475 | +1.2810 |

### Raw Model Comparison

| Metric | V3 (56 features) | V2 (46 features) | Difference |
|--------|------------------|-------------------|------------|
| F1 | 0.9042 | 0.9381 | +0.0339 |
| Precision | 0.9199 | 0.9521 | +0.0322 |
| Recall | 0.8978 | 0.9321 | +0.0343 |
| Sensitivity | 0.8978 | 0.9321 | +0.0343 |
| Specificity | 0.9519 | 0.9652 | +0.0133 |
| Pct Swaps Resolved | 89.7775 | 93.2086 | +3.4311 |
| Pct Frames Clean Post | 92.9993 | 95.2394 | +2.2400 |

## Next Steps

1. **Analyze Results**: Compare performance metrics, stability, and feature importance
2. **Make Decision**: Determine if feature reduction is beneficial
3. **Further Optimization**: Consider additional feature engineering based on results

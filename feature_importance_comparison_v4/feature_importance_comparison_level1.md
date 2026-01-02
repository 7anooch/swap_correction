# Feature Importance Comparison: LEVEL1 Model (Including V4)

## Overview

This report compares feature importance across four feature extraction approaches:

1. **Original Features** (56 features): Baseline feature set
2. **Features V2** (46 features): Removed redundant features
3. **Features V3** (39 features): Improved calculations + new features
4. **Features V4** (~40-42 features): Phase 1 & 2 improvements

## Top Features Comparison

### Top 20 Features

| Rank | Feature | Original | V2 | V3 | V4 |
|------|---------|----------|----|----|----|
| 1 | alignment_angle_mean_50 | 5 (0.3745) | 5 (0.4822) | 5 (0.5075) | 4 (0.3400) |
| 2 | alignment_angle_mean_10 | - | 2 (0.0558) | - | 2 (0.2865) |
| 3 | alignment_angle_mean_20 | 3 (0.2651) | 3 (0.1154) | 3 (0.1429) | 3 (0.1005) |
| 4 | alignment_angle | 1 (0.0387) | 1 (0.0336) | 1 (0.0364) | 1 (0.0340) |
| 5 | head_speed_std_50 | 28 (0.0267) | 26 (0.0290) | 23 (0.0364) | 20 (0.0327) |
| 6 | tail_speed | 43 (0.0174) | 36 (0.0210) | 33 (0.0157) | 31 (0.0173) |
| 7 | head_speed_mean_50 | 24 (0.0190) | 22 (0.0198) | 21 (0.0176) | 18 (0.0130) |
| 8 | tail_speed_std_50 | 51 (0.0142) | 44 (0.0172) | 39 (0.0183) | 36 (0.0123) |
| 9 | tail_speed_mean_50 | 47 (0.0136) | 40 (0.0167) | 37 (0.0178) | 34 (0.0179) |
| 10 | head_speed | 20 (0.0129) | 18 (0.0134) | 17 (0.0151) | 15 (0.0096) |
| 11 | speed_ratio | 39 (0.0127) | 32 (0.0123) | 28 (0.0135) | 26 (0.0104) |
| 12 | tail_speed_mean_20 | 45 (0.0129) | - | 35 (0.0108) | - |
| 13 | head_tail_distance | 29 (0.0084) | 27 (0.0119) | 25 (0.0121) | 22 (0.0101) |
| 14 | tail_speed_mean_5 | - | 39 (0.0077) | 36 (0.0121) | - |
| 15 | tail_mid_distance | 41 (0.0070) | 34 (0.0102) | 30 (0.0120) | 29 (0.0084) |
| 16 | head_mid_distance | - | 15 (0.0104) | 14 (0.0101) | 13 (0.0083) |
| 17 | centroid_y | 12 (0.0101) | - | - | - |
| 18 | cumulative_tail_distance | - | 13 (0.0096) | 12 (0.0099) | 10 (0.0080) |
| 19 | tail_velocity_magnitude | 52 (0.0099) | - | - | - |
| 20 | cumulative_head_distance | - | 12 (0.0095) | 11 (0.0098) | 9 (0.0074) |

## Statistics

### Feature Count
- Original: 56 features
- V2: 46 features
- V3: 39 features
- V4: 36 features

### Importance Statistics

#### Original
- Mean importance: 0.017857
- Std importance: 0.059941
- Max importance: 0.374451
- Min importance: 0.000251
- Top feature: alignment_angle_mean_50 (0.374451)

#### V2
- Mean importance: 0.021739
- Std importance: 0.071831
- Max importance: 0.482178
- Min importance: 0.000847
- Top feature: alignment_angle_mean_50 (0.482178)

#### V3
- Mean importance: 0.025641
- Std importance: 0.082418
- Max importance: 0.507538
- Min importance: 0.000000
- Top feature: alignment_angle_mean_50 (0.507538)

#### V4
- Mean importance: 0.027778
- Std importance: 0.072548
- Max importance: 0.339996
- Min importance: 0.000906
- Top feature: alignment_angle_mean_50 (0.339996)

## New Features in V4

The following features are new in V4:

- **head_acceleration**: Rank 11, Importance 0.001044
- **relative_acceleration**: Rank 24, Importance 0.000982
- **tail_acceleration**: Rank 27, Importance 0.003298

## Removed Features in V4 (from V3)

- **alignment_angle_mean_5**: Rank 4, Importance 0.008655
- **collapsed_keypoints**: Rank 9, Importance 0.000000
- **head_path_curvature**: Rank 16, Importance 0.001642
- **head_speed_mean_5**: Rank 20, Importance 0.004897
- **tail_path_curvature**: Rank 32, Importance 0.000757
- **tail_speed_mean_5**: Rank 36, Importance 0.012099
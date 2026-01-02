# Feature Importance Comparison: LEVEL1 Model

## Overview

This report compares feature importance across three feature extraction approaches:

1. **Original Features** (56 features): Baseline feature set
2. **Features V2** (46 features): Removed redundant features
3. **Features V3** (39 features): Improved calculations + new features

## Top Features Comparison

### Top 20 Features

| Rank | Original | V2 | V3 |
|------|----------|----|----|
| 1 | alignment_angle | 1 (0.0387) | 1 (0.0336) | 1 (0.0364) |
| 2 | alignment_angle_mean_10 | - | 2 (0.0558) | - |
| 3 | alignment_angle_mean_20 | 3 (0.2651) | 3 (0.1154) | 3 (0.1429) |
| 4 | alignment_angle_mean_5 | 4 (0.0072) | - | 4 (0.0087) |
| 5 | alignment_angle_mean_50 | 5 (0.3745) | 5 (0.4822) | 5 (0.5075) |
| 6 | body_orientation_angle | - | 10 (0.0087) | 8 (0.0089) |
| 7 | cumulative_head_distance | - | 12 (0.0095) | 11 (0.0098) |
| 8 | centroid_y | 12 (0.0101) | - | - |
| 9 | cumulative_tail_distance | - | 13 (0.0096) | 12 (0.0099) |
| 10 | head_mid_distance | - | 15 (0.0104) | 14 (0.0101) |
| 11 | head_speed | 20 (0.0129) | 18 (0.0134) | 17 (0.0151) |
| 12 | head_speed_mean_20 | 22 (0.0084) | 20 (0.0094) | - |
| 13 | head_speed_mean_50 | 24 (0.0190) | 22 (0.0198) | 21 (0.0176) |
| 14 | head_speed_std_50 | 28 (0.0267) | 26 (0.0290) | 23 (0.0364) |
| 15 | head_tail_distance | 29 (0.0084) | 27 (0.0119) | 25 (0.0121) |
| 16 | position_in_trial | - | 30 (0.0091) | 26 (0.0093) |
| 17 | head_velocity_magnitude | 30 (0.0089) | - | - |
| 18 | head_x | 33 (0.0086) | - | - |
| 19 | speed_ratio | 39 (0.0127) | 32 (0.0123) | 28 (0.0135) |
| 20 | tail_mid_distance | 41 (0.0070) | 34 (0.0102) | 30 (0.0120) |

## Statistics

### Feature Count
- Original: 56 features
- V2: 46 features
- V3: 39 features

### Importance Statistics

#### Original
- Mean importance: 0.017857
- Std importance: 0.059941
- Max importance: 0.374451
- Min importance: 0.000251

#### V2
- Mean importance: 0.021739
- Std importance: 0.071831
- Max importance: 0.482178
- Min importance: 0.000847

#### V3
- Mean importance: 0.025641
- Std importance: 0.082418
- Max importance: 0.507538
- Min importance: 0.000000

## New Features in V3

The following features are new in V3:

- **collapsed_keypoints**: Rank 9, Importance 0.000000
- **head_tail_curvature_ratio**: Rank 24, Importance 0.002177
- **tail_path_curvature**: Rank 32, Importance 0.000757

## Removed Features

### Removed in V2

- **centroid_x**: Rank 11, Importance 0.006792
- **centroid_y**: Rank 12, Importance 0.010128
- **head_velocity_magnitude**: Rank 30, Importance 0.008948
- **head_x**: Rank 33, Importance 0.008636
- **head_y**: Rank 34, Importance 0.005349
- **mid_x**: Rank 35, Importance 0.006448
- **mid_y**: Rank 36, Importance 0.006646
- **tail_velocity_magnitude**: Rank 52, Importance 0.009881
- **tail_x**: Rank 55, Importance 0.008298
- **tail_y**: Rank 56, Importance 0.006504

### Removed in V3 (from V2)

- **alignment_angle_std_10**: Rank 6, Importance 0.002477
- **alignment_angle_std_5**: Rank 8, Importance 0.000847
- **head_speed_std_10**: Rank 23, Importance 0.001652
- **head_speed_std_5**: Rank 25, Importance 0.001552
- **head_velocity_x**: Rank 28, Importance 0.006175
- **head_velocity_y**: Rank 29, Importance 0.004330
- **tail_speed_std_10**: Rank 41, Importance 0.001448
- **tail_speed_std_5**: Rank 43, Importance 0.002240
- **tail_velocity_x**: Rank 45, Importance 0.003246
- **tail_velocity_y**: Rank 46, Importance 0.002442
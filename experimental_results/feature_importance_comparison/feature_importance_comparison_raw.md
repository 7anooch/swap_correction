# Feature Importance Comparison: RAW Model

## Overview

This report compares feature importance across three feature extraction approaches:

1. **Original Features** (56 features): Baseline feature set
2. **Features V2** (46 features): Removed redundant features
3. **Features V3** (39 features): Improved calculations + new features

## Top Features Comparison

### Top 20 Features

| Rank | Original | V2 | V3 |
|------|----------|----|----|
| 1 | alignment_angle | 1 (0.0391) | 1 (0.0311) | 1 (0.0325) |
| 2 | alignment_angle_mean_10 | - | 2 (0.0230) | - |
| 3 | alignment_angle_mean_20 | 3 (0.1787) | 3 (0.0727) | 3 (0.0969) |
| 4 | alignment_angle_mean_50 | 5 (0.3178) | 5 (0.4119) | 5 (0.4234) |
| 5 | alignment_angle_std_50 | 9 (0.0065) | - | - |
| 6 | cumulative_head_distance | - | 12 (0.0076) | 11 (0.0081) |
| 7 | cumulative_tail_distance | - | 13 (0.0083) | 12 (0.0081) |
| 8 | head_mid_distance | - | 15 (0.0081) | 14 (0.0080) |
| 9 | head_speed_mean_10 | 21 (0.0119) | 19 (0.0129) | 18 (0.0073) |
| 10 | head_speed_mean_20 | 22 (0.0801) | 20 (0.0592) | 19 (0.0748) |
| 11 | head_speed_mean_50 | 24 (0.0221) | 22 (0.0309) | 21 (0.0299) |
| 12 | head_speed_std_20 | 26 (0.0137) | 24 (0.0308) | 22 (0.0152) |
| 13 | head_speed_std_10 | 25 (0.0165) | - | - |
| 14 | head_speed_std_50 | 28 (0.0366) | 26 (0.0453) | 23 (0.0436) |
| 15 | head_tail_distance | 29 (0.0112) | 27 (0.0135) | 25 (0.0139) |
| 16 | position_in_trial | - | 30 (0.0079) | 26 (0.0083) |
| 17 | tail_mid_distance | - | - | 30 (0.0077) |
| 18 | speed_ratio | 39 (0.0208) | 32 (0.0251) | 28 (0.0240) |
| 19 | head_y | 34 (0.0061) | - | - |
| 20 | mid_x | 35 (0.0066) | - | - |

## Statistics

### Feature Count
- Original: 56 features
- V2: 46 features
- V3: 39 features

### Importance Statistics

#### Original
- Mean importance: 0.017857
- Std importance: 0.048480
- Max importance: 0.317836
- Min importance: 0.000952

#### V2
- Mean importance: 0.021739
- Std importance: 0.061013
- Max importance: 0.411904
- Min importance: 0.001568

#### V3
- Mean importance: 0.025641
- Std importance: 0.068555
- Max importance: 0.423369
- Min importance: 0.001355

## New Features in V3

The following features are new in V3:

- **collapsed_keypoints**: Rank 9, Importance 0.002238
- **head_tail_curvature_ratio**: Rank 24, Importance 0.003733
- **tail_path_curvature**: Rank 32, Importance 0.001802

## Removed Features

### Removed in V2

- **centroid_x**: Rank 11, Importance 0.006028
- **centroid_y**: Rank 12, Importance 0.005789
- **head_velocity_magnitude**: Rank 30, Importance 0.001581
- **head_x**: Rank 33, Importance 0.005001
- **head_y**: Rank 34, Importance 0.006080
- **mid_x**: Rank 35, Importance 0.006594
- **mid_y**: Rank 36, Importance 0.006062
- **tail_velocity_magnitude**: Rank 52, Importance 0.003622
- **tail_x**: Rank 55, Importance 0.005965
- **tail_y**: Rank 56, Importance 0.005922

### Removed in V3 (from V2)

- **alignment_angle_std_10**: Rank 6, Importance 0.002540
- **alignment_angle_std_5**: Rank 8, Importance 0.002645
- **head_speed_std_10**: Rank 23, Importance 0.002349
- **head_speed_std_5**: Rank 25, Importance 0.002864
- **head_velocity_x**: Rank 28, Importance 0.002221
- **head_velocity_y**: Rank 29, Importance 0.003790
- **tail_speed_std_10**: Rank 41, Importance 0.002302
- **tail_speed_std_5**: Rank 43, Importance 0.001943
- **tail_velocity_x**: Rank 45, Importance 0.003247
- **tail_velocity_y**: Rank 46, Importance 0.001568
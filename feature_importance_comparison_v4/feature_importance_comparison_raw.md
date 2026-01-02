# Feature Importance Comparison: RAW Model (Including V4)

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
| 1 | alignment_angle_mean_50 | 5 (0.3178) | 5 (0.4119) | 5 (0.4234) | 4 (0.3492) |
| 2 | alignment_angle_mean_20 | 3 (0.1787) | 3 (0.0727) | 3 (0.0969) | 3 (0.0808) |
| 3 | alignment_angle_mean_10 | - | 2 (0.0230) | - | 2 (0.1174) |
| 4 | head_speed_mean_20 | 22 (0.0801) | 20 (0.0592) | 19 (0.0748) | 17 (0.0815) |
| 5 | tail_speed_mean_20 | 45 (0.0316) | 38 (0.0423) | 35 (0.0466) | 33 (0.0409) |
| 6 | head_speed_std_50 | 28 (0.0366) | 26 (0.0453) | 23 (0.0436) | 20 (0.0389) |
| 7 | tail_speed_std_50 | 51 (0.0334) | 44 (0.0390) | 39 (0.0435) | 36 (0.0341) |
| 8 | alignment_angle | 1 (0.0391) | 1 (0.0311) | 1 (0.0325) | 1 (0.0346) |
| 9 | head_speed_mean_50 | 24 (0.0221) | 22 (0.0309) | 21 (0.0299) | 18 (0.0256) |
| 10 | head_speed_std_20 | 26 (0.0137) | 24 (0.0308) | 22 (0.0152) | 19 (0.0150) |
| 11 | speed_ratio | 39 (0.0208) | 32 (0.0251) | 28 (0.0240) | 26 (0.0232) |
| 12 | head_speed_std_10 | 25 (0.0165) | - | - | - |
| 13 | tail_speed_mean_50 | 47 (0.0138) | 40 (0.0141) | 37 (0.0141) | 34 (0.0162) |
| 14 | head_tail_distance | 29 (0.0112) | 27 (0.0135) | 25 (0.0139) | 22 (0.0117) |
| 15 | head_speed_mean_10 | 21 (0.0119) | 19 (0.0129) | 18 (0.0073) | 16 (0.0085) |
| 16 | tail_speed | 43 (0.0121) | 36 (0.0096) | - | 31 (0.0099) |
| 17 | alignment_angle_std_50 | 9 (0.0065) | - | - | 6 (0.0115) |
| 18 | tail_speed_std_20 | 49 (0.0083) | - | - | 35 (0.0111) |
| 19 | cumulative_tail_distance | - | 13 (0.0083) | 12 (0.0081) | 10 (0.0076) |
| 20 | position_in_trial | - | 30 (0.0079) | 26 (0.0083) | - |

## Statistics

### Feature Count
- Original: 56 features
- V2: 46 features
- V3: 39 features
- V4: 36 features

### Importance Statistics

#### Original
- Mean importance: 0.017857
- Std importance: 0.048480
- Max importance: 0.317836
- Min importance: 0.000952
- Top feature: alignment_angle_mean_50 (0.317836)

#### V2
- Mean importance: 0.021739
- Std importance: 0.061013
- Max importance: 0.411904
- Min importance: 0.001568
- Top feature: alignment_angle_mean_50 (0.411904)

#### V3
- Mean importance: 0.025641
- Std importance: 0.068555
- Max importance: 0.423369
- Min importance: 0.001355
- Top feature: alignment_angle_mean_50 (0.423369)

#### V4
- Mean importance: 0.027778
- Std importance: 0.060822
- Max importance: 0.349240
- Min importance: 0.001397
- Top feature: alignment_angle_mean_50 (0.349240)

## New Features in V4

The following features are new in V4:

- **head_acceleration**: Rank 11, Importance 0.002463
- **relative_acceleration**: Rank 24, Importance 0.001397
- **tail_acceleration**: Rank 27, Importance 0.002914

## Removed Features in V4 (from V3)

- **alignment_angle_mean_5**: Rank 4, Importance 0.004754
- **collapsed_keypoints**: Rank 9, Importance 0.002238
- **head_path_curvature**: Rank 16, Importance 0.001355
- **head_speed_mean_5**: Rank 20, Importance 0.005106
- **tail_path_curvature**: Rank 32, Importance 0.001802
- **tail_speed_mean_5**: Rank 36, Importance 0.007423
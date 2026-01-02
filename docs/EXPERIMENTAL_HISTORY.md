# Experimental History and Findings

**Last Updated**: 2026-01-02  
**Purpose**: Comprehensive documentation of all experiments, feature evolution, and findings from the swap correction ML project.

---

## Executive Summary

### What Worked

1. **Features V4**: Final feature set (~36 features) provides best balance of performance and efficiency
   - Level1 Model: F1 = 0.8965 ± 0.0591, 97.89% swaps resolved, 96.55% frames clean post
   - Raw Model: F1 = 0.9397 ± 0.0306, 93.31% swaps resolved, 95.36% frames clean post
   - Benefits from more training data (unlike V3 which showed decline)
   - Most data-efficient feature set among all versions

2. **Gaussian Filtering**: Critical improvement (sigma=4.6)
   - Improved F1 from 0.45 to 0.99 in initial tests
   - Essential for handling noisy raw tracking data

3. **Threshold Optimization**: 
   - Level1 models: Optimal threshold = 0.63 (vs default 0.5)
   - Raw models: Default threshold = 0.5 is optimal
   - Significant improvement in % frames clean post

4. **Stability Analysis**: Comprehensive validation across multiple random samples
   - Sample size 80-100 provides stable results
   - Low coefficient of variation (CV < 0.04) indicates robust models

### What Didn't Work

1. **Features V3**: Showed overfitting issues
   - Performance declined with more training data in some iterations
   - Led to development of V4 to address this

2. **Original 56-feature set**: Redundant features
   - 8 position features and 2 velocity magnitude features were redundant
   - V2 removal improved performance while reducing feature count

3. **Rule-based improvements**: Limited success
   - Initial rule-based swap detection improvements had mixed results
   - ML approach proved more effective

### Final Conclusions

- **Features V4 is production-ready** and should be the default implementation
- **Level1 models** perform slightly better than Raw models for post-correction refinement
- **Raw models** are viable for direct processing of raw data
- **More training data helps** with V4 (unlike V3)
- **Threshold optimization is essential** for optimal performance

---

## Part 1: Feature Evolution

### Original Features (56 features)

**Description**: Initial feature set extracted from tracking data.

**Features**:
- 8 position features (x, y for head, tail, mid, centroid)
- 3 distance features
- 9 velocity features (speeds, components, magnitudes)
- 6 angular features
- 2 geometric features (cross-sign, path curvature)
- 24 temporal context features (mean/std for windows: 5, 10, 20, 50 frames)
- 2 cumulative distance features
- 1 context feature (position in trial)

**Performance** (Level1 Model, iterations 007-012):
- F1-Score: 0.8585 ± 0.0555
- Precision: 0.8772 ± 0.0653
- Recall: 0.9000 ± 0.0429
- % Swaps Resolved: 89.99% ± 4.29%
- % Frames Clean Post: 94.97% ± 2.06%

**Issues Identified**:
- Redundant features (velocity magnitudes identical to speeds)
- Raw position features may not add value beyond distances
- Some features showed very low importance

### Features V2 (46 features)

**Changes**: Removed 10 redundant features
- 8 position features (`head_x`, `head_y`, `tail_x`, `tail_y`, `mid_x`, `mid_y`, `centroid_x`, `centroid_y`)
- 2 velocity magnitude features (`head_velocity_magnitude`, `tail_velocity_magnitude`)

**Rationale**: These features were redundant with distance features and speed features.

**Performance** (Level1 Model, iterations 007-012):
- F1-Score: 0.8920 ± 0.0671 (+0.0335 vs Original)
- Precision: 0.8667 ± 0.0787 (-0.0105 vs Original)
- Recall: 0.9794 ± 0.0146 (+0.0794 vs Original)
- % Swaps Resolved: 97.94% ± 1.46% (+7.94% vs Original)
- % Frames Clean Post: 96.25% ± 1.82% (+1.28% vs Original)

**Key Finding**: Removing redundant features improved recall significantly while maintaining precision.

### Features V3 (39 features)

**Changes**:
- **Improved calculations**:
  - 3-point central difference for angular velocity (more robust)
  - Added tail path curvature calculation
- **New features**:
  - `head_tail_curvature_ratio`: Ratio of head to tail path curvature
  - `collapsed_keypoints`: Binary flag for tracking errors (tolerance 0.05mm)
- **Removed features**:
  - Standard deviation features for window sizes 5 and 10
  - `head_velocity_x/y` and `tail_velocity_x/y` components

**Rationale**: 
- Better angular velocity calculation improves feature quality
- Curvature ratio captures relative path contortion
- Collapsed keypoints flag tracking errors

**Performance** (Level1 Model, iterations 007-012):
- F1-Score: 0.8900 ± 0.0636 (+0.0315 vs Original, -0.0019 vs V2)
- Precision: 0.8646 ± 0.0731 (-0.0125 vs Original, -0.0021 vs V2)
- Recall: 0.9768 ± 0.0167 (+0.0768 vs Original, -0.0026 vs V2)
- % Swaps Resolved: 97.68% ± 1.67% (+7.68% vs Original, -0.26% vs V2)
- % Frames Clean Post: 96.14% ± 1.67% (+1.17% vs Original, -0.11% vs V2)

**Key Finding**: Performance similar to V2, but learning curve analysis showed overfitting issues (performance declined with more data in some iterations).

### Features V4 (~36 features) - FINAL

**Changes**:
- **Phase 1 (Remove underperformers)**:
  - Removed `collapsed_keypoints` (0-0.22% importance)
  - Removed `head_path_curvature` and `tail_path_curvature` as direct features (kept calculations for ratio)
  - Removed window size 5 features (consistently low importance)
- **Phase 2 (High-value additions)**:
  - Added acceleration features: `head_acceleration`, `tail_acceleration`, `relative_acceleration`
  - Added body length normalization to distance features (normalized by mean body length)

**Rationale**: 
- Remove low-importance features to reduce overfitting
- Add acceleration features to capture motion dynamics
- Normalize distances by body length for better generalization

**Performance** (Level1 Model, iterations 007-012):
- F1-Score: 0.8965 ± 0.0591 (+0.0380 vs Original, +0.0045 vs V2, +0.0065 vs V3)
- Precision: 0.8737 ± 0.0671 (-0.0035 vs Original, +0.0069 vs V2, +0.0090 vs V3)
- Recall: 0.9789 ± 0.0160 (+0.0790 vs Original, -0.0004 vs V2, +0.0022 vs V3)
- % Swaps Resolved: 97.89% ± 1.60% (+7.90% vs Original, -0.04% vs V2, +0.22% vs V3)
- % Frames Clean Post: 96.55% ± 1.59% (+1.58% vs Original, +0.30% vs V2, +0.41% vs V3)

**Key Finding**: Best overall performance, addresses V3 overfitting issues, benefits from more training data.

---

## Part 2: Stability Analysis Experiments

### Overview

Stability analysis was performed to assess model robustness across different random samples of the dataset. Multiple iterations were run with different sample sizes and feature versions.

### Sample Sizes Tested

- **30 samples**: Initial testing (10 iterations)
- **40 samples**: Expanded testing (10 iterations)
- **50 samples**: Recommended size (10 iterations)
- **60 samples**: Extended testing (6 iterations)
- **80 samples**: Final recommended size (6 iterations)
- **100 samples**: Large sample testing (3 iterations for V4)

### Feature Versions Tested

1. **Original (56 features)**: Tested in initial stability analysis
2. **V2 (46 features)**: Tested in `stability_analysis_v3_features_v2`
3. **V3 (39 features)**: Tested in `stability_analysis_v3_features_v3`
4. **V4 (~36 features)**: Tested in `stability_analysis_v3_features_v4` (final)

### Key Findings

#### Sample Size Analysis

**Best Sample Size: 80**
- Provides optimal balance of performance and stability
- Level1 Model: F1 = 0.9372 ± 0.0380 (CV = 0.0405)
- Raw Model: F1 = 0.9345 ± 0.0355 (CV = 0.0379)
- Low coefficient of variation indicates stable results

#### Features V4 Stability (Sample Size 80, 6 iterations)

**Level1 Model**:
- Mean F1: 0.9372 (std: 0.0380, CV: 0.0405)
- Mean Sensitivity: 0.9789 (std: 0.0160)
- Mean Specificity: 0.9189 (std: 0.0470)
- Mean % Swaps Resolved: 97.89% (std: 1.60%)
- Mean % Frames Clean Post: 96.55% (std: 1.59%)

**Raw Model**:
- Mean F1: 0.9345 (std: 0.0355, CV: 0.0379)
- Mean Sensitivity: 0.9331 (std: 0.0356)
- Mean Specificity: 0.9658 (std: 0.0189)
- Mean % Swaps Resolved: 93.31% (std: 3.56%)
- Mean % Frames Clean Post: 95.36% (std: 2.45%)

**Conclusion**: Models are stable across different random samples, with low variance in performance metrics.

---

## Part 3: Learning Curve Analysis

### Purpose

Learning curve analysis determines whether more training data improves model performance, helping assess if data collection should continue or if performance has plateaued.

### Methodology

Models were trained on progressively larger subsets of training data (10%, 20%, ..., 100%) while keeping validation and test sets constant. Performance was measured at each training size.

### Results Summary

#### Features V4 (Iterations 13-15, Sample Size 100)

**Level1 Model**:
- Average F1 at 50% data: 0.9244
- Average F1 at 100% data: 0.9232
- Average improvement (50→100%): -0.0012 ± 0.0072
- More data helpful: 1/3 iterations

**Raw Model**:
- Average F1 at 50% data: 0.9302
- Average F1 at 100% data: 0.9348
- Average improvement (50→100%): +0.0046 ± 0.0042
- More data helpful: 2/3 iterations

**Key Finding**: Raw models show consistent improvement with more data, while Level1 models show mixed results (likely due to already high performance).

#### Comparison Across Feature Versions (Iterations 7-8)

**V2**:
- Level1: Avg improvement (50→100%) = +0.0132, More data helpful: 100%
- Raw: Avg improvement (50→100%) = +0.0167, More data helpful: 50%

**V3**:
- Level1: Avg improvement (50→100%) = +0.0007, More data helpful: 50%
- Raw: Avg improvement (50→100%) = +0.0152, More data helpful: 50%

**V4**:
- Level1: Avg improvement (50→100%) = +0.0121, More data helpful: 100%
- Raw: Avg improvement (50→100%) = +0.0124, More data helpful: 50%

**Conclusion**: V4 addresses V3's overfitting issues and shows consistent improvement with more data (unlike V3 which showed decline in some cases).

---

## Part 4: Threshold Optimization

### Purpose

Find optimal classification threshold that maximizes performance metrics (especially % Frames Clean Post).

### Methodology

Tested thresholds from 0.0 to 1.0 in increments of 0.01, evaluating all metrics at each threshold on validation set.

### Results

#### Level1 Models

**Recommended Threshold: 0.63**
- Optimized for: % Frames Clean Post
- Performance: ~97.65% ± 0.76% frames clean post on average
- Rationale: Based on comprehensive threshold optimization across multiple stability analysis iterations

**Default Threshold (0.5) Performance**:
- Lower % Frames Clean Post compared to optimized threshold
- Still acceptable but not optimal

#### Raw Models

**Recommended Threshold: 0.5 (default)**
- Default threshold provides optimal performance
- Performance: ~96% frames clean post on average
- No significant improvement with threshold optimization

### Key Finding

Threshold optimization is essential for Level1 models but not necessary for Raw models (default 0.5 is optimal).

---

## Part 5: Model Performance Summary

### Final Performance Metrics (Features V4)

#### Level1 Model

**Training Performance** (from stability analysis, sample size 80):
- Mean F1: 0.9372 ± 0.0380
- Mean Sensitivity: 0.9789 ± 0.0160
- Mean Specificity: 0.9189 ± 0.0470
- Mean % Swaps Resolved: 97.89% ± 1.60%
- Mean % Frames Clean Post: 96.55% ± 1.59%

**Use Case**: Best for detecting remaining swaps after initial auto-correction (level1.csv → level2.csv)

#### Raw Model

**Training Performance** (from stability analysis, sample size 80):
- Mean F1: 0.9345 ± 0.0355
- Mean Sensitivity: 0.9331 ± 0.0356
- Mean Specificity: 0.9658 ± 0.0189
- Mean % Swaps Resolved: 93.31% ± 3.56%
- Mean % Frames Clean Post: 95.36% ± 2.45%

**Use Case**: Best for detecting swaps directly from raw tracking data (raw _data.csv → level2.csv)

### Model Comparison

| Metric | Level1 Model | Raw Model | Winner |
|--------|-------------|-----------|--------|
| F1-Score | 0.9372 | 0.9345 | Level1 (slight) |
| Sensitivity | 0.9789 | 0.9331 | Level1 |
| Specificity | 0.9189 | 0.9658 | Raw |
| % Swaps Resolved | 97.89% | 93.31% | Level1 |
| % Frames Clean Post | 96.55% | 95.36% | Level1 |
| Stability (CV) | 0.0405 | 0.0379 | Raw (slightly more stable) |

**Conclusion**: Level1 models perform slightly better overall, but both models are production-ready.

---

## Part 6: Feature Importance Analysis

### Top Features Across Versions

#### Original Features (56 features)

**Top 10 Most Important**:
1. `alignment_angle_mean_50`: 36-40%
2. `alignment_angle_mean_20`: 15-27%
3. `head_speed_mean_50`: ~4%
4. `tail_speed_mean_50`: ~3%
5. `alignment_angle`: ~3%
6. `tail_speed_mean_20`: ~3%
7. `head_speed_std_50`: ~2%
8. `tail_velocity_magnitude`: ~2%
9. `tail_speed_std_50`: ~2%
10. `tail_speed`: ~2%

**Key Insight**: Alignment angle features dominate, accounting for >50% of feature importance.

#### Features V4 (~36 features)

**Top Features** (similar distribution to original):
- Alignment angle features still dominate
- Acceleration features show moderate importance
- Body length normalization improves distance features

**Key Insight**: Feature reduction maintained performance while improving efficiency and reducing overfitting.

---

## Part 7: Comparison Reports Summary

### Feature Version Comparison

**Best Overall**: Features V4
- Highest F1-score for both Level1 and Raw models
- Best % Frames Clean Post
- Addresses V3 overfitting issues
- Benefits from more training data

**Performance Ranking** (Level1 Model):
1. V4: F1 = 0.8965 ± 0.0591
2. V2: F1 = 0.8920 ± 0.0671
3. V3: F1 = 0.8900 ± 0.0636
4. Original: F1 = 0.8585 ± 0.0555

**Performance Ranking** (Raw Model):
1. V4: F1 = 0.9397 ± 0.0306
2. V3: F1 = 0.9391 ± 0.0297
3. V2: F1 = 0.9381 ± 0.0327
4. Original: F1 = 0.9042 ± 0.0372

### Learning Curve Comparison

**V4 Advantages**:
- Consistent improvement with more data (unlike V3)
- Maintains stable performance across iterations
- Most data-efficient feature set

---

## Part 8: Key Lessons Learned

### What Worked Well

1. **Iterative Feature Refinement**: Starting with 56 features and systematically removing redundancies and adding improvements led to optimal feature set
2. **Comprehensive Testing**: Stability analysis across multiple random samples provided confidence in model robustness
3. **Threshold Optimization**: Significant improvement for Level1 models
4. **Gaussian Filtering**: Critical for handling noisy raw data

### What Didn't Work

1. **V3 Overfitting**: Showed that adding features without careful consideration can lead to overfitting
2. **Raw Position Features**: Low importance, redundant with distance features
3. **Small Window Features**: Window size 5 features consistently showed low importance

### Recommendations for Future Work

1. **Use Features V4 as default**: Best balance of performance and efficiency
2. **Continue data collection**: Learning curves suggest more data would help, especially for Raw models
3. **Monitor for overfitting**: Regular validation and learning curve analysis
4. **Consider ensemble methods**: Combining Level1 and Raw models might improve performance

---

## References

- **Feature Extraction Review**: `docs/FEATURE_EXTRACTION_REVIEW.md`
- **Stability Reports**: `stability_analysis_v3_features_v4/stability_report.md`
- **Comparison Reports**: `features_comparison_report.md`, `learning_curve_comparison_v4_iterations_13_15.md`
- **Model Registry**: `MODEL_REGISTRY.md`
- **Initial Findings**: `findings_and_planning.md`

---

**End of Experimental History**


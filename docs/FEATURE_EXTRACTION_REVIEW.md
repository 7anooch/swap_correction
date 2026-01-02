# Feature Extraction Review and Optimization

## 1. Executive Summary

### Overview

The current feature extraction system for ML-based swap detection uses an optimized implementation (`extract_all_frame_features_optimized()`) that achieves a 400x speedup through pre-computation and vectorized operations. The system extracts 56 frame-level features from animal tracking data, with Gaussian filtering (sigma=4.6) applied to raw positions before feature extraction.

### Key Findings

1. **Dominant Features**: Alignment angle features dominate importance, with `alignment_angle_mean_50` accounting for ~36-40% of feature importance and `alignment_angle_mean_20` for ~15-27%.

2. **Feature Redundancy**: Several features are redundant:
   - `head_velocity_magnitude` and `tail_velocity_magnitude` are identical to `head_speed` and `tail_speed`
   - Raw position features (x, y coordinates) may not add value beyond distance features

3. **Low-Value Features**: Some features show surprisingly low importance:
   - `cross_sign`: ~0.04% (despite being key in rule-based detection)
   - `head_path_curvature`: ~0.04%
   - Angular velocities: ~0.03%

4. **Missing Features**: Several potentially valuable features are absent:
   - Acceleration features (rate of speed change)
   - Body length normalization
   - Cross-sign temporal consistency
   - Swap persistence features

### High-Level Recommendations

1. **Immediate**: Remove redundant features (velocity magnitudes, potentially raw positions)
2. **Short-term**: Add missing critical features (acceleration, body length normalization)
3. **Medium-term**: Improve existing features (better angular velocity, more temporal statistics)
4. **Long-term**: Feature selection and advanced engineering (interactions, frequency domain)

---

## 2. Current Feature Extraction Architecture

### Function: `extract_all_frame_features_optimized()`

**Location**: `swap_correction/ml/features/features.py`

**Key Optimizations**:
- Pre-computation of expensive operations (speeds, cross-sign, cumulative distances, alignment angles)
- Vectorized NumPy operations instead of per-frame calculations
- Pandas rolling windows for temporal statistics
- **Result**: 400x speedup compared to naive implementation

### Pre-computation Strategy

The function pre-computes:
1. Speeds for all frames (head and tail)
2. Cross-sign for all frames
3. Cumulative distances (O(n) incremental calculation)
4. Alignment angles for all frames
5. Velocity vectors (x, y components)
6. Angular velocities
7. Path curvature
8. Temporal context statistics (mean/std for windows: 5, 10, 20, 50 frames)

### Gaussian Filtering

- **Default**: `apply_filtering=False`, `filter_sigma=4.5`
- **Optimal**: `filter_sigma=4.6` (determined through grid search)
- Applied to all position columns: `xhead`, `yhead`, `xtail`, `ytail`, `xmid`, `ymid`, `xctr`, `yctr`
- **Impact**: Dramatically improved model performance (F1 from 0.45 to 0.99)

### Feature Categories

Total: **56 features** organized into 8 categories:

1. **Position Features** (8): Raw coordinates
2. **Distance Features** (3): Head-tail, head-mid, tail-mid distances
3. **Velocity Features** (9): Speeds, velocity components, magnitudes, relative velocity
4. **Angular Features** (6): Orientation, motion angles, alignment angle, angular velocities
5. **Geometric Features** (2): Cross-sign, path curvature
6. **Temporal Context Features** (24): Mean/std for speeds and alignment angles across 4 window sizes
7. **Cumulative Features** (2): Cumulative head and tail distances
8. **Context Features** (1): Position in trial (normalized frame index)

---

## 3. Feature Catalog (56 Features)

### 3.1 Position Features (8 features)

**Features**:
- `head_x`, `head_y`: Head position coordinates
- `tail_x`, `tail_y`: Tail position coordinates
- `mid_x`, `mid_y`: Midpoint position coordinates
- `centroid_x`, `centroid_y`: Centroid position coordinates

**Critical Review**:
- **Question**: Are raw positions useful or redundant with distance features?
- **Observation**: Position features show low-medium importance (~0.7-1.0% each)
- **Consideration**: Distances capture spatial relationships; raw positions may be trial-specific
- **Recommendation**: Test removal - distances may be sufficient

### 3.2 Distance Features (3 features)

**Features**:
- `head_tail_distance`: Distance between head and tail
- `head_mid_distance`: Distance from head to midpoint
- `tail_mid_distance`: Distance from tail to midpoint

**Critical Review**:
- **Importance**: Medium (~0.7-0.8% each)
- **Issue**: Not normalized by body length (affects different animals/sizes)
- **Recommendation**: Add body length normalization

### 3.3 Velocity Features (9 features)

**Features**:
- Instantaneous: `head_speed`, `tail_speed`, `speed_ratio`
- Vector components: `head_velocity_x`, `head_velocity_y`, `tail_velocity_x`, `tail_velocity_y`
- Magnitudes: `head_velocity_magnitude`, `tail_velocity_magnitude`
- Relative: `relative_velocity_magnitude`

**Critical Review**:
- **Redundancy**: `head_velocity_magnitude` = `head_speed` (identical calculation)
- **Redundancy**: `tail_velocity_magnitude` = `tail_speed` (identical calculation)
- **Importance**: Speed features show higher importance (~1.6-1.9%) than velocity magnitudes (~1.3-1.7%)
- **Recommendation**: **Remove velocity magnitude features** - they are redundant

### 3.4 Angular Features (6 features)

**Features**:
- `body_orientation_angle`: Angle of body vector (tail to midpoint)
- `head_motion_angle`: Direction of head motion
- `tail_motion_angle`: Direction of tail motion
- `alignment_angle`: **Key feature** - angle between body orientation and tail motion (0° = forward, 180° = backward/swap)
- `head_angular_velocity`: Rate of change of head direction
- `tail_angular_velocity`: Rate of change of tail direction

**Critical Review**:
- **Alignment Angle**: Dominant feature (~3-4% instantaneous, ~36-40% for mean_50 window)
- **Angular Velocity**: Very low importance (~0.03%)
- **Issue**: Angular velocity calculated from 2-frame differences (noisy)
- **Recommendation**: Improve angular velocity calculation (smoothing, longer window)

### 3.5 Geometric Features (2 features)

**Features**:
- `cross_sign`: Cross product sign (used in rule-based detection)
- `head_path_curvature`: Curvature of head path

**Critical Review**:
- **Cross-sign**: Surprisingly low importance (~0.04%) despite being key in rule-based methods
- **Path Curvature**: Very low importance (~0.04%)
- **Issue**: Only head curvature, not tail
- **Recommendation**: Add tail path curvature; investigate why cross-sign is low (may need temporal consistency)

### 3.6 Temporal Context Features (24 features)

**Window Sizes**: 5, 10, 20, 50 frames

**For each window size**:
- `head_speed_mean_{window}`, `head_speed_std_{window}`
- `tail_speed_mean_{window}`, `tail_speed_std_{window}`
- `alignment_angle_mean_{window}`, `alignment_angle_std_{window}`

**Critical Review**:
- **Dominant**: `alignment_angle_mean_50` (~36-40% importance)
- **Secondary**: `alignment_angle_mean_20` (~15-27% importance)
- **Small Windows**: Features with window=5, 10 show lower importance
- **Missing**: Only mean/std, no min/max/range/median
- **Recommendation**: 
  - Evaluate necessity of all window sizes
  - Add more temporal statistics (min, max, range, median)

### 3.7 Cumulative Features (2 features)

**Features**:
- `cumulative_head_distance`: Total distance traveled by head
- `cumulative_tail_distance`: Total distance traveled by tail

**Critical Review**:
- **Importance**: Medium (~0.5-1.0%)
- **Utility**: May help detect persistent swaps (head travels more than tail)
- **Issue**: Not normalized by trial length or body length
- **Recommendation**: Keep, but consider normalization

### 3.8 Context Features (1 feature)

**Features**:
- `position_in_trial`: Normalized frame index (0.0 to 1.0)

**Critical Review**:
- **Utility**: Captures trial-specific context
- **Issue**: May reduce generalizability across trials
- **Recommendation**: Evaluate impact on cross-trial performance

---

## 4. Feature Importance Analysis

### Data Sources

- Stability analysis v2 (30 iterations, sample sizes 30/40/50)
- Stability analysis v3 (18 iterations, sample sizes 60/80/100)
- Model comparison reports

### 4.1 Top Features (Consistently High Importance)

**Level1 Model** (from stability analysis):
1. `alignment_angle_mean_50`: ~36-40% importance (dominant)
2. `alignment_angle_mean_20`: ~15-27% importance
3. `alignment_angle`: ~3-4% importance
4. `head_speed_mean_50`: ~1.9-4.2% importance
5. `tail_speed_std_50`: ~2.2% importance
6. `tail_speed`: ~1.5-1.9% importance
7. `head_speed_std_50`: ~1.9-2.5% importance

**Raw Model** (from stability analysis):
1. `alignment_angle_mean_50`: ~35-40% importance
2. `alignment_angle_mean_20`: ~15-27% importance
3. `head_speed_mean_50`: ~3.3-10.3% importance
4. `alignment_angle`: ~3-4% importance
5. `head_speed_mean_20`: ~4.2% importance
6. `head_speed_std_50`: ~3.3% importance

**Key Insight**: Alignment angle features dominate, especially with larger temporal windows (50 frames).

### 4.2 Medium Importance Features

- `tail_speed_mean_50`: ~1.6-3.3%
- `tail_speed_mean_20`: ~1.6-2.7%
- `speed_ratio`: ~1.5% importance
- Position features: ~0.7-1.0% each
- Distance features: ~0.7-0.8% each
- Cumulative distances: ~0.5-1.0%

### 4.3 Low Importance Features

- `cross_sign`: ~0.04% (surprisingly low given its use in rule-based detection)
- `head_path_curvature`: ~0.04%
- Angular velocities: ~0.03%
- Many temporal features with small windows (5, 10 frames): <0.5%
- `alignment_angle_mean_5`: ~0.9%
- `alignment_angle_std_*`: Generally lower than mean features

**Key Insight**: Small temporal windows and instantaneous geometric features show low importance compared to longer-term temporal patterns.

---

## 5. Critical Review

### 5.1 Strengths

1. **Efficient Pre-computation**: 400x speedup through smart pre-computation
2. **Comprehensive Coverage**: 56 features capture multiple aspects of motion
3. **Temporal Context**: Multiple window sizes capture different time scales
4. **Gaussian Filtering**: Dramatically improves signal quality (sigma=4.6 optimal)
5. **Vectorized Operations**: Fast NumPy/Pandas operations

### 5.2 Weaknesses and Issues

#### 5.2.1 Feature Redundancy

**Issues**:
- `head_velocity_magnitude` = `head_speed` (identical calculation: `sqrt(vx² + vy²)`)
- `tail_velocity_magnitude` = `tail_speed` (identical calculation)
- Multiple window sizes may be redundant (5, 10 frames show low importance)
- Raw position features may not add value beyond distance features

**Impact**: 
- Wastes computation and memory
- May confuse feature importance analysis
- Increases model complexity without benefit

#### 5.2.2 Missing Features

**Critical Missing Features**:
1. **Acceleration**: Rate of speed change (head/tail acceleration, relative acceleration)
2. **Body Length Normalization**: Normalize distances by body length (important for different animals)
3. **Cross-sign Temporal Consistency**: Rolling window consistency of cross-sign (may explain low importance)
4. **Swap Persistence**: Count of consecutive swapped frames
5. **Relative Position**: Head-tail vector direction (normalized)

**Moderate Priority Missing Features**:
- Temporal min/max/range statistics (currently only mean/std)
- Tail path curvature (currently only head)
- Motion direction consistency features

#### 5.2.3 Feature Engineering Issues

**Angular Velocity**:
- Calculated from 2-frame differences (very noisy)
- Low importance (~0.03%) may be due to noise
- **Recommendation**: Use smoothing or longer window

**Path Curvature**:
- Only calculated for head, not tail
- Very low importance (~0.04%)
- **Recommendation**: Add tail curvature; investigate why importance is low

**Cross-sign**:
- Key feature in rule-based detection
- Very low importance in ML (~0.04%)
- **Hypothesis**: May need temporal consistency (persistence over time)
- **Recommendation**: Add cross-sign temporal features

#### 5.2.4 Scale and Normalization

**Issues**:
- Features not normalized before importance calculation
- Different scales may bias importance (e.g., distances vs. angles)
- Body length not normalized (affects distance features across animals)
- Position features are trial-specific (may reduce generalizability)

**Impact**: 
- Feature importance may be biased by scale
- Model may not generalize well across different animals/sizes

#### 5.2.5 Temporal Window Selection

**Issues**:
- Window sizes (5, 10, 20, 50) may not be optimal
- No adaptive window sizing based on trial characteristics
- Only mean/std statistics, missing min/max/range/median
- Small windows (5, 10) show low importance

**Recommendation**: 
- Evaluate optimal window sizes
- Consider adaptive windows based on trial length or motion characteristics
- Add more temporal statistics

---

## 6. Recommendations for Improvement

### 6.1 High Priority

#### 6.1.1 Remove Redundant Features

**Actions**:
1. Remove `head_velocity_magnitude` (redundant with `head_speed`)
2. Remove `tail_velocity_magnitude` (redundant with `tail_speed`)
3. Test removal of raw position features (`head_x`, `head_y`, `tail_x`, `tail_y`, `mid_x`, `mid_y`, `centroid_x`, `centroid_y`)
4. Evaluate necessity of small window sizes (5, 10 frames)

**Expected Impact**:
- Reduce feature count from 56 to ~47-48 features
- Faster training and inference
- Clearer feature importance analysis

#### 6.1.2 Add Missing Critical Features

**Actions**:
1. **Acceleration Features**:
   - `head_acceleration`: Rate of change of head speed
   - `tail_acceleration`: Rate of change of tail speed
   - `relative_acceleration`: Difference in acceleration

2. **Body Length Normalization**:
   - Calculate body length (mean head-tail distance)
   - Normalize distance features by body length
   - Add normalized distance features

3. **Cross-sign Temporal Consistency**:
   - Rolling window consistency of cross-sign
   - Count of consistent cross-sign in window
   - May explain why instantaneous cross-sign has low importance

4. **Swap Persistence Features**:
   - Count of consecutive frames with swap indicators
   - May help detect persistent swaps

**Expected Impact**:
- Better detection of swap patterns
- Improved generalization across animals
- Better understanding of temporal patterns

#### 6.1.3 Improve Existing Features

**Actions**:
1. **Better Angular Velocity**:
   - Use smoothing (moving average)
   - Longer window for calculation
   - May improve importance from 0.03% to higher values

2. **Add Tail Path Curvature**:
   - Currently only head curvature exists
   - May provide complementary information

3. **Add Temporal Statistics**:
   - Min, max, range, median for temporal windows
   - Currently only mean/std

4. **Normalize Features**:
   - Body length normalization for distances
   - Consider robust scaling

**Expected Impact**:
- More informative features
- Better generalization
- Improved feature importance

### 6.2 Medium Priority

#### 6.2.1 Feature Selection

**Actions**:
1. Analyze feature importance across all stability iterations
2. Remove consistently low-importance features (<0.1%)
3. Consider automated feature selection (recursive feature elimination)
4. Test model performance with reduced feature set

**Expected Impact**:
- Simpler models
- Faster training
- Potentially better generalization

#### 6.2.2 Temporal Feature Enhancement

**Actions**:
1. Add more temporal statistics (min, max, range, median)
2. Consider adaptive window sizes based on trial characteristics
3. Add features for temporal patterns (e.g., "speed increasing/decreasing")

**Expected Impact**:
- Richer temporal information
- Better capture of motion patterns

#### 6.2.3 Domain-Specific Features

**Actions**:
1. Features for collapsed keypoint detection
2. Features for tracking quality assessment
3. Features for motion direction consistency

**Expected Impact**:
- Better handling of edge cases
- Improved robustness

### 6.3 Low Priority

#### 6.3.1 Advanced Feature Engineering

**Actions**:
1. Polynomial features for key interactions
2. Feature interactions (e.g., `alignment_angle × speed_ratio`)
3. Frequency domain features (FFT of speed/angle signals)

**Expected Impact**:
- Potentially better performance
- More complex models

#### 6.3.2 Feature Normalization

**Actions**:
1. Body length normalization
2. Trial-specific normalization
3. Robust scaling (median/IQR instead of mean/std)

**Expected Impact**:
- Better generalization
- More robust models

---

## 7. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

**Goal**: Remove redundancy, add critical missing features

**Tasks**:
1. Remove redundant features (`velocity_magnitude`, test raw positions)
2. Add acceleration features
3. Add body length normalization

**Deliverable**: `features_v2.py` with ~47-50 features

### Phase 2: Feature Enhancement (3-5 days)

**Goal**: Improve existing features, add temporal enhancements

**Tasks**:
1. Improve angular velocity calculation
2. Add temporal min/max/range statistics
3. Add cross-sign consistency features
4. Add swap persistence features
5. Add tail path curvature

**Deliverable**: `features_v3.py` with enhanced features

### Phase 3: Feature Selection (2-3 days)

**Goal**: Optimize feature set based on importance

**Tasks**:
1. Analyze feature importance across all stability iterations
2. Remove consistently low-importance features
3. Test model performance with reduced feature set

**Deliverable**: Optimized feature set, performance comparison

### Phase 4: Advanced Features (1-2 weeks)

**Goal**: Domain-specific and advanced features

**Tasks**:
1. Domain-specific features (collapsed keypoints, tracking quality)
2. Feature interactions
3. Adaptive temporal windows

**Deliverable**: `features_v4.py` with advanced features

---

## 8. Expected Impact

### Performance

- **Training Speed**: Removing redundant features may improve training speed by 5-10%
- **Inference Speed**: Fewer features = faster predictions
- **Model Size**: Smaller feature set = smaller models

### Accuracy

- **Adding Missing Features**: Acceleration and body length normalization should improve detection accuracy
- **Improving Features**: Better angular velocity and temporal statistics should capture more patterns
- **Expected Improvement**: 1-3% F1 score improvement possible

### Interpretability

- **Reduced Feature Set**: Easier to understand model decisions
- **Clearer Importance**: Removing redundancy clarifies which features matter

### Generalization

- **Body Length Normalization**: Should improve cross-animal performance
- **Better Normalization**: More robust across different trial conditions

---

## 9. Metrics for Evaluation

### Model Performance

- F1-score, precision, recall (before/after changes)
- Sensitivity, specificity
- % Swaps Resolved
- % Frames Clean (pre/post correction)

### Feature Analysis

- Feature importance distribution
- Feature count (before/after)
- Feature importance stability across iterations

### Computational Performance

- Training time
- Inference time
- Feature extraction time
- Memory usage

### Model Characteristics

- Model size
- Overfitting indicators (train/val/test gaps)
- Generalization across trials

---

## 10. Code Examples

### Current Feature Extraction

```python
from swap_correction.ml.features import extract_all_frame_features_optimized

# Extract features with Gaussian filtering
features = extract_all_frame_features_optimized(
    trial_data,
    fps=30,
    apply_filtering=True,
    filter_sigma=4.6
)
# Returns: DataFrame with 56 features
```

### Redundant Features to Remove

```python
# These are redundant:
features['head_velocity_magnitude']  # = features['head_speed']
features['tail_velocity_magnitude']  # = features['tail_speed']

# These may be redundant:
features[['head_x', 'head_y', 'tail_x', 'tail_y', 
          'mid_x', 'mid_y', 'centroid_x', 'centroid_y']]
```

### Missing Features to Add

```python
# Acceleration (rate of speed change)
head_acceleration = np.diff(head_speed) * fps
tail_acceleration = np.diff(tail_speed) * fps

# Body length normalization
body_length = np.nanmean(head_tail_distance)
normalized_distance = head_tail_distance / body_length

# Cross-sign temporal consistency
cross_sign_consistency = rolling_window_consistency(cross_sign, window=20)
```

---

## 11. Version 3 Improvements

### 11.1 Overview

Version 3 (`features_v3.py`) builds on v2 with improved calculations and further feature optimization:

- **Improved Angular Velocity**: 3-point central difference method (more accurate, less noisy)
- **Added Tail Path Curvature**: Complement to head path curvature
- **Added Curvature Ratio**: `head_path_curvature / tail_path_curvature` (captures head vs tail motion differences)
- **Added Collapsed Keypoints**: Binary feature flagging tracking error frames
- **Removed Low-Value Features**: Standard deviation for small windows (5, 10) and velocity components

### 11.2 Improved Angular Velocity Calculation

**Previous (v2)**: 2-frame difference using points i-2, i-1, i
- Computed angle between vectors from i-2→i-1 and i-1→i
- Very noisy, low importance (~0.03%)

**New (v3)**: 3-point central difference using points i-1, i, i+1
- Computes angle between vectors from i-1→i and i→i+1
- More accurate derivative approximation
- Less noisy, should improve feature importance
- First and last frames are NaN (need 3 points)

**Expected Impact**: Reduced noise should increase feature importance and improve model performance

### 11.3 New Features

#### Tail Path Curvature
- **Feature**: `tail_path_curvature`
- **Calculation**: Same method as head path curvature (angle change between consecutive velocity vectors)
- **Rationale**: Head and tail may have different curvature patterns; tail curvature provides complementary information

#### Curvature Ratio
- **Feature**: `head_tail_curvature_ratio`
- **Calculation**: `head_path_curvature / max(tail_path_curvature, 0.001)`
- **Epsilon**: 0.001 to handle division by zero
- **Rationale**: Head typically has more contorted path than tail; ratio captures this relationship

#### Collapsed Keypoints
- **Feature**: `collapsed_keypoints` (binary: 0 or 1)
- **Calculation**: Uses `detect_collapsed_keypoints()` with tolerance 0.05mm
- **Detection**: Flags frames where:
  - Head/centroid collapse (tracking can't resolve head)
  - Tail/centroid collapse (tracking can't resolve tail)
  - Three or more keypoints collapse
- **Rationale**: Frames with collapsed keypoints indicate tracking errors and may affect swap detection reliability

### 11.4 Removed Features

#### Standard Deviation for Small Windows
- **Removed**: `head_speed_std_5`, `head_speed_std_10`
- **Removed**: `tail_speed_std_5`, `tail_speed_std_10`
- **Removed**: `alignment_angle_std_5`, `alignment_angle_std_10`
- **Kept**: Mean features for all windows (5, 10, 20, 50)
- **Kept**: Standard deviation for larger windows (20, 50)
- **Rationale**: Small window std features show low importance; larger windows capture more meaningful variation

#### Velocity Component Features
- **Removed**: `head_velocity_x`, `head_velocity_y`
- **Removed**: `tail_velocity_x`, `tail_velocity_y`
- **Kept**: Speed features (magnitude), relative velocity
- **Note**: Velocity components still computed internally for `relative_velocity_magnitude` calculation
- **Rationale**: Velocity components are less informative than speed magnitude; reduce redundancy

### 11.5 Feature Count

- **v2**: 46 features
- **v3**: ~39 features (46 - 10 removed + 3 new = 39)
- **Reduction**: 7 fewer features than v2, 17 fewer than original (56)

### 11.6 Expected Impact

#### Performance
- **Improved Angular Velocity**: Should increase importance from ~0.03% to higher values
- **New Features**: Curvature ratio and collapsed keypoints may capture additional patterns
- **Feature Reduction**: Faster training and inference, potentially better generalization

#### Accuracy
- **Better Angular Velocity**: More accurate derivative should improve detection
- **Curvature Ratio**: May help distinguish head vs tail motion patterns
- **Collapsed Keypoints**: May help model identify unreliable frames

#### Generalization
- **Fewer Features**: Reduced model complexity may improve cross-trial performance
- **Better Calculations**: More accurate features should improve robustness

### 11.7 Comparison Strategy

Results from v3 should be compared with:
1. **Original features (56 features)**: Baseline performance
2. **features_v2 (46 features)**: Impact of initial feature reduction
3. **features_v3 (39 features)**: Impact of improved calculations and further optimization

## 12. Analysis of v2/v3 Results and v4 Recommendations

### 12.1 Critical Findings from Feature Importance Comparison

**Feature Importance Concentration Problem**:
- `alignment_angle_mean_50` accounts for 50.75% importance (v3 level1), 42.34% (v3 raw)
- Combined with `alignment_angle_mean_20`: ~65-70% of total importance
- Concentration INCREASED from v2 (48.22%) to v3 (50.75%)
- **Implication**: Model over-relies on a single feature type, suggesting potential overfitting and missing complementary information

**V3 New Features Underperformed**:
- `collapsed_keypoints`: 0% (level1), 0.22% (raw) - essentially unused
- `head_tail_curvature_ratio`: 0.22% (level1), 0.37% (raw) - very low but kept (encodes more information than raw curvatures)
- `tail_path_curvature`: 0.08% (level1), 0.18% (raw) - very low
- `head_path_curvature`: Not included in v3, but raw curvatures less informative than ratio

### 12.2 Learning Curve Analysis Insights

**V3 Overfitting Indicators**:
- V3 level1 model shows performance DECLINE with more data
  - Best performance at 50% data: F1 = 0.9542
  - Performance at 100% data: F1 = 0.9317 (-0.0225)
- V2 shows consistent improvement: 0.9528 → 0.9566 (+0.0038)
- **Implication**: V3 may be overfitting; new features or reduced feature set causing instability

**Data Efficiency**:
- V2 benefits from more training data (positive slope)
- V3 shows negative returns with more data (overfitting)
- **Recommendation**: V2 is more robust and should be used as baseline for v4

### 12.3 Stability Analysis Implications

**Feature Reduction Impact**:
- Removing features increased importance concentration:
  - Original: Mean 0.0179, Max 0.3745, Std 0.0600
  - V2: Mean 0.0217, Max 0.4822, Std 0.0718
  - V3: Mean 0.0256, Max 0.5075, Std 0.0824
- **Implication**: Fewer features = higher concentration (expected), but concentration may be TOO high, indicating over-reliance

**Performance Comparison**:
- V2 improved over original: +3.35% F1 (level1), +3.39% (raw)
- V3 vs V2: Essentially no improvement (slight regression for level1)
- **Implication**: Feature reduction helped, but v3 changes didn't improve performance

### 12.4 Proposed v4 Improvements

**Phase 1: Quick Wins (Remove Underperformers)**:
1. Remove `collapsed_keypoints` (0-0.22% importance)
2. Remove `head_path_curvature` and `tail_path_curvature` (raw curvatures, less informative)
3. **Keep** `head_tail_curvature_ratio` (0.22-0.37% - encodes relative relationship)
4. Remove window size 5 features (consistently low importance)

**Phase 2: High-Value Additions**:
1. **Acceleration features**:
   - `head_acceleration`: Rate of speed change
   - `tail_acceleration`: Rate of speed change
   - `relative_acceleration`: Difference in acceleration
   - **Rationale**: Captures motion dynamics, may complement speed features

2. **Body length normalization**:
   - Normalize distance features by mean body length
   - Keep same feature names (`head_tail_distance`, etc.)
   - **Rationale**: Improves generalization across different animals/sizes

**Expected Feature Count**:
- V3: 39 features
- V4: ~40-42 features (remove 3, add 3-6)

### 12.5 Success Metrics for v4

**Performance Targets**:
- F1-score: Maintain or improve upon v2 (0.8920 level1, 0.9381 raw)
- Feature importance distribution: Reduce concentration (target: max feature <45%)
- Learning curve: Show improvement or plateau (not decline) with more data
- Stability: Maintain or improve CV (<0.04)

**Feature Quality Targets**:
- No features with 0% importance
- Mean feature importance: 0.022-0.025 (balanced distribution)
- Top 3 features: <60% combined importance (vs current ~70%)

### 12.6 Version 4 Implementation

**Phase 1 Changes**:
- Removed `collapsed_keypoints` (underperformed)
- Removed `head_path_curvature` and `tail_path_curvature` as features (kept calculation for ratio)
- Kept `head_tail_curvature_ratio` (more informative than raw curvatures)
- Removed window size 5 features (changed from [5, 10, 20, 50] to [10, 20, 50])

**Phase 2 Changes**:
- Added `head_acceleration`, `tail_acceleration`, `relative_acceleration` features
- Added body length normalization to distance features (normalized by mean body length, same names)

**Testing Strategy**:
- Test on same 6 iterations (007-012) as v2 and v3
- Compare against v2 (baseline) and v3
- Evaluate using success metrics above

## References

- Stability Analysis v2: 30 iterations, sample sizes 30/40/50
- Stability Analysis v3: 18 iterations, sample sizes 60/80/100
- Model Comparison Reports: `ml_analysis/model_comparison_report.md`
- Feature Importance Files: `ml_models/feature_importance.csv`, `ml_models_raw/feature_importance.csv`
- Features v2 Testing: `stability_analysis_v3_features_v2/`
- Features v3 Testing: `stability_analysis_v3_features_v3/`
- Features v4 Testing: `stability_analysis_v3_features_v4/`


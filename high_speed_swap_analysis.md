# Analysis of High-Speed Swap Correction Limitations

## Overview
This document analyzes why the current swap correction system has difficulty handling high-speed swap events, based on code examination of `swap_correct.py` and its dependencies.

## Key Limitations

### 1. Speed-Based Detection Thresholds
- The system uses fixed speed thresholds (24 mm/s) to detect discontinuities
- During high-speed events, legitimate movements may exceed these thresholds
- This causes the system to either:
  a) Miss actual swaps because they don't trigger the threshold
  b) Flag legitimate fast movements as potential swaps

### 2. Overlap Detection Issues
- The overlap detector (`detect_overlaps`) uses a fixed distance threshold (`OVERLAP_THRESH = 0`)
- During high-speed movement:
  - Head and tail points may not fully overlap due to motion blur
  - The distance between points may remain above threshold despite being swapped
  - This causes the system to miss swaps that don't result in perfect overlaps

### 3. Movement Pattern Assumptions
The corrector makes several assumptions that break down at high speeds:

1. **Delta Mismatches**
   - Assumes head and tail should move at similar speeds (`detect_delta_mismatches`)
   - During rapid turns or accelerations, this assumption is invalid
   - Head may legitimately move faster than tail during quick movements

2. **Sign Reversals**
   - Uses fixed angle thresholds (π/2 for general, π/4 for overlaps)
   - Fast movements can cause legitimate large angle changes
   - System may misclassify rapid turns as swaps

3. **Movement Ratios**
   - `get_swapped_segments` uses fixed thresholds (0.95-1.05) for movement ratios
   - High-speed movements often have different head/tail movement patterns
   - These ratios become unreliable during rapid acceleration/deceleration

### 4. Temporal Resolution Limitations

1. **Frame Rate Dependencies**
   - Many detection methods don't properly account for frame rate variations
   - High-speed events may occur between frames
   - Position interpolation may be inaccurate during rapid movements

2. **Fixed Window Sizes**
   - Filtering methods use fixed window sizes
   - High-speed events require smaller windows to maintain temporal precision
   - Current windows may over-smooth rapid movements

### 5. Validation Weaknesses

1. **Ambiguity Resolution**
   - `_merge_ambiguous_flags` and `_swap_following_ambiguous_flags` use simple heuristics
   - These heuristics don't account for high-speed movement patterns
   - May incorrectly resolve ambiguous cases during rapid movement

2. **Alignment Checks**
   - `_get_alignment_angles` assumes relatively stable orientations
   - High-speed movements can cause rapid orientation changes
   - Alignment-based validation may fail during quick turns

## Recommendations for Improvement

1. **Adaptive Thresholds**
   - Implement speed-dependent thresholds for overlap and discontinuity detection
   - Consider local movement patterns when setting thresholds
   - Use acceleration and jerk information to adjust sensitivity

2. **Enhanced Movement Analysis**
   - Add specific detection patterns for high-speed scenarios
   - Consider temporal sequences rather than single-frame metrics
   - Implement momentum-based prediction for expected positions

3. **Multi-Scale Processing**
   - Use different window sizes for different movement speeds
   - Implement hierarchical analysis for different temporal scales
   - Consider both local and global movement patterns

4. **Improved Validation**
   - Add specific validation rules for high-speed segments
   - Implement confidence scoring for swap detection
   - Consider multiple frames before/after potential swaps

5. **Machine Learning Integration**
   - Train a classifier specifically for high-speed swap detection
   - Use features that capture dynamic movement patterns
   - Incorporate behavioral context in decision making

## Next Steps
1. Implement adaptive thresholds based on local speed
2. Add specific high-speed movement pattern detection
3. Enhance validation for rapid movements
4. Test improvements on high-speed swap dataset
5. Consider adding ML-based classification for ambiguous cases 
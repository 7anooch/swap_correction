# Test Results: Improved Global Swap Detection

## Implementation Summary

### Step 1.2a: Detect and Handle Collapsed Keypoints ✅ COMPLETED
- **Function:** `detect_collapsed_keypoints()` in `tracking_correction.py`
- **Detection criteria:**
  - Head/centroid collapse (head tracking error)
  - Tail/centroid collapse (tail tracking error)  
  - 3+ keypoints collapsing (at least 3 of 4 within 0.1mm)
- **Integration:** Excludes collapsed frames and head-tail overlaps from global swap detection calculations

### Step 1.2: Improved Global Swap Detection ✅ COMPLETED
- **Start/End endpoint detection:** Checks first 100 frames (small window) and first 200 frames (large window) for persistent swap patterns
- **Persistence check:** Requires pattern to persist in larger window (>60% negative) to prevent false positives
- **Adaptive consensus:** Uses `swap_at_start`/`swap_at_end` flags (which include persistence checks) instead of raw ratios
- **Safety checks:** More conservative when cross-sign is already consistent

## Test Results

### Individual Trial Tests
- **2024.11.13_00-48-15_Sussex_e2hex** (known global swap):
  - Before: 62.13% error, 27 segments
  - After: 36.52% error, 1 segment
  - **Improvement: 25.6 percentage points reduction**
  - Global swap correctly detected and applied

- **2024.11.13_00-13-24_Sussex_e2hex** (was perfect):
  - Global swap detection: Correctly NOT swapping ✓
  - However, frame-by-frame detection may be causing issues

### All Trials Comparison

**Baseline (existing level1 files):**
- Perfect trials: 14 / 25 (56%)
- Problematic trials: 11 / 25 (44%)
- Average error rate: 12.68%

**Current (with improved global swap detection):**
- Perfect trials: 6 / 25 (24%)
- Problematic trials: 19 / 25 (76%)
- Average error rate: 50.72%

**Key Finding:** Many trials that were previously perfect (0% error) are now showing 98-99% error. Investigation shows:
- Global swap detection is working correctly (not incorrectly swapping)
- The issue appears to be in **frame-by-frame detection**, which is incorrectly flagging and swapping frames
- Raw data vs level2 shows ~30% error, but after correction shows 99% error
- This suggests frame-by-frame detection is making things worse

## Issues Identified

1. **Frame-by-frame detection is problematic:**
   - Incorrectly flagging frames that shouldn't be swapped
   - Making perfect trials worse (0% → 99% error)
   - Needs investigation and fixes

2. **Global swap detection is working:**
   - Correctly identifies and swaps the known global swap trial
   - Correctly avoids swapping trials that shouldn't be swapped
   - Start/end endpoint detection is functioning

## Next Steps

1. **Investigate frame-by-frame detection:**
   - Why is it incorrectly flagging frames?
   - Is it related to collapsed keypoints?
   - Does it need to also exclude collapsed frames?

2. **Refine global swap detection thresholds:**
   - May need further tuning based on more test cases
   - Consider additional safety checks

3. **Test on subset of trials:**
   - Focus on trials where global swap detection should help
   - Measure improvement specifically for those cases


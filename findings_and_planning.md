# Swap Correction Algorithm Improvements
## Findings and Planning Document

**Date:** 2024  
**Goal:** Eliminate need for manual corrections (make level1.csv and level2.csv effectively identical)

---

## Executive Summary

**Current Status:**
- 14 out of 25 trials (56%) are perfect - algorithms work well for most cases
- 11 trials have errors: 1-21 swap segments each, 11-41% error rate
- Main issue: **Segment-level detection rate is 0-10%** - algorithms detect individual frames but miss contiguous segments

**Critical Findings:**
1. **Global swap detection fails when speeds are similar** - must use cross-sign consistency instead
2. **Cross-sign is the most reliable indicator** - works even when speeds are similar
3. **Frame-by-frame detection is too sparse** - detects 8-20 frames but misses 1-3000 frame segments
4. **Long segments (1000+ frames) are completely missed** - need window-based analysis

**Priority Fixes:**
1. **CRITICAL:** Add cross-sign consistency check to global swap detection
2. **HIGH:** Implement cross-sign-based segment detection (sliding window)
3. **HIGH:** Fix segment formation from detected frames (expand and merge)
4. **MEDIUM:** Fix speed ratio calculation bug (affects reporting, not detection)

**Expected Impact:**
- Global swap detection: 0% → 100% (currently fails on similar-speed cases)
- Segment detection rate: 0-10% → >90%
- Error rate: 11-41% → <1% for problematic trials
- Manual corrections: Eliminated for >95% of cases

---

## Step 1 Analysis Results

### 1. Speed Ratio Calculation Bug

**Issue Identified:**
- Maximum ratio error: **47,492,246.50** (extremely large values)
- 92,750 frames with near-zero tail speed (< 0.01 mm/s)
- 13,295 frames with extreme ratio errors (>1000)

**Root Cause:**
- When tail speed is very small (near zero), even with `1e-6` addition, ratios become extreme
- In swapped segments, tail may legitimately have near-zero speed (animal stopped, tail stationary)
- The `1e-6` epsilon is too small for practical use
- Error calculation: `h2_ratio - h1_ratio` amplifies the problem when both ratios are large

**Location:**
- `error_analysis.py` line 398-400: `h1_ratio = h1_speed / (t1_speed + 1e-6)`
- This is used for error reporting, not detection, but indicates underlying issues

**Fix Required:**
- Use larger epsilon or cap ratios at reasonable maximum (e.g., 100)
- Consider using log ratios or alternative metrics
- Filter out frames with near-zero speeds before calculating ratios

### 2. Segment-Level Pattern Analysis

**Key Findings:**

#### Global Swap Trials
- **1 trial** with single large segment >30% of trial:
  - `2024.11.13_00-48-15_Sussex_e2hex`: 1 segment of 3287 frames (36% of trial)
  - Cross-sign match rate: 0.631 (low, indicating swap)
  - **Current global swap detection WORKS for this case** (detected correctly)
  - However, detection only found 12 frames, not the entire segment

#### Multi-Segment Trials
- **10 trials** with multiple segments (1-21 segments each)
- Segment lengths vary: 1-3128 frames
- Examples:
  - `2024.11.13_22-30-49_Sussex_e2hex`: 21 segments, max 1196 frames
  - `2024.11.13_01-12-20_Sussex_e2hex`: 3 segments, max 3128 frames (one very long)
  - `2024.11.13_00-19-29_Sussex_e2hex`: 18 segments, max 1509 frames

#### Perfect Trials
- **14 out of 25 trials (56%) are perfect** - algorithms work well!
- These trials have 0% error rate, 0 segments
- Suggests algorithms are fundamentally sound, but fail on specific patterns

### 3. Detection Failure Analysis

**Critical Finding: Segment Detection Rate is Near Zero**

Analysis of 5 problematic trials:
- **Trial 1:** 4 GT segments, 10 detected frames → **0% segment detection rate**
- **Trial 2:** 18 GT segments, 20 detected frames → **5.56% segment detection rate**
- **Trial 3:** 10 GT segments, 16 detected frames → **10% segment detection rate**
- **Trial 4:** 1 GT segment (3287 frames), 12 detected frames → **0% segment detection rate**
- **Trial 5:** 3 GT segments, 10 detected frames → **0% segment detection rate**

**Why Detection Fails:**

1. **Frame-by-frame detection is too sparse:**
   - Detects 8-20 individual frames
   - But ground truth has 1-21 segments (each segment is 1-3000+ frames)
   - Detected frames don't form coherent segments that match ground truth

2. **Global swap detection works but doesn't correct:**
   - For the 3287-frame global swap, detection correctly identifies it should be swapped
   - But `correct_global_swap()` only swaps if `mean_tail_speed > mean_head_speed`
   - May miss cases where speeds are similar but swap still exists

3. **Long segments between overlaps:**
   - Current detection relies on overlaps to anchor segments
   - Long segments (1000+ frames) between overlaps are completely missed
   - Segment-based validation is disabled, so these never get caught

4. **Detection methods are too conservative:**
   - Minimum delta mismatch: Only flags when minimum is clearly wrong
   - Sign reversal: Requires π/2 threshold, misses gradual changes
   - Overlap-based: Only works near overlaps

### 4. Comparison: Perfect vs Problematic Trials

**Perfect Trials (14):**
- 0% error rate
- 0 swap segments
- Algorithms work correctly

**Problematic Trials (11):**
- 11-41% error rate
- 1-21 swap segments
- Average: ~8 segments per trial
- Segment lengths: 1-3287 frames

**Key Difference:**
- Perfect trials likely have:
  - Few or no overlaps
  - Clear forward motion patterns
  - Consistent head-leading behavior
- Problematic trials likely have:
  - Many overlaps (confusing detection)
  - Complex motion patterns
  - Periods of stationary behavior
  - Global misidentification at start

### 5. Detailed Analysis of Specific Trials

#### Global Swap Trial: `2024.11.13_00-48-15_Sussex_e2hex`
- **Segment:** 1 segment of 3287 frames (36% of trial)
- **Cross-sign analysis:**
  - Raw vs Level2 match: **0.617** (should be ~1.0 if correct)
  - Level1 vs Level2 match: **0.631** (swap not fully corrected)
  - **Conclusion:** Cross-sign clearly indicates global swap
- **Speed analysis:**
  - Raw: head=6.274 mm/s, tail=5.978 mm/s (very similar!)
  - Level1: head=5.520 mm/s, tail=5.411 mm/s
  - Level2: head=5.650 mm/s, tail=5.280 mm/s
  - **Current detection:** `mean_tail (5.978) > mean_head (6.274)` → **False** (no swap)
  - **Should swap:** No (speeds are similar, but cross-sign says swap)
  - **Problem:** Speed-based detection fails when speeds are similar, but cross-sign is reliable

#### Multi-Segment Trial: `2024.11.13_00-19-29_Sussex_e2hex`
- **Segments:** 18 segments, lengths 1-1509 frames
- **Segment 1 (1509 frames - very long!):**
  - Cross-sign match: **0.007** (almost completely wrong)
  - Level1 speeds: head=5.526, tail=5.244
  - Level2 speeds: head=5.244, tail=5.526 (swapped!)
  - **Insight:** When speeds are similar, cross-sign is the only reliable indicator
- **Segment 2 (143 frames):**
  - Cross-sign match: **0.000** (completely wrong)
  - Level1: head=7.391, tail=7.621
  - Level2: head=7.621, tail=7.391 (swapped!)
  - **Pattern:** Consistent - cross-sign mismatch indicates swap even when speeds are similar

**Key Finding:** Cross-sign consistency is more reliable than speed for detecting swaps when speeds are similar.

---

## Updated Implementation Plan

Based on Step 1 findings, here's the revised plan:

### Phase 1: Critical Fixes (Week 1) - HIGHEST PRIORITY

#### 1.1 Fix Speed Ratio Calculation
- **Issue:** Extreme values due to near-zero tail speeds
- **Fix:** 
  - Use larger epsilon (e.g., 0.1 mm/s minimum)
  - Cap ratios at reasonable maximum (e.g., 100)
  - Consider using median instead of mean for ratios
  - Filter out stationary frames before ratio calculation

#### 1.2 Improve Global Swap Detection - CRITICAL FIX
- **Current:** Only checks `mean_tail_speed > mean_head_speed`
- **Problem:** 
  - **Fails when speeds are similar** (e.g., head=6.274, tail=5.978 → no swap detected)
  - But cross-sign clearly indicates swap (match rate = 0.617, should be ~1.0)
- **Solution:**
  - **Primary:** Cross-sign consistency check - if cross-sign match < 0.7 → global swap
  - **Secondary:** Speed check - if `mean_tail_speed > mean_head_speed` → swap
  - **Tertiary:** Head-leading motion check (for forward motion segments)
  - Use consensus: if 2 out of 3 indicate swap → swap entire trajectory
  - **Apply BEFORE frame-by-frame detection** (may eliminate need for it in many cases)

#### 1.3 Fix Segment Formation from Detected Frames
- **Current:** Detects 8-20 frames but doesn't form proper segments
- **Problem:** Detected frames are sparse, don't match ground truth segments
- **Solution:**
  - After detection, expand flagged frames to form segments:
    - If frame N is flagged, check frames N-10 to N+10
    - Merge nearby flagged regions (within 50 frames)
    - Use temporal consistency: if 3+ frames in 10-frame window suggest swap, flag entire window

### Phase 2: Long Segment Detection (Week 2) - HIGH PRIORITY

#### 2.1 Cross-Sign Consistency Tracker - HIGHEST PRIORITY
- **Why critical:** Cross-sign is the most reliable indicator when speeds are similar
- **Evidence:** 
  - Global swap: cross-sign match = 0.617 (should swap)
  - Long segment (1509 frames): cross-sign match = 0.007 (clearly swapped)
- **Implementation:**
  - Calculate cross-sign match rate over entire trajectory (baseline check)
  - If overall match < 0.7 → likely global swap, swap entire trajectory
  - Sliding window (100-200 frames) for segment-level detection
  - Flag windows where match rate < 0.6 (indicates consistent swap)
  - Merge overlapping flagged windows into segments
  - **Priority over speed-based methods** when speeds are similar

#### 2.2 Velocity-Based Segment Detection
- **Implementation:**
  - Calculate head/tail velocity ratios over windows (not single frames)
  - Flag windows where tail velocity consistently > head velocity
  - Use median/percentile thresholds (more robust than mean)
  - Minimum window size: 50 frames

#### 2.3 Trajectory Smoothness Analysis
- **Implementation:**
  - Calculate local tortuosity for head vs tail over windows
  - Flag regions where tail trajectory is smoother than head (suggests swap)
  - Window size: 100 frames

### Phase 3: Enable Segment-Based Validation (Week 3) - HIGH PRIORITY

#### 3.1 Investigate Why Disabled
- **Current:** VALIDATE=False, "not recommended"
- **Action:** Test on problematic trials to understand failures
- **Hypothesis:** Thresholds too strict, or fails on long segments

#### 3.2 Fix and Improve
- Lower `minTime` threshold (currently 1 second = 30 frames)
- Multi-metric approach: combine alignment + speed + distance
- Handle cases with no overlaps (check entire trajectory)
- Better handling of very long segments (>1000 frames)

#### 3.3 Enable and Tune
- Set VALIDATE=True by default
- Tune parameters based on error analysis
- Test on all problematic trials

### Phase 4: Integration and Testing (Week 4)

- Integrate all improvements
- Run on all 25 test trials
- Measure segment-level detection performance
- Compare new level1 to level2
- Iterate based on results

### Phase 5: Refinement (Week 5)

- Post-processing improvements
- Iterative correction
- Final tuning
- Ensure no regressions on perfect trials

---

## Success Metrics (Updated)

### Segment-Level Detection
- **Current:** 0-10% segment detection rate
- **Target:** > 90% segment detection rate
- **Global swaps:** 100% detection (currently works but needs to actually correct)
- **Long segments (>500 frames):** > 80% detection

### Error Reduction
- **Current:** 11 trials with errors (1-21 segments, 11-41% error rate)
- **Target:** < 3 trials with errors
- **Segment elimination:**
  - Eliminate all segments >1000 frames
  - Reduce average segment count from 1-21 to < 2 per problematic trial
- **Error rate:** Reduce from 11-41% to < 1% for problematic trials

### Manual Correction Elimination
- **Current:** level1 differs from level2 by 11-41% in problematic trials
- **Target:** level1.csv matches level2.csv for > 99% of frames
- **Position error:** Remaining differences < 0.3mm
- **No systematic errors:** No consistent patterns of missed swaps

---

## Key Insights from Step 1

1. **Algorithms work for 56% of trials** - we're on the right track, just need to handle edge cases

2. **Global swap detection has a critical flaw:**
   - **Finding:** For global swap trial (3287 frames), speeds are nearly identical (head=6.274, tail=5.978)
   - **Problem:** Speed-based detection (`mean_tail > mean_head`) fails when speeds are similar
   - **Solution:** Must use cross-sign consistency (raw vs level2 match = 0.617, clearly indicates swap)
   - **Current status:** Detection logic says "don't swap" but should swap based on cross-sign

3. **Cross-sign is the key metric for global swaps:**
   - Global swap trial: cross-sign match = 0.617 (should be ~1.0 if correct)
   - Multi-segment trial, segment 1 (1509 frames): cross-sign match = 0.007 (almost completely wrong)
   - **Insight:** Cross-sign consistency is more reliable than speed for detecting swaps when speeds are similar

4. **Main problem: Frame-by-frame detection doesn't form segments**
   - Detects individual frames but misses the contiguous nature of swaps
   - Need segment-level thinking, not frame-level
   - Detected frames are sparse (8-20 frames) but ground truth has contiguous segments (1-3000+ frames)

5. **Long segments are the biggest issue:**
   - 1000+ frame segments are completely missed
   - Example: 1509-frame segment with cross-sign match = 0.007 (almost completely wrong)
   - These segments have consistent cross-sign mismatch throughout
   - Need window-based analysis, not single-frame

6. **Speed-based detection fails when speeds are similar:**
   - Multi-segment trial shows: when head and tail speeds are similar, speed-based methods fail
   - But cross-sign clearly shows the swap (match rate < 0.01)
   - Need to prioritize cross-sign over speed for detection

7. **Speed ratio bug is a red herring:**
   - Affects error reporting, not detection
   - But indicates need for better handling of stationary periods

---

## Step 1 Completion Summary

**Completed:**
- ✅ Analyzed speed ratio calculation bug (found extreme values due to near-zero speeds)
- ✅ Identified segment-level patterns (1 global swap, 10 multi-segment trials)
- ✅ Analyzed detection failures (0-10% segment detection rate)
- ✅ Compared perfect vs problematic trials (14 perfect, 11 problematic)
- ✅ Detailed analysis of specific trials (global swap and multi-segment)
- ✅ Identified critical finding: cross-sign is more reliable than speed when speeds are similar

**Key Discoveries:**
1. Global swap detection fails when speeds are similar (head=6.274, tail=5.978) but cross-sign clearly indicates swap (match=0.617)
2. Long segments (1509 frames) have cross-sign match < 0.01, indicating consistent swap
3. Frame-by-frame detection is too sparse (8-20 frames) to catch contiguous segments (1-3000+ frames)
4. Cross-sign consistency is the most reliable indicator for swaps when speeds are similar

**Next Steps:**
1. **Step 2 (Immediate):**
   - Fix speed ratio calculation bug (affects reporting)
   - **CRITICAL:** Add cross-sign consistency check to global swap detection
   - Fix segment formation from detected frames (expand and merge)

2. **Step 3 (Short-term):**
   - Implement cross-sign-based segment detection (sliding window)
   - Add velocity-based segment detection (windowed)
   - Enable and fix segment-based validation

3. **Step 4-5 (Medium-term):**
   - Integration and testing on all 25 trials
   - Measure improvements (target: >90% segment detection rate)
   - Refinement based on results

---

## Files to Create/Modify

### New Files
- `findings_and_planning.md` (this file) - Planning and findings documentation
- `swap_correction/step1_analysis.py` - Analysis script (already created)

### Files to Modify
1. `swap_correction/tracking_correction.py`
   - Improve `correct_global_swap()`
   - Add segment-level detection functions
   - Fix segment formation logic

2. `swap_correction/error_analysis.py`
   - Fix speed ratio calculation
   - Add segment-level analysis functions

3. `swap_correction/metrics.py`
   - Add windowed metric calculations

4. `config.yaml`
   - Document new parameters

---

## Testing Strategy

1. **Unit Tests:**
   - Test global swap detection on known global swap trials
   - Test segment formation logic
   - Test windowed metrics

2. **Integration Tests:**
   - Run on all 25 test trials
   - Measure segment-level detection performance
   - Compare to ground truth

3. **Validation:**
   - Use error_analysis suite to measure improvements
   - Track segment-level metrics
   - Ensure perfect trials stay perfect


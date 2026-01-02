# Performance Metric Calculations

This document provides detailed explanations of how each performance metric is calculated in the ML swap detection evaluation.

## Overview

The evaluation metrics are computed by comparing model predictions against ground truth labels. For each trial, we have:
- **Ground Truth (GT)**: Binary array indicating which frames are swapped (1) or not swapped (0) according to manual correction (level2.csv)
- **Predictions**: Binary array indicating which frames the model predicts as swapped (1) or not swapped (0)

From these, we compute a confusion matrix:
- **TP (True Positives)**: Frames correctly predicted as swapped
- **FP (False Positives)**: Frames incorrectly predicted as swapped (not actually swapped)
- **FN (False Negatives)**: Frames incorrectly predicted as not swapped (actually swapped)
- **TN (True Negatives)**: Frames correctly predicted as not swapped

## Core Metrics

### Precision
**Formula**: `TP / (TP + FP)`

**Interpretation**: Of all frames predicted as swapped, what fraction are actually swapped?

**Range**: 0.0 to 1.0 (higher is better)

**Example**: If the model predicts 100 frames as swapped, and 80 of them are actually swapped, precision = 0.80

### Recall (Sensitivity)
**Formula**: `TP / (TP + FN)`

**Interpretation**: Of all frames that are actually swapped, what fraction did the model detect?

**Range**: 0.0 to 1.0 (higher is better)

**Example**: If there are 100 swapped frames in ground truth, and the model detects 90 of them, recall = 0.90

**Note**: Recall and Sensitivity are identical metrics.

### Specificity
**Formula**: `TN / (TN + FP)`

**Interpretation**: Of all frames that are actually not swapped, what fraction did the model correctly identify as not swapped?

**Range**: 0.0 to 1.0 (higher is better)

**Example**: If there are 1000 non-swapped frames, and the model correctly identifies 950 as not swapped, specificity = 0.95

### F1-Score
**Formula**: `2 * (Precision * Recall) / (Precision + Recall)`

**Interpretation**: Harmonic mean of precision and recall, providing a balanced metric.

**Range**: 0.0 to 1.0 (higher is better)

## Percentage Metrics

### % Swaps Resolved
**Formula**: `(TP / (TP + FN)) * 100`

**Interpretation**: Percentage of ground truth swaps that were correctly detected. This is equivalent to Recall/Sensitivity expressed as a percentage.

**Range**: 0% to 100% (higher is better)

**Example**: 
- If there are 100 swapped frames (TP + FN = 100)
- And the model correctly detects 99 of them (TP = 99)
- Then % Swaps Resolved = 99%

**Important Note**: This metric only considers swaps (TP and FN). It does not account for false positives (FP). A model can have 99% swaps resolved but still have a lower % Frames Clean Post if it makes false positive predictions.

### % Frames Clean (Pre-correction)
**Formula**: `((Total Frames - GT Swapped Frames) / Total Frames) * 100`

**Interpretation**: Percentage of frames that are NOT swapped in the raw/level1 data (before any correction).

**Range**: 0% to 100% (higher indicates less swapping in input data)

**Example**:
- If a trial has 1000 frames total
- And 200 frames are swapped according to ground truth
- Then % Frames Clean Pre = (1000 - 200) / 1000 * 100 = 80%

**Note**: This metric describes the input data quality, not model performance.

### % Frames Clean (Post-correction)
**Formula**: `((TP + TN) / Total Frames) * 100`

**Interpretation**: Percentage of frames that are correctly classified. This is equivalent to Accuracy.

**Range**: 0% to 100% (higher is better)

**Example**:
- If a trial has 1000 frames total
- And the model correctly classifies 980 frames (TP + TN = 980)
- Then % Frames Clean Post = 98%

**Important Note**: This metric considers ALL frames (TP, TN, FP, FN). False positives (FP) reduce this metric even if all swaps are resolved.

## Why % Swaps Resolved and % Clean Post Can Differ

It is **normal and expected** for these metrics to differ. Here's why:

### Example Scenario

Consider a trial with:
- **Total frames**: 1000
- **Ground truth swaps**: 100 frames (10% of trial)
- **Model predictions**: 110 frames predicted as swapped

**Confusion Matrix**:
- **TP**: 99 (correctly detected swaps)
- **FN**: 1 (missed swap)
- **FP**: 11 (false positives - predicted as swapped but not actually swapped)
- **TN**: 889 (correctly identified as not swapped)

**Calculations**:
- **% Swaps Resolved** = TP / (TP + FN) * 100 = 99 / (99 + 1) * 100 = **99%**
- **% Frames Clean Post** = (TP + TN) / Total * 100 = (99 + 889) / 1000 * 100 = **98.8%**

**Explanation**: 
- The model resolved 99% of the swaps (only 1 missed)
- But it also made 11 false positive predictions
- These false positives reduce the overall accuracy (% Clean Post)
- The difference: 99% - 98.8% = 0.2% reflects the false positive rate

### Key Insight

**% Swaps Resolved** focuses only on detecting existing swaps (TP vs FN).
**% Frames Clean Post** considers the entire classification (TP + TN vs FP + FN).

A model can have:
- High % Swaps Resolved (detects most swaps) but
- Lower % Clean Post (due to false positives)

Or:
- High % Clean Post (overall accurate) but
- Lower % Swaps Resolved (misses some swaps but makes few false positives)

## Edge Cases

### No Swaps in Ground Truth
- **GT Swaps = 0, Pred Swaps = 0**: 
  - % Swaps Resolved = 100% (no swaps to resolve)
  - % Clean Post = 100% (all frames correctly classified)
  
- **GT Swaps = 0, Pred Swaps > 0**:
  - % Swaps Resolved = 100% (no swaps to resolve)
  - % Clean Post < 100% (false positives reduce accuracy)

### All Frames Swapped
- **GT Swaps = Total Frames**:
  - % Clean Pre = 0% (no clean frames in input)
  - % Swaps Resolved = TP / (TP + FN) * 100
  - % Clean Post = TP / Total * 100

## Mathematical Relationships

### Relationship Between Metrics

```
% Swaps Resolved = Recall = Sensitivity = TP / (TP + FN)

% Frames Clean Post = Accuracy = (TP + TN) / Total
                    = (TP + TN) / (TP + TN + FP + FN)

% Frames Clean Pre = (TN + FN) / Total
                  = (Frames not swapped in GT) / Total
```

### Why They Can Differ

The difference between % Swaps Resolved and % Clean Post is:

```
Difference = % Swaps Resolved - % Clean Post
           = [TP / (TP + FN)] - [(TP + TN) / Total]
```

This difference reflects:
- **False Positives (FP)**: Reduce % Clean Post but don't affect % Swaps Resolved
- **True Negatives (TN)**: Increase % Clean Post but don't affect % Swaps Resolved
- **Trial composition**: If most frames are not swapped, TN dominates % Clean Post

## Recommendations for Interpretation

1. **% Swaps Resolved**: Use to assess how well the model detects existing swaps
   - High value (>95%): Model catches most swaps
   - Low value (<80%): Model misses many swaps

2. **% Frames Clean Post**: Use to assess overall classification accuracy
   - High value (>95%): Model is generally accurate
   - Low value (<90%): Model makes many classification errors (FP or FN)

3. **Compare Both Metrics**: 
   - High % Swaps Resolved + High % Clean Post: Excellent model
   - High % Swaps Resolved + Low % Clean Post: Detects swaps but makes false positives
   - Low % Swaps Resolved + High % Clean Post: Conservative model, misses swaps but few false positives
   - Low % Swaps Resolved + Low % Clean Post: Poor model performance

4. **Consider Precision**: If % Swaps Resolved is high but % Clean Post is lower, check precision
   - Low precision indicates false positives are the issue
   - High precision indicates the difference is due to trial composition

## Implementation Details

All metrics are calculated in `swap_correction/ml/api/batch_processor.py` in the `evaluate_on_dataset` method.

The calculations handle edge cases explicitly:
- Division by zero protection
- Special handling when GT has no swaps
- Special handling when model predicts no swaps


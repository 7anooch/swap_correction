# How Swap Correction Works

## Overview

Swap correction is the process of detecting and fixing identity swaps in animal tracking data—most commonly, head-tail swaps in linear animals (like larvae or worms) tracked over time. The goal is to ensure that the head and tail positions are consistently labeled throughout the dataset, even when the tracking system makes mistakes.

The swap correction pipeline in this package is robust and multi-step, combining several detection and correction strategies to maximize accuracy.

---

## Main Steps in Swap Correction

### 1. **Preprocessing and Edge Cleaning**

- **Remove Edge Frames:**  
  Frames at the beginning or end of the recording where all position values are identical (often zeros or artifacts) are set to NaN to avoid spurious corrections.

### 2. **Initial Swap Detection and Correction**

- **Flagging Swaps:**  
  The function `flag_all_swaps` identifies frames where a swap is likely, using several criteria:
  - **Minimum Delta Mismatches:** Checks if the smallest movement between frames is between head and tail, rather than head-to-head or tail-to-tail.
  - **Sign Reversals:** Looks for sudden changes in the direction of the animal's body.
  - **Overlap Mismatches:** Detects when head and tail positions overlap or cross.

- **Correcting Swapped Segments:**  
  The function `correct_tracking_errors` uses the flagged frames to define segments where swaps likely occurred. It then calls `correct_swapped_segments`, which swaps the head and tail coordinates for all frames in those segments.

- **Global Swap Correction:**  
  Sometimes, the entire recording is globally swapped (e.g., the head and tail are reversed for all frames). The function `correct_global_swap` compares the movement of the head and tail; if the tail appears to move more, it swaps all head and tail data.

### 3. **Validation and Secondary Correction**

- **Validation:**  
  After the initial correction, `validate_corrected_data` checks for any remaining swaps that may have been missed, especially in segments between overlaps. It uses assumptions about forward movement and segment metrics (like alignment, speed, or distance) to detect subtle swaps.

- **Segment-Based Metrics:**  
  The function `get_swapped_segments` divides the trajectory into segments between overlaps and computes metrics for each segment:
    - **Alignment:** Compares the direction of movement to the animal's orientation.
    - **Speed/Distance Ratios:** Compares head and tail movement.
  Segments that deviate from expected values are flagged as swapped and corrected.

### 4. **Error Removal and Interpolation**

- **Remove Overlaps:**  
  Frames where the head and tail overlap are set to NaN, and, if possible, only the erroneous point (head or tail) is removed based on discontinuities.

- **Interpolation (Optional):**  
  Short gaps (NaNs) in the data can be interpolated to produce smooth trajectories.

- **Filtering (Optional):**  
  Position data can be smoothed using filters (Gaussian, Savitzky-Golay, etc.) to reduce noise.

---

## Key Functions and Their Roles

- **`tracking_correction`**  
  The main entry point. Orchestrates the entire correction pipeline, calling the steps above in sequence.

- **`flag_all_swaps`**  
  Aggregates multiple swap-detection strategies to robustly flag likely swap frames.

- **`correct_swapped_segments`**  
  Swaps head and tail coordinates for all frames in the specified segments.

- **`correct_global_swap`**  
  Detects and corrects cases where the entire recording is globally swapped.

- **`get_swapped_segments`**  
  Uses segment-based metrics to detect subtle or ambiguous swaps.

- **`remove_overlaps`**  
  Removes or interpolates frames where head and tail overlap.

---

## Example Workflow

```python
from swap_correction import tracking_correction

# Load your tracking data (as a pandas DataFrame)
data = tracking_correction.load_data("tracking.csv")

# Apply swap correction
corrected = tracking_correction.tracking_correction(data, fps=30)
```

---

## Advanced Notes

- **Ambiguous Segments:**  
  The pipeline includes logic to handle ambiguous cases, such as segments sandwiched between swapped segments or following a swap.

- **Debugging:**  
  Most functions have a `debug` flag to print detailed information about detected swaps and corrections.

- **Custom Parameters:**  
  You can adjust thresholds and modes (e.g., use speed or distance instead of alignment) for fine-tuning on your data.

---

## Summary

Swap correction in this package is a multi-step, robust process that combines:
- Direct detection of swap events,
- Segment-based validation using behavioral metrics,
- Correction of both local and global swaps,
- Careful handling of edge cases and artifacts.

This ensures that your tracking data is as accurate and reliable as possible for downstream analysis. 
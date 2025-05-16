# Swap Correction Process Documentation

## 1. Main Entry Point: `swap_corrector.py`

### Function Signature
```python
def swap_corrector(
    data: pd.DataFrame,      # Raw tracking data
    fps: float,             # Frames per second
    filterData: bool = False,    # Whether to apply filtering
    swapCorrection: bool = True, # Whether to correct head-tail swaps
    validate: bool = False,      # Whether to validate corrections
    removeErrors: bool = True,   # Whether to remove error frames
    interp: bool = True,         # Whether to interpolate gaps
    debug: bool = False          # Whether to print debug messages
) -> pd.DataFrame
```

### Purpose
This is the primary entry point for the swap correction process. It serves as the main orchestrator, coordinating all the individual steps and applying them in the correct sequence.

### Process Flow
1. **Data Validation**
   - Checks if input data is valid
   - Verifies required columns are present
   - Ensures data types are correct

2. **Initial Processing**
   - Removes edge frames
   - Applies filtering if requested
   - Prepares data for correction

3. **Correction Pipeline**
   - Calls `tracking_correction` for the main correction process
   - Handles any errors or exceptions
   - Returns the corrected data

## 2. Tracking Correction Module

### Function: `tracking_correction`
```python
def tracking_correction(
    data: pd.DataFrame,
    fps: float,
    filterData: bool = False,
    swapCorrection: bool = True,
    validate: bool = False,
    removeErrors: bool = True,
    interp: bool = True,
    debug: bool = False
) -> pd.DataFrame
```

### Process Flow
1. **Edge Frame Removal**
   ```python
   data = remove_edge_frames(data)
   ```
   - Identifies frames where head or tail positions are at the edge
   - Sets these frames to NaN
   - Prevents false swap detections

2. **Data Filtering** (Optional)
   ```python
   if filterData:
       data = filtering.filter_data(data)
   ```
   - Applies Gaussian filter by default (sigma=3)
   - Alternative filters available:
     - Savitzky-Golay filter
     - Mean-median filter
     - Median filter

3. **Tracking Error Correction**
   ```python
   data = correct_tracking_errors(data, fps, swapCorrection, validate, debug)
   ```
   - Detects potential swap frames using multiple methods
   - Corrects identified swaps
   - Validates corrections if requested

4. **Error Frame Removal** (Optional)
   ```python
   if removeErrors:
       data = remove_overlaps(data, debug)
   ```
   - Removes frames with head-tail overlaps
   - Sets problematic frames to NaN

5. **Gap Interpolation** (Optional)
   ```python
   if interp:
       data = interpolate_gaps(data, debug)
   ```
   - Fills gaps in the data using linear interpolation
   - Maintains data continuity

## 3. Swap Detection Methods

### 1. Overlap Detection
```python
def flag_overlaps(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```
- Identifies frames where head and tail positions overlap
- Uses a threshold of 0.5mm for separation
- Returns array of frame indices where overlaps occur

### 2. Sign Reversal Detection
```python
def flag_sign_reversals(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```
- Detects sudden reversals in head-tail delta direction
- Uses dot product between consecutive deltas
- Flags frames where direction changes abruptly

### 3. Delta Mismatch Detection
```python
def flag_delta_mismatches(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```
- Identifies frames where head-tail delta changes abruptly
- Uses magnitude of delta changes
- Threshold of 2.0mm for flagging

### 4. Overlap Sign Reversal Detection
```python
def flag_overlap_sign_reversals(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```
- Specialized detection for reversals during overlaps
- Combines overlap and sign reversal detection
- More sensitive to swap detection during overlaps

### 5. Overlap Minimum Mismatch Detection
```python
def flag_overlap_minimum_mismatches(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```
- Detects unexpected minimum separations during overlaps
- Helps identify frames where swaps occur during overlaps

## 4. Correction Process

### 1. Segment Identification
```python
segments = utils.get_consecutive_ranges(swap_frames)
```
- Groups detected swap frames into consecutive segments
- Handles multiple swaps in sequence

### 2. Segment Correction
```python
for start, end in segments:
    data = correct_swapped_segments(data, start, end, debug)
```
- Corrects each segment of swapped frames
- Swaps head and tail positions for the segment

### 3. Global Swap Correction
```python
data = correct_global_swap(data, debug)
```
- Checks for global head-tail swaps
- Corrects if mean separation is negative

## 5. Helper Functions

### Position Calculations
1. `get_delta_in_frame`: Calculates separation between points
2. `get_vectors_between`: Calculates vectors between points
3. `get_orientation_vectors`: Gets orientation vectors

### Angle Calculations
1. `get_head_angle`: Calculates internal angle of the animal
2. `get_orientation`: Calculates global body angle
3. `get_bearing`: Calculates angle relative to source

### Motion Calculations
1. `get_speed_from_df`: Calculates speed from position data
2. `get_motion_vector`: Calculates motion vectors
3. `get_local_tortuosity`: Calculates local path tortuosity

## 6. Output
The final output is a corrected DataFrame with:
- Corrected head-tail positions
- Interpolated gaps (if requested)
- Removed error frames (if requested)
- Smoothed data (if filtering was applied)

## 7. Usage Example

```python
import pandas as pd
from swap_correction import swap_corrector

# Load your tracking data
data = pd.read_csv('tracking_data.csv')

# Apply swap correction
corrected_data = swap_corrector(
    data=data,
    fps=30.0,
    filterData=True,
    swapCorrection=True,
    validate=True,
    removeErrors=True,
    interp=True,
    debug=False
)

# Save corrected data
corrected_data.to_csv('corrected_tracking_data.csv', index=False)
```

## 8. Notes
- The correction process is designed to be robust against various types of tracking errors
- Multiple detection methods are used to ensure comprehensive swap detection
- The process can be customized through various parameters
- Debug mode can be enabled for detailed logging of the correction process 
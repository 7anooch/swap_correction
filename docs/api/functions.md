# Modern Code: Function-by-Function Developer Reference

This document provides a detailed, granular breakdown of each major function in the modern Swap Correction codebase. Functions are organized by module. For each function: purpose, parameters, return values, workflow, and special notes are described.

---

## Main Module

### `swap_corrector`
```python
def swap_corrector(
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
**Purpose:**
- Main entry point for the swap correction process.

**Parameters:**
- `data`: Raw tracking data (DataFrame)
- `fps`: Frames per second
- `filterData`, `swapCorrection`, `validate`, `removeErrors`, `interp`, `debug`: Control pipeline steps

**Returns:**
- Corrected tracking data (DataFrame)

**Workflow:**
1. Optionally filters data
2. Detects and corrects swaps
3. Optionally validates corrections
4. Removes error frames
5. Interpolates gaps
6. Returns processed data

**Notes:**
- Raises `ValueError` or `SwapCorrectionError` on failure

---

## Tracking Module

### `tracking_correction`
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
**Purpose:**
- Main function for correcting tracking data.

**Parameters:**
- See above

**Returns:**
- Corrected tracking data (DataFrame)

**Workflow:**
1. Optionally filters data
2. Detects and corrects swaps
3. Optionally validates corrections
4. Removes error frames
5. Interpolates gaps

---

### `remove_edge_frames`
```python
def remove_edge_frames(data: pd.DataFrame) -> pd.DataFrame
```
**Purpose:**
- Removes frames where head or tail is at the edge of the frame.

**Parameters:**
- `data`: Input tracking data

**Returns:**
- Data with edge frames set to NaN

---

### `correct_tracking_errors`
```python
def correct_tracking_errors(
    data: pd.DataFrame,
    fps: float,
    swapCorrection: bool = True,
    validate: bool = False,
    debug: bool = False
) -> pd.DataFrame
```
**Purpose:**
- Corrects tracking errors in the data.

**Parameters:**
- See above

**Returns:**
- Corrected tracking data

---

### `validate_corrected_data`
```python
def validate_corrected_data(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> pd.DataFrame
```
**Purpose:**
- Validates corrections by checking for remaining errors.

**Returns:**
- Validated tracking data

---

### `remove_overlaps`
```python
def remove_overlaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame
```
**Purpose:**
- Removes frames where head and tail positions overlap.

**Returns:**
- Data with overlap frames set to NaN

---

### `interpolate_gaps`
```python
def interpolate_gaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame
```
**Purpose:**
- Interpolates over gaps in the data.

**Returns:**
- Data with gaps interpolated

---

## Flagging Module

### `flag_all_swaps`
```python
def flag_all_swaps(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> np.ndarray
```
**Purpose:**
- Flags all potential head-tail swaps using multiple detection methods.

**Returns:**
- Array of frame indices where swaps are detected

---

### `flag_overlaps`, `flag_sign_reversals`, `flag_delta_mismatches`, `flag_overlap_sign_reversals`, `flag_overlap_minimum_mismatches`
```python
def flag_overlaps(data: pd.DataFrame, debug: bool = False) -> np.ndarray
# ... (similar signatures for other flagging functions)
```
**Purpose:**
- Detect specific types of swap or error frames.

**Returns:**
- Array of frame indices for each detection type

---

## Filtering Module

### `filter_data`, `filter_sgolay`, `filter_gaussian`, `filter_meanmed`, `filter_median`
```python
def filter_data(rawData: pd.DataFrame) -> pd.DataFrame
# ... (see API for other filter signatures)
```
**Purpose:**
- Apply various filters to tracking data.

**Returns:**
- Filtered tracking data

**Notes:**
- Parameters control window size, sigma, etc.

---

## Metrics Module

### `calculate_metrics`
```python
def calculate_metrics(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> pd.DataFrame
```
**Purpose:**
- Calculates all metrics for the tracking data.

**Returns:**
- Data with calculated metrics

---

### `calculate_separation`, `calculate_vectors`, `calculate_head_angle`, `calculate_orientation`, `calculate_speed`, `calculate_tortuosity`
```python
def calculate_separation(data: pd.DataFrame) -> np.ndarray
# ... (see API for other metric signatures)
```
**Purpose:**
- Calculate specific metrics (separation, angles, speed, tortuosity, etc.)

**Returns:**
- Array of calculated values

---

## Utils Module

### `validate_input_data`
```python
def validate_input_data(
    data: pd.DataFrame,
    required_columns: list = None,
    debug: bool = False
) -> bool
```
**Purpose:**
- Validates that the input data has the required columns and data types.

**Returns:**
- True if data is valid, False otherwise

---

### `load_data`, `save_data`, `remove_nan_rows`, `interpolate_nan`, `load_config`, `save_config`
```python
def load_data(file_path: str, **kwargs) -> pd.DataFrame
# ... (see API for other utility signatures)
```
**Purpose:**
- File operations and data cleaning utilities.

**Returns:**
- DataFrame or None, depending on function

---

## Special Notes
- See the API reference for full details and additional helper functions.
- All modules are designed to be used together or independently for custom workflows. 
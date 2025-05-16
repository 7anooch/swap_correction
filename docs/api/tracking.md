# Tracking Module API Reference

This document provides detailed API documentation for the tracking module of the Swap Correction package.

## Main Functions

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

Main function for correcting tracking data.

#### Parameters

- `data` (pd.DataFrame): Raw tracking data
- `fps` (float): Frames per second
- `filterData` (bool, optional): Whether to apply filtering
- `swapCorrection` (bool, optional): Whether to correct head-tail swaps
- `validate` (bool, optional): Whether to validate corrections
- `removeErrors` (bool, optional): Whether to remove error frames
- `interp` (bool, optional): Whether to interpolate gaps
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Corrected tracking data

#### Example

```python
from swap_correction.tracking import tracking_correction

corrected_data = tracking_correction(
    data=data,
    fps=30.0,
    filterData=True,
    swapCorrection=True,
    validate=True
)
```

## Correction Functions

### `remove_edge_frames`

```python
def remove_edge_frames(data: pd.DataFrame) -> pd.DataFrame
```

Removes frames where head or tail is at the edge of the frame.

#### Parameters

- `data` (pd.DataFrame): Input tracking data

#### Returns

- `pd.DataFrame`: Data with edge frames set to NaN

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

Corrects tracking errors in the data.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `fps` (float): Frames per second
- `swapCorrection` (bool, optional): Whether to correct head-tail swaps
- `validate` (bool, optional): Whether to validate corrections
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Corrected tracking data

### `validate_corrected_data`

```python
def validate_corrected_data(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> pd.DataFrame
```

Validates corrections by checking for remaining errors.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `fps` (float): Frames per second
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Validated tracking data

### `remove_overlaps`

```python
def remove_overlaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame
```

Removes frames where head and tail positions overlap.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Data with overlap frames set to NaN

### `interpolate_gaps`

```python
def interpolate_gaps(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame
```

Interpolates over gaps in the data.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Data with gaps interpolated

### `correct_global_swap`

```python
def correct_global_swap(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame
```

Corrects a global head-tail swap.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Data with global swap corrected

### `correct_swapped_segments`

```python
def correct_swapped_segments(
    data: pd.DataFrame,
    start: int,
    end: int,
    debug: bool = False
) -> pd.DataFrame
```

Corrects a segment of swapped frames.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `start` (int): Start frame index
- `end` (int): End frame index
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Data with segment corrected

### `get_swapped_segments`

```python
def get_swapped_segments(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> list
```

Gets segments of frames that need to be swapped.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `fps` (float): Frames per second
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `list`: List of (start, end) frame index pairs

## Helper Functions

### `get_consecutive_ranges`

```python
def get_consecutive_ranges(indices: np.ndarray) -> list
```

Gets ranges of consecutive indices.

#### Parameters

- `indices` (np.ndarray): Array of frame indices

#### Returns

- `list`: List of (start, end) index pairs

### `filter_array`

```python
def filter_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray
```

Filters an array using a boolean mask.

#### Parameters

- `arr` (np.ndarray): Input array
- `mask` (np.ndarray): Boolean mask

#### Returns

- `np.ndarray`: Filtered array

### `merge`

```python
def merge(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray
```

Merges two arrays of indices.

#### Parameters

- `arr1` (np.ndarray): First array of indices
- `arr2` (np.ndarray): Second array of indices

#### Returns

- `np.ndarray`: Merged array of indices

## Notes

1. The tracking module provides the core functionality for correcting tracking data.
2. Functions are designed to be used in sequence as part of the correction pipeline.
3. Debug mode can be enabled for detailed logging.
4. Helper functions are available for common operations.

## See Also

- [Main Module API](main.md)
- [Flagging Module API](flagging.md)
- [Filtering Module API](filtering.md)
- [Metrics Module API](metrics.md)
- [Utils Module API](utils.md) 
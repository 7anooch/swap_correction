# Flagging Module API Reference

This document provides detailed API documentation for the flagging module of the Swap Correction package.

## Main Functions

### `flag_all_swaps`

```python
def flag_all_swaps(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> np.ndarray
```

Flags all potential head-tail swaps using multiple detection methods.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `fps` (float): Frames per second
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `np.ndarray`: Array of frame indices where swaps are detected

#### Example

```python
from swap_correction.tracking import flagging

swap_frames = flagging.flag_all_swaps(
    data=data,
    fps=30.0,
    debug=True
)
```

## Detection Functions

### `flag_overlaps`

```python
def flag_overlaps(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```

Flags frames where head and tail positions overlap.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `np.ndarray`: Array of frame indices where overlaps occur

### `flag_sign_reversals`

```python
def flag_sign_reversals(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```

Flags frames where head-tail delta direction reverses.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `np.ndarray`: Array of frame indices where sign reversals occur

### `flag_delta_mismatches`

```python
def flag_delta_mismatches(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```

Flags frames where head-tail delta changes abruptly.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `np.ndarray`: Array of frame indices where delta mismatches occur

### `flag_overlap_sign_reversals`

```python
def flag_overlap_sign_reversals(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```

Flags frames where head-tail delta direction reverses during overlaps.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `np.ndarray`: Array of frame indices where overlap sign reversals occur

### `flag_overlap_minimum_mismatches`

```python
def flag_overlap_minimum_mismatches(data: pd.DataFrame, debug: bool = False) -> np.ndarray
```

Flags frames where minimum head-tail separation occurs at unexpected times.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `np.ndarray`: Array of frame indices where minimum mismatches occur

## Helper Functions

### `get_overlap_edges`

```python
def get_overlap_edges(overlaps: np.ndarray) -> tuple
```

Gets the start and end frames of overlap segments.

#### Parameters

- `overlaps` (np.ndarray): Array of overlap frame indices

#### Returns

- `tuple`: (starts, ends) arrays of segment boundaries

### `get_all_overlap_edges`

```python
def get_all_overlap_edges(data: pd.DataFrame, debug: bool = False) -> tuple
```

Gets all overlap segment edges in the data.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `tuple`: (starts, ends) arrays of segment boundaries

### `get_all_deltas`

```python
def get_all_deltas(data: pd.DataFrame) -> np.ndarray
```

Gets all head-tail delta vectors.

#### Parameters

- `data` (pd.DataFrame): Input tracking data

#### Returns

- `np.ndarray`: Array of delta vectors

## Constants

### `OVERLAP_THRESHOLD`

```python
OVERLAP_THRESHOLD = 0.5  # mm
```

Threshold for overlap detection.

### `DELTA_MISMATCH_THRESHOLD`

```python
DELTA_MISMATCH_THRESHOLD = 2.0  # mm
```

Threshold for delta mismatch detection.

## Notes

1. The flagging module provides functions for detecting potential head-tail swaps.
2. Multiple detection methods are used to ensure comprehensive swap detection.
3. Debug mode can be enabled for detailed logging.
4. Helper functions are available for common operations.

## See Also

- [Main Module API](main.md)
- [Tracking Module API](tracking.md)
- [Filtering Module API](filtering.md)
- [Metrics Module API](metrics.md)
- [Utils Module API](utils.md) 
# Metrics Module API Reference

This document provides detailed API documentation for the metrics module of the Swap Correction package.

## Main Functions

### `calculate_metrics`

```python
def calculate_metrics(
    data: pd.DataFrame,
    fps: float,
    debug: bool = False
) -> pd.DataFrame
```

Calculates all metrics for the tracking data.

#### Parameters

- `data` (pd.DataFrame): Tracking data
- `fps` (float): Frames per second
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `pd.DataFrame`: Data with calculated metrics

#### Example

```python
from swap_correction.tracking import metrics

data_with_metrics = metrics.calculate_metrics(
    data=data,
    fps=30.0,
    debug=True
)
```

## Position Metrics

### `calculate_separation`

```python
def calculate_separation(data: pd.DataFrame) -> np.ndarray
```

Calculates the separation between head and tail.

#### Parameters

- `data` (pd.DataFrame): Tracking data

#### Returns

- `np.ndarray`: Array of separation distances

### `calculate_vectors`

```python
def calculate_vectors(data: pd.DataFrame) -> tuple
```

Calculates head-tail and center vectors.

#### Parameters

- `data` (pd.DataFrame): Tracking data

#### Returns

- `tuple`: (head_tail_vectors, center_vectors)

## Angle Metrics

### `calculate_head_angle`

```python
def calculate_head_angle(data: pd.DataFrame) -> np.ndarray
```

Calculates the head angle relative to the x-axis.

#### Parameters

- `data` (pd.DataFrame): Tracking data

#### Returns

- `np.ndarray`: Array of head angles in degrees

### `calculate_orientation`

```python
def calculate_orientation(data: pd.DataFrame) -> np.ndarray
```

Calculates the orientation of the animal.

#### Parameters

- `data` (pd.DataFrame): Tracking data

#### Returns

- `np.ndarray`: Array of orientation angles in degrees

## Motion Metrics

### `calculate_speed`

```python
def calculate_speed(
    data: pd.DataFrame,
    fps: float
) -> np.ndarray
```

Calculates the speed of the animal.

#### Parameters

- `data` (pd.DataFrame): Tracking data
- `fps` (float): Frames per second

#### Returns

- `np.ndarray`: Array of speeds in mm/s

### `calculate_tortuosity`

```python
def calculate_tortuosity(
    data: pd.DataFrame,
    window: int = 10
) -> np.ndarray
```

Calculates the tortuosity of the animal's path.

#### Parameters

- `data` (pd.DataFrame): Tracking data
- `window` (int, optional): Window size for calculation

#### Returns

- `np.ndarray`: Array of tortuosity values

## Helper Functions

### `normalize_angle`

```python
def normalize_angle(angle: float) -> float
```

Normalizes an angle to the range [-180, 180] degrees.

#### Parameters

- `angle` (float): Input angle in degrees

#### Returns

- `float`: Normalized angle in degrees

### `calculate_angle_between`

```python
def calculate_angle_between(
    v1: np.ndarray,
    v2: np.ndarray
) -> float
```

Calculates the angle between two vectors.

#### Parameters

- `v1` (np.ndarray): First vector
- `v2` (np.ndarray): Second vector

#### Returns

- `float`: Angle between vectors in degrees

## Constants

### `DEFAULT_METRIC_PARAMS`

```python
DEFAULT_METRIC_PARAMS = {
    'tortuosity_window': 10,
    'speed_window': 5
}
```

Default parameters for metric calculations.

## Notes

1. The metrics module provides functions for calculating various metrics from tracking data.
2. Position metrics include separation and vectors.
3. Angle metrics include head angle and orientation.
4. Motion metrics include speed and tortuosity.
5. Helper functions are available for common calculations.

## See Also

- [Main Module API](main.md)
- [Tracking Module API](tracking.md)
- [Flagging Module API](flagging.md)
- [Filtering Module API](filtering.md)
- [Utils Module API](utils.md) 
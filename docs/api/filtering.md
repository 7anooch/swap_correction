# Filtering Module API Reference

This document provides detailed API documentation for the filtering module of the Swap Correction package.

## Main Functions

### `filter_data`

```python
def filter_data(rawData: pd.DataFrame) -> pd.DataFrame
```

Applies the default filter used by the analysis pipeline.

#### Parameters

- `rawData` (pd.DataFrame): Raw tracking data

#### Returns

- `pd.DataFrame`: Filtered tracking data

#### Example

```python
from swap_correction.tracking import filtering

filtered_data = filtering.filter_data(data)
```

## Filter Functions

### `filter_sgolay`

```python
def filter_sgolay(
    rawData: pd.DataFrame,
    window: int = 45,
    order: int = 4
) -> pd.DataFrame
```

Applies Savitzky-Golay filter to the position data.

#### Parameters

- `rawData` (pd.DataFrame): Raw tracking data
- `window` (int, optional): Window size for the filter
- `order` (int, optional): Polynomial order for the filter

#### Returns

- `pd.DataFrame`: Filtered tracking data

#### Notes

- The Savitzky-Golay filter is good for preserving higher moments of the data
- Larger window sizes provide more smoothing
- Higher order polynomials can better preserve peak shapes

### `filter_gaussian`

```python
def filter_gaussian(
    rawData: pd.DataFrame,
    sigma: float = 3
) -> pd.DataFrame
```

Applies Gaussian filter to the position data.

#### Parameters

- `rawData` (pd.DataFrame): Raw tracking data
- `sigma` (float, optional): Standard deviation for the Gaussian kernel

#### Returns

- `pd.DataFrame`: Filtered tracking data

#### Notes

- The Gaussian filter is good for general smoothing
- Larger sigma values provide more smoothing
- This is the default filter used by `filter_data`

### `filter_meanmed`

```python
def filter_meanmed(
    rawData: pd.DataFrame,
    medWin: int = 15,
    meanWin: int = None
) -> pd.DataFrame
```

Filters the position data by taking a rolling median followed by a rolling mean.

#### Parameters

- `rawData` (pd.DataFrame): Raw tracking data
- `medWin` (int, optional): Window size for median filter
- `meanWin` (int, optional): Window size for mean filter

#### Returns

- `pd.DataFrame`: Filtered tracking data

#### Notes

- The median filter helps remove outliers
- The mean filter provides additional smoothing
- If `meanWin` is None, it uses the same window size as `medWin`

### `filter_median`

```python
def filter_median(
    rawData: pd.DataFrame,
    win: int = 5
) -> pd.DataFrame
```

Filters the position data using a rolling median.

#### Parameters

- `rawData` (pd.DataFrame): Raw tracking data
- `win` (int, optional): Window size for the median filter

#### Returns

- `pd.DataFrame`: Filtered tracking data

#### Notes

- The median filter is good for removing outliers
- Smaller window sizes preserve more detail
- Larger window sizes provide more smoothing

## Helper Functions

### `apply_filter_to_column`

```python
def apply_filter_to_column(
    data: np.ndarray,
    filter_func: callable,
    **kwargs
) -> np.ndarray
```

Applies a filter function to a single column of data.

#### Parameters

- `data` (np.ndarray): Input data column
- `filter_func` (callable): Filter function to apply
- `**kwargs`: Additional arguments for the filter function

#### Returns

- `np.ndarray`: Filtered data column

### `get_filter_params`

```python
def get_filter_params(filter_type: str) -> dict
```

Gets default parameters for a filter type.

#### Parameters

- `filter_type` (str): Type of filter ('gaussian', 'sgolay', 'meanmed', 'median')

#### Returns

- `dict`: Default parameters for the filter

## Constants

### `DEFAULT_FILTER_PARAMS`

```python
DEFAULT_FILTER_PARAMS = {
    'gaussian': {'sigma': 3},
    'sgolay': {'window': 45, 'order': 4},
    'meanmed': {'medWin': 15, 'meanWin': None},
    'median': {'win': 5}
}
```

Default parameters for different filter types.

## Notes

1. The filtering module provides functions for smoothing tracking data.
2. Different filters are suitable for different types of data and noise.
3. Filter parameters can be adjusted to control the amount of smoothing.
4. The default filter (Gaussian) is a good general-purpose choice.

## See Also

- [Main Module API](main.md)
- [Tracking Module API](tracking.md)
- [Flagging Module API](flagging.md)
- [Metrics Module API](metrics.md)
- [Utils Module API](utils.md) 
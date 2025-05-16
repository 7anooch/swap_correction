# Utils Module API Reference

This document provides detailed API documentation for the utils module of the Swap Correction package.

## Data Validation

### `validate_input_data`

```python
def validate_input_data(
    data: pd.DataFrame,
    required_columns: list = None,
    debug: bool = False
) -> bool
```

Validates that the input data has the required columns and data types.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `required_columns` (list, optional): List of required column names
- `debug` (bool, optional): Whether to print debug messages

#### Returns

- `bool`: True if data is valid, False otherwise

#### Raises

- `DataValidationError`: If data validation fails

#### Example

```python
from swap_correction.tracking import utils

is_valid = utils.validate_input_data(
    data=data,
    required_columns=['xhead', 'yhead', 'xtail', 'ytail'],
    debug=True
)
```

## Data Processing

### `remove_nan_rows`

```python
def remove_nan_rows(data: pd.DataFrame) -> pd.DataFrame
```

Removes rows containing NaN values from the data.

#### Parameters

- `data` (pd.DataFrame): Input tracking data

#### Returns

- `pd.DataFrame`: Data with NaN rows removed

### `interpolate_nan`

```python
def interpolate_nan(
    data: pd.DataFrame,
    method: str = 'linear'
) -> pd.DataFrame
```

Interpolates NaN values in the data.

#### Parameters

- `data` (pd.DataFrame): Input tracking data
- `method` (str, optional): Interpolation method ('linear', 'cubic', etc.)

#### Returns

- `pd.DataFrame`: Data with interpolated values

## File Operations

### `load_data`

```python
def load_data(
    file_path: str,
    **kwargs
) -> pd.DataFrame
```

Loads tracking data from a file.

#### Parameters

- `file_path` (str): Path to the data file
- `**kwargs`: Additional arguments for pandas read function

#### Returns

- `pd.DataFrame`: Loaded tracking data

#### Raises

- `FileNotFoundError`: If file does not exist
- `ValueError`: If file format is not supported

### `save_data`

```python
def save_data(
    data: pd.DataFrame,
    file_path: str,
    **kwargs
) -> None
```

Saves tracking data to a file.

#### Parameters

- `data` (pd.DataFrame): Tracking data to save
- `file_path` (str): Path to save the data
- `**kwargs`: Additional arguments for pandas write function

#### Raises

- `ValueError`: If file format is not supported

## Configuration

### `load_config`

```python
def load_config(
    config_path: str,
    default_config: dict = None
) -> dict
```

Loads configuration from a file.

#### Parameters

- `config_path` (str): Path to the config file
- `default_config` (dict, optional): Default configuration values

#### Returns

- `dict`: Loaded configuration

### `save_config`

```python
def save_config(
    config: dict,
    config_path: str
) -> None
```

Saves configuration to a file.

#### Parameters

- `config` (dict): Configuration to save
- `config_path` (str): Path to save the config

## Constants

### `DEFAULT_CONFIG`

```python
DEFAULT_CONFIG = {
    'filter_type': 'gaussian',
    'filter_params': {
        'sigma': 3
    },
    'debug': False
}
```

Default configuration values.

## Notes

1. The utils module provides helper functions for common operations.
2. Data validation ensures input data meets requirements.
3. Data processing functions handle NaN values and interpolation.
4. File operations support loading and saving data.
5. Configuration functions manage settings.

## See Also

- [Main Module API](main.md)
- [Tracking Module API](tracking.md)
- [Flagging Module API](flagging.md)
- [Filtering Module API](filtering.md)
- [Metrics Module API](metrics.md) 
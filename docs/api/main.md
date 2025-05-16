# Main Module API Reference

This document provides detailed API documentation for the main module of the Swap Correction package.

## Main Functions

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

The main entry point for the swap correction process.

#### Parameters

- `data` (pd.DataFrame): Raw tracking data
  - Required columns: xhead, yhead, xtail, ytail
  - Optional columns: xctr, yctr, xmid, ymid
  - All position coordinates should be numeric
  - Missing values should be represented as NaN

- `fps` (float): Frames per second
  - Used for velocity calculations
  - Affects swap detection thresholds

- `filterData` (bool, optional): Whether to apply filtering
  - Default: False
  - If True, applies Gaussian filter with sigma=3

- `swapCorrection` (bool, optional): Whether to correct head-tail swaps
  - Default: True
  - If False, only performs filtering and error removal

- `validate` (bool, optional): Whether to validate corrections
  - Default: False
  - If True, performs additional validation after correction

- `removeErrors` (bool, optional): Whether to remove error frames
  - Default: True
  - If True, removes frames with head-tail overlaps

- `interp` (bool, optional): Whether to interpolate gaps
  - Default: True
  - If True, fills gaps using linear interpolation

- `debug` (bool, optional): Whether to print debug messages
  - Default: False
  - If True, prints detailed information about the correction process

#### Returns

- `pd.DataFrame`: Corrected tracking data
  - Same structure as input data
  - Head-tail swaps corrected
  - Gaps interpolated (if requested)
  - Error frames removed (if requested)
  - Data filtered (if requested)

#### Raises

- `ValueError`: If input data is invalid
- `SwapCorrectionError`: If correction process fails

#### Example

```python
import pandas as pd
from swap_correction import swap_corrector

# Load tracking data
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

## Helper Functions

### `validate_input_data`

```python
def validate_input_data(data: pd.DataFrame) -> bool
```

Validates the input data structure and types.

#### Parameters

- `data` (pd.DataFrame): Input tracking data

#### Returns

- `bool`: True if data is valid, False otherwise

#### Raises

- `ValueError`: If required columns are missing
- `TypeError`: If data types are incorrect

### `create_default_config`

```python
def create_default_config() -> None
```

Creates a default configuration file.

#### Parameters

None

#### Returns

None

#### Example

```python
from swap_correction import config

# Create default configuration
config.create_default_config()
```

## Constants

### `POSDICT`

```python
POSDICT = {
    'head': ('xhead', 'yhead'),
    'tail': ('xtail', 'ytail'),
    'ctr': ('xctr', 'yctr'),
    'mid': ('xmid', 'ymid')
}
```

Dictionary mapping point names to their coordinate column names.

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

## Error Classes

### `SwapCorrectionError`

```python
class SwapCorrectionError(Exception):
    """Base class for swap correction errors."""
    pass
```

Base exception class for swap correction errors.

### `DataValidationError`

```python
class DataValidationError(SwapCorrectionError):
    """Raised when input data validation fails."""
    pass
```

Raised when input data validation fails.

### `CorrectionError`

```python
class CorrectionError(SwapCorrectionError):
    """Raised when correction process fails."""
    pass
```

Raised when correction process fails.

## Notes

1. The main function `swap_corrector` is designed to be the primary entry point for most users.
2. Helper functions are available for more advanced usage.
3. Constants can be modified to change default behavior.
4. Error classes can be used for custom error handling.

## See Also

- [Tracking Module API](tracking.md)
- [Flagging Module API](flagging.md)
- [Filtering Module API](filtering.md)
- [Metrics Module API](metrics.md)
- [Utils Module API](utils.md) 
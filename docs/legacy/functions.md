# Legacy Code: Function-by-Function Developer Reference

This document provides a detailed, granular breakdown of each function in the legacy `swap_correct.py` script, including commented-out functions, constants, and the main execution block.

---

## Constants

- `FILE_NAME`, `FILE_SUFFIX`: Output file naming.
- `FIX_SWAPS`, `VALIDATE`, `REMOVE_ERRORS`, `INTERPOLATE`, `FILTER_DATA`: Control processing steps.
- `DEBUG`, `DIAGNOSTIC_PLOTS`, `SHOW_PLOTS`, `TIMES`: Control verbosity and plotting.

---

## Function Reference

### `compare_filtered_trajectories`
```python
def compare_filtered_trajectories(mainPath: str, outputPath: str = None, fileName: str = 'compare_trajectories.png', times: tuple = None, show: bool = True) -> None:
```
**Purpose:**
- Compare and plot trajectories from raw and filtered position data.

**Parameters:**
- `mainPath`: Directory containing data for one sample.
- `outputPath`: Directory to export image file to (optional).
- `fileName`: Name of the image file (default: 'compare_trajectories.png').
- `times`: Range of times to display data for (optional).
- `show`: Whether to display the figure after saving.

**Returns:**
- None (saves a figure to disk).

**Workflow:**
1. Loads raw and processed data using `loader`.
2. Plots both trajectories side-by-side using `plotting.plot_trajectory`.
3. Sets axis limits and titles.
4. Saves the figure using `plotting.save_figure`.

---

### `compare_filtered_distributions` *(commented out)*
```python
# def compare_filtered_distributions(mainPath: str, outputPath: str = None, fileName: str = 'compare_distributions.png', show: bool = True) -> None:
```
**Purpose:**
- Compare distributions (separation, orientation, reorientation rate) between raw and processed data.

**Parameters:**
- `mainPath`, `outputPath`, `fileName`, `show` (see above).

**Returns:**
- None (would save a figure to disk).

**Workflow:**
1. Loads raw and processed data.
2. Calculates metrics (separation, orientation, reorientation rate) using `metrics`.
3. Plots histograms for each metric.
4. Saves the figure.

**Notes:**
- This function is commented out in the legacy script.

---

### `examine_flags` *(commented out)*
```python
# def examine_flags(mainPath: str, outputPath: str = None, show: bool = True, fileName: str = 'flags.png', times: tuple = None, labelFrames: bool = False) -> None:
```
**Purpose:**
- Compare flagged frames and verified swap frames for diagnostics.

**Parameters:**
- `mainPath`, `outputPath`, `show`, `fileName`, `times`, `labelFrames` (see above).

**Returns:**
- None (would save a figure to disk).

**Workflow:**
1. Loads raw and processed data.
2. Flags discontinuities, overlaps, sign reversals, delta mismatches, and overlap-related events using `tc` (tracking_correction).
3. Filters and processes flagged frames.
4. Intended to plot and compare flagged vs. verified swap frames.

**Notes:**
- This function is commented out in the legacy script.

---

## Main Execution Block

### `if __name__ == '__main__':`
**Purpose:**
- Main workflow for batch processing of sample directories.

**Workflow:**
1. Opens a file dialog for the user to select a directory.
2. Identifies all sample folders (or treats the selected folder as a single sample).
3. For each sample:
   - Loads raw tracking data and framerate.
   - Runs the correction pipeline using `tracking_correction` with parameters set by script constants.
   - Exports the processed data to a new CSV file with a suffix (e.g., `level1`).
   - Generates diagnostic plots comparing raw and processed trajectories.
4. Prints a finished message when all samples are processed.

---

## Special Notes
- Some functions are commented out and were used for diagnostics or development.
- The script is designed for batch processing and diagnostic visualization.
- For migration to the new system, see the migration guide. 
# Legacy Code Developer Reference

This document provides a detailed reference for the legacy `swap_correct.py` script, including an overview and documentation for each function.

---

## Overview

The legacy `swap_correct.py` script was designed to process animal tracking data, correct head-tail swaps, handle errors, and generate diagnostic plots. It can process one or more experiment folders and was typically run as a standalone script with a GUI file dialog for selecting data folders.

---

## Functions

### 1. `compare_filtered_trajectories`
**Purpose:**
- Plots and compares the animal's trajectory before and after processing.

**Parameters:**
- `mainPath` (str): Path to the sample data directory.
- `outputPath` (str, optional): Directory to export the image file to. If None, saves in the sample directory.
- `fileName` (str, optional): Name of the image file (default: 'compare_trajectories.png').
- `times` (tuple, optional): Range of times to display data for. If None, displays the entire sample.
- `show` (bool, optional): Whether to display the figure after saving.

**Workflow:**
- Loads both raw and processed data.
- Plots trajectories side-by-side.
- Sets axis limits and titles.
- Saves the figure to disk.

---

### 2. `compare_filtered_distributions` *(commented out)*
**Purpose:**
- Would compare distributions (e.g., head-tail separation, orientation, reorientation rate) between raw and processed data.

**Parameters:**
- `mainPath` (str): Path to the sample data directory.
- `outputPath` (str, optional): Directory to export the image file to.
- `fileName` (str, optional): Name of the image file.
- `show` (bool, optional): Whether to display the figure after saving.

**Workflow:**
- Loads raw and processed data.
- Calculates metrics (separation, orientation, reorientation rate).
- Plots histograms for each metric, for both raw and processed data.
- Saves the figure to disk.

---

### 3. `examine_flags` *(commented out)*
**Purpose:**
- Would compare flagged frames and verified swap frames for diagnostic purposes.

**Parameters:**
- `mainPath` (str): Path to the sample data directory.
- `outputPath` (str, optional): Directory to export the image file to.
- `show` (bool, optional): Whether to display the figure after saving.
- `fileName` (str, optional): Name of the image file.
- `times` (tuple, optional): Range of times to display data for.
- `labelFrames` (bool, optional): Whether to label the x-axis in frames instead of seconds.

**Workflow:**
- Loads raw and processed data.
- Flags discontinuities, overlaps, sign reversals, delta mismatches, and overlap-related events.
- Filters and processes flagged frames.
- Intended to plot and compare flagged vs. verified swap frames.

---

### 4. Main Execution Block (`if __name__ == '__main__':`)
**Purpose:**
- Provides the main workflow for batch processing of sample directories.

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

## Constants and Configuration

- `FILE_NAME`, `FILE_SUFFIX`: Control output file naming.
- `FIX_SWAPS`, `VALIDATE`, `REMOVE_ERRORS`, `INTERPOLATE`, `FILTER_DATA`: Control which processing steps are applied.
- `DEBUG`, `DIAGNOSTIC_PLOTS`, `SHOW_PLOTS`, `TIMES`: Control verbosity and plotting.

---

## External Dependencies

- `loader`: Handles data loading and exporting, as well as retrieving settings.
- `plotting`: Used for plotting trajectories and saving figures.
- `metrics`: Used for calculating bounds and other metrics for plotting.
- `tracking_correction`: The main function for correcting tracking data (imported from the package).

---

## Notes
- Some functions are commented out and were used for diagnostics or development.
- The script is designed for batch processing and diagnostic visualization.
- For migration to the new system, see the migration guide. 
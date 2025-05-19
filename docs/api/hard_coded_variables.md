# Hard-Coded Variables Reference

This document lists all hard-coded variables in the Swap Correction project, organized by file/module. Tracking these variables is important for maintainability, configurability, and reproducibility. Consider moving critical values to configuration files or exposing them as parameters.

---

## swap_correction/pivr_loader.py
- `ANALYZED_DATA = 'analysis.csv'`
- `FILTERED_DATA = 'data_filtered.csv'`
- `MANUAL_DATA = 'filtered_data.csv'`
- `DISTANCE_TO_SOURCE = 'distance_to_source.csv'`
- `PIVR_SETTINGS = 'experiment_settings.json'`
- `PIVR_SETTINGS_UPDATED = 'experiment_settings_updated.json'`
- `SWAP_DATA = 'swaps.json'`
- `PIVRCOLS = [...]` (list of raw PiVR file columns)
- `NEWCOLS = [...]` (list of new column names)
- `POSCOLS = NEWCOLS[:8]`
- `_retrieve_raw_data(..., dataTag = 'data.csv', exclude = ['filtered','omit'])`
- `suffix: str = 'level1'` (in export_to_PiVR)
- In `flag_discontinuities`: `threshold = 5.0  # mm/s`
- In `flag_delta_mismatches`: `threshold = 2.0  # mm`

---

## swap_correction/tracking/correction/correction.py
- In `remove_edge_frames`:
  - `width = 1920  # TODO: Get from settings`
  - `height = 1080  # TODO: Get from settings`

---

## swap_correction/tracking/filtering/filters.py
- In `filter_sgolay`: `window: int = 45, order: int = 4`
- In `filter_gaussian`: `sigma: float = 3`
- In `filter_meanmed`: `medWin: int = 15, meanWin: int | None = None`
- In `filter_median`: `win: int = 5`

---

## swap_correction/tracking/flagging/flags.py
- In `flag_discontinuities`: `threshold = 5.0  # mm/s`
- In `flag_delta_mismatches`: `threshold = 2.0  # mm`
- In `flag_overlaps`: `threshold = 0.5  # mm`

---

## swap_correction/swap_correct.py (and legacy version)
- `FILE_NAME = loader.FILTERED_DATA`
- `FILE_SUFFIX = 'level1'`
- `FIX_SWAPS = True`
- `VALIDATE = False`
- `REMOVE_ERRORS = True`
- `INTERPOLATE = True`
- `FILTER_DATA = False`
- `DEBUG = False`
- `DIAGNOSTIC_PLOTS = True`
- `SHOW_PLOTS = False`
- `TIMES = None #(200,230)`

---

## General Patterns
- Many function parameters have default values that are hard-coded (e.g., thresholds, window sizes).
- File names, suffixes, and some settings are hard-coded as module-level constants or function defaults.
- Some settings (e.g., frame width/height) are marked as TODOs to be made configurable.

---

## Recommendations
- Move critical values to a configuration file (JSON, YAML, or Python config module).
- Pass values as parameters or load them at runtime.
- Expose thresholds and window sizes as function arguments with sensible defaults.

---

**Maintaining a list of hard-coded variables helps ensure your codebase is flexible, maintainable, and easy to adapt for new experiments or users.** 
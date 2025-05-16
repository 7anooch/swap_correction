# Migration Guide: Legacy to Modern Pipeline

This guide helps users and developers migrate from the legacy `swap_correct.py` workflow to the new modular Swap Correction pipeline.

---

## Why Migrate?
- Improved modularity and maintainability
- More robust error handling and validation
- Enhanced filtering, flagging, and correction options
- Better documentation and developer support

---

## Mapping: Legacy to Modern Functions

| Legacy Feature/Function         | Modern Equivalent/Module                |
|---------------------------------|-----------------------------------------|
| `compare_filtered_trajectories` | Use new plotting + metrics modules      |
| `compare_filtered_distributions`| Use new metrics and plotting            |
| `examine_flags`                 | Use `flagging` and `validation` modules |
| `tracking_correction`           | `tracking.tracking_correction`          |
| `VALIDATE` flag                 | `validate=True` in new pipeline         |
| `FILTER_DATA` flag              | `filterData=True` in new pipeline       |
| `pivr_loader`                   | Use `utils.load_data`/`save_data`       |
| `plotting` (legacy)             | Use new plotting utilities              |
| `metrics` (legacy)              | Use `metrics` module                    |

---

## Step-by-Step Migration

1. **Data Loading**
   - **Legacy:** `pivr_loader.load_raw_data(mainPath)`
   - **Modern:** `from swap_correction.tracking import utils; data = utils.load_data(path)`

2. **Correction Pipeline**
   - **Legacy:**
     ```python
     data = tracking_correction(
         data, fps,
         filterData=FILTER_DATA,
         swapCorrection=FIX_SWAPS,
         validate=VALIDATE,
         removeErrors=REMOVE_ERRORS,
         interp=INTERPOLATE,
         debug=DEBUG
     )
     ```
   - **Modern:**
     ```python
     from swap_correction.tracking import tracking_correction
     corrected_data = tracking_correction(
         data=data,
         fps=30.0,
         filterData=True,
         swapCorrection=True,
         validate=True,
         removeErrors=True,
         interp=True,
         debug=False
     )
     ```

3. **Exporting Data**
   - **Legacy:** `loader.export_to_PiVR(sample, data, suffix=FILE_SUFFIX)`
   - **Modern:** `utils.save_data(corrected_data, 'output.csv')`

4. **Plotting and Diagnostics**
   - **Legacy:** Custom plotting functions in `plotting` module
   - **Modern:** Use new plotting utilities and metrics for diagnostics

5. **Flagging and Validation**
   - **Legacy:** Custom flagging in `examine_flags`
   - **Modern:** Use `flagging` module for swap detection and validation

---

## Tips
- Review the [Core Concepts](../guides/core_concepts.md) and [API Reference](../api/main.md) for new usage patterns.
- Use the new modular functions for better maintainability.
- Update scripts to use new data loading, correction, and export utilities.
- For custom workflows, combine modules as needed.

---

## Need Help?
- See the [FAQ](../faq.md) or open an issue on GitHub. 
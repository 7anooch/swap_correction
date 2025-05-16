# Deprecated Features in Legacy Code

This document lists all deprecated or removed features and functions from the legacy `swap_correct.py` and related code.

| Feature/Function                | Purpose                                              | Status                | Replacement/Notes                |
|---------------------------------|------------------------------------------------------|-----------------------|----------------------------------|
| `compare_filtered_distributions`| Compare distributions pre/post filtering             | Deprecated (commented)| Use new metrics/plotting modules |
| `examine_flags`                 | Compare flagged and verified swap frames             | Deprecated (commented)| Use new flagging/validation      |
| `VALIDATE` flag                 | Segment-based swap correction validation             | Deprecated            | Not recommended in legacy        |
| `FILTER_DATA` flag              | Filtering before export                             | Deprecated            | Use new filtering pipeline       |
| `pivr_loader`                   | PiVR-specific data loading/export                   | Obsolete              | Use new data I/O utilities       |
| `plotting` module (legacy)      | Diagnostic plotting                                 | Obsolete              | Use new plotting/metrics         |
| `metrics` (legacy)              | Legacy metrics calculations                         | Obsolete              | Use new metrics module           |

## Notes
- Deprecated features are not maintained and may be removed in future versions.
- Use the new modular pipeline for all new analyses.
- See the migration guide for how to update legacy workflows. 
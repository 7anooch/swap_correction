# Experimental Results Directory

This directory contains all experimental analysis results organized by analysis type.

## Structure

```
experimental_results/
├── stability_analysis/
│   ├── current/          # Current production version (Features V4)
│   └── legacy/           # Previous versions (original, v2, v3, features_v2, features_v3)
├── threshold_analysis/
│   ├── current/          # Current threshold analysis (Features V4)
│   └── legacy/           # Previous threshold analyses (v2, original, iteration-specific)
├── learning_curve_analysis/
│   ├── current/          # Current learning curve analyses (Features V4)
│   └── legacy/           # Previous learning curve analyses (original, v2, v3)
└── feature_importance_comparison/  # Feature importance comparisons
```

## Current Production Version

**Stability Analysis**: `stability_analysis/current/stability_analysis_v3_features_v4/`
- Features V4 (~36 features)
- Sample size 80-100
- 6-15 iterations
- **This is the reference implementation**

**Threshold Analysis**: `threshold_analysis/current/threshold_analysis_aggregated_v4/`
- Features V4 threshold optimization results
- Recommended thresholds: Level1=0.63, Raw=0.5

**Learning Curve Analysis**: `learning_curve_analysis/current/learning_curve_analysis_v4_iter*/`
- Features V4 learning curves
- Iterations 7, 8, 13, 14, 15

## Legacy Versions

Legacy versions are preserved for reference and historical comparison. See `docs/EXPERIMENTAL_HISTORY.md` for detailed information about each version.

## Accessing Results

### Stability Analysis
```bash
# Current production version (Features V4)
cat experimental_results/stability_analysis/current/stability_analysis_v3_features_v4/stability_report.md

# Legacy versions (summaries only, iteration data cleaned)
ls experimental_results/stability_analysis/legacy/
```

### Threshold Analysis
```bash
# Current aggregated results (Features V4)
ls experimental_results/threshold_analysis/current/threshold_analysis_aggregated_v4/

# Legacy results (summaries only)
ls experimental_results/threshold_analysis/legacy/
```

### Learning Curve Analysis
```bash
# Current results (Features V4)
ls experimental_results/learning_curve_analysis/current/

# Legacy results (summaries and plots only)
ls experimental_results/learning_curve_analysis/legacy/
```

### Feature Importance
```bash
# Feature importance comparisons
ls experimental_results/feature_importance_comparison/
ls experimental_results/feature_importance_comparison_v4/
```

## Notes

- All detailed iteration data has been cleaned from legacy directories (only summaries kept)
- Current production version (`stability_analysis_v3_features_v4`) contains full iteration data
- See `docs/CLEANUP_SUMMARY.md` for details on what was cleaned and why

---

**Last Updated**: 2026-01-02


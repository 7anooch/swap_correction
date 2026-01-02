# ML Model Registry

**Generated**: 2025-12-31 17:53:47

This registry documents all trained ML models for swap detection in animal tracking data.

---

## Model 1: Level1 Model

### Overview
- **Purpose**: Detect remaining swaps in level1.csv (auto-corrected) data
- **Training Data**: level1.csv vs level2.csv (ground truth)
- **Model Type**: XGBoost Classifier
- **Status**: ✅ Available

### Performance Metrics

#### Test Set (Primary Metric)
- **F1-Score**: 0.9895 (98.95%)
- **Precision**: 0.9916 (99.16%)
- **Recall**: 0.9874 (98.74%)
- **ROC-AUC**: 0.9998 (99.98%)

#### All Splits
| Split | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| Train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Val | 0.9901 | 0.9974 | 0.9830 | 0.9999 |
| Test | 0.9895 | 0.9916 | 0.9874 | 0.9998 |

### Confusion Matrix (Test Set)
```
                Predicted
              No Swap  Swap
Actual No Swap  38513     54
       Swap        81   6352
```

- **False Positive Rate**: 0.14%
- **False Negative Rate**: 1.26%

### Training Data Characteristics
- **Total Training Frames**: 153,000
- **Swapped Frames**: 20,145 (13.17%)
- **Test Set Size**: 45,000
- **Test Swapped Frames**: 6,433 (14.30%)

### Model Configuration
- **Algorithm**: XGBoost
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **N Estimators**: 200 (with early stopping)
- **Subsample**: 0.8
- **Column Sample by Tree**: 0.8
- **Gaussian Filter Sigma**: 4.6 (optimal from grid search)

### Feature Set
- **Total Features**: ~36 (Features V4 - current default)
- **Feature Types**:
  - Distance features (head-tail, head-mid, tail-mid, normalized by body length)
  - Speed features (head, tail, ratio)
  - Acceleration features (head, tail, relative)
  - Angular features (orientation, motion angles, angular velocity with 3-point derivative)
  - Geometric features (cross-sign, alignment angle, head/tail curvature ratio)
  - Temporal context features (mean/std over windows: 10, 20, 50 frames)
  - Cumulative distance features
- **Note**: This model uses Features V4, which is the current default implementation. Legacy feature versions (V2, V3, Original 56-feature set) are available in `swap_correction/ml/features/legacy/` for reference.

### Top 10 Most Important Features
1. **alignment_angle_mean_50**: 0.3994
2. **alignment_angle_mean_20**: 0.2232
3. **head_speed_mean_50**: 0.0420
4. **tail_speed_mean_50**: 0.0329
5. **alignment_angle**: 0.0319
6. **tail_speed_mean_20**: 0.0269
7. **head_speed_std_50**: 0.0245
8. **tail_velocity_magnitude**: 0.0160
9. **tail_speed_std_50**: 0.0158
10. **tail_speed**: 0.0154

### File Locations
- **Model**: `ml_models/swap_detector_xgb.pkl`
- **Scaler**: `ml_models/feature_scaler.pkl`
- **Imputer**: `ml_models/feature_imputer.pkl`
- **Feature Names**: `ml_models/feature_names.pkl`
- **Feature Importance**: `ml_models/feature_importance.csv`
- **Training Results**: `ml_models/training_results.json`

### Usage
```python
from swap_correction.ml.api import SwapPredictor

# Simple usage with recommended threshold
predictor = SwapPredictor(model_type='level1')
predictions = predictor.predict(trial_data, fps=30, threshold=0.63)  # Recommended threshold
segments = predictor.predict_segments(trial_data, fps=30, threshold=0.63)
```

### Recommended Threshold
- **Threshold**: 0.63 (optimized for % Frames Clean Post)
- **Performance**: Achieves ~97.65% ± 0.76% frames clean post on average
- **Rationale**: Based on comprehensive threshold optimization across multiple stability analysis iterations

### Use Case
**Best for**: Detecting remaining swaps after initial auto-correction
- Higher precision (99.16%) and overall performance (98.95% F1)
- Lower false positive rate
- Use when you have level1.csv files and want to improve them further

---

## Model 2: Raw Data Model

### Overview
- **Purpose**: Detect swaps directly from raw tracking data
- **Training Data**: raw _data.csv vs level2.csv (ground truth)
- **Model Type**: XGBoost Classifier
- **Status**: ✅ Available

### Performance Metrics

#### Test Set (Primary Metric)
- **F1-Score**: 0.9744 (97.44%)
- **Precision**: 0.9720 (97.20%)
- **Recall**: 0.9769 (97.69%)
- **ROC-AUC**: 0.9981 (99.81%)

#### All Splits
| Split | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| Train | 0.9988 | 0.9986 | 0.9990 | 1.0000 |
| Val | 0.9837 | 0.9762 | 0.9913 | 0.9992 |
| Test | 0.9744 | 0.9720 | 0.9769 | 0.9981 |

### Confusion Matrix (Test Set)
```
                Predicted
              No Swap  Swap
Actual No Swap  22869    605
       Swap       498  21028
```

- **False Positive Rate**: 2.58%
- **False Negative Rate**: 2.31%

### Training Data Characteristics
- **Total Training Frames**: 153,000
- **Swapped Frames**: 77,475 (50.64%)
- **Test Set Size**: 45,000
- **Test Swapped Frames**: 21,526 (47.84%)

### Model Configuration
- **Algorithm**: XGBoost
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **N Estimators**: 200 (with early stopping)
- **Subsample**: 0.8
- **Column Sample by Tree**: 0.8
- **Gaussian Filter Sigma**: 4.6 (optimal from grid search)

### Feature Set
- **Total Features**: ~36 (Features V4 - current default, same as Level1 model)
- **Feature Types**: Same as Level1 model
- **Note**: This model uses Features V4, which is the current default implementation.

### Top 10 Most Important Features
1. **alignment_angle_mean_50**: 0.3486
2. **alignment_angle_mean_20**: 0.1506
3. **head_speed_mean_50**: 0.1026
4. **head_speed_mean_20**: 0.0421
5. **head_speed_std_50**: 0.0327
6. **speed_ratio**: 0.0313
7. **tail_speed_mean_50**: 0.0311
8. **alignment_angle**: 0.0275
9. **tail_speed_mean_20**: 0.0275
10. **tail_speed_std_50**: 0.0161

### File Locations
- **Model**: `ml_models_raw/swap_detector_xgb.pkl`
- **Scaler**: `ml_models_raw/feature_scaler.pkl`
- **Imputer**: `ml_models_raw/feature_imputer.pkl`
- **Feature Names**: `ml_models_raw/feature_names.pkl`
- **Feature Importance**: `ml_models_raw/feature_importance.csv`
- **Training Results**: `ml_models_raw/training_results.json`

### Usage
```python
from swap_correction.ml.api import SwapPredictor

# Simple usage (default threshold 0.5 is optimal)
predictor = SwapPredictor(model_type='raw')
predictions = predictor.predict(trial_data, fps=30)  # threshold=0.5 is default and optimal
segments = predictor.predict_segments(trial_data, fps=30)
```

### Recommended Threshold
- **Threshold**: 0.5 (default, optimal for this model)
- **Performance**: Achieves ~96% frames clean post on average
- **Rationale**: Default threshold provides optimal performance for raw data models

### Use Case
**Best for**: Detecting swaps directly from raw tracking data
- Can work directly on raw data (no need for level1 correction first)
- Still achieves good performance (97.44% F1)
- Handles higher swap rate in raw data
- Use when you want to skip the level1 correction step entirely

---

## Model Comparison Summary

| Aspect | Level1 Model | Raw Data Model |
|--------|--------------|----------------|
| **Test F1-Score** | 0.9895 | 0.9744 |
| **Test Precision** | 0.9916 | 0.9720 |
| **Test Recall** | 0.9874 | 0.9769 |
| **Training Swap Rate** | 13.17% | 50.64% |
| **Best For** | Post-correction refinement | Direct raw data processing |

## Recommendations

1. **Use Level1 Model** if:
   - You already have level1.csv files
   - Maximum accuracy is required
   - Lower false positive rate is critical

2. **Use Raw Data Model** if:
   - You want to process raw data directly
   - You want to skip level1 correction step
   - You're building a new pipeline from scratch

Both models are production-ready and can be used for automated swap detection.

---

## Additional Resources

- **Usage Guide**: `docs/ML_USAGE_GUIDE.md`
- **API Reference**: `docs/ML_API_REFERENCE.md`
- **Training Guide**: `docs/ML_TRAINING_GUIDE.md`
- **Overfitting Analysis**: `ml_analysis/model_overfitting_analysis.json`
- **Learning Curves**: `ml_analysis/learning_curves_data.json`
- **Model Comparison Report**: `ml_analysis/model_comparison_report.md`

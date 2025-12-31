# Machine Learning Approach to Head-Tail Swap Detection

## Executive Summary

This document describes our machine learning (ML) approach to automatically detect and correct head-tail swaps in animal tracking data. We use supervised learning to train a classifier that can identify swapped frames more accurately than rule-based heuristics.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Why Machine Learning?](#why-machine-learning)
3. [Overview of Machine Learning](#overview-of-machine-learning)
4. [Data Preparation](#data-preparation)
5. [Feature Engineering](#feature-engineering)
6. [Model Selection and Training](#model-selection-and-training)
7. [Evaluation](#evaluation)
8. [Integration](#integration)
9. [Future Improvements](#future-improvements)

---

## Problem Statement

### The Swap Detection Challenge

In animal tracking data, the head and tail of an animal are tracked as separate points. Occasionally, the tracking system misidentifies which point is the head and which is the tail, causing a "swap" where the labels are reversed. This creates errors in downstream analysis.

**Current Approach**: We use rule-based heuristics (if-then rules) to detect swaps:
- "If tail speed > head speed, likely swapped"
- "If alignment angle > 90°, likely swapped"
- etc.

**Limitations**:
- Rules are manually tuned and may miss complex patterns
- Different swap patterns require different rules
- Hard to balance false positives vs. false negatives
- Rules don't adapt to new data patterns

**Goal**: Train a machine learning model that can automatically learn patterns that distinguish swapped from non-swapped frames, potentially outperforming rule-based methods.

---

## Why Machine Learning?

### Advantages of ML Approach

1. **Automatic Pattern Discovery**: ML models can discover complex, non-obvious patterns in the data that humans might miss.

2. **Data-Driven**: The model learns from actual examples of swaps (ground truth data), rather than relying on assumptions.

3. **Adaptability**: Once trained, the model can be retrained on new data to adapt to different experimental conditions.

4. **Holistic Analysis**: ML models consider many features simultaneously, finding optimal combinations that rules might miss.

5. **Quantifiable Performance**: We can measure precision, recall, and other metrics to understand exactly how well the model performs.

### When ML is Appropriate

ML is a good fit for this problem because:
- ✅ We have labeled training data (ground truth from manual corrections)
- ✅ The problem is well-defined (binary classification: swapped or not)
- ✅ Patterns are complex but learnable
- ✅ We have sufficient data (225,000 frames across 25 trials)

---

## Overview of Machine Learning

### What is Machine Learning?

**Traditional Programming**: Write explicit rules
```
IF tail_speed > head_speed THEN swap = True
```

**Machine Learning**: Learn rules from examples
```
Given many examples of (features → swap_label),
learn a function: swap = f(features)
```

### Supervised Learning

We use **supervised learning**, which means:
- **Input**: Features describing each frame (speed, angles, positions, etc.)
- **Output**: Label (swapped = True/False)
- **Training**: Show the model many examples of (features, label) pairs
- **Goal**: Learn to predict labels for new, unseen frames

### Binary Classification

Our problem is **binary classification**:
- **Class 0**: Frame is NOT swapped (normal)
- **Class 1**: Frame IS swapped (error)

The model outputs a probability (0.0 to 1.0) that a frame is swapped, and we use a threshold (e.g., 0.5) to make the final decision.

---

## Data Preparation

### Ground Truth Labels

We have three levels of data for each trial:
- **`_data.csv`**: Raw tracking data (has errors)
- **`_level1.csv`**: Auto-corrected using rule-based methods (still has some errors)
- **`_level2.csv`**: Manually corrected (ground truth - the "correct answer")

### Creating Training Labels

**Step 1: Compare level1 vs level2**
- For each frame, check if head/tail positions differ between level1 and level2
- If positions differ by >0.5mm, label that frame as "swapped" (label = 1)
- Otherwise, label as "not swapped" (label = 0)

**Step 2: Extract frame-level labels**
- Create a dataset with one row per frame
- Columns: `trial`, `frame_idx`, `is_swapped`

**Step 3: Extract segment-level labels**
- Group consecutive swapped frames into segments
- Create a dataset with one row per swap segment
- Columns: `trial`, `start_frame`, `end_frame`, `is_swapped`

### Data Statistics

From our 25 trials:
- **Total frames**: 225,000
- **Swapped frames**: 28,519 (12.68%)
- **Non-swapped frames**: 196,481 (87.32%)
- **Perfect trials** (0% error): 14 trials (56%)
- **Problematic trials** (>0% error): 11 trials (44%)

**Class Imbalance**: We have many more non-swapped frames than swapped frames. This is common in ML and we handle it using class weights (see Model Training section).

### Train/Validation/Test Split

We split our data into three sets:
- **Training (70%)**: Used to train the model
- **Validation (15%)**: Used to tune hyperparameters and prevent overfitting
- **Test (15%)**: Used for final evaluation (never seen during training)

**Important**: We split by **trial**, not by frame, to ensure the model generalizes to new trials rather than just memorizing specific frames.

---

## Feature Engineering

### What are Features?

**Features** are measurable properties of each frame that help the model distinguish swapped from non-swapped frames. Think of them as "clues" the model uses to make predictions.

### Frame-Level Features

For each frame, we extract ~56 features across several categories:

#### 1. Position Features (8 features)
- **Head position**: `head_x`, `head_y`
- **Tail position**: `tail_x`, `tail_y`
- **Midpoint position**: `mid_x`, `mid_y`
- **Centroid position**: `centroid_x`, `centroid_y`
- **Distances**: `head_tail_distance`, `head_mid_distance`, `tail_mid_distance`

**Why useful**: Swapped frames may have unusual spatial relationships between keypoints.

#### 2. Velocity Features (9 features)
- **Speeds**: `head_speed`, `tail_speed` (instantaneous speed in mm/s)
- **Speed ratio**: `speed_ratio` = head_speed / tail_speed
- **Velocity vectors**: `head_velocity_x`, `head_velocity_y`, `tail_velocity_x`, `tail_velocity_y`
- **Velocity magnitudes**: `head_velocity_magnitude`, `tail_velocity_magnitude`
- **Relative velocity**: `relative_velocity_magnitude`

**Why useful**: In normal motion, the head typically moves faster than the tail. Swapped frames often show the opposite pattern.

#### 3. Angular Features (7 features)
- **Body orientation angle**: Angle of tail-to-midpoint vector (body direction)
- **Motion direction angles**: Direction of head and tail movement
- **Alignment angle**: Angle between body orientation and tail motion direction
  - Small angle (<90°): Forward motion (normal)
  - Large angle (>90°): Backward motion (likely swapped)
- **Angular velocities**: Rate of change of direction for head and tail

**Why useful**: Swapped frames often show "backwards" motion patterns where the tail appears to lead.

#### 4. Geometric Features (4 features)
- **Cross-sign consistency**: Sign of cross product (head-tail × tail-midpoint)
  - Consistent sign: Normal motion
  - Inconsistent sign: Likely swapped
- **Path curvature**: How sharply the head path bends
- **Cumulative distances**: Total distance traveled by head and tail

**Why useful**: The head typically follows a more contorted path than the tail.

#### 5. Temporal Context Features (~28 features)

For each of several window sizes (5, 10, 20, 50 frames), we calculate:
- **Mean and standard deviation** of:
  - Head speed
  - Tail speed
  - Alignment angle

**Why useful**: Swaps often persist over multiple frames. Looking at patterns over time helps distinguish real swaps from temporary noise.

#### 6. Context Features (1 feature)
- **Position in trial**: Where in the trial this frame occurs (0.0 = start, 1.0 = end)

**Why useful**: Some swap patterns are more common at the beginning or end of trials.

### Segment-Level Features

For swap segments (contiguous groups of swapped frames), we extract:
- **Aggregate statistics**: Mean, median, std, min, max, percentiles of all frame-level features
- **Pattern features**: Consistency metrics, transition features (how segment differs from surrounding frames)
- **Context features**: Position in trial, proximity to other swaps

**Note**: We focus primarily on frame-level classification, but segment-level features can be used for validation and refinement.

### Feature Extraction Process

1. **Load tracking data** for a trial
2. **Optional: Apply Gaussian filtering** (if data is noisy)
   - Filter position coordinates with Gaussian filter (sigma = 4-5)
   - Smooths out tracking noise before computing speeds/angles
   - Can improve feature quality if raw data is particularly noisy
3. **For each frame**:
   - Extract position, velocity, angular features
   - Calculate temporal context using sliding windows
   - Combine into a feature vector (56 values)
4. **Handle missing data**: Fill NaN values with median of that feature
5. **Normalize features**: Scale to mean=0, std=1 (important for ML models)

**Note on Filtering**: Raw tracking data is often noisy. If model performance is poor, try enabling Gaussian filtering (`apply_filtering=True`, `filter_sigma=4.5`) in feature extraction. This smooths position data before computing derivatives (speeds, angles), which can improve feature quality.

---

## Model Selection and Training

### Why XGBoost?

We chose **XGBoost** (Extreme Gradient Boosting) for several reasons:

1. **Excellent Performance**: XGBoost consistently performs well on tabular data (data in rows/columns format)

2. **Handles Mixed Data Types**: Works well with our mix of continuous features (speeds, angles) and handles missing values

3. **Interpretability**: Provides feature importance scores, so we can understand which features are most predictive

4. **Efficiency**: Fast to train and make predictions, important for processing large datasets

5. **Robust to Overfitting**: Built-in regularization helps prevent memorizing training data

### What is XGBoost?

**Gradient Boosting** is an ensemble method:
- Creates many "weak" decision trees (simple rules)
- Each tree tries to correct mistakes of previous trees
- Combines all trees into one strong model

**Example** (simplified):
```
Tree 1: "If speed_ratio < 0.9, predict swap"
Tree 2: "If Tree 1 says no swap BUT alignment_angle > 100°, predict swap"
Tree 3: "If Trees 1-2 disagree AND cross_sign is negative, predict swap"
...
Final prediction = weighted vote of all trees
```

### Training Process

#### Step 1: Prepare Data
- Load features and labels
- Split into train/validation/test sets
- Handle missing values (impute with median)
- Scale features to mean=0, std=1 (StandardScaler)

#### Step 2: Handle Class Imbalance
- **Problem**: 87% non-swapped, 13% swapped frames
- **Solution**: Use `scale_pos_weight` parameter
  - Tells XGBoost to weight swapped frames more heavily
  - Formula: `scale_pos_weight = n_negative / n_positive`
  - Example: If 87% negative, 13% positive: weight = 87/13 ≈ 6.7

#### Step 3: Set Hyperparameters
Key hyperparameters we tune:
- **`max_depth`**: Maximum depth of decision trees (default: 6)
  - Deeper = more complex, but risk of overfitting
- **`learning_rate`**: How much each tree contributes (default: 0.1)
  - Lower = more trees needed, but better generalization
- **`n_estimators`**: Number of trees (default: 200)
- **`subsample`**: Fraction of data each tree sees (default: 0.8)
  - Helps prevent overfitting
- **`colsample_bytree`**: Fraction of features each tree sees (default: 0.8)
  - Adds diversity to trees

#### Step 4: Train Model
- XGBoost builds trees iteratively
- After each tree, evaluates on validation set
- Stops early if validation performance doesn't improve (early stopping)
- Saves best model based on validation performance

#### Step 5: Make Predictions
- For each frame, model outputs probability (0.0 to 1.0)
- We use threshold = 0.5: if probability > 0.5, predict "swapped"
- Can adjust threshold to balance precision vs. recall

### Model Output

The trained model provides:
1. **Predictions**: Binary (swapped/not swapped) for each frame
2. **Probabilities**: Confidence score (0.0 to 1.0) for each prediction
3. **Feature Importance**: Which features are most predictive

---

## Evaluation

### Metrics

We evaluate the model using several metrics:

#### 1. Precision
**Definition**: Of all frames predicted as swapped, how many were actually swapped?

```
Precision = True Positives / (True Positives + False Positives)
```

**Example**: If model predicts 100 frames as swapped, and 90 are actually swapped:
- Precision = 90/100 = 0.90 (90%)

**Interpretation**: High precision = low false positive rate (fewer false alarms)

#### 2. Recall
**Definition**: Of all actually swapped frames, how many did we detect?

```
Recall = True Positives / (True Positives + False Negatives)
```

**Example**: If there are 1000 swapped frames, and model detects 800:
- Recall = 800/1000 = 0.80 (80%)

**Interpretation**: High recall = low false negative rate (catches most swaps)

#### 3. F1-Score
**Definition**: Harmonic mean of precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretation**: Balances precision and recall. Higher is better (max = 1.0).

#### 4. ROC-AUC
**Definition**: Area under the Receiver Operating Characteristic curve

**Interpretation**: 
- 1.0 = Perfect classifier
- 0.5 = Random guessing
- >0.8 = Good performance

### Confusion Matrix

A 2×2 table showing prediction vs. reality:

```
                Predicted
              Not Swapped  Swapped
Actual
Not Swapped      TN         FP
Swapped          FN         TP
```

- **TN (True Negative)**: Correctly identified as not swapped
- **TP (True Positive)**: Correctly identified as swapped
- **FP (False Positive)**: Incorrectly predicted as swapped (false alarm)
- **FN (False Negative)**: Missed a swap (false negative)

### Success Criteria

Our model should achieve:
- **Precision ≥ 90%**: Low false positive rate (don't corrupt good data)
- **Recall ≥ 80%**: Catch most swaps
- **Preserve ≥93% of perfect trials**: Don't introduce errors in trials that were already correct
- **Improve ≥50% of problematic trials**: Reduce errors in trials with known swaps

### Evaluation Process

1. **Train model** on training set
2. **Evaluate on validation set** during training (for hyperparameter tuning)
3. **Final evaluation on test set** (never seen during training)
4. **Compare to baseline**: Rule-based method performance
5. **Error analysis**: Examine false positives and false negatives to understand failure modes

---

## Integration

### Using the Trained Model

Once trained, the model is saved as:
- `ml_models/swap_detector_xgb.pkl`: The trained model
- `ml_models/feature_scaler.pkl`: Feature normalization parameters
- `ml_models/feature_imputer.pkl`: Missing value imputation parameters
- `ml_models/feature_names.pkl`: List of feature names (for reference)

### Integration with Existing Pipeline

The ML detector can be integrated into `correct_tracking_errors()`:

```python
def correct_tracking_errors(rawData, fps, use_ml_detection=True):
    # ... existing rule-based detection ...
    
    if use_ml_detection:
        # Extract features
        features = extract_all_frame_features(rawData, fps)
        
        # Load model and preprocessors
        model = load_model('ml_models/swap_detector_xgb.pkl')
        scaler = load_scaler('ml_models/feature_scaler.pkl')
        imputer = load_imputer('ml_models/feature_imputer.pkl')
        
        # Preprocess and predict
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]
        
        # Convert to swap segments
        swapped_frames = np.where(predictions == 1)[0]
        # ... apply corrections ...
```

### Hybrid Approach

We can combine ML and rule-based methods:
1. **Rule-based detection** as baseline (catches obvious cases)
2. **ML detection** for refinement (catches complex patterns)
3. **Consensus**: Only swap if both methods agree (very conservative)
4. **Union**: Swap if either method detects (more aggressive)

---

## Future Improvements

### Potential Enhancements

1. **Data Preprocessing**
   - **Gaussian Filtering**: Apply smoothing to position data before feature extraction
     - Reduces noise in raw tracking data
     - Use sigma = 4-5 for typical tracking data
     - Can improve feature quality, especially for speed and angle calculations
     - Already implemented as optional parameter in feature extraction functions

2. **Deep Learning Models**
   - **LSTM/GRU**: Sequence models that capture temporal dependencies
   - **1D CNN**: Convolutional layers to detect patterns in time series
   - May improve performance but require more data and computation

2. **Feature Engineering**
   - Add more domain-specific features
   - Feature selection to remove redundant features
   - Feature interactions (combinations of features)

3. **Active Learning**
   - Identify frames where model is uncertain
   - Manually label these "hard" examples
   - Retrain with additional labeled data

4. **Transfer Learning**
   - Train on all available trials
   - Fine-tune on specific experimental conditions
   - Adapt to new data types without full retraining

5. **Ensemble Methods**
   - Combine multiple models (XGBoost + Random Forest + Neural Network)
   - Voting or stacking for final prediction
   - Often improves robustness

6. **Real-time Processing**
   - Optimize feature extraction for speed
   - Use lighter models for real-time applications
   - Batch processing for offline analysis

---

## Glossary

- **Feature**: A measurable property of data (e.g., speed, angle)
- **Label**: The correct answer (swapped or not swapped)
- **Training**: The process of teaching a model using labeled examples
- **Prediction**: The model's guess for a new, unseen example
- **Overfitting**: When a model memorizes training data but fails on new data
- **Hyperparameter**: A setting that controls how the model learns (not learned from data)
- **Class Imbalance**: When one class (swapped) is much rarer than another (not swapped)
- **Precision**: Accuracy of positive predictions (fewer false alarms)
- **Recall**: Coverage of positive examples (catches most swaps)
- **F1-Score**: Balance between precision and recall

---

## References and Further Reading

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **Introduction to Machine Learning**: "Hands-On Machine Learning" by Aurélien Géron
- **Feature Engineering**: "Feature Engineering for Machine Learning" by Alice Zheng

---

## Appendix: Technical Details

### Feature Extraction Implementation

See `swap_correction/ml_features.py` for implementation details:
- `extract_frame_features()`: Extract features for a single frame
- `extract_all_frame_features()`: Extract features for all frames in a trial
- `extract_segment_features()`: Extract aggregate features for a segment

### Training Script

See `train_ml_swap_detector.py` for:
- Data loading and preprocessing
- Model training with hyperparameter tuning
- Evaluation and metrics calculation
- Model persistence

### Model Files

Trained models are saved in `ml_models/`:
- Model weights and structure
- Feature preprocessing parameters
- Evaluation results and feature importance

---

*Document Version: 1.0*  
*Last Updated: 2024*


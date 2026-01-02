# Learning Curve Analysis: Iterations 13-15 (Features V4)

## Overview

This report analyzes learning curves for iterations 13-15 using features_v4 (~36 features).
Each iteration used a sample size of 100 trials, randomly selected from the main dataset.

## Summary Statistics

### Per-Iteration Results

| Iteration | Model Type | Best Test F1 | Best Test % Clean | F1 at 50% | F1 at 100% | Improvement (50→100%) | Improvement (Last 20%) | More Data Helpful |
|:----------|:-----------|--------------:|------------------:|----------:|------------:|----------------------:|----------------------:|:------------------|
| 013 | LEVEL1 | 0.9603 | 98.00% | 0.9603 | 0.9506 | -0.0096 | -0.0026 | No |
| 013 | RAW | 0.9595 | 96.28% | 0.9572 | 0.9559 | -0.0013 | -0.0030 | No |
| 014 | LEVEL1 | 0.8847 | 94.03% | 0.8828 | 0.8807 | -0.0021 | -0.0024 | No |
| 014 | RAW | 0.9028 | 92.88% | 0.8961 | 0.9028 | +0.0067 | +0.0022 | Yes |
| 015 | LEVEL1 | 0.9383 | 96.80% | 0.9303 | 0.9383 | +0.0080 | +0.0022 | Yes |
| 015 | RAW | 0.9461 | 95.19% | 0.9374 | 0.9458 | +0.0083 | +0.0012 | Yes |

## Average Across Iterations

| Model Type | Avg Best Test F1 | Avg F1 at 50% | Avg F1 at 100% | Avg Improvement (50→100%) | Avg Improvement (Last 20%) | More Data Helpful (Count) |
|:-----------|-----------------:|--------------:|---------------:|--------------------------:|---------------------------:|:--------------------------|
| LEVEL1 | 0.9277 ± 0.0317 | 0.9244 | 0.9232 | -0.0012 ± 0.0072 | -0.0009 ± 0.0022 | 1/3 |
| RAW | 0.9361 ± 0.0242 | 0.9302 | 0.9348 | +0.0046 ± 0.0042 | +0.0002 ± 0.0023 | 2/3 |

## Key Findings

### Learning Curve Characteristics:

1. **Data Efficiency**: Analysis of how performance changes with training data size
2. **Consistency**: Comparison of learning curves across different random samples
3. **Model Comparison**: Level1 vs Raw model performance trends

### Recommendations:

- Based on the learning curves, determine if more training data would improve performance
- Assess the stability of learning curves across different random samples
- Compare Level1 and Raw models to understand their data requirements

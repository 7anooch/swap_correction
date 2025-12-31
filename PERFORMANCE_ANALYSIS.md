# Feature Extraction Performance Analysis

## Executive Summary

Feature extraction is slow (~32 frames/second, ~2 hours for 225,000 frames) due to several major bottlenecks. The primary issues are:

1. **O(n²) cumulative distance calculation** - Recalculates from frame 0 for each frame
2. **Redundant temporal context calculations** - Computes window statistics 4x per frame
3. **Repeated expensive function calls** - Calls `get_speed_from_df()` and `get_ht_cross_sign()` on full data for each frame
4. **Pandas indexing overhead** - Many `.iloc[]` and `.get()` calls

## Detailed Bottleneck Analysis

### 1. Cumulative Distance Calculation (CRITICAL - O(n²) complexity)

**Location**: `ml_features.py` lines 332-361

**Problem**: 
- For each frame `i`, the code loops through frames 1 to `i` to calculate cumulative distance
- For frame 9000, this does 9000 iterations
- Total operations: 1 + 2 + 3 + ... + n = n(n+1)/2 ≈ O(n²)

**Current Implementation**:
```python
if frame_idx > 0:
    head_dist = 0
    tail_dist = 0
    for i in range(1, frame_idx + 1):  # Loops through ALL previous frames!
        # Calculate distance for frame i
        head_dist += distance
```

**Impact**: 
- Frame 100: ~0.1 ms
- Frame 1000: ~1 ms  
- Frame 5000: ~5 ms
- Frame 9000: ~9 ms
- **Total for 9000 frames: ~40 seconds just for cumulative distance!**

**Solution**: Calculate incrementally (O(1) per frame):
```python
# Pre-compute cumulative distances once for all frames
cumulative_head_dist = np.zeros(len(data))
cumulative_tail_dist = np.zeros(len(data))
for i in range(1, len(data)):
    # Add distance from previous frame
    cumulative_head_dist[i] = cumulative_head_dist[i-1] + frame_distance
```

### 2. Temporal Context Features (MAJOR - 4x redundant calculations per frame)

**Location**: `ml_features.py` lines 363-422

**Problem**:
- For each frame, computes statistics for 4 different window sizes (5, 10, 20, 50 frames)
- Each window calculation:
  - Calls `get_speed_from_df()` on windowed data
  - Loops through window to compute alignment angles frame-by-frame
- This is called **4 times per frame** (once per window size)

**Current Implementation**:
```python
for window_size in window_sizes:  # [5, 10, 20, 50] - 4 iterations!
    window_data = data.iloc[start_idx:end_idx]
    window_hspd = metrics.get_speed_from_df(window_data, 'head', fps=fps)  # Expensive!
    # ... compute alignment angles in loop ...
```

**Impact**:
- Each window calculation: ~0.5-1 ms
- 4 windows × 9000 frames = 36,000 window calculations
- **Total: ~18-36 seconds**

**Solution**: Pre-compute speeds and alignment angles once, then use sliding window statistics:
```python
# Pre-compute once for all frames
all_speeds = metrics.get_speed_from_df(data, 'head', fps=fps)
all_alignment_angles = compute_alignment_angles(data)

# For each frame, just compute statistics on pre-computed arrays
for window_size in window_sizes:
    window_speeds = all_speeds[start_idx:end_idx]
    features[f'head_speed_mean_{window_size}'] = np.mean(window_speeds)
```

### 3. Repeated Expensive Function Calls

**Location**: Throughout `extract_frame_features()`

**Problems**:

#### a) `get_speed_from_df()` called multiple times
- Line 116: Called for head speed (full data)
- Line 117: Called for tail speed (full data)  
- Line 372: Called for each window size (4x) for head
- Line 373: Called for each window size (4x) for tail
- **Total: 10 calls per frame!**

**Impact**: Even though each call is fast (~0.01 ms), the overhead adds up:
- 10 calls × 9000 frames = 90,000 function calls
- **Total: ~1-2 seconds**

**Solution**: Pre-compute speeds once:
```python
# Pre-compute once
hspd_all = metrics.get_speed_from_df(data, 'head', fps=fps)
tspd_all = metrics.get_speed_from_df(data, 'tail', fps=fps)

# Then just index into arrays
features['head_speed'] = hspd_all[frame_idx]
```

#### b) `get_ht_cross_sign()` called once per frame
- Line 293: Called on full data for each frame
- Processes entire dataframe each time

**Impact**: 
- ~0.1 ms per call × 9000 frames = **~1 second**

**Solution**: Pre-compute once:
```python
cross_sign_all = metrics.get_ht_cross_sign(data)
features['cross_sign'] = cross_sign_all[frame_idx]
```

### 4. Pandas Indexing Overhead

**Problem**: Many `.iloc[]` and `.get()` calls have overhead
- `data.iloc[frame_idx]` - creates a Series object
- `frame.get('xhead')` - dictionary-like lookup
- `data.iloc[start_idx:end_idx]` - creates DataFrame slice

**Impact**: 
- Profiling shows ~2.5 seconds spent in pandas indexing operations for 100 frames
- **For 9000 frames: ~225 seconds (3.75 minutes) just in indexing!**

**Solution**: Convert to NumPy arrays for faster access:
```python
# Convert once to NumPy
xhead = data['xhead'].values
yhead = data['yhead'].values
# ... etc

# Then use array indexing (much faster)
features['head_x'] = xhead[frame_idx]
```

## Performance Breakdown (Estimated)

For a 9000-frame trial:

| Operation | Time | Percentage |
|-----------|------|------------|
| Cumulative distance (O(n²)) | ~40s | 20% |
| Temporal context (4 windows) | ~30s | 15% |
| Pandas indexing overhead | ~225s | 60% |
| Speed/angle calculations | ~5s | 3% |
| Other features | ~2s | 1% |
| **Total** | **~302s (5 minutes)** | **100%** |

For 25 trials × 9000 frames = 225,000 frames:
- **Estimated total time: ~2.1 hours**

## Optimization Recommendations

### Priority 1: Fix Cumulative Distance (10-20x speedup potential)

**Change from O(n²) to O(n)**:
- Pre-compute cumulative distances once for entire trial
- Store in arrays, index per frame

**Expected improvement**: Reduces time from ~40s to ~0.1s per trial

### Priority 2: Pre-compute Expensive Functions (5-10x speedup)

**Pre-compute once**:
- `get_speed_from_df()` for head and tail (full data)
- `get_ht_cross_sign()` (full data)
- Alignment angles (full data)

**Then index into arrays** instead of calling functions per frame.

**Expected improvement**: Reduces redundant calculations

### Priority 3: Optimize Temporal Context (3-5x speedup)

**Instead of**:
- Computing window statistics 4x per frame

**Do**:
- Pre-compute all speeds and alignment angles
- Use NumPy sliding window functions (`np.convolve`, `pd.rolling`)
- Or vectorized operations with broadcasting

**Expected improvement**: Reduces from ~30s to ~5s per trial

### Priority 4: Reduce Pandas Overhead (2-3x speedup)

**Convert to NumPy arrays**:
- Extract position columns to NumPy arrays once
- Use array indexing instead of DataFrame operations
- Only use pandas when necessary

**Expected improvement**: Reduces indexing overhead significantly

## Expected Overall Improvement

With all optimizations:
- **Current**: ~2 hours for 225,000 frames
- **Optimized**: ~10-15 minutes for 225,000 frames
- **Speedup**: **8-12x faster**

## Implementation Strategy

1. **Create optimized version** of `extract_all_frame_features()`:
   - Pre-compute all expensive operations once
   - Use vectorized NumPy operations
   - Minimize pandas operations

2. **Keep current version** as fallback (for debugging/comparison)

3. **Test** optimized version to ensure feature values match

4. **Benchmark** to verify speedup

## Code Structure for Optimized Version

```python
def extract_all_frame_features_optimized(trial_data, fps=30):
    """
    Optimized version that pre-computes expensive operations.
    """
    n_frames = len(trial_data)
    
    # Pre-compute once (O(n) operations)
    hspd_all = metrics.get_speed_from_df(trial_data, 'head', fps=fps)
    tspd_all = metrics.get_speed_from_df(trial_data, 'tail', fps=fps)
    cross_sign_all = metrics.get_ht_cross_sign(trial_data)
    
    # Convert to NumPy for faster access
    xhead = trial_data['xhead'].values
    yhead = trial_data['yhead'].values
    # ... etc
    
    # Pre-compute cumulative distances (O(n))
    cumulative_head_dist = np.zeros(n_frames)
    cumulative_tail_dist = np.zeros(n_frames)
    # ... compute incrementally ...
    
    # Pre-compute alignment angles (O(n))
    alignment_angles_all = compute_alignment_angles_vectorized(trial_data)
    
    # For each frame, just assemble features from pre-computed arrays
    all_features = []
    for i in range(n_frames):
        features = {}
        # Fast array indexing
        features['head_x'] = xhead[i]
        features['head_speed'] = hspd_all[i]
        features['cross_sign'] = cross_sign_all[i]
        features['cumulative_head_distance'] = cumulative_head_dist[i]
        # ... etc
        
        # Temporal context: use pre-computed arrays with sliding windows
        for window_size in window_sizes:
            window_speeds = hspd_all[max(0, i-window_size+1):i+1]
            features[f'head_speed_mean_{window_size}'] = np.nanmean(window_speeds)
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)
```

## Conclusion

The main performance issues are:
1. **O(n²) cumulative distance** - Biggest bottleneck
2. **Redundant temporal context calculations** - Second biggest
3. **Pandas indexing overhead** - Third biggest
4. **Repeated function calls** - Minor but adds up

With proper optimization (pre-computation + vectorization), we can achieve **8-12x speedup**, reducing training data preparation from ~2 hours to ~10-15 minutes.


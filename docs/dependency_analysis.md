# Swap Correction Codebase Dependency Analysis

## Function-Level Dependency Tree

### Main Entry Point (`swap_correct.py`)
```
main()
├── utils.get_dirs()
├── loader.load_raw_data()
│   ├── loader._retrieve_raw_data()
│   │   ├── utils.find_file()
│   │   └── utils.read_csv()
│   └── loader.get_all_settings()
│       └── json.load()
└── tc.tracking_correction()
    ├── remove_edge_frames()
    │   ├── utils.flatten()
    │   └── utils.get_consecutive_ranges()
    ├── correct_tracking_errors()
    │   ├── flag_all_swaps()
    │   │   ├── flag_overlaps()
    │   │   ├── flag_sign_reversals()
    │   │   ├── flag_delta_mismatches()
    │   │   └── flag_overlap_sign_reversals()
    │   ├── utils.indices_to_segments()
    │   ├── correct_swapped_segments()
    │   └── correct_global_swap()
    │       ├── filter_data()
    │       └── metrics.get_speed_from_df()
    ├── validate_corrected_data() [DISABLED]
    │   ├── get_swapped_segments()
    │   └── correct_swapped_segments()
    ├── remove_overlaps()
    │   ├── get_overlap_edges()
    │   ├── flag_discontinuities()
    │   └── utils.filter_array()
    ├── interpolate_gaps() [DISABLED]
    │   ├── utils.get_value_segments()
    │   └── utils.ranges_to_list()
    └── filter_data() [DISABLED]
        └── filter_gaussian()
```

### Utility Functions (`utils.py`)
```
utils.py
├── get_dirs()
├── find_file()
├── read_csv()
├── flatten()
├── get_consecutive_ranges()
├── indices_to_segments()
├── filter_array()
├── get_value_segments()
└── ranges_to_list()
```

### Metrics Functions (`metrics.py`)
```
metrics.py
├── get_speed_from_df()
├── get_delta_in_frame()
├── get_delta_between_frames()
├── get_cross_segment_deltas()
├── get_segment_distance()
├── get_motion_vector()
├── get_orientation_vectors()
├── get_head_angle()
└── get_ht_cross_sign()
```

## Unused Code Analysis

### Completely Unused Functions
1. **Commented Out Functions**:
   - `compare_filtered_distributions()`
   - `examine_flags()`

2. **Unused Filter Functions**:
   - `filter_gaussian()` - defined but never called directly
   - `filter_meanmed()` - defined but never called
   - `filter_median()` - defined but never called

3. **Redundant Flag Functions**:
   - `flag_min_delta_mismatches()` - redundant with `flag_delta_mismatches()`
   - `flag_overlap_mismatches()` - not used in main correction flow
   - `flag_overlap_minimum_mismatches()` - not used in main correction flow

4. **Legacy Functions**:
   - `_get_alignment_angles_legacy()` - newer version exists

### Disabled Features
1. **Configuration Parameters** (in `swap_correct.py`):
   ```python
   VALIDATE = False      # validate_corrected_data() is not used
   INTERPOLATE = False   # interpolate_gaps() is not used
   FILTER_DATA = False   # filter_data() is not used
   SHOW_PLOTS = False    # Plotting functionality is disabled
   ```

## Recommendations for Streamlining

### 1. Code Removal
- Remove all commented out functions
- Remove unused filter functions or consolidate into a single filtering module
- Remove redundant flag functions or document their purpose
- Remove legacy functions or mark them with proper deprecation warnings

### 2. Code Organization
- Split `tracking_correction.py` into smaller modules:
  - `flagging.py` for all flag-related functions
  - `filtering.py` for all filter-related functions
  - `correction.py` for core correction logic

### 3. Configuration Management
- Remove unused configuration parameters or document their purpose
- Consider using a configuration class instead of global variables
- Add validation for configuration parameters

### 4. Documentation
- Add clear documentation about which functions are actively used
- Mark deprecated functions with proper deprecation warnings
- Document the purpose of disabled configuration parameters
- Add docstrings to all functions explaining their role in the correction pipeline

### 5. Testing
- Add tests for all actively used functions
- Remove tests for removed functions
- Add integration tests for the main correction pipeline

## Implementation Priority
1. Remove commented out and unused functions
2. Reorganize code into smaller modules
3. Update documentation
4. Clean up configuration
5. Update tests

## Notes
- The main correction pipeline is relatively streamlined
- Most unused code is in supporting functions
- The core functionality is well-tested
- Configuration could be more flexible 
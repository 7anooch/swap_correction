"""
Tests for the utils module.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from swap_corrector.utils import (
    get_dirs,
    find_file,
    read_csv,
    export_json,
    load_json,
    file_exists,
    get_bounds,
    get_min_length,
    get_max_length,
    create_sample_matrix,
    flatten,
    merge,
    filter_array,
    match_arrays,
    mismatch_arrays,
    get_time_axis,
    where,
    get_consecutive_values,
    ranges_to_list,
    ranges_to_edges,
    invert_ranges,
    fuse_ranges,
    filter_ranges,
    segment_lengths,
    segments_of_length,
    get_consecutive_ranges,
    get_value_segments,
    get_indices_steps,
    get_intervals,
    indices_to_segments,
    metrics_by_segment,
    unit_vector,
    get_angle,
    get_cross_sign,
    get_speed,
    ddt,
    get_distance,
    get_cross_segment_deltas,
    rolling_mean,
    rolling_median
)

@pytest.fixture
def sample_dir(tmp_path):
    """Create a sample directory with test files."""
    # Create test files
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    data.to_csv(tmp_path / "test.csv", index=False)
    
    json_data = {'key': 'value'}
    with open(tmp_path / "test.json", 'w') as f:
        json.dump(json_data, f)
    
    # Create subdirectories
    os.makedirs(tmp_path / "subdir1")
    os.makedirs(tmp_path / "subdir2")
    
    return str(tmp_path)

def test_get_dirs(sample_dir):
    """Test getting directories from a path."""
    dirs = get_dirs(sample_dir)
    assert len(dirs) == 2
    assert all(os.path.isdir(d) for d in dirs)

def test_find_file(sample_dir):
    """Test finding a file by substring."""
    file_path = find_file(sample_dir, "test.csv")
    assert os.path.exists(file_path)
    assert "test.csv" in file_path

def test_read_csv(sample_dir):
    """Test reading a CSV file."""
    data = read_csv(os.path.join(sample_dir, "test.csv"))
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert 'col1' in data.columns

def test_export_json(sample_dir):
    """Test exporting to JSON."""
    data = {'test': 'value'}
    export_json(sample_dir, "export.json", data)
    assert os.path.exists(os.path.join(sample_dir, "export.json"))

def test_load_json(sample_dir):
    """Test loading a JSON file."""
    data = load_json(sample_dir, "test.json")
    assert isinstance(data, dict)
    assert data['key'] == 'value'

def test_file_exists(sample_dir):
    """Test checking if a file exists."""
    assert file_exists(sample_dir, "test.csv")
    assert not file_exists(sample_dir, "nonexistent.csv")

def test_get_bounds():
    """Test getting bounds of an array."""
    data = np.array([1, 2, 3, 4, 5])
    bounds = get_bounds(data, buffer=0.1)
    assert len(bounds) == 2
    assert bounds[0] < bounds[1]

def test_get_min_length():
    """Test getting minimum length of arrays."""
    arrays = [np.array([1, 2, 3]), np.array([1, 2])]
    assert get_min_length(arrays) == 2

def test_get_max_length():
    """Test getting maximum length of arrays."""
    arrays = [np.array([1, 2, 3]), np.array([1, 2])]
    assert get_max_length(arrays) == 3

def test_create_sample_matrix():
    """Test creating a sample matrix."""
    samples = [np.array([1, 2, 3]), np.array([4, 5])]
    matrix = create_sample_matrix(samples, length=3)
    assert matrix.shape == (2, 3)
    assert np.isnan(matrix[1, 2])

def test_flatten():
    """Test flattening a jagged list."""
    jagged = [[1, 2], [3, 4, 5], [6]]
    flat = flatten(jagged)
    assert len(flat) == 6
    assert all(isinstance(x, np.int64) for x in flat)

def test_merge():
    """Test merging arrays."""
    a = [1, 2, 3]
    b = [3, 4, 5]
    merged = merge(a, b)
    assert len(merged) == 5
    assert all(x in merged for x in [1, 2, 3, 4, 5])

def test_filter_array():
    """Test filtering arrays."""
    a = [1, 2, 3, 4]
    b = [2, 4]
    filtered = filter_array(a, b)
    assert len(filtered) == 2
    assert all(x in filtered for x in [1, 3])

def test_match_arrays():
    """Test matching arrays."""
    a = [1, 2, 3, 4]
    b = [2, 4, 5]
    matched = match_arrays(a, b)
    assert len(matched) == 2
    assert all(x in matched for x in [2, 4])

def test_mismatch_arrays():
    """Test finding mismatches between arrays."""
    a = [1, 2, 3, 4]
    b = [2, 4, 5]
    mismatched = mismatch_arrays(a, b)
    assert len(mismatched) == 3
    assert all(x in mismatched for x in [1, 3, 5])

def test_get_time_axis():
    """Test creating a time axis."""
    time = get_time_axis(10, 30)
    assert len(time) == 10
    assert time[0] == 0
    assert np.allclose(time[-1], 9/30, rtol=1e-10)

def test_where():
    """Test finding indices where condition is met."""
    arr = np.array([1, 0, 1, 0, 1])
    idx = where(arr == 1)
    assert len(idx) == 3
    assert all(x in idx for x in [0, 2, 4])

def test_get_consecutive_values():
    """Test getting consecutive values."""
    nums = [1, 2, 3, 5, 6, 8]
    consecutive = get_consecutive_values(nums)
    assert len(consecutive) == 3
    assert len(consecutive[0]) == 3
    assert len(consecutive[1]) == 2
    assert len(consecutive[2]) == 1

def test_ranges_to_list():
    """Test converting ranges to list."""
    ranges = [(1, 3), (5, 5), (7, 8)]
    lst = ranges_to_list(ranges)
    assert len(lst) == 6
    assert all(x in lst for x in [1, 2, 3, 5, 7, 8])

def test_ranges_to_edges():
    """Test converting ranges to edges."""
    ranges = [(3, 5), (11, 15)]
    edges = ranges_to_edges(ranges)
    assert len(edges) == 4
    assert all(x in edges for x in [3, 6, 11, 16])

def test_invert_ranges():
    """Test inverting ranges."""
    ranges = [(1, 2), (5, 7)]
    inverted = invert_ranges(ranges, 10)
    assert len(inverted) == 3
    assert all(x in inverted for x in [(0, 0), (3, 4), (8, 9)])

def test_fuse_ranges():
    """Test fusing ranges."""
    ranges1 = [(7, 9), (2, 3), (16, 18)]
    ranges2 = [(14, 15), (1, 4), (9, 12)]
    fused = fuse_ranges(ranges1, ranges2)
    assert len(fused) == 3
    assert all(x in fused for x in [(1, 4), (7, 12), (14, 18)])

def test_get_speed():
    """Test calculating speed."""
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    speed = get_speed(x, y, fps=30, npoints=2)
    
    # Speed array should be same length as input
    assert len(speed) == len(x)
    # All values should be approximately sqrt(2) * fps
    assert np.allclose(speed, np.sqrt(2) * 30)

def test_get_distance():
    """Test calculating distance."""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    distance = get_distance(x, y, origin=(0, 0))
    assert len(distance) == 3
    assert np.allclose(distance, np.sqrt(2) * np.array([1, 2, 3]))

def test_rolling_mean():
    """Test calculating rolling mean."""
    x = np.array([1, 2, 3, 4, 5])
    mean = rolling_mean(x, w=3)
    assert len(mean) == 3
    assert np.allclose(mean, [2, 3, 4])

def test_rolling_median():
    """Test calculating rolling median."""
    x = np.array([1, 2, 3, 4, 5])
    median = rolling_median(x, w=3)
    assert len(median) == 3
    assert np.allclose(median, [2, 3, 4]) 
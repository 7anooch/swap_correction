"""Tests for the utils module."""
import os
import numpy as np
import pytest
from swap_correction import utils


@pytest.fixture
def sample_ranges():
    """Sample ranges for testing."""
    return [(1, 5), (7, 10), (13, 20)]


@pytest.fixture
def sample_filter_ranges():
    """Sample filter ranges for testing."""
    return [(10, 15), (2, 3), (19, 20)]


def test_get_dirs(tmp_path):
    """Test directory listing functionality."""
    # Create test directories
    test_dirs = ["dir1", "dir2", "Omit_dir"]
    for d in test_dirs:
        (tmp_path / d).mkdir()
    
    # Test without exclude
    dirs = utils.get_dirs(str(tmp_path))
    assert len(dirs) == 2  # Should exclude 'Omit_dir'
    assert all('Omit' not in d for d in dirs)
    
    # Test with custom exclude
    dirs = utils.get_dirs(str(tmp_path), exclude=['dir1', 'Omit'])
    assert len(dirs) == 1
    assert all('dir1' not in d and 'Omit' not in d for d in dirs)


def test_find_file(tmp_path):
    """Test file finding functionality."""
    # Create test files
    test_files = ["test_data.csv", "other_file.txt"]
    for f in test_files:
        (tmp_path / f).touch()
    
    # Test finding file
    found_file = utils.find_file(str(tmp_path), "data.csv")
    assert os.path.basename(found_file) == "test_data.csv"


def test_filter_ranges(sample_ranges, sample_filter_ranges):
    """Test range filtering functionality."""
    filtered = utils.filter_ranges(sample_ranges, sample_filter_ranges)
    expected = [(1, 1), (4, 5), (7, 9), (16, 18)]
    assert filtered == expected


def test_merge():
    """Test array merging functionality."""
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    merged = utils.merge(arr1, arr2)
    assert np.array_equal(merged, np.array([1, 2, 3, 4, 5, 6]))


def test_get_consecutive_ranges():
    """Test consecutive range detection."""
    arr = np.array([1, 2, 3, 5, 6, 8, 9, 10])
    ranges = utils.get_consecutive_ranges(arr)
    assert ranges == [(1, 3), (5, 6), (8, 10)] 
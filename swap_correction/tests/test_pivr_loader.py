import os
import numpy as np
import pandas as pd
import pytest
from unittest import mock
from swap_correction import pivr_loader

# --- Fixtures and Mocks ---
@pytest.fixture
def fake_dirs():
    return ["/path/sample1", "/path/sample2"]

@pytest.fixture
def fake_raw_data():
    return pd.DataFrame({
        'X-Head': [1, 2], 'Y-Head': [3, 4], 'X-Tail': [5, 6], 'Y-Tail': [7, 8],
        'X-Midpoint': [9, 10], 'Y-Midpoint': [11, 12], 'X-Centroid': [13, 14], 'Y-Centroid': [15, 16],
        'Xmin-bbox': [17, 18], 'Ymin-bbox': [19, 20], 'Xmax-bbox': [21, 22], 'Ymax-bbox': [23, 24],
        'stimulation': [0, 1]
    })

@pytest.fixture
def fake_settings():
    return {'Framerate': 30, 'Pixel per mm': 2, 'Source x': 10, 'Source y': 20}

@pytest.fixture
def fake_settings_updated():
    return {'Pixel per mm': 3}

# --- Tests ---
def test_get_sample_directories(fake_dirs):
    with mock.patch('swap_correction.utils.get_dirs', return_value=fake_dirs), \
         mock.patch('swap_correction.utils.find_file', return_value='data.csv'):
        result = pivr_loader.get_sample_directories('/source')
        assert result == fake_dirs

    # Test with one folder missing data.csv
    def find_file_side_effect(folder, fname):
        if folder == fake_dirs[0]:
            return 'data.csv'
        else:
            raise FileNotFoundError
    with mock.patch('swap_correction.utils.get_dirs', return_value=fake_dirs), \
         mock.patch('swap_correction.utils.find_file', side_effect=find_file_side_effect):
        result = pivr_loader.get_sample_directories('/source')
        assert result == [fake_dirs[0]]

def test_load_raw_data(fake_raw_data, fake_settings):
    # Patch utils.read_csv, get_settings, and _retrieve_raw_data
    with mock.patch('swap_correction.utils.read_csv', return_value=fake_raw_data), \
         mock.patch('swap_correction.pivr_loader.get_settings', return_value=(30, 2, np.array([10, 20]))), \
         mock.patch('swap_correction.pivr_loader._retrieve_raw_data', return_value=(fake_raw_data, 'data.csv')):
        df = pivr_loader.load_raw_data('/main')
        assert isinstance(df, pd.DataFrame)
        assert 'stimulus' in df.columns
        assert 'xhead' in df.columns
        assert np.allclose(df['xhead'], (fake_raw_data['X-Head'] - 10) / 2)

    # Test with px2mm=False
    with mock.patch('swap_correction.utils.read_csv', return_value=fake_raw_data), \
         mock.patch('swap_correction.pivr_loader.get_settings', return_value=(30, 2, np.array([10, 20]))), \
         mock.patch('swap_correction.pivr_loader._retrieve_raw_data', return_value=(fake_raw_data, 'data.csv')):
        df = pivr_loader.load_raw_data('/main', px2mm=False)
        assert np.allclose(df['xhead'], fake_raw_data['X-Head'])

def test__retrieve_raw_data(fake_raw_data):
    with mock.patch('swap_correction.utils.find_file', return_value='data.csv'), \
         mock.patch('swap_correction.utils.read_csv', return_value=fake_raw_data):
        df, path = pivr_loader._retrieve_raw_data('/main')
        assert isinstance(df, pd.DataFrame)
        assert path == 'data.csv'

def test_import_analysed_data(fake_raw_data):
    with mock.patch('swap_correction.utils.read_csv', return_value=fake_raw_data):
        df = pivr_loader.import_analysed_data('/source')
        assert isinstance(df, pd.DataFrame)
        assert 'X-Head' in df.columns

def test_export_to_PiVR(tmp_path, fake_raw_data):
    # Patch get_settings and _retrieve_raw_data
    with mock.patch('swap_correction.pivr_loader.get_settings', return_value=(30, 2, np.array([10, 20]))), \
         mock.patch('swap_correction.pivr_loader._retrieve_raw_data', return_value=(fake_raw_data.copy(), str(tmp_path/'data.csv'))):
        data = pd.DataFrame({
            'xhead': [1, 2], 'yhead': [3, 4], 'xtail': [5, 6], 'ytail': [7, 8],
            'xmid': [9, 10], 'ymid': [11, 12], 'xctr': [13, 14], 'yctr': [15, 16],
            'xmin': [17, 18], 'ymin': [19, 20], 'xmax': [21, 22], 'ymax': [23, 24]
        })
        pivr_loader.export_to_PiVR(str(tmp_path), data, suffix='test', mm2px=True)
        # Check that the file was created
        files = list(tmp_path.glob('*.csv'))
        assert any('test' in f.name for f in files)

def test_get_all_settings(tmp_path):
    # Create a fake settings file
    settings = {'Framerate': 30, 'Pixel per mm': 2, 'Source x': 10, 'Source y': 20}
    settings_path = tmp_path / 'experiment_settings.json'
    with open(settings_path, 'w') as f:
        import json
        json.dump(settings, f)
    # Should load successfully
    result = pivr_loader.get_all_settings(str(tmp_path))
    assert result == settings
    # Should return None if file missing or invalid
    assert pivr_loader.get_all_settings(str(tmp_path), fileName='nonexistent.json') is None

def test_get_settings(tmp_path):
    # Create both settings and updated settings
    settings = {'Framerate': 30, 'Pixel per mm': 2, 'Source x': 10, 'Source y': 20}
    updated = {'Pixel per mm': 3}
    settings_path = tmp_path / 'experiment_settings.json'
    updated_path = tmp_path / 'experiment_settings_updated.json'
    import json
    with open(settings_path, 'w') as f:
        json.dump(settings, f)
    with open(updated_path, 'w') as f:
        json.dump(updated, f)
    fps, ppmm, source = pivr_loader.get_settings(str(tmp_path))
    assert fps == 30
    assert ppmm == 3  # uses updated value
    assert np.allclose(source, [10, 20])
    # Test mmSource=True
    fps, ppmm, source = pivr_loader.get_settings(str(tmp_path), mmSource=True)
    assert np.allclose(source, [10/3, 20/3])
    # Test with no source keys
    settings2 = {'Framerate': 10, 'Pixel per mm': 1}
    with open(settings_path, 'w') as f:
        json.dump(settings2, f)
    fps, ppmm, source = pivr_loader.get_settings(str(tmp_path))
    assert np.allclose(source, [0, 0])

def test_import_distance_data(tmp_path):
    # Create fake data and distance file
    data = pd.DataFrame({'Frame': [2, 3, 4]})
    distance = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [10, 11, 12, 13, 14]})
    distance_path = tmp_path / 'distance_to_source.csv'
    distance.to_csv(distance_path, index=False)
    with mock.patch('swap_correction.utils.read_csv', return_value=distance):
        result = pivr_loader.import_distance_data(data, str(tmp_path) + os.sep, fileName='distance_to_source.csv')
        # Should skip first two rows (startFrame=2), and take column 1
        assert np.allclose(result.values, [12, 13, 14])

def test_get_fps():
    # Test collection (multiple samples)
    with mock.patch('swap_correction.utils.get_dirs', return_value=['/sample1']), \
         mock.patch('swap_correction.pivr_loader.get_all_settings', return_value={'Framerate': 42}):
        fps = pivr_loader.get_fps('/parent')
        assert fps == 42
    # Test single sample
    with mock.patch('swap_correction.utils.get_dirs', return_value=[]), \
         mock.patch('swap_correction.pivr_loader.get_all_settings', return_value={'Framerate': 24}):
        fps = pivr_loader.get_fps('/single')
        assert fps == 24

def test_get_led_data():
    # Patch get_dirs and import_analysed_data
    with mock.patch('swap_correction.utils.get_dirs', return_value=['/sample1']), \
         mock.patch('swap_correction.pivr_loader.import_analysed_data', return_value=pd.DataFrame({'stimulus': [0, 1, 0]})):
        led = pivr_loader.get_led_data('/parent')
        assert np.array_equal(led, [0, 1, 0]) 
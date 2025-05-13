import pytest
from unittest import mock
import numpy as np
import pandas as pd
from swap_correction import swap_correct

def test_compare_filtered_trajectories_basic(tmp_path):
    # Mock loader, plotting, and metrics
    fake_df = pd.DataFrame({'xhead': [0, 1], 'yhead': [0, 1], 'xtail': [0, 1], 'ytail': [0, 1], 'xctr': [0, 1], 'yctr': [0, 1]})
    with mock.patch('swap_correction.swap_correct.loader._retrieve_raw_data', return_value=(None, 'data.csv')) as m_retrieve, \
         mock.patch('swap_correction.swap_correct.loader.get_all_settings', return_value={'Framerate': 30}), \
         mock.patch('swap_correction.swap_correct.loader.load_raw_data', return_value=fake_df), \
         mock.patch('swap_correction.swap_correct.metrics.get_df_bounds', return_value=(0, 1)), \
         mock.patch('swap_correction.swap_correct.plotting.plot_trajectory') as m_plot_traj, \
         mock.patch('swap_correction.swap_correct.plotting.save_figure') as m_save_fig:
        swap_correct.compare_filtered_trajectories(str(tmp_path), outputPath=str(tmp_path), fileName='test.png', times=(0, 1), show=False)
        assert m_retrieve.called
        assert m_plot_traj.call_count == 2
        assert m_save_fig.called


def test_compare_filtered_trajectories_no_output_path(tmp_path):
    fake_df = pd.DataFrame({'xhead': [0, 1], 'yhead': [0, 1], 'xtail': [0, 1], 'ytail': [0, 1], 'xctr': [0, 1], 'yctr': [0, 1]})
    with mock.patch('swap_correction.swap_correct.loader._retrieve_raw_data', return_value=(None, 'data.csv')), \
         mock.patch('swap_correction.swap_correct.loader.get_all_settings', return_value={'Framerate': 30}), \
         mock.patch('swap_correction.swap_correct.loader.load_raw_data', return_value=fake_df), \
         mock.patch('swap_correction.swap_correct.metrics.get_df_bounds', return_value=(0, 1)), \
         mock.patch('swap_correction.swap_correct.plotting.plot_trajectory'), \
         mock.patch('swap_correction.swap_correct.plotting.save_figure') as m_save_fig:
        swap_correct.compare_filtered_trajectories(str(tmp_path), outputPath=None, fileName='test.png', times=None, show=True)
        assert m_save_fig.called


def test_compare_filtered_trajectories_missing_settings(tmp_path):
    fake_df = pd.DataFrame({'xhead': [0, 1], 'yhead': [0, 1], 'xtail': [0, 1], 'ytail': [0, 1], 'xctr': [0, 1], 'yctr': [0, 1]})
    with mock.patch('swap_correction.swap_correct.loader._retrieve_raw_data', return_value=(None, 'data.csv')), \
         mock.patch('swap_correction.swap_correct.loader.get_all_settings', return_value=None), \
         mock.patch('swap_correction.swap_correct.loader.load_raw_data', return_value=fake_df), \
         mock.patch('swap_correction.swap_correct.metrics.get_df_bounds', return_value=(0, 1)), \
         mock.patch('swap_correction.swap_correct.plotting.plot_trajectory'), \
         mock.patch('swap_correction.swap_correct.plotting.save_figure') as m_save_fig:
        with pytest.raises(TypeError):
            swap_correct.compare_filtered_trajectories(str(tmp_path), outputPath=None, fileName='test.png', times=None, show=True) 
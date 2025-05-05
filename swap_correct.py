"""
Main script for PiVR swap correction pipeline.
"""

import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Tuple

from swap_corrector import (
    plotting, utils, metrics,
    tracking_correction as tc,
    pivr_loader as loader,
    config,
    logger
)

# Initialize configuration
cfg = config.SwapCorrectionConfig()

def compare_filtered_trajectories(main_path: str, output_path: Optional[str] = None,
            file_name: str = 'compare_trajectories.png', times: Optional[Tuple[float, float]] = None,
            show: bool = True) -> None:
    """
    Compare trajectories from raw and filtered position data.
    Note: analysis csv file must have been generated.

    Args:
        main_path: Directory containing data for one sample
        output_path: Directory to export image file to; if None, do not save the image
        file_name: Name of the image file
        times: Range of times to display data for; if None, display data for entire sample
        show: Display the figure after saving
    """
    logger.info(f"Comparing trajectories for {main_path}")
    
    # Data Retrieval
    suffix = 'level1'
    _, data_path = loader._retrieve_raw_data(main_path)
    raw_data_filename = os.path.basename(data_path)
    name = raw_data_filename.split('.csv')[0]
    new_file_name = f"{name}_{suffix}.csv"

    fps = loader.get_all_settings(main_path)['Framerate']
    raw_data = loader.load_raw_data(main_path)
    processed_data = loader.load_raw_data(main_path, new_file_name)

    # Figure
    fig, axs = plt.subplots(1, 2, squeeze=True, figsize=(8, 4))

    titles = ['Raw', 'Processed']
    data = [raw_data, processed_data]

    # Plot
    for i, ax in enumerate(axs):
        plotting.plot_trajectory(ax, data[i], fps, times)
        xlim = metrics.get_df_bounds([processed_data], ['xhead', 'xtail'])
        ylim = metrics.get_df_bounds([processed_data], ['yhead', 'ytail'])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis('square')
        ax.set_title(titles[i])

    # Finish
    out_path = output_path if output_path else main_path
    plotting.save_figure(fig, file_name, out_path, show=show)
    logger.info(f"Saved comparison plot to {os.path.join(out_path, file_name)}")

def compare_filtered_distributions(main_path: str, output_path: Optional[str] = None,
            file_name: str = 'compare_distributions.png', show: bool = True) -> None:
    """
    Compare distributions from raw and filtered position data.
    Note: analysis csv file must have been generated.

    Args:
        main_path: Directory containing data for one sample
        output_path: Directory to export image file to; if None, do not save the image
        file_name: Name of the image file
        show: Display the figure after saving
    """
    logger.info(f"Comparing distributions for {main_path}")
    
    # Data Retrieval
    suffix = 'level1'
    _, data_path = loader._retrieve_raw_data(main_path)
    raw_data_filename = os.path.basename(data_path)
    name = raw_data_filename.split('.csv')[0]
    new_file_name = f"{name}_{suffix}.csv"

    raw_data = loader.load_raw_data(main_path)
    processed_data = loader.load_raw_data(main_path, new_file_name)
    data = [raw_data, processed_data]

    # Figure
    fig, axs = plt.subplots(3, 2, squeeze=True, figsize=(8, 4))
    fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.90, wspace=0.3, hspace=0.6)

    titles = ['Head-Tail Separation (mm)', 'Body Orientation (rad)', 'Reorientation Rate (rad/s)']
    spans = [(0, 5), (0, np.pi), (0, 1)]

    # Plot
    for j, df in enumerate(data):
        # Calculations
        dist = metrics.get_delta_in_frame(df, 'head', 'tail')
        ba = metrics.get_orientation(df)
        rrate = np.abs(np.diff(ba))
        vals = [dist, ba, rrate]

        # Figure
        for i, ax in enumerate(axs[:, j]):
            plotting.histogram(ax, vals[i], 100, spans[i], True)
            ax.set_title(titles[i])
            if i == 2:
                ax.set_ylim(0, 100)

    # Finish
    out_path = output_path if output_path else main_path
    plotting.save_figure(fig, file_name, out_path, show=show)
    logger.info(f"Saved distribution comparison plot to {os.path.join(out_path, file_name)}")

def examine_flags(main_path: str, output_path: Optional[str] = None, show: bool = True,
            file_name: str = 'flags.png', times: Optional[Tuple[float, float]] = None,
            label_frames: bool = False) -> None:
    """
    Compare flagged frames and verified swap frames.

    Args:
        main_path: Directory containing data for one sample
        output_path: Directory to export image file to; if None, do not save the image
        show: Display the figure after saving
        file_name: Name of the image file
        times: Range of times to display data for; if None, display data for entire sample
        label_frames: Label x-axis in units of frames instead of seconds
    """
    logger.info(f"Examining flags for {main_path}")
    
    # Data Retrieval
    fps = loader.get_all_settings(main_path)['Framerate']
    raw_data = loader.load_raw_data(main_path)
    processed_data = loader.load_raw_data(main_path, cfg.filtered_data_filename)

    raw_data = tc.remove_edge_frames(raw_data)

    # Initialize flag detector
    flag_detector = tc.FlagDetector()
    flags = flag_detector.detect_all_flags(raw_data, fps, debug=cfg.debug)

    # Filter out overlaps
    filt = utils.merge(flags['overlaps'], flags['overlaps'] + 1)
    flags['delta_mismatches'] = utils.filter_array(flags['delta_mismatches'], filt)
    flags['sign_reversals'] = utils.filter_array(flags['sign_reversals'], filt)

    # Convert overlap frames to segments
    overlaps = utils.get_consecutive_ranges(flags['overlaps'])

    # Curl Detection & Validation
    int_data = tc.correct_tracking_errors(raw_data)
    curls = tc.get_curl_segments(int_data, overlaps=False, debug=cfg.debug)
    rem_swaps = tc.flag_swaps_after_curl(int_data, debug=cfg.debug)

    # Figure
    fig, axs = plt.subplots(2, 1, squeeze=True, figsize=(12, 6))
    fig.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.95, wspace=0.3, hspace=0.6)

    err_frames = [[], [], flags['delta_mismatches'], flags['overlap_sign_reversals'],
                 flags['overlap_minimum_mismatches'], flags['sign_reversals']]
    segments = [curls, rem_swaps, [], [], overlaps, []]
    cols = ['black', 'crimson', 'fuchsia', 'cyan', 'blue', 'lime']

    time = np.arange(0, len(raw_data['xhead'])) if label_frames else utils.get_time_axis(len(raw_data['xhead']), fps)
    dt = time[1] - time[0]
    delta = dt / 20
    titles = ['Flags', 'Overlaps and Curls']
    axes = ['Raw BA', 'Corrected BA']
    data = [raw_data, processed_data]

    # Plot
    for j, ax in enumerate(axs):
        # Line plot for reference
        ba = metrics.get_orientation(data[j])
        ax.plot(time, ba, c='grey')

        # Use vertical lines to indicate where errors detected
        if j == 0:
            for k, flag in enumerate(err_frames):
                for fr in flag:
                    ax.axvline(time[fr] + delta * k, c=cols[k], alpha=1.0, lw=1.5)

        # Use shaded regions to indicate overlaps, curls, etc
        else:
            for k, segs in enumerate(segments):
                for a, b in segs:
                    c = min(b + 1, raw_data.shape[0] - 1)
                    ax.fill_between(time[a:c], np.pi, 0, color=cols[k], alpha=0.2)

        if times:
            if label_frames:
                ax.set_xlim(times[0] * fps, times[1] * fps)
            else:
                ax.set_xlim(times)
        ax.set_ylim([0, np.pi])
        ax.set_xlabel('Frame' if label_frames else 'Time (s)')
        ax.set_ylabel(axes[j])
        ax.set_title(titles[j])

    # Finish
    out_path = output_path if output_path else main_path
    plotting.save_figure(fig, file_name, out_path, show=show)
    logger.info(f"Saved flag examination plot to {os.path.join(out_path, file_name)}")

if __name__ == '__main__':
    # Open dialogue to get target directory
    msg = 'Select a PiVR trial folder or a parent folder containing multiple trials.'
    source_dir = fd.askdirectory(title=msg)
    logger.info(f"Selected source directory: {source_dir}")

    # Retrieve sample directories / detect if target directory is single sample
    samples = utils.get_dirs(source_dir)
    if len(samples) == 0:
        samples = [source_dir]
    n_samples = len(samples)
    
    # Process each sample
    logger.info(f"Processing {n_samples} samples")
    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i+1}/{n_samples}: {sample}")

        if True:  # set to false to skip filtering and just plot
            data = loader.load_raw_data(sample)
            fps = loader.get_all_settings(sample)['Framerate']
            data = tc.tracking_correction(
                data, fps,
                filterData=cfg.filter_data,
                swapCorrection=cfg.fix_swaps,
                validate=cfg.validate,
                removeErrors=cfg.remove_errors,
                interp=cfg.interpolate,
                debug=cfg.debug
            )
            
            # Save processed data
            loader.save_data(data, sample, cfg.filtered_data_filename)
            logger.info(f"Saved processed data for {sample}")

        # Generate diagnostic plots if enabled
        if cfg.diagnostic_plots:
            compare_filtered_trajectories(sample, times=cfg.times, show=cfg.show_plots)
            compare_filtered_distributions(sample, show=cfg.show_plots)
            examine_flags(sample, times=cfg.times, show=cfg.show_plots)

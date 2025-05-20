import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from swap_correction.tracking.correction import correction
from swap_correction.tracking.flagging import flags
from swap_correction.tracking.filtering import filters
from swap_correction.metrics import Metrics

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'tests', 'test_data')
SAMPLE_EXPERIMENT = None  # Set to None to select randomly
SAMPLE_PREFIX = None

# --- Helper to add x, y, angle columns ---
def add_xy_angle_columns(df):
    df = df.copy()
    # No need to add x/y columns, just ensure angle is present
    dx = df['X-Midpoint'] - df['X-Tail']
    dy = df['Y-Midpoint'] - df['Y-Tail']
    df['angle'] = np.arctan2(dy, dx)
    return df

def get_random_experiment_dir():
    dirs = [d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d)) and not d.startswith('.')]
    if not dirs:
        print('No experiment directories found in test_data_dir, using default.')
        return "2024.11.13_00-48-15_Sussex_e2hex"
    return random.choice(dirs)

# --- Data Loading ---
def load_data():
    global SAMPLE_EXPERIMENT, SAMPLE_PREFIX
    if SAMPLE_EXPERIMENT is None:
        SAMPLE_EXPERIMENT = get_random_experiment_dir()
    SAMPLE_PREFIX = SAMPLE_EXPERIMENT.split('_Sussex')[0] if '_Sussex' in SAMPLE_EXPERIMENT else SAMPLE_EXPERIMENT.split('_')[0]
    print(f'Using experiment: {SAMPLE_EXPERIMENT}')
    exp_dir = os.path.join(TEST_DATA_DIR, SAMPLE_EXPERIMENT)
    raw_path = os.path.join(exp_dir, f"{SAMPLE_PREFIX}_data.csv")
    gt_path = os.path.join(exp_dir, f"{SAMPLE_PREFIX}_data_level2.csv")
    raw = pd.read_csv(raw_path)
    gt = pd.read_csv(gt_path)
    # Drop unnamed columns
    raw = raw.loc[:, ~raw.columns.str.startswith('Unnamed')]
    gt = gt.loc[:, ~gt.columns.str.startswith('Unnamed')]
    raw = add_xy_angle_columns(raw)
    gt = add_xy_angle_columns(gt)
    return raw, gt

# --- Run Pipeline ---
def run_pipeline(raw, fps=30):
    # Use the correct flagging and correction functions
    # 1. Flag swaps and errors
    swap_frames = flags.flag_all_swaps(raw, fps)
    # For demonstration, treat swaps as segments
    from swap_correction import utils
    swap_segments = utils.get_consecutive_ranges(swap_frames)
    # 2. Correct tracking errors (including swaps)
    corrected = correction.tracking_correction(raw.copy(), fps=fps, swapCorrection=True)
    # 3. Filter data
    filtered = filters.filter_data(corrected)
    # For visualization, return swaps as the only flag type
    flag_dict = {'swaps': swap_segments}
    return filtered, flag_dict

# --- Metrics ---
def compute_metrics(raw, filtered, gt):
    metrics = Metrics()
    raw_err = metrics.calculate_error_metrics(raw, gt)
    filtered_err = metrics.calculate_error_metrics(filtered, gt)
    return raw_err, filtered_err

# --- Visualization ---
def plot_trajectories(raw, filtered, gt, save_path=None):
    plt.figure(figsize=(8, 8))
    # Plot only tail trajectories with solid lines
    plt.plot(raw['X-Tail'], raw['Y-Tail'], label='Raw Tail', color='tab:blue', alpha=0.8, linestyle='-')
    plt.plot(filtered['X-Tail'], filtered['Y-Tail'], label='Corrected Tail', color='tab:orange', alpha=0.8, linestyle='-')
    plt.plot(gt['X-Tail'], gt['Y-Tail'], label='Ground Truth Tail', color='tab:green', alpha=0.8, linestyle='-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Tail Trajectories')
    plt.legend()
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_flagged_frames_time_series(flags, n_frames, flag_type, save_path=None):
    flagged = np.zeros(n_frames, dtype=bool)
    for start, end in flags:
        flagged[start:end+1] = True
    plt.figure(figsize=(12, 2))
    plt.plot(flagged.astype(int), drawstyle='steps-post', color='red', lw=2)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Frame')
    plt.ylabel('Flagged')
    plt.title(f'Flagged Frames (Time Series) - {flag_type}')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_flagged_on_trajectory(df, flags, flag_type, save_path=None):
    plt.figure(figsize=(8, 8))
    plt.plot(df['X-Tail'], df['Y-Tail'], color='gray', alpha=0.5, label='Tail Trajectory')
    # Overlay flagged frames
    flagged_idx = np.zeros(len(df), dtype=bool)
    for start, end in flags:
        flagged_idx[start:end+1] = True
    plt.scatter(df.loc[flagged_idx, 'X-Tail'], df.loc[flagged_idx, 'Y-Tail'],
                color='red', s=20, label=f'Flagged ({flag_type})', zorder=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Tail Trajectory with Flagged Frames ({flag_type})')
    plt.legend()
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_velocity_ratio(df):
    # Compute instantaneous velocity for head and tail
    vx_head = np.diff(df['X-Head'])
    vy_head = np.diff(df['Y-Head'])
    v_head = np.sqrt(vx_head**2 + vy_head**2)

    vx_tail = np.diff(df['X-Tail'])
    vy_tail = np.diff(df['Y-Tail'])
    v_tail = np.sqrt(vx_tail**2 + vy_tail**2)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = v_head / v_tail
        ratio[v_tail == 0] = np.nan
    return ratio

def plot_velocity_ratio(raw, filtered, gt, save_path=None):
    ratio_raw = compute_velocity_ratio(raw)
    ratio_filtered = compute_velocity_ratio(filtered)
    ratio_gt = compute_velocity_ratio(gt)
    plt.figure(figsize=(12, 4))
    plt.plot(ratio_raw, label='Raw', color='tab:blue', alpha=0.7)
    plt.plot(ratio_filtered, label='Corrected', color='tab:orange', alpha=0.7)
    plt.plot(ratio_gt, label='Ground Truth', color='tab:green', alpha=0.7)
    plt.ylabel('Velocity Ratio (Head/Tail)')
    plt.xlabel('Frame')
    plt.title('Instantaneous Velocity Ratio (Head/Tail)')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# --- Main Analysis ---
def main():
    print('Loading data...')
    raw, gt = load_data()
    print('Running correction pipeline...')
    filtered, flags = run_pipeline(raw)
    print('Computing metrics...')
    raw_err, filtered_err = compute_metrics(raw, filtered, gt)
    print('\n--- Summary Metrics ---')
    print('Raw Position Error:', raw_err['position_error'])
    print('Corrected Position Error:', filtered_err['position_error'])
    print('Raw Angle Error:', raw_err['angle_error'])
    print('Corrected Angle Error:', filtered_err['angle_error'])
    print('Flags:', flags)
    print('\nPlotting trajectories...')
    plot_trajectories(raw, filtered, gt, save_path='trajectories.png')
    # Visualize each flag type
    n_frames = len(raw)
    for flag_type, flag_ranges in flags.items():
        print(f'Plotting flagged frames for {flag_type}...')
        plot_flagged_frames_time_series(flag_ranges, n_frames, flag_type, save_path=f'flagged_{flag_type}_timeseries.png')
        plot_flagged_on_trajectory(raw, flag_ranges, flag_type, save_path=f'flagged_{flag_type}_trajectory.png')
    print('Figures saved as trajectories.png and flagged_*_timeseries.png/flagged_*_trajectory.png')

if __name__ == '__main__':
    main() 
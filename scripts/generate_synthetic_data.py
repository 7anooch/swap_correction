"""
Generate synthetic data with known swap patterns for testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

def generate_trajectory(n_frames: int, fps: float = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a smooth trajectory."""
    t = np.linspace(0, n_frames/fps, n_frames)
    
    # Generate smooth x and y coordinates using sine waves
    x = 100 + 50 * np.sin(0.5 * t) + 20 * np.sin(2 * t)
    y = 100 + 50 * np.cos(0.5 * t) + 20 * np.cos(2 * t)
    
    return x, y

def generate_orientation(trajectory: Tuple[np.ndarray, np.ndarray], n_frames: int) -> np.ndarray:
    """Generate orientation based on trajectory."""
    x, y = trajectory
    dx = np.gradient(x)
    dy = np.gradient(y)
    orientation = np.arctan2(dy, dx)
    return orientation

def generate_head_tail_positions(trajectory: Tuple[np.ndarray, np.ndarray], 
                               orientation: np.ndarray,
                               body_length: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate head and tail positions based on trajectory and orientation."""
    x, y = trajectory
    
    # Calculate head and tail positions
    x_head = x + body_length/2 * np.cos(orientation)
    y_head = y + body_length/2 * np.sin(orientation)
    x_tail = x - body_length/2 * np.cos(orientation)
    y_tail = y - body_length/2 * np.sin(orientation)
    
    return x_head, y_head, x_tail, y_tail

def introduce_swaps(x_head: np.ndarray, y_head: np.ndarray,
                   x_tail: np.ndarray, y_tail: np.ndarray,
                   swap_frames: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Introduce head-tail swaps at specified frames."""
    x_head_swapped = x_head.copy()
    y_head_swapped = y_head.copy()
    x_tail_swapped = x_tail.copy()
    y_tail_swapped = y_tail.copy()
    
    for start, end in swap_frames:
        # Swap head and tail positions
        x_head_swapped[start:end], x_tail_swapped[start:end] = x_tail[start:end], x_head[start:end]
        y_head_swapped[start:end], y_tail_swapped[start:end] = y_tail[start:end], y_head[start:end]
    
    return x_head_swapped, y_head_swapped, x_tail_swapped, y_tail_swapped

def generate_synthetic_data(n_frames: int = 1000, fps: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic tracking data with known swap patterns.
    
    Args:
        n_frames: Number of frames to generate
        fps: Frame rate
        
    Returns:
        Tuple of (raw_data, ground_truth) DataFrames
    """
    # Generate base trajectory
    t = np.linspace(0, n_frames/fps, n_frames)
    
    # Generate ground truth data
    head_x = 100 + 50 * np.sin(2 * np.pi * 0.5 * t)
    head_y = 100 + 50 * np.cos(2 * np.pi * 0.5 * t)
    tail_x = 100 + 30 * np.sin(2 * np.pi * 0.5 * t + np.pi)
    tail_y = 100 + 30 * np.cos(2 * np.pi * 0.5 * t + np.pi)
    
    # Calculate midpoint positions
    mid_x = (head_x + tail_x) / 2
    mid_y = (head_y + tail_y) / 2
    
    # Create ground truth DataFrame
    ground_truth = pd.DataFrame({
        'X-Head': head_x,
        'Y-Head': head_y,
        'X-Tail': tail_x,
        'Y-Tail': tail_y,
        'X-Midpoint': mid_x,
        'Y-Midpoint': mid_y
    })
    
    # Create raw data with swaps
    # Define swap regions (start_frame, end_frame)
    swap_frames = [
        (100, 150),   # Early swap
        (300, 350),   # Middle swap
        (600, 650),   # Late swap
        (800, 850)    # Very late swap
    ]
    
    # Introduce swaps
    head_x_raw, head_y_raw, tail_x_raw, tail_y_raw = introduce_swaps(
        head_x, head_y, tail_x, tail_y, swap_frames
    )
    
    # Calculate midpoint for raw data
    mid_x_raw = (head_x_raw + tail_x_raw) / 2
    mid_y_raw = (head_y_raw + tail_y_raw) / 2
    
    # Create raw DataFrame
    raw_data = pd.DataFrame({
        'X-Head': head_x_raw,
        'Y-Head': head_y_raw,
        'X-Tail': tail_x_raw,
        'Y-Tail': tail_y_raw,
        'X-Midpoint': mid_x_raw,
        'Y-Midpoint': mid_y_raw
    })
    
    return raw_data, ground_truth

def save_synthetic_data(raw_data: pd.DataFrame, ground_truth: pd.DataFrame, output_dir: str):
    """Save synthetic data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    raw_data.to_csv(output_path / 'synthetic_data.csv', index=False)
    
    # Save ground truth
    ground_truth.to_csv(output_path / 'synthetic_data_level2.csv', index=False)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot raw trajectory
    ax.plot(raw_data['X-Head'], raw_data['Y-Head'], 'b-', alpha=0.5, label='Raw Head')
    ax.plot(raw_data['X-Tail'], raw_data['Y-Tail'], 'r-', alpha=0.5, label='Raw Tail')
    ax.plot(raw_data['X-Midpoint'], raw_data['Y-Midpoint'], 'g-', alpha=0.5, label='Raw Midpoint')
    
    # Plot ground truth trajectory
    ax.plot(ground_truth['X-Head'], ground_truth['Y-Head'], 'b--', alpha=0.5, label='Ground Truth Head')
    ax.plot(ground_truth['X-Tail'], ground_truth['Y-Tail'], 'r--', alpha=0.5, label='Ground Truth Tail')
    ax.plot(ground_truth['X-Midpoint'], ground_truth['Y-Midpoint'], 'g--', alpha=0.5, label='Ground Truth Midpoint')
    
    ax.set_title('Synthetic Data Trajectories')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.axis('equal')
    
    plt.savefig(output_path / 'synthetic_data_visualization.png')
    plt.close()

def main():
    # Generate synthetic data
    raw_data, ground_truth = generate_synthetic_data()
    
    # Save data
    save_synthetic_data(raw_data, ground_truth, 'data/synthetic')

if __name__ == '__main__':
    main() 
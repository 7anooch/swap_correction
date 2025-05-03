import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, NamedTuple

class SwapEvent:
    """Detailed information about a swap event."""
    def __init__(self, data: pd.DataFrame, start_frame: int, end_frame: int, fps: int = 30):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.duration = end_frame - start_frame + 1
        
        # Get data for the event plus context
        context_window = 30  # frames
        self.start_with_context = max(0, start_frame - context_window)
        self.end_with_context = min(len(data), end_frame + context_window)
        self.data = data.iloc[self.start_with_context:self.end_with_context+1].copy()
        
        # Calculate various metrics
        self._calculate_speeds(fps)
        self._calculate_accelerations(fps)
        self._calculate_trajectories()
        self._calculate_body_metrics()
    
    def _calculate_speeds(self, fps: int):
        """Calculate speeds for head, tail, and midpoint."""
        for part in ['Head', 'Tail', 'Midpoint']:
            dx = np.diff(self.data[f'X-{part}'].values)
            dy = np.diff(self.data[f'Y-{part}'].values)
            speeds = np.sqrt(dx**2 + dy**2) * fps
            self.data[f'{part.lower()}_speed'] = np.concatenate([speeds, [speeds[-1]]])
    
    def _calculate_accelerations(self, fps: int):
        """Calculate accelerations and jerks."""
        for part in ['head', 'tail', 'midpoint']:
            speed = self.data[f'{part}_speed'].values
            self.data[f'{part}_acceleration'] = np.gradient(speed) * fps
            self.data[f'{part}_jerk'] = np.gradient(self.data[f'{part}_acceleration'].values) * fps
    
    def _calculate_trajectories(self):
        """Calculate trajectory properties."""
        # Calculate path curvature
        dx = np.gradient(self.data['X-Midpoint'].values)
        dy = np.gradient(self.data['Y-Midpoint'].values)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
        self.data['path_curvature'] = curvature
        
        # Calculate distance from arena center
        center_x, center_y = 1296/2, 972/2  # Image dimensions
        self.data['distance_from_center'] = np.sqrt(
            (self.data['X-Midpoint'] - center_x)**2 +
            (self.data['Y-Midpoint'] - center_y)**2
        )
    
    def _calculate_body_metrics(self):
        """Calculate body shape and orientation metrics."""
        # Body length
        self.data['body_length'] = np.sqrt(
            (self.data['X-Head'] - self.data['X-Tail'])**2 +
            (self.data['Y-Head'] - self.data['Y-Tail'])**2
        )
        
        # Body angle relative to movement direction
        dx = np.gradient(self.data['X-Midpoint'].values)
        dy = np.gradient(self.data['Y-Midpoint'].values)
        movement_angle = np.arctan2(dy, dx)
        
        body_dx = self.data['X-Head'] - self.data['X-Tail']
        body_dy = self.data['Y-Head'] - self.data['Y-Tail']
        body_angle = np.arctan2(body_dy, body_dx)
        
        self.data['body_movement_angle'] = np.abs(
            np.mod(movement_angle - body_angle + np.pi, 2*np.pi) - np.pi
        )

def find_swap_segments(data1: pd.DataFrame, data2: pd.DataFrame) -> List[Tuple[int, int]]:
    """Find segments where head/tail assignments differ between datasets."""
    # Calculate distances for normal and swapped configurations
    normal_dist = np.sqrt(
        (data1['X-Head'].values - data2['X-Head'].values)**2 +
        (data1['Y-Head'].values - data2['Y-Head'].values)**2 +
        (data1['X-Tail'].values - data2['X-Tail'].values)**2 +
        (data1['Y-Tail'].values - data2['Y-Tail'].values)**2
    )
    
    swapped_dist = np.sqrt(
        (data1['X-Head'].values - data2['X-Tail'].values)**2 +
        (data1['Y-Head'].values - data2['Y-Tail'].values)**2 +
        (data1['X-Tail'].values - data2['X-Head'].values)**2 +
        (data1['Y-Tail'].values - data2['Y-Head'].values)**2
    )
    
    # Find frames where positions are swapped
    is_swapped = normal_dist > swapped_dist
    swap_frames = np.where(is_swapped)[0]
    
    if len(swap_frames) == 0:
        return []
    
    # Find continuous segments
    segments = []
    segment_start = swap_frames[0]
    prev_frame = swap_frames[0]
    
    for frame in swap_frames[1:]:
        if frame > prev_frame + 1:
            segments.append((segment_start, prev_frame))
            segment_start = frame
        prev_frame = frame
    
    segments.append((segment_start, prev_frame))
    return segments

def analyze_high_speed_swaps(exp_dir: Path) -> List[SwapEvent]:
    """Analyze high-speed swap events in an experiment."""
    # Load data
    timestamp = exp_dir.name[:19]
    raw_data = pd.read_csv(exp_dir / f"{timestamp}_data.csv")
    level1_data = pd.read_csv(exp_dir / f"{timestamp}_data_level1.csv")
    level2_data = pd.read_csv(exp_dir / f"{timestamp}_data_level2.csv")
    
    # Calculate speeds for raw data
    dx = np.diff(raw_data['X-Midpoint'].values)
    dy = np.diff(raw_data['Y-Midpoint'].values)
    speeds = np.sqrt(dx**2 + dy**2) * 30  # 30 fps
    raw_data['midpoint_speed'] = np.concatenate([speeds, [speeds[-1]]])
    
    # Find swap segments
    raw_segments = find_swap_segments(raw_data, level2_data)
    level1_segments = find_swap_segments(level1_data, level2_data)
    
    # Analyze each raw swap
    high_speed_events = []
    for start, end in raw_segments:
        event = SwapEvent(raw_data, start, end)
        
        # Check if this is a high-speed event
        mean_speed = event.data['midpoint_speed'].mean()
        if mean_speed > np.percentile(raw_data['midpoint_speed'], 90):
            # Check if it was corrected in level1
            is_corrected = not any(
                l1_start <= start <= l1_end or l1_start <= end <= l1_end
                for l1_start, l1_end in level1_segments
            )
            event.was_corrected = is_corrected
            high_speed_events.append(event)
    
    return high_speed_events

def plot_event_analysis(event: SwapEvent, output_dir: Path, index: int):
    """Create detailed visualizations for a high-speed swap event."""
    # Create event directory
    event_dir = output_dir / f"event_{index}"
    event_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Trajectory
    plt.figure(figsize=(10, 10))
    plt.plot(event.data['X-Midpoint'], event.data['Y-Midpoint'], 'b-', label='Path')
    plt.plot(event.data['X-Midpoint'].iloc[0], event.data['Y-Midpoint'].iloc[0], 'go', label='Start')
    plt.plot(event.data['X-Midpoint'].iloc[-1], event.data['Y-Midpoint'].iloc[-1], 'ro', label='End')
    
    # Plot head-tail during swap
    swap_start = event.start_frame - event.start_with_context
    swap_end = event.end_frame - event.start_with_context
    plt.plot(event.data['X-Head'].iloc[swap_start:swap_end],
             event.data['Y-Head'].iloc[swap_start:swap_end], 'r.', label='Head')
    plt.plot(event.data['X-Tail'].iloc[swap_start:swap_end],
             event.data['Y-Tail'].iloc[swap_start:swap_end], 'b.', label='Tail')
    
    plt.title(f"Trajectory (Duration: {event.duration} frames)")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.axis('equal')
    plt.savefig(event_dir / 'trajectory.png')
    plt.close()
    
    # Plot 2: Time series of key metrics
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # Speed plot
    axes[0].plot(event.data.index, event.data['head_speed'], 'r-', label='Head')
    axes[0].plot(event.data.index, event.data['tail_speed'], 'b-', label='Tail')
    axes[0].axvspan(event.start_frame, event.end_frame, color='gray', alpha=0.2)
    axes[0].set_ylabel('Speed (px/s)')
    axes[0].legend()
    
    # Acceleration plot
    axes[1].plot(event.data.index, event.data['head_acceleration'], 'r-', label='Head')
    axes[1].plot(event.data.index, event.data['tail_acceleration'], 'b-', label='Tail')
    axes[1].axvspan(event.start_frame, event.end_frame, color='gray', alpha=0.2)
    axes[1].set_ylabel('Acceleration (px/s²)')
    
    # Body metrics
    axes[2].plot(event.data.index, event.data['body_length'], 'g-')
    axes[2].axvspan(event.start_frame, event.end_frame, color='gray', alpha=0.2)
    axes[2].set_ylabel('Body Length (px)')
    
    # Path curvature
    axes[3].plot(event.data.index, event.data['path_curvature'], 'k-')
    axes[3].axvspan(event.start_frame, event.end_frame, color='gray', alpha=0.2)
    axes[3].set_ylabel('Path Curvature')
    axes[3].set_xlabel('Frame')
    
    plt.tight_layout()
    plt.savefig(event_dir / 'metrics.png')
    plt.close()
    
    # Save event data
    event.data.to_csv(event_dir / 'event_data.csv')

def analyze_all_experiments(data_dir: str):
    """Analyze high-speed swaps across all experiments."""
    data_dir = Path(data_dir)
    output_dir = Path("high_speed_analysis")
    output_dir.mkdir(exist_ok=True)
    
    all_events = []
    
    # Process each experiment directory
    for exp_dir in data_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        try:
            print(f"\nAnalyzing {exp_dir.name}...")
            events = analyze_high_speed_swaps(exp_dir)
            all_events.extend(events)
            print(f"Found {len(events)} high-speed swaps")
            
            # Plot each event
            for i, event in enumerate(events):
                plot_event_analysis(event, output_dir / exp_dir.name, i)
            
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # Analyze patterns in high-speed swaps
    print("\nHigh-Speed Swap Analysis:")
    print(f"Total high-speed swaps found: {len(all_events)}")
    
    corrected = [e for e in all_events if e.was_corrected]
    uncorrected = [e for e in all_events if not e.was_corrected]
    
    print(f"Corrected: {len(corrected)} ({len(corrected)/len(all_events)*100:.1f}%)")
    print(f"Uncorrected: {len(uncorrected)} ({len(uncorrected)/len(all_events)*100:.1f}%)")
    
    # Compare characteristics of corrected vs uncorrected swaps
    metrics = {
        'Duration': lambda e: e.duration,
        'Mean Speed': lambda e: e.data['midpoint_speed'].mean(),
        'Max Speed': lambda e: e.data['midpoint_speed'].max(),
        'Mean Acceleration': lambda e: e.data['midpoint_acceleration'].mean(),
        'Mean Curvature': lambda e: e.data['path_curvature'].mean(),
        'Mean Body Length': lambda e: e.data['body_length'].mean(),
        'Mean Body-Movement Angle': lambda e: e.data['body_movement_angle'].mean()
    }
    
    print("\nMetric Comparison (Corrected vs Uncorrected):")
    for name, func in metrics.items():
        corrected_vals = [func(e) for e in corrected]
        uncorrected_vals = [func(e) for e in uncorrected]
        print(f"\n{name}:")
        print(f"  Corrected: {np.mean(corrected_vals):.2f} ± {np.std(corrected_vals):.2f}")
        print(f"  Uncorrected: {np.mean(uncorrected_vals):.2f} ± {np.std(uncorrected_vals):.2f}")

if __name__ == "__main__":
    analyze_all_experiments("data") 
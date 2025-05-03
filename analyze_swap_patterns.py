import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

class SwapContext(NamedTuple):
    """Context information for a swap segment."""
    start_frame: int
    end_frame: int
    duration: int
    body_angles: np.ndarray  # Angles between head-midpoint-tail
    head_tail_distances: np.ndarray  # Distance between head and tail
    speeds: np.ndarray  # Overall speed of the animal
    accelerations: np.ndarray  # Rate of speed change
    jerks: np.ndarray  # Rate of acceleration change
    curvature: np.ndarray  # Local curvature of the body
    distance_to_border: np.ndarray  # Distance to nearest image border
    body_length: np.ndarray  # Distance from head to tail
    overlap_proximity: bool  # Whether swap occurs near point overlap
    border_following: bool  # Whether animal is following arena border
    is_corrected_level1: bool  # Whether Level 1 corrected this swap

def load_data_files(exp_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw, level1, and level2 data files for an experiment."""
    exp_dir = Path(exp_dir)
    timestamp = exp_dir.name[:19]
    
    raw_file = exp_dir / f"{timestamp}_data.csv"
    level1_file = exp_dir / f"{timestamp}_data_level1.csv"
    level2_file = exp_dir / f"{timestamp}_data_level2.csv"
    
    if not all(f.exists() for f in [raw_file, level1_file, level2_file]):
        raise FileNotFoundError(f"Missing data files in {exp_dir}")
    
    raw_data = pd.read_csv(raw_file)
    level1_data = pd.read_csv(level1_file)
    level2_data = pd.read_csv(level2_file)
    
    return raw_data, level1_data, level2_data

def calculate_body_angle(x_head: float, y_head: float, 
                        x_mid: float, y_mid: float,
                        x_tail: float, y_tail: float) -> float:
    """Calculate angle between head-midpoint-tail."""
    vec1 = np.array([x_head - x_mid, y_head - y_mid])
    vec2 = np.array([x_tail - x_mid, y_tail - y_mid])
    
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    # Normalize vectors
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2
    
    # Calculate angle
    dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    return angle

def calculate_curvature(x: np.ndarray, y: np.ndarray, window: int = 5) -> float:
    """Calculate local curvature using a window of points."""
    if len(x) < window:
        return 0
    
    # Use a sliding window to calculate average curvature
    curvatures = []
    for i in range(len(x) - window + 1):
        points = np.column_stack((x[i:i+window], y[i:i+window]))
        
        # Fit a circle to the points
        x_m = np.mean(points[:, 0])
        y_m = np.mean(points[:, 1])
        u = points[:, 0] - x_m
        v = points[:, 1] - y_m
        
        # Calculate the curvature using the fitted circle
        Suv = np.sum(u * v)
        Suu = np.sum(u ** 2)
        Svv = np.sum(v ** 2)
        Suuv = np.sum(u ** 2 * v)
        Suvv = np.sum(u * v ** 2)
        Suuu = np.sum(u ** 3)
        Svvv = np.sum(v ** 3)
        
        # Solve the system of equations
        A = np.array([[Suu, Suv], [Suv, Svv]])
        B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
        
        try:
            uc, vc = np.linalg.solve(A, B)
            r = np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(points))
            curvatures.append(1/r if r > 0 else 0)
        except np.linalg.LinAlgError:
            curvatures.append(0)
    
    return np.mean(curvatures)

def get_distance_to_border(x: float, y: float, image_width: float = 1296, image_height: float = 972) -> float:
    """Calculate distance to nearest image border."""
    dist_left = x
    dist_right = image_width - x
    dist_top = y
    dist_bottom = image_height - y
    return min(dist_left, dist_right, dist_top, dist_bottom)

def calculate_kinematics(positions: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate speed, acceleration, and jerk from position data."""
    # Calculate velocities (first derivative)
    velocities = np.gradient(positions, axis=0) * fps
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Calculate accelerations (second derivative)
    accelerations = np.gradient(velocities, axis=0) * fps
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)
    
    # Calculate jerks (third derivative)
    jerks = np.gradient(accelerations, axis=0) * fps
    jerk_magnitudes = np.linalg.norm(jerks, axis=1)
    
    return speeds, accel_magnitudes, jerk_magnitudes

def detect_border_following(positions: np.ndarray, distances: np.ndarray, 
                          min_frames: int = 10, max_distance: float = 50) -> bool:
    """Detect if the animal is following the arena border."""
    # Check if animal stays near border for extended period
    near_border = distances < max_distance
    
    if not np.any(near_border):
        return False
    
    # Find continuous segments near border
    segments = np.where(near_border)[0]
    if len(segments) < min_frames:
        return False
    
    # Calculate movement parallel to nearest border
    parallel_movement = 0
    for i in range(1, len(positions)):
        if near_border[i] and near_border[i-1]:
            movement = positions[i] - positions[i-1]
            # Get nearest border (left, right, top, bottom)
            if distances[i] == positions[i][0]:  # Left border
                parallel_movement += abs(movement[1])
            elif distances[i] == 1296 - positions[i][0]:  # Right border
                parallel_movement += abs(movement[1])
            elif distances[i] == positions[i][1]:  # Top border
                parallel_movement += abs(movement[0])
            else:  # Bottom border
                parallel_movement += abs(movement[0])
    
    # Consider it border following if significant parallel movement
    return parallel_movement > min_frames * 2  # At least 2 pixels per frame

def detect_point_overlap(data: pd.DataFrame, window: int = 30) -> np.ndarray:
    """Detect frames where head and tail points overlap or nearly overlap."""
    head_tail_dist = np.sqrt(
        (data['X-Head'] - data['X-Tail'])**2 +
        (data['Y-Head'] - data['Y-Tail'])**2
    ).values
    
    # Points considered overlapping if distance is very small
    is_overlapping = head_tail_dist < 1e-6
    
    # Create a window around overlapping points
    near_overlap = np.zeros_like(is_overlapping)
    for i in range(len(is_overlapping)):
        start = max(0, i - window)
        end = min(len(is_overlapping), i + window)
        if np.any(is_overlapping[start:end]):
            near_overlap[i] = True
    
    return near_overlap

def analyze_swap_context(data: pd.DataFrame, start_frame: int, end_frame: int) -> SwapContext:
    """Analyze the context of a swap segment."""
    # Get data for the swap segment plus some context
    context_window = 30  # frames
    start_with_context = max(0, start_frame - context_window)
    end_with_context = min(len(data), end_frame + context_window)
    segment_data = data.iloc[start_with_context:end_with_context+1]
    
    # Calculate basic metrics
    angles = np.array([
        calculate_body_angle(
            row['X-Head'], row['Y-Head'],
            row['X-Midpoint'], row['Y-Midpoint'],
            row['X-Tail'], row['Y-Tail']
        ) for _, row in segment_data.iterrows()
    ])
    
    # Calculate head-tail distances and body length
    distances = np.sqrt(
        (segment_data['X-Head'] - segment_data['X-Tail'])**2 +
        (segment_data['Y-Head'] - segment_data['Y-Tail'])**2
    ).values
    
    # Calculate kinematics using midpoint
    positions = np.column_stack((
        segment_data['X-Midpoint'].values,
        segment_data['Y-Midpoint'].values
    ))
    speeds, accelerations, jerks = calculate_kinematics(positions)
    
    # Calculate curvature
    curvature = np.array([
        calculate_curvature(
            segment_data['X-Midpoint'].values,
            segment_data['Y-Midpoint'].values
        )
    ])
    
    # Calculate distance to border
    border_distances = np.array([
        get_distance_to_border(x, y)
        for x, y in zip(segment_data['X-Midpoint'], segment_data['Y-Midpoint'])
    ])
    
    # Check for nearby point overlaps
    near_overlap = detect_point_overlap(segment_data)
    has_nearby_overlap = np.any(near_overlap[
        context_window:context_window+end_frame-start_frame+1
    ])
    
    # Check for border following behavior
    is_following_border = detect_border_following(
        positions[context_window:context_window+end_frame-start_frame+1],
        border_distances[context_window:context_window+end_frame-start_frame+1]
    )
    
    return SwapContext(
        start_frame=start_frame,
        end_frame=end_frame,
        duration=end_frame - start_frame + 1,
        body_angles=angles[context_window:context_window+end_frame-start_frame+1],
        head_tail_distances=distances[context_window:context_window+end_frame-start_frame+1],
        speeds=speeds[context_window:context_window+end_frame-start_frame+1],
        accelerations=accelerations[context_window:context_window+end_frame-start_frame+1],
        jerks=jerks[context_window:context_window+end_frame-start_frame+1],
        curvature=curvature,
        distance_to_border=border_distances[context_window:context_window+end_frame-start_frame+1],
        body_length=distances[context_window:context_window+end_frame-start_frame+1],
        overlap_proximity=has_nearby_overlap,
        border_following=is_following_border,
        is_corrected_level1=False  # Will be set later
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

def analyze_experiment_patterns(raw_data: pd.DataFrame, level1_data: pd.DataFrame, 
                              level2_data: pd.DataFrame) -> List[SwapContext]:
    """Analyze swap patterns in an experiment."""
    # Find swap segments (comparing against level2 as ground truth)
    raw_segments = find_swap_segments(raw_data, level2_data)
    level1_segments = find_swap_segments(level1_data, level2_data)
    
    # Analyze context for each raw swap
    swap_contexts = []
    for start, end in raw_segments:
        context = analyze_swap_context(raw_data, start, end)
        
        # Check if this swap was corrected in level1
        is_corrected = not any(
            l1_start <= start <= l1_end or l1_start <= end <= l1_end
            for l1_start, l1_end in level1_segments
        )
        
        swap_contexts.append(context._replace(is_corrected_level1=is_corrected))
    
    return swap_contexts

def classify_swap_scenario(context: SwapContext) -> List[str]:
    """Classify the likely scenario(s) causing a swap."""
    scenarios = []
    
    # Check for curled body position
    mean_angle = np.mean(context.body_angles)
    if mean_angle < np.pi/2:  # Less than 90 degrees
        scenarios.append('curled_body')
    
    # Check for close head-tail proximity
    mean_distance = np.mean(context.head_tail_distances)
    if mean_distance < 20:  # Threshold in pixels
        scenarios.append('close_proximity')
    
    # Check for high speed
    mean_speed = np.mean(context.speeds)
    speed_threshold = np.percentile(context.speeds, 90) if len(context.speeds) > 0 else 0
    if mean_speed > speed_threshold:
        scenarios.append('high_speed')
    
    # Check for high acceleration/jerk (sudden movements)
    mean_accel = np.mean(context.accelerations)
    mean_jerk = np.mean(context.jerks)
    accel_threshold = np.percentile(context.accelerations, 90) if len(context.accelerations) > 0 else 0
    jerk_threshold = np.percentile(context.jerks, 90) if len(context.jerks) > 0 else 0
    if mean_accel > accel_threshold or mean_jerk > jerk_threshold:
        scenarios.append('sudden_movement')
    
    # Check for high curvature
    mean_curvature = np.mean(context.curvature)
    curv_threshold = np.percentile(context.curvature, 90) if len(context.curvature) > 0 else 0
    if mean_curvature > curv_threshold:
        scenarios.append('high_curvature')
    
    # Check for border proximity and following
    mean_border_dist = np.mean(context.distance_to_border)
    if mean_border_dist < 50:  # Threshold in pixels
        scenarios.append('near_border')
    if context.border_following:
        scenarios.append('border_following')
    
    # Check for rapid direction change
    angle_changes = np.abs(np.diff(context.body_angles))
    if len(angle_changes) > 0 and np.any(angle_changes > np.pi/2):  # More than 90 degrees
        scenarios.append('rapid_turn')
    
    # Check for body shape changes
    body_length_changes = np.abs(np.diff(context.body_length))
    if len(body_length_changes) > 0:
        length_change_threshold = np.percentile(body_length_changes, 90)
        if np.any(body_length_changes > length_change_threshold):
            scenarios.append('body_shape_change')
    
    # Check for proximity to point overlaps
    if context.overlap_proximity:
        scenarios.append('near_overlap')
    
    return scenarios if scenarios else ['unknown']

def analyze_all_experiments(data_dir: str) -> pd.DataFrame:
    """Analyze swap patterns across all experiments."""
    data_dir = Path(data_dir)
    all_contexts = []
    
    for exp_dir in data_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
        
        try:
            print(f"\nAnalyzing {exp_dir.name}...")
            raw_data, level1_data, level2_data = load_data_files(exp_dir)
            
            # Analyze swap patterns
            contexts = analyze_experiment_patterns(raw_data, level1_data, level2_data)
            
            # Classify scenarios
            for context in contexts:
                scenarios = classify_swap_scenario(context)
                for scenario in scenarios:
                    all_contexts.append({
                        'experiment': exp_dir.name,
                        'start_frame': context.start_frame,
                        'duration': context.duration,
                        'scenario': scenario,
                        'body_angle': np.mean(context.body_angles),
                        'head_tail_distance': np.mean(context.head_tail_distances),
                        'speed': np.mean(context.speeds),
                        'curvature': np.mean(context.curvature),
                        'border_distance': np.mean(context.distance_to_border),
                        'is_corrected_level1': context.is_corrected_level1
                    })
            
            print(f"Found {len(contexts)} swaps")
            
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    return pd.DataFrame(all_contexts)

def plot_scenario_analysis(results: pd.DataFrame, output_dir: str):
    """Create visualizations of swap scenarios."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Scenario distribution
    plt.figure(figsize=(12, 6))
    scenario_counts = results['scenario'].value_counts()
    plt.bar(scenario_counts.index, scenario_counts.values)
    plt.title('Distribution of Swap Scenarios')
    plt.xlabel('Scenario')
    plt.ylabel('Number of Swaps')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_distribution.png')
    plt.close()
    
    # Plot 2: Correction success by scenario
    plt.figure(figsize=(12, 6))
    correction_rates = results.groupby('scenario')['is_corrected_level1'].mean() * 100
    plt.bar(correction_rates.index, correction_rates.values)
    plt.title('Level 1 Correction Success by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Correction Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'correction_by_scenario.png')
    plt.close()
    
    # Plot 3: Swap duration by scenario
    plt.figure(figsize=(12, 6))
    plt.boxplot([
        results[results['scenario'] == scenario]['duration']
        for scenario in scenario_counts.index
    ], labels=scenario_counts.index)
    plt.title('Swap Duration by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Duration (frames)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'duration_by_scenario.png')
    plt.close()
    
    # Plot 4: Correlations between metrics
    metrics = ['body_angle', 'head_tail_distance', 'speed', 'curvature', 'border_distance']
    plt.figure(figsize=(15, 15))
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i >= j:
                continue
            plt.subplot(len(metrics)-1, len(metrics)-1, (i * (len(metrics)-1)) + j)
            plt.scatter(results[metric1], results[metric2], alpha=0.1)
            plt.xlabel(metric1)
            plt.ylabel(metric2)
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_correlations.png')
    plt.close()

def main():
    data_dir = "data"
    output_dir = "swap_analysis"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Analyze all experiments
    results = analyze_all_experiments(data_dir)
    
    if len(results) == 0:
        print("No swaps found. Please check the data directory.")
        return
    
    # Save results
    results.to_csv(Path(output_dir) / "swap_patterns.csv", index=False)
    
    # Create visualizations
    plot_scenario_analysis(results, output_dir)
    
    # Print summary statistics
    print("\nSwap Scenario Summary:")
    scenario_stats = results.groupby('scenario').agg({
        'duration': ['count', 'mean', 'std'],
        'is_corrected_level1': 'mean'
    })
    print(scenario_stats)
    
    # Print correlations between metrics
    metrics = ['body_angle', 'head_tail_distance', 'speed', 'curvature', 'border_distance']
    print("\nMetric Correlations:")
    print(results[metrics].corr())

if __name__ == "__main__":
    main() 
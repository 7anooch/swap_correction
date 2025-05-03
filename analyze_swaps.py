import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

def get_speed(data: pd.DataFrame, part: str, fps: int = 30) -> np.ndarray:
    """Calculate speed of a body part (head or tail)."""
    x = data[f'X-{part}'].values
    y = data[f'Y-{part}'].values
    
    # Calculate displacement between consecutive frames
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    
    # Calculate speed (pixels per frame)
    speed = np.sqrt(dx**2 + dy**2)
    
    return speed * fps  # Convert to pixels per second

def get_angle(data: pd.DataFrame) -> np.ndarray:
    """Calculate the angle of the body axis relative to x-axis."""
    dx = data['X-Head'].values - data['X-Tail'].values
    dy = data['Y-Head'].values - data['Y-Tail'].values
    return np.arctan2(dy, dx)

def load_data_files(exp_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw, level1, and level2 data files for an experiment."""
    exp_dir = Path(exp_dir)
    # Extract timestamp from directory name (first 19 characters: YYYY.MM.DD_HH-MM-SS)
    timestamp = exp_dir.name[:19]
    
    # Find the data files with correct naming pattern
    raw_file = exp_dir / f"{timestamp}_data.csv"
    level1_file = exp_dir / f"{timestamp}_data_level1.csv"
    level2_file = exp_dir / f"{timestamp}_data_level2.csv"
    
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_file}")
    if not level1_file.exists():
        raise FileNotFoundError(f"Level 1 data file not found: {level1_file}")
    if not level2_file.exists():
        raise FileNotFoundError(f"Level 2 data file not found: {level2_file}")
    
    try:
        # Load the data
        raw_data = pd.read_csv(raw_file)
        level1_data = pd.read_csv(level1_file)
        level2_data = pd.read_csv(level2_file)
        
        # Verify required columns exist
        required_cols = ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']
        for df, name in [(raw_data, 'raw'), (level1_data, 'level1'), (level2_data, 'level2')]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {name} data: {missing_cols}")
        
        return raw_data, level1_data, level2_data
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty data file found in {exp_dir}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV files in {exp_dir}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading data from {exp_dir}: {str(e)}")

def positions_match(data1: pd.DataFrame, data2: pd.DataFrame) -> np.ndarray:
    """
    Compare head/tail positions between two datasets.
    Returns boolean array where True means positions match (no swap).
    """
    # Calculate distances between head positions and tail positions
    head_to_head = np.sqrt(
        (data1['X-Head'].values - data2['X-Head'].values)**2 +
        (data1['Y-Head'].values - data2['Y-Head'].values)**2
    )
    tail_to_tail = np.sqrt(
        (data1['X-Tail'].values - data2['X-Tail'].values)**2 +
        (data1['Y-Tail'].values - data2['Y-Tail'].values)**2
    )
    
    # Calculate distances if positions were swapped
    head_to_tail = np.sqrt(
        (data1['X-Head'].values - data2['X-Tail'].values)**2 +
        (data1['Y-Head'].values - data2['Y-Tail'].values)**2
    )
    tail_to_head = np.sqrt(
        (data1['X-Tail'].values - data2['X-Head'].values)**2 +
        (data1['Y-Tail'].values - data2['Y-Head'].values)**2
    )
    
    # Positions match if normal distances are smaller than swapped distances
    normal_dist = head_to_head + tail_to_tail
    swapped_dist = head_to_tail + tail_to_head
    
    return normal_dist <= swapped_dist

def find_swap_segments(matches: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find continuous segments where positions don't match (swaps).
    Returns list of (start_frame, end_frame) tuples.
    """
    # Find frames where positions don't match (swaps)
    swap_frames = np.where(~matches)[0]
    
    if len(swap_frames) == 0:
        return []
    
    # Find continuous segments
    segments = []
    segment_start = swap_frames[0]
    prev_frame = swap_frames[0]
    
    for frame in swap_frames[1:]:
        if frame > prev_frame + 1:  # Gap found, end current segment
            segments.append((segment_start, prev_frame))
            segment_start = frame
        prev_frame = frame
    
    # Add the last segment
    segments.append((segment_start, prev_frame))
    
    return segments

def analyze_experiment(raw_data: pd.DataFrame, level1_data: pd.DataFrame, 
                      level2_data: pd.DataFrame) -> Dict:
    """
    Analyze swap corrections for one experiment by comparing against level2 (ground truth).
    """
    # Compare raw and level1 against level2 (ground truth)
    raw_matches = positions_match(raw_data, level2_data)
    level1_matches = positions_match(level1_data, level2_data)
    
    # Find swap segments
    raw_segments = find_swap_segments(raw_matches)
    level1_segments = find_swap_segments(level1_matches)
    
    # Calculate statistics
    total_swaps = len(raw_segments)
    remaining_after_level1 = len(level1_segments)
    corrected_in_level1 = total_swaps - remaining_after_level1
    
    # Calculate average duration of swaps
    raw_durations = [end - start + 1 for start, end in raw_segments]
    avg_duration = np.mean(raw_durations) if raw_durations else 0
    
    return {
        'total_swaps': total_swaps,
        'corrected_in_level1': corrected_in_level1,
        'remaining_after_level1': remaining_after_level1,
        'percentage_corrected_level1': (corrected_in_level1 / total_swaps * 100) if total_swaps > 0 else 0,
        'percentage_remaining_level1': (remaining_after_level1 / total_swaps * 100) if total_swaps > 0 else 0,
        'average_swap_duration': avg_duration
    }

def analyze_all_experiments(data_dir: str) -> pd.DataFrame:
    """Analyze all experiments in the data directory."""
    data_dir = Path(data_dir)
    results = []
    
    # Process each experiment directory
    for exp_dir in data_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue
            
        try:
            print(f"\nProcessing {exp_dir.name}...")
            raw_data, level1_data, level2_data = load_data_files(exp_dir)
            print(f"Successfully loaded data files")
            print(f"Raw data shape: {raw_data.shape}")
            print(f"Level 1 data shape: {level1_data.shape}")
            print(f"Level 2 data shape: {level2_data.shape}")
            
            print("Analyzing swaps...")
            stats = analyze_experiment(raw_data, level1_data, level2_data)
            stats['experiment'] = exp_dir.name
            results.append(stats)
            
            print(f"Found {stats['total_swaps']} total swaps")
            print(f"Level 1 corrected: {stats['corrected_in_level1']} swaps ({stats['percentage_corrected_level1']:.1f}%)")
            print(f"Average swap duration: {stats['average_swap_duration']:.1f} frames")
            
        except Exception as e:
            print(f"Error processing {exp_dir.name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

def plot_results(results: pd.DataFrame, output_dir: str):
    """Create visualizations of the analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Correction percentages
    plt.figure(figsize=(10, 6))
    plt.bar(['Corrected in Level 1', 'Remaining after Level 1'], 
            [results['percentage_corrected_level1'].mean(),
             results['percentage_remaining_level1'].mean()])
    plt.title('Average Correction Percentages')
    plt.ylabel('Percentage of Swaps')
    plt.savefig(output_dir / 'correction_percentages.png')
    plt.close()
    
    # Plot 2: Distribution of total swaps per experiment
    plt.figure(figsize=(10, 6))
    plt.hist(results['total_swaps'], bins=20)
    plt.title('Distribution of Total Swaps per Experiment')
    plt.xlabel('Number of Swap Segments')
    plt.ylabel('Number of Experiments')
    plt.savefig(output_dir / 'swap_distribution.png')
    plt.close()
    
    # Plot 3: Swaps per experiment
    plt.figure(figsize=(15, 6))
    experiments = results['experiment'].apply(lambda x: x[:19])  # Show only timestamps
    plt.bar(experiments, results['total_swaps'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Total Swap Segments per Experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Number of Swap Segments')
    plt.tight_layout()
    plt.savefig(output_dir / 'swaps_per_experiment.png')
    plt.close()
    
    # Plot 4: Swap duration distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['average_swap_duration'], bins=20)
    plt.title('Distribution of Average Swap Duration')
    plt.xlabel('Average Duration (frames)')
    plt.ylabel('Number of Experiments')
    plt.savefig(output_dir / 'swap_duration_distribution.png')
    plt.close()

def main():
    data_dir = "data"
    output_dir = "analysis_results"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Analyze all experiments
    results = analyze_all_experiments(data_dir)
    
    if len(results) == 0:
        print("No valid experiments were processed. Please check the data directory and file names.")
        return
    
    # Save results
    results.to_csv(Path(output_dir) / "swap_analysis.csv", index=False)
    
    # Create visualizations
    plot_results(results, output_dir)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total experiments analyzed: {len(results)}")
    print(f"Average swaps per experiment: {results['total_swaps'].mean():.2f}")
    print(f"Average swap duration (frames): {results['average_swap_duration'].mean():.2f}")
    print(f"Average percentage corrected in Level 1: {results['percentage_corrected_level1'].mean():.2f}%")
    print(f"Average percentage remaining after Level 1: {results['percentage_remaining_level1'].mean():.2f}%")

if __name__ == "__main__":
    main() 
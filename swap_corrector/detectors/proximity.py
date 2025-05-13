"""
Proximity-based swap detector.
"""

import numpy as np
import pandas as pd
from .base import SwapDetector
from ..config import SwapConfig
from typing import Dict, Any, List, Tuple

class ProximityDetector(SwapDetector):
    """Detector that identifies swaps based on proximity patterns."""
    
    def __init__(self, config: SwapConfig):
        """Initialize proximity detector.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.window_sizes = {
            'distance': 5,
            'velocity': 3,
            'outlier': 5
        }
        
    def _remove_isolated_detections(self, detections: np.ndarray, window: int = 2) -> np.ndarray:
        """Remove isolated detections that are likely false positives.
        
        Args:
            detections: Array of boolean detection flags
            window: Window size for checking neighboring frames
            
        Returns:
            Array of boolean detection flags with isolated detections removed
        """
        result = detections.copy()
        
        # Remove isolated detections
        for i in range(window, len(detections) - window):
            if detections[i]:
                # Keep detection if there are other detections nearby
                if not any(detections[i-window:i]) and not any(detections[i+1:i+window+1]):
                    result[i] = False
        
        # Extend segments to fill gaps
        for i in range(1, len(detections) - 1):
            if detections[i-1] and detections[i+1]:
                result[i] = True  # Fill gaps
            if detections[i-1] or detections[i+1]:  # Extend segments by one frame
                result[i] = True
        
        # Remove short segments
        min_segment_length = 2
        start_idx = None
        for i in range(len(detections)):
            if result[i]:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                if i - start_idx < min_segment_length:
                    result[start_idx:i] = False
                start_idx = None
        
        # Handle case where last segment extends to end
        if start_idx is not None and len(detections) - start_idx < min_segment_length:
            result[start_idx:] = False
            
        return result
        
    def _smooth_flags(self, flags: np.ndarray, min_duration: int = 2) -> np.ndarray:
        """Smooth detection flags to remove noise and fill gaps.
        
        Args:
            flags: Array of boolean detection flags
            min_duration: Minimum duration for a valid detection segment
            
        Returns:
            Smoothed array of boolean detection flags
        """
        # Remove isolated detections
        smoothed = self._remove_isolated_detections(flags)
        
        # Ensure minimum segment duration
        start_idx = None
        for i in range(len(smoothed)):
            if smoothed[i]:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                if i - start_idx < min_duration:
                    smoothed[start_idx:i] = False
                start_idx = None
        
        # Handle case where last segment extends to end
        if start_idx is not None and len(smoothed) - start_idx < min_duration:
            smoothed[start_idx:] = False
            
        return smoothed
        
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for detection.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate body length
        body_length = np.sqrt(
            (data['X-Head'] - data['X-Tail'])**2 +
            (data['Y-Head'] - data['Y-Tail'])**2
        )
        mean_body_length = body_length.mean()
        metrics['body_length'] = body_length
        metrics['mean_body_length'] = mean_body_length
        
        # Calculate frame-to-frame position changes
        head_dx = data['X-Head'] - data['X-Head'].shift(1)
        head_dy = data['Y-Head'] - data['Y-Head'].shift(1)
        tail_dx = data['X-Tail'] - data['X-Tail'].shift(1)
        tail_dy = data['Y-Tail'] - data['Y-Tail'].shift(1)
        
        # Calculate distances between consecutive positions
        head_distances = np.sqrt(head_dx**2 + head_dy**2)
        tail_distances = np.sqrt(tail_dx**2 + tail_dy**2)
        metrics['head_distances'] = head_distances
        metrics['tail_distances'] = tail_distances
        
        # Calculate relative distances with safe division
        metrics['relative_head_dist'] = np.where(
            mean_body_length > 0,
            head_distances / mean_body_length,
            0
        )
        metrics['relative_tail_dist'] = np.where(
            mean_body_length > 0,
            tail_distances / mean_body_length,
            0
        )
        
        # Calculate velocities
        head_velocity = np.sqrt(
            np.gradient(data['X-Head'])**2 + np.gradient(data['Y-Head'])**2
        )
        tail_velocity = np.sqrt(
            np.gradient(data['X-Tail'])**2 + np.gradient(data['Y-Tail'])**2
        )
        metrics['head_velocity'] = head_velocity
        metrics['tail_velocity'] = tail_velocity
        
        # Calculate relative velocities with safe division
        metrics['relative_head_velocity'] = np.where(
            mean_body_length > 0,
            head_velocity / mean_body_length,
            0
        )
        metrics['relative_tail_velocity'] = np.where(
            mean_body_length > 0,
            tail_velocity / mean_body_length,
            0
        )
        
        # Calculate velocity ratio with safe division
        max_velocity = np.maximum(head_velocity, tail_velocity)
        min_velocity = np.minimum(head_velocity, tail_velocity)
        metrics['velocity_ratio'] = np.where(max_velocity > 0, min_velocity / max_velocity, 1.0)
        
        # Calculate accelerations
        head_accel = np.abs(np.gradient(head_velocity))
        tail_accel = np.abs(np.gradient(tail_velocity))
        metrics['head_accel'] = head_accel
        metrics['tail_accel'] = tail_accel
        
        # Calculate acceleration ratio with safe division
        max_accel = np.maximum(head_accel, tail_accel)
        min_accel = np.minimum(head_accel, tail_accel)
        metrics['accel_ratio'] = np.where(max_accel > 0, min_accel / max_accel, 1.0)
        
        # Calculate angles
        dx = data['X-Head'] - data['X-Tail']
        dy = data['Y-Head'] - data['Y-Tail']
        angles = np.arctan2(dy, dx)
        metrics['angle'] = angles
        metrics['angle_change'] = np.abs(np.diff(angles, prepend=angles[0]))
        
        # Calculate relative position changes
        head_pos_prev = np.column_stack([data['X-Head'].shift(1), data['Y-Head'].shift(1)])
        tail_pos_prev = np.column_stack([data['X-Tail'].shift(1), data['Y-Tail'].shift(1)])
        head_pos_curr = np.column_stack([data['X-Head'], data['Y-Head']])
        tail_pos_curr = np.column_stack([data['X-Tail'], data['Y-Tail']])
        
        # Calculate distances between current and previous positions
        head_to_head = np.sqrt(np.sum((head_pos_curr - head_pos_prev)**2, axis=1))
        tail_to_tail = np.sqrt(np.sum((tail_pos_curr - tail_pos_prev)**2, axis=1))
        head_to_tail = np.sqrt(np.sum((head_pos_curr - tail_pos_prev)**2, axis=1))
        tail_to_head = np.sqrt(np.sum((tail_pos_curr - head_pos_prev)**2, axis=1))
        
        # Calculate relative position metrics
        metrics['head_to_head'] = head_to_head
        metrics['tail_to_tail'] = tail_to_tail
        metrics['head_to_tail'] = head_to_tail
        metrics['tail_to_head'] = tail_to_head
        
        # Calculate swap likelihood based on relative positions
        metrics['swap_likelihood'] = np.where(
            (head_to_tail < head_to_head * 0.3) & (tail_to_head < tail_to_tail * 0.3),  # Further reduced threshold
            1.0,
            0.0
        )
        
        return metrics
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect potential swaps based on proximity.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Array of boolean values indicating potential swaps
        """
        # Setup detector with data if not already done
        if self.data is None or len(self.data) != len(data) or not np.array_equal(self.data, data):
            self.setup(data)
        
        # Get metrics
        metrics = self._calculate_metrics(data)
        
        # Calculate smoothness metrics
        head_pos = np.column_stack([data['X-Head'], data['Y-Head']])
        tail_pos = np.column_stack([data['X-Tail'], data['Y-Tail']])
        
        # Calculate velocities
        head_vel = np.gradient(head_pos, axis=0)
        tail_vel = np.gradient(tail_pos, axis=0)
        
        # Calculate accelerations
        head_acc = np.gradient(head_vel, axis=0)
        tail_acc = np.gradient(tail_vel, axis=0)
        
        # Calculate jerk (rate of change of acceleration)
        head_jerk = np.gradient(head_acc, axis=0)
        tail_jerk = np.gradient(tail_acc, axis=0)
        
        # Calculate magnitudes
        head_vel_mag = np.linalg.norm(head_vel, axis=1)
        tail_vel_mag = np.linalg.norm(tail_vel, axis=1)
        head_acc_mag = np.linalg.norm(head_acc, axis=1)
        tail_acc_mag = np.linalg.norm(tail_acc, axis=1)
        head_jerk_mag = np.linalg.norm(head_jerk, axis=1)
        tail_jerk_mag = np.linalg.norm(tail_jerk, axis=1)
        
        # Normalize by body length
        mean_body_length = metrics['mean_body_length']
        head_vel_rel = np.where(mean_body_length > 0, head_vel_mag / mean_body_length, 0)
        tail_vel_rel = np.where(mean_body_length > 0, tail_vel_mag / mean_body_length, 0)
        head_acc_rel = np.where(mean_body_length > 0, head_acc_mag / mean_body_length, 0)
        tail_acc_rel = np.where(mean_body_length > 0, tail_acc_mag / mean_body_length, 0)
        head_jerk_rel = np.where(mean_body_length > 0, head_jerk_mag / mean_body_length, 0)
        tail_jerk_rel = np.where(mean_body_length > 0, tail_jerk_mag / mean_body_length, 0)
        
        # Calculate smoothness scores (lower is smoother)
        head_smoothness = np.where(head_vel_rel > 0, head_jerk_rel / head_vel_rel, 0)
        tail_smoothness = np.where(tail_vel_rel > 0, tail_jerk_rel / tail_vel_rel, 0)
        
        # Detect potential swaps
        potential_swaps = np.zeros(len(data), dtype=bool)
        
        # Pattern 1: Sudden position changes
        pattern1 = (
            (metrics['relative_head_dist'] > 0.005) |  # Further reduced threshold
            (metrics['relative_tail_dist'] > 0.005)
        )
        
        # Pattern 2: High velocity with low smoothness
        pattern2 = (
            ((head_vel_rel > 0.01) & (head_smoothness > 0.4)) |  # Further reduced thresholds
            ((tail_vel_rel > 0.01) & (tail_smoothness > 0.4))
        )
        
        # Pattern 3: High acceleration
        pattern3 = (
            (head_acc_rel > 0.03) |  # Further reduced threshold
            (tail_acc_rel > 0.03)
        )
        
        # Pattern 4: High jerk
        pattern4 = (
            (head_jerk_rel > 0.05) |  # Further reduced threshold
            (tail_jerk_rel > 0.05)
        )
        
        # Pattern 5: Relative position change
        pattern5 = (
            (metrics['head_to_tail'] < metrics['head_to_head'] * 0.3) &  # Further reduced threshold
            (metrics['tail_to_head'] < metrics['tail_to_tail'] * 0.3)
        )
        
        # Pattern 6: Angle change
        pattern6 = metrics['angle_change'] > np.pi / 6  # Further reduced threshold
        
        # Pattern 7: Velocity direction change
        head_vel_dir = np.arctan2(head_vel[:, 1], head_vel[:, 0])
        tail_vel_dir = np.arctan2(tail_vel[:, 1], tail_vel[:, 0])
        vel_dir_change = np.abs(np.diff(head_vel_dir - tail_vel_dir, prepend=0))
        pattern7 = vel_dir_change > np.pi / 6  # Further reduced threshold
        
        # Combine patterns with weights
        weights = {
            'pattern1': 0.25,  # Increased weight for position changes
            'pattern2': 0.15,
            'pattern3': 0.15,
            'pattern4': 0.1,
            'pattern5': 0.25,  # Increased weight for relative position
            'pattern6': 0.05,
            'pattern7': 0.05
        }
        
        weighted_sum = (
            pattern1 * weights['pattern1'] +
            pattern2 * weights['pattern2'] +
            pattern3 * weights['pattern3'] +
            pattern4 * weights['pattern4'] +
            pattern5 * weights['pattern5'] +
            pattern6 * weights['pattern6'] +
            pattern7 * weights['pattern7']
        )
        
        # Convert to binary with lower threshold
        potential_swaps = weighted_sum > 0.1  # Further reduced threshold
        
        # Post-process detections
        potential_swaps = self._remove_isolated_detections(potential_swaps)
        potential_swaps = self._smooth_flags(potential_swaps, min_duration=2)
        
        return potential_swaps
        
    def get_confidence(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate confidence scores for detections.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Array of confidence scores between 0 and 1
        """
        # Setup detector with data if not already done
        if self.data is None or len(self.data) != len(data) or not np.array_equal(self.data, data):
            self.setup(data)
            
        # Get metrics
        metrics = self._calculate_metrics(data)
        
        # Calculate position change confidence
        head_pos_change = np.sqrt(
            np.gradient(data['X-Head'])**2 + np.gradient(data['Y-Head'])**2
        )
        tail_pos_change = np.sqrt(
            np.gradient(data['X-Tail'])**2 + np.gradient(data['Y-Tail'])**2
        )
        mean_body_length = metrics['mean_body_length']
        
        # Calculate relative position changes with safe division
        relative_head_change = np.where(
            mean_body_length > 0,
            head_pos_change / mean_body_length,
            0
        )
        relative_tail_change = np.where(
            mean_body_length > 0,
            tail_pos_change / mean_body_length,
            0
        )
        
        # Calculate distance change confidence
        distance_changes = np.abs(np.gradient(np.sqrt(
            (data['X-Head'] - data['X-Tail'])**2 + (data['Y-Head'] - data['Y-Tail'])**2
        )))
        distance_conf = np.where(
            mean_body_length > 0,
            np.minimum(distance_changes / mean_body_length, 1),
            0
        )
        
        # Calculate velocity confidence
        velocity_conf = 1 - np.minimum(metrics['velocity_ratio'], 1)
        
        # Calculate acceleration confidence
        accel_conf = 1 - np.minimum(metrics['accel_ratio'], 1)
        
        # Calculate position confidence
        position_conf = np.minimum(
            np.maximum(relative_head_change, relative_tail_change),
            1
        )
        
        # Calculate relative position confidence
        rel_pos_conf = np.where(
            (metrics['head_to_tail'] < metrics['head_to_head']) &
            (metrics['tail_to_head'] < metrics['tail_to_tail']),
            1 - np.maximum(
                metrics['head_to_tail'] / metrics['head_to_head'],
                metrics['tail_to_head'] / metrics['tail_to_tail']
            ),
            0
        )
        
        # Combine confidence factors with weights
        weights = {
            'position': 0.3,
            'distance': 0.15,
            'velocity': 0.15,
            'acceleration': 0.1,
            'angle': 0.1,
            'rel_position': 0.2  # Added weight for relative position confidence
        }
        
        confidence = (
            position_conf * weights['position'] +
            distance_conf * weights['distance'] +
            velocity_conf * weights['velocity'] +
            accel_conf * weights['acceleration'] +
            metrics['angle_change'] / np.pi * weights['angle'] +
            rel_pos_conf * weights['rel_position']  # Added relative position confidence
        )
        
        # Apply sigmoid function to boost mid-range confidences
        confidence = 1 / (1 + np.exp(-10 * (confidence - 0.5)))
        
        # Ensure confidence is between 0 and 1
        confidence = np.clip(confidence, 0, 1)
        
        # Handle NaN values
        confidence = np.nan_to_num(confidence, nan=0.5)
        
        return confidence

    def get_swap_segments(self, data: pd.DataFrame) -> np.ndarray:
        """Get segments where swaps are detected.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Array of (start, end) indices for swap segments
        """
        # Detect potential swaps
        potential_swaps = self.detect(data)
        
        # Find consecutive swap segments
        segments = []
        start_idx = None
        
        for i in range(len(potential_swaps)):
            if potential_swaps[i]:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                segments.append((start_idx, i - 1))
                start_idx = None
        
        # Handle case where last segment extends to end
        if start_idx is not None:
            segments.append((start_idx, len(potential_swaps) - 1))
        
        return np.array(segments)

    def _calculate_segment_metrics(self, segment: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for a segment of data.
        
        Args:
            segment: DataFrame segment
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate distances
        dx = segment['X-Head'] - segment['X-Tail']
        dy = segment['Y-Head'] - segment['Y-Tail']
        metrics['distance'] = np.sqrt(dx**2 + dy**2)
        
        # Calculate distance changes
        metrics['distance_change'] = np.abs(np.gradient(metrics['distance']))
        
        # Calculate velocities
        head_dx = np.gradient(segment['X-Head'])
        head_dy = np.gradient(segment['Y-Head'])
        tail_dx = np.gradient(segment['X-Tail'])
        tail_dy = np.gradient(segment['Y-Tail'])
        
        metrics['head_velocity'] = np.sqrt(head_dx**2 + head_dy**2) * self.config.fps
        metrics['tail_velocity'] = np.sqrt(tail_dx**2 + tail_dy**2) * self.config.fps
        
        # Calculate relative velocity
        metrics['relative_velocity'] = np.abs(metrics['head_velocity'] - metrics['tail_velocity'])
        
        # Calculate velocity ratio
        metrics['velocity_ratio'] = np.minimum(metrics['head_velocity'], metrics['tail_velocity']) / \
                                   np.maximum(metrics['head_velocity'], metrics['tail_velocity'])
        
        # Calculate accelerations
        metrics['head_accel'] = np.gradient(metrics['head_velocity']) * self.config.fps
        metrics['tail_accel'] = np.gradient(metrics['tail_velocity']) * self.config.fps
        
        # Calculate relative acceleration
        metrics['relative_accel'] = np.abs(metrics['head_accel'] - metrics['tail_accel'])
        
        # Calculate acceleration ratio
        metrics['accel_ratio'] = np.minimum(np.abs(metrics['head_accel']), np.abs(metrics['tail_accel'])) / \
                                np.maximum(np.abs(metrics['head_accel']), np.abs(metrics['tail_accel']))
        
        # Calculate angles
        metrics['angle'] = np.arctan2(dy, dx)
        metrics['angle_change'] = np.abs(np.gradient(metrics['angle']))
        
        return metrics 

    def validate_swap(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """Validate a potential swap segment.
        
        Args:
            data: DataFrame with trajectory data
            start_idx: Start index of swap segment
            end_idx: End index of swap segment
            
        Returns:
            Whether the swap is valid
        """
        # Get segment data with padding
        pad = 2  # Add 2 frames before and after for context
        start = max(0, start_idx - pad)
        end = min(len(data), end_idx + pad + 1)
        segment = data.iloc[start:end]
        
        # Calculate body length
        body_length = np.sqrt(
            (segment['X-Head'] - segment['X-Tail'])**2 +
            (segment['Y-Head'] - segment['Y-Tail'])**2
        )
        mean_body_length = body_length.mean()
        
        # Calculate position differences
        head_diff = np.sqrt(
            np.gradient(segment['X-Head'])**2 + np.gradient(segment['Y-Head'])**2
        )
        tail_diff = np.sqrt(
            np.gradient(segment['X-Tail'])**2 + np.gradient(segment['Y-Tail'])**2
        )
        
        # Calculate relative position changes
        relative_head_change = head_diff / mean_body_length
        relative_tail_change = tail_diff / mean_body_length
        
        # Calculate velocities
        head_velocity = np.sqrt(
            np.gradient(segment['X-Head'])**2 + np.gradient(segment['Y-Head'])**2
        )
        tail_velocity = np.sqrt(
            np.gradient(segment['X-Tail'])**2 + np.gradient(segment['Y-Tail'])**2
        )
        
        # Calculate velocity ratio
        velocity_ratio = np.minimum(head_velocity, tail_velocity) / np.maximum(head_velocity, tail_velocity)
        velocity_ratio = np.nan_to_num(velocity_ratio, nan=1.0)
        
        # Calculate angle changes
        dx = segment['X-Head'] - segment['X-Tail']
        dy = segment['Y-Head'] - segment['Y-Tail']
        angles = np.arctan2(dy, dx)
        angle_changes = np.abs(np.diff(angles))
        
        # Primary conditions (must be met)
        primary_conditions = [
            np.max(relative_head_change) > 0.2,  # Significant head movement
            np.max(relative_tail_change) > 0.2,  # Significant tail movement
            np.mean(velocity_ratio) < 0.6,  # Asymmetric velocities
            np.max(angle_changes) > np.pi/6  # Some angle change
        ]
        
        # Secondary conditions (some must be met)
        secondary_conditions = [
            np.max(relative_head_change) > 0.4,  # Large head movement
            np.max(relative_tail_change) > 0.4,  # Large tail movement
            np.mean(velocity_ratio) < 0.4,  # Very asymmetric velocities
            np.max(angle_changes) > np.pi/3,  # Large angle change
            np.mean(body_length) > self._current_thresholds['proximity']  # Points are far apart
        ]
        
        # Validate if all primary conditions and some secondary conditions are met
        valid = (
            all(primary_conditions) and  # All primary conditions must be met
            sum(secondary_conditions) >= 2  # At least 2 out of 5 secondary conditions
        )
        
        return valid 
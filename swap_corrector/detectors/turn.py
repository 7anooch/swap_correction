"""
Turn-based swap detector.
"""

import numpy as np
import pandas as pd
from .base import SwapDetector
from ..config import SwapConfig
from typing import Dict, Any

class TurnDetector(SwapDetector):
    """Detector that identifies swaps based on turn patterns."""
    
    def __init__(self, config: SwapConfig):
        """Initialize turn detector.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.window_sizes = {
            'curvature': 5,
            'angle': 3,
            'outlier': 5
        }
        self.analysis_scales = {
            'short_term': {'window': 2, 'metrics': ['curvature', 'angular_velocity']},
            'medium_term': {'window': 10, 'metrics': ['mean_curvature', 'mean_angular_velocity']},
            'long_term': {'window': 30, 'metrics': ['trend_curvature', 'trend_angular_velocity']}
        }
        
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean of data.
        
        Args:
            data: Input array
            window: Window size
            
        Returns:
            Array of rolling means
        """
        return pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for detection.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Handle small segments
        if len(data) < 3:
            return {
                'head_velocity': np.zeros(len(data)),
                'tail_velocity': np.zeros(len(data)),
                'velocity_ratio': np.ones(len(data)),
                'turn_angle': np.zeros(len(data)),
                'curvature': np.zeros(len(data)),
                'radius': np.full(len(data), np.inf),
                'angle': np.zeros(len(data)),
                'turn_rate': np.zeros(len(data))
            }
        
        # Calculate velocities
        dx_head = np.gradient(data['X-Head'].values)
        dy_head = np.gradient(data['Y-Head'].values)
        dx_tail = np.gradient(data['X-Tail'].values)
        dy_tail = np.gradient(data['Y-Tail'].values)
        
        # Calculate speeds
        head_velocity = np.sqrt(dx_head**2 + dy_head**2)
        tail_velocity = np.sqrt(dx_tail**2 + dy_tail**2)
        
        # Calculate velocity ratio with safe division
        max_velocity = np.maximum(head_velocity, tail_velocity)
        min_velocity = np.minimum(head_velocity, tail_velocity)
        metrics['velocity_ratio'] = np.where(max_velocity > 0, min_velocity / max_velocity, 1.0)
        
        # Calculate angles
        head_angle = np.arctan2(dy_head, dx_head)
        tail_angle = np.arctan2(dy_tail, dx_tail)
        metrics['turn_angle'] = np.abs(head_angle - tail_angle)
        
        # Calculate body angle
        dx = data['X-Head'] - data['X-Tail']
        dy = data['Y-Head'] - data['Y-Tail']
        metrics['angle'] = np.arctan2(dy, dx)
        
        # Calculate turn rate
        metrics['turn_rate'] = np.gradient(metrics['angle']) * self.config.fps
        
        # Calculate curvature
        epsilon = 1e-10  # Small value to avoid division by zero
        dx_head_2 = np.gradient(dx_head)
        dy_head_2 = np.gradient(dy_head)
        dx_tail_2 = np.gradient(dx_tail)
        dy_tail_2 = np.gradient(dy_tail)
        
        # Calculate curvature for head and tail
        head_speed_squared = dx_head**2 + dy_head**2
        tail_speed_squared = dx_tail**2 + dy_tail**2
        
        head_curvature = np.where(
            head_speed_squared > epsilon,
            np.abs(dx_head * dy_head_2 - dy_head * dx_head_2) / (head_speed_squared**1.5 + epsilon),
            0.0
        )
        
        tail_curvature = np.where(
            tail_speed_squared > epsilon,
            np.abs(dx_tail * dy_tail_2 - dy_tail * dx_tail_2) / (tail_speed_squared**1.5 + epsilon),
            0.0
        )
        
        # Use maximum curvature
        metrics['curvature'] = np.maximum(head_curvature, tail_curvature)
        
        # Calculate radius of curvature with safe division
        metrics['radius'] = np.where(metrics['curvature'] > epsilon, 1.0 / metrics['curvature'], np.inf)
        
        # Store raw velocities
        metrics['head_velocity'] = head_velocity
        metrics['tail_velocity'] = tail_velocity
        
        return metrics
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect potential swaps based on turn patterns.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Array of boolean values indicating potential swaps
        """
        # Setup detector with data if not already done
        if self.data is None or len(self.data) != len(data) or not np.array_equal(self.data, data):
            self.setup(data)
        
        # Detect potential swaps
        potential_swaps = np.zeros(len(data), dtype=bool)
        
        # Calculate changes with padding to match array size
        curvature_changes = np.abs(np.diff(self.metrics['curvature'], prepend=self.metrics['curvature'][0]))
        angle_changes = np.abs(np.diff(self.metrics['angle'], prepend=self.metrics['angle'][0]))
        turn_rate_changes = np.abs(np.diff(self.metrics['turn_rate'], prepend=self.metrics['turn_rate'][0]))
        
        # Set thresholds
        base_turn = self._current_thresholds['turn']
        
        # Calculate mean curvature for threshold
        mean_curvature = np.mean(self.metrics['curvature'])
        curvature_threshold = mean_curvature * 0.5  # 50% of mean curvature
        
        # Calculate mean angle for threshold
        mean_angle = np.mean(np.abs(self.metrics['angle']))
        angle_threshold = mean_angle * 0.5  # 50% of mean angle
        
        # Calculate mean turn radius for threshold
        mean_radius = np.mean(self.metrics['radius'])
        radius_threshold = mean_radius * 0.5  # 50% of mean radius
        
        # Pattern 1: Sudden turn with angle change
        pattern1 = (
            (curvature_changes > curvature_threshold) &  # Sudden curvature change
            (angle_changes > angle_threshold)  # With angle change
        )
        
        # Pattern 2: High turn rate with angle change
        pattern2 = (
            (turn_rate_changes > base_turn * 0.8) &  # High turn rate change
            (angle_changes > angle_threshold * 0.5)  # With angle change
        )
        
        # Pattern 3: Sharp turn with small radius
        pattern3 = (
            (self.metrics['radius'] < radius_threshold) &  # Small turn radius
            (self.metrics['curvature'] > mean_curvature)  # High curvature
        )
        
        # Pattern 4: Sudden angle change
        pattern4 = angle_changes > angle_threshold * 1.2  # Very large angle change
        
        # Pattern 5: Rapid turn rate change
        pattern5 = turn_rate_changes > base_turn * 1.5  # Very rapid turn
        
        # Combine patterns
        potential_swaps |= pattern1
        potential_swaps |= pattern2
        potential_swaps |= pattern3
        potential_swaps |= pattern4
        potential_swaps |= pattern5
        
        return potential_swaps
        
    def _calculate_turn_radius(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate turn radius for each frame.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Array of turn radii
        """
        # Calculate body length
        body_length = np.sqrt(
            (data['X-Head'] - data['X-Tail'])**2 + 
            (data['Y-Head'] - data['Y-Tail'])**2
        ).mean()
        
        # Calculate midpoints
        mid_x = (data['X-Head'] + data['X-Tail']) / 2
        mid_y = (data['Y-Head'] + data['Y-Tail']) / 2
        
        # Calculate velocities
        vx = np.gradient(mid_x)
        vy = np.gradient(mid_y)
        speed = np.sqrt(vx**2 + vy**2)
        
        # Calculate accelerations
        ax = np.gradient(vx)
        ay = np.gradient(vy)
        
        # Calculate curvature
        epsilon = 1e-10
        numerator = np.abs(vx * ay - vy * ax)
        denominator = speed**3 + epsilon
        curvature = numerator / denominator
        
        # Calculate radius
        radius = np.where(curvature > epsilon, 1 / curvature, np.inf)
        
        # Normalize by body length
        radius = radius / body_length
        
        # Replace infinite values with maximum radius
        max_radius = 1.0  # Maximum normalized radius
        radius = np.where(np.isinf(radius) | (radius > max_radius), max_radius, radius)
        
        return radius
        
    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence score based on turn metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Confidence score between 0 and 0.95
        """
        if self._current_thresholds is None:
            return 0.5
        
        # Calculate confidence based on curvature and angular velocity
        curvature = metrics['curvature'].mean()
        turn_rate = metrics['turn_rate'].mean()
        
        # Calculate base confidence from curvature
        base_conf = 1 / (1 + np.exp(-5 * (curvature - 0.3)))  # Steeper sigmoid centered at 0.3
        
        # Calculate turn rate confidence
        turn_conf = 1 / (1 + np.exp(-5 * (turn_rate - 0.3)))
        
        # Calculate velocity ratio confidence
        vel_ratio = metrics['velocity_ratio'].mean()
        vel_conf = 1 - vel_ratio  # Lower ratio means higher confidence
        
        # Calculate turn radius confidence
        radius = metrics['radius'].mean()
        radius_conf = 1 / (1 + np.exp(-5 * (1.0 - radius)))  # Higher confidence for smaller radius
        
        # Weight the different factors
        confidence = 0.4 * base_conf + 0.3 * turn_conf + 0.2 * vel_conf + 0.1 * radius_conf
        
        # Scale confidence to prefer higher values
        confidence = 1 / (1 + np.exp(-5 * (confidence - 0.4)))
        
        # Boost confidence for large turns and rapid changes
        if curvature > 0.5 or turn_rate > 0.5:  # Significant turn or rapid change
            confidence = min(1.0, confidence * 1.5)
        
        # Ensure confidence is between 0 and 0.95
        confidence = np.clip(confidence, 0.0, 0.95)
        
        return float(confidence)

    def _analyze_movement_scales(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze movement patterns at different time scales.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Dictionary of metrics at different scales
        """
        # Calculate head and tail positions
        head_pos = data[['X-Head', 'Y-Head']].values
        tail_pos = data[['X-Tail', 'Y-Tail']].values
        
        # Calculate midpoints
        midpoints = (head_pos + tail_pos) / 2
        
        # Calculate velocities
        velocities = np.diff(midpoints, axis=0)
        velocities = np.pad(velocities, ((1, 0), (0, 0)), mode='edge')
        
        # Calculate angles and angular velocities
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        angular_velocities = np.diff(angles)
        angular_velocities = np.pad(angular_velocities, (1, 0), mode='edge')
        
        # Calculate curvatures
        speeds = np.sqrt(np.sum(velocities**2, axis=1))
        curvatures = np.abs(angular_velocities) / (speeds + 1e-6)
        
        # Define window sizes for different scales
        short_window = 2
        medium_window = 5
        long_window = 10
        
        # Calculate metrics at different scales
        metrics = {
            'short_term': {
                'curvature': self._rolling_mean(curvatures, short_window),
                'angular_velocity': self._rolling_mean(angular_velocities, short_window)
            },
            'medium_term': {
                'curvature': self._rolling_mean(curvatures, medium_window),
                'angular_velocity': self._rolling_mean(angular_velocities, medium_window)
            },
            'long_term': {
                'curvature': self._rolling_mean(curvatures, long_window),
                'angular_velocity': self._rolling_mean(angular_velocities, long_window)
            }
        }
        
        return metrics 

    def _calculate_confidence_scores(self, data: pd.DataFrame, potential_swaps: np.ndarray,
                                  scale_metrics: Dict[str, Any]) -> np.ndarray:
        """Calculate confidence scores for potential swaps.
        
        Args:
            data: DataFrame with trajectory data
            potential_swaps: Array of boolean values indicating potential swaps
            scale_metrics: Dictionary of metrics at different scales
            
        Returns:
            Array of confidence scores
        """
        scores = np.zeros(len(data))
        
        # Calculate confidence for each potential swap
        for i in range(len(data)):
            if potential_swaps[i]:
                # Get metrics for this frame
                metrics = {
                    'curvature': scale_metrics['short_term']['curvature'][i],
                    'angular_velocity': scale_metrics['short_term']['angular_velocity'][i],
                    'mean_curvature': scale_metrics['medium_term']['curvature'][i],
                    'mean_angular_velocity': scale_metrics['medium_term']['angular_velocity'][i],
                    'trend_curvature': scale_metrics['long_term']['curvature'][i],
                    'trend_angular_velocity': scale_metrics['long_term']['angular_velocity'][i]
                }
                
                # Calculate confidence score
                scores[i] = self._calculate_confidence(metrics)
        
        return scores 

    def _calculate_segment_metrics(self, segment: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for a segment of data.
        
        Args:
            segment: DataFrame segment
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate body length
        dx = segment['X-Head'] - segment['X-Tail']
        dy = segment['Y-Head'] - segment['Y-Tail']
        metrics['body_length'] = np.sqrt(dx**2 + dy**2)
        
        # Calculate angles
        metrics['angle'] = np.arctan2(dy, dx)
        metrics['angle_change'] = np.abs(np.gradient(metrics['angle']))
        
        # Calculate turn radius
        metrics['radius'] = self._calculate_turn_radius(segment)
        
        # Calculate turn rate
        metrics['turn_rate'] = np.gradient(metrics['radius']) * self.config.fps
        
        # Calculate curvature
        metrics['curvature'] = 1 / (metrics['radius'] + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Calculate relative turn metrics
        metrics['relative_turn'] = np.abs(metrics['radius'] - np.roll(metrics['radius'], 1))
        metrics['relative_curvature'] = np.abs(metrics['curvature'] - np.roll(metrics['curvature'], 1))
        
        # Calculate turn ratio
        metrics['turn_ratio'] = np.minimum(metrics['radius'], np.roll(metrics['radius'], 1)) / \
                               np.maximum(metrics['radius'], np.roll(metrics['radius'], 1))
        
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
        
        # Calculate max values
        metrics['max_curvature'] = np.max(metrics['curvature'])
        metrics['max_angle_change'] = np.max(metrics['angle_change'])
        metrics['max_turn_rate'] = np.max(metrics['turn_rate'])
        
        return metrics 
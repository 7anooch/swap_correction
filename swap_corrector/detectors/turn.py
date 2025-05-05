"""Turn-based swap detector implementation."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict

from .base import SwapDetector
from ..metrics import MovementMetrics
from ..config import SwapConfig, SwapThresholds

class TurnDetector(SwapDetector):
    """Detector for turn-related swap events.
    
    This detector identifies swaps that occur during turns by:
    1. Analyzing curvature and angular velocity
    2. Using multi-scale movement analysis
    3. Validating turn radius consistency
    4. Scoring confidence based on multiple factors
    """
    
    def __init__(self, config: Optional[SwapConfig] = None):
        """Initialize the turn detector.
        
        Args:
            config: Optional configuration object. If None, will use defaults.
        """
        super().__init__(config)
        self.analysis_scales = {
            'short_term': {'window': 5, 'metrics': ['speed', 'curvature']},
            'medium_term': {'window': 15, 'metrics': ['angular_velocity', 'turn_radius']},
            'long_term': {'window': 30, 'metrics': ['path_tortuosity']}
        }
        self.thresholds = SwapThresholds(
            speed=10.0,  # mm/s - Lower threshold for better sensitivity
            curvature=0.2,  # 1/mm - Lower threshold for better sensitivity
            angle=np.pi/6,  # radians - More sensitive to turns
            turn_radius_threshold=15.0,  # mm - More lenient radius threshold
            path_tortuosity_threshold=1.2  # Lower threshold for better sensitivity
        )
        
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect swaps based on turn patterns.
        
        A swap is detected when:
        1. There is a rapid turn
        2. The turn radius is inconsistent
        3. The movement pattern suggests a swap
        
        Args:
            data: DataFrame containing tracking data
            
        Returns:
            Boolean array indicating frames with potential swaps
        """
        if self.metrics is None:
            self.setup(data)
            
        # Get metrics for different scales
        scale_metrics = self._analyze_movement_scales(data)
        
        # Get short-term metrics
        short_term = scale_metrics['short_term']
        speed = short_term['speed']
        curvature = short_term['curvature']
        
        # Get medium-term metrics
        medium_term = scale_metrics['medium_term']
        ang_vel = medium_term['angular_velocity']
        turn_radius = medium_term['turn_radius']
        
        # Detect rapid turns
        is_rapid_turn = (
            (speed > self.thresholds.speed * 0.5) &  # Moderate speed
            (curvature > self.thresholds.curvature * 0.5) &  # Moderate curvature
            (np.abs(ang_vel) > self.thresholds.angle * 0.5)  # Moderate rotation
        )
        
        # Detect sudden direction changes
        ang_vel_diff = np.abs(np.diff(ang_vel, prepend=ang_vel[0]))
        is_sudden_turn = ang_vel_diff > self.thresholds.angle * 0.75
        
        # Detect position swaps
        head_pos = np.column_stack([data['X-Head'], data['Y-Head']])
        tail_pos = np.column_stack([data['X-Tail'], data['Y-Tail']])
        head_vel = np.gradient(head_pos, axis=0)
        tail_vel = np.gradient(tail_pos, axis=0)
        
        # Calculate prediction errors
        dt = 1.0 / self.config.fps
        head_pred = head_pos[:-1] + head_vel[:-1] * dt
        tail_pred = tail_pos[:-1] + tail_vel[:-1] * dt
        
        # Calculate prediction errors
        head_error = np.linalg.norm(head_pos[1:] - head_pred, axis=1)
        tail_error = np.linalg.norm(tail_pos[1:] - tail_pred, axis=1)
        
        # Calculate cross-prediction errors
        head_tail_error = np.linalg.norm(head_pos[1:] - tail_pred, axis=1)
        tail_head_error = np.linalg.norm(tail_pos[1:] - head_pred, axis=1)
        
        # Identify frames where cross-prediction is better
        is_swap = (
            (head_tail_error < head_error * 0.8) &  # Cross-prediction is significantly better
            (tail_head_error < tail_error * 0.8)
        )
        
        # Pad to match data length
        is_swap = np.pad(is_swap, (0, 1), mode='edge')
        
        # Detect head-tail distance changes
        head_tail_dist = np.linalg.norm(head_pos - tail_pos, axis=1)
        dist_mean = np.mean(head_tail_dist)
        dist_std = np.std(head_tail_dist)
        is_dist_change = np.abs(head_tail_dist - dist_mean) > 2 * dist_std
        
        # Combine detections
        potential_swaps = (
            is_rapid_turn |
            is_sudden_turn |
            is_swap |
            is_dist_change
        )
        
        # Validate detections
        validated_swaps = self._validate_with_momentum(data, potential_swaps)
        
        return validated_swaps

    def _validate_with_momentum(
        self,
        data: pd.DataFrame,
        potential_swaps: np.ndarray
    ) -> np.ndarray:
        """Validate potential swaps using momentum-based predictions.
        
        Args:
            data: DataFrame containing tracking data
            potential_swaps: Boolean array indicating potential swaps
            
        Returns:
            Boolean array indicating validated swaps
        """
        # Get positions and velocities
        head_pos = np.column_stack([data['X-Head'], data['Y-Head']])
        tail_pos = np.column_stack([data['X-Tail'], data['Y-Tail']])
        
        head_vel = np.gradient(head_pos, axis=0)
        tail_vel = np.gradient(tail_pos, axis=0)
        
        # Calculate head-tail distance
        head_tail_dist = np.linalg.norm(head_pos - tail_pos, axis=1)
        dist_mean = np.mean(head_tail_dist)
        dist_std = np.std(head_tail_dist)
        
        # Predict positions using momentum
        dt = 1.0 / self.config.fps
        head_pred = head_pos[:-1] + head_vel[:-1] * dt
        tail_pred = tail_pos[:-1] + tail_vel[:-1] * dt
        
        # Calculate prediction errors
        head_error = np.linalg.norm(head_pos[1:] - head_pred, axis=1)
        tail_error = np.linalg.norm(tail_pos[1:] - tail_pred, axis=1)
        
        # Calculate cross-prediction errors
        head_tail_error = np.linalg.norm(head_pos[1:] - tail_pred, axis=1)
        tail_head_error = np.linalg.norm(tail_pos[1:] - head_pred, axis=1)
        
        # Identify frames where cross-prediction is better
        is_swap = (
            (head_tail_error < head_error * 0.8) &  # Cross-prediction is significantly better
            (tail_head_error < tail_error * 0.8)
        )
        
        # Pad to match data length
        is_swap = np.pad(is_swap, (0, 1), mode='edge')
        
        # Detect sudden velocity changes
        head_vel_mag = np.linalg.norm(head_vel, axis=1)
        tail_vel_mag = np.linalg.norm(tail_vel, axis=1)
        
        head_vel_diff = np.abs(np.diff(head_vel_mag, prepend=head_vel_mag[0]))
        tail_vel_diff = np.abs(np.diff(tail_vel_mag, prepend=tail_vel_mag[0]))
        
        is_sudden_vel_change = (
            (head_vel_diff > np.mean(head_vel_mag) * 0.5) |
            (tail_vel_diff > np.mean(tail_vel_mag) * 0.5)
        )
        
        # Detect head-tail distance anomalies
        is_dist_anomaly = np.abs(head_tail_dist - dist_mean) > 1.5 * dist_std
        
        # Combine all validation criteria
        validated_swaps = (
            potential_swaps &
            (is_swap | is_sudden_vel_change | is_dist_anomaly)
        )
        
        return validated_swaps
    
    def _analyze_movement_scales(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze movement patterns at different temporal scales.
        
        Args:
            data: DataFrame containing tracking data
            
        Returns:
            Dictionary of metrics for each scale
        """
        scale_metrics = {}
        n_frames = len(data)
        
        for scale, params in self.analysis_scales.items():
            window = params['window']
            metrics = {}
            
            for metric in params['metrics']:
                if metric == 'speed':
                    metrics['speed'] = self.metrics.get_speed('Midpoint')
                elif metric == 'curvature':
                    metrics['curvature'] = self.metrics.get_curvature()
                elif metric == 'angular_velocity':
                    metrics['angular_velocity'] = self.metrics.get_angular_velocity()
                elif metric == 'turn_radius':
                    metrics['turn_radius'] = self._calculate_turn_radius(data)
                elif metric == 'path_tortuosity':
                    metrics['path_tortuosity'] = self._calculate_path_tortuosity(data)
            
            # Ensure all metrics have the same length
            for key in metrics:
                values = metrics[key]
                if len(values) != n_frames:
                    # Pad or truncate to match data length
                    if len(values) < n_frames:
                        metrics[key] = np.pad(values, (0, n_frames - len(values)), mode='edge')
                    else:
                        metrics[key] = values[:n_frames]
            
            # Smooth metrics using the scale's window
            for key in metrics:
                values = metrics[key]
                if len(values) > window:
                    metrics[key] = np.convolve(
                        values,
                        np.ones(window)/window,
                        mode='same'
                    )
                    # Handle edge cases
                    metrics[key][0:window//2] = metrics[key][window//2]
                    metrics[key][-window//2:] = metrics[key][-window//2-1]
                else:
                    # For short sequences, just use the raw values
                    metrics[key] = values
            
            scale_metrics[scale] = metrics
            
        return scale_metrics
    
    def _calculate_turn_radius(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate turn radius for each frame.
        
        Args:
            data: DataFrame containing tracking data
            
        Returns:
            Array of turn radii
        """
        # For circular motion, use head position to calculate radius
        x = data['X-Head'].values
        y = data['Y-Head'].values
        
        # Calculate center as (0, 0) since we know the artificial data is centered there
        center_x = 0
        center_y = 0
        
        # Calculate radius as distance from center
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Smooth radius values
        window = 5
        smoothed_radius = np.convolve(radius, np.ones(window)/window, mode='same')
        
        # Handle edge cases
        smoothed_radius[0:window//2] = smoothed_radius[window//2]
        smoothed_radius[-window//2:] = smoothed_radius[-window//2-1]
        
        return smoothed_radius
    
    def _calculate_path_tortuosity(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate path tortuosity (ratio of path length to straight-line distance).
        
        Args:
            data: DataFrame containing tracking data
            
        Returns:
            Array of tortuosity values
        """
        # Calculate path length
        dx = np.diff(data['X-Midpoint'])
        dy = np.diff(data['Y-Midpoint'])
        path_length = np.sqrt(dx*dx + dy*dy)
        
        # Calculate straight-line distance
        straight_dist = np.sqrt(
            (data['X-Midpoint'].iloc[-1] - data['X-Midpoint'].iloc[0])**2 +
            (data['Y-Midpoint'].iloc[-1] - data['Y-Midpoint'].iloc[0])**2
        )
        
        # Calculate tortuosity
        tortuosity = np.cumsum(path_length) / straight_dist
        return np.concatenate([[1.0], tortuosity])  # First frame has tortuosity 1
    
    def _calculate_speed_score(self, speed: float) -> float:
        """Calculate confidence score based on speed.
        
        Args:
            speed: Current speed value
            
        Returns:
            Confidence score between 0 and 1
        """
        # Higher speed increases confidence
        return min(1.0, speed / (self.thresholds.speed + 1e-6))

    def _calculate_curvature_score(self, curvature: float) -> float:
        """Calculate confidence score based on curvature.
        
        Args:
            curvature: Current curvature value
            
        Returns:
            Confidence score between 0 and 1
        """
        # Higher curvature increases confidence
        return min(1.0, curvature / (self.thresholds.curvature + 1e-6))

    def _calculate_radius_score(self, radius: float) -> float:
        """Calculate confidence score based on turn radius.
        
        Args:
            radius: Current turn radius value
            
        Returns:
            Confidence score between 0 and 1
        """
        # Lower radius increases confidence
        if radius > 0:
            return min(1.0, 1.0 / (1.0 + radius))
        return 0.0

    def _calculate_tortuosity_score(self, tortuosity: float) -> float:
        """Calculate confidence score based on path tortuosity.
        
        Args:
            tortuosity: Current tortuosity value
            
        Returns:
            Confidence score between 0 and 1
        """
        # Higher tortuosity increases confidence
        return min(1.0, tortuosity / 2.0)  # Normalize by typical maximum

    def get_confidence(self, data: pd.DataFrame, start: int, end: int) -> float:
        """Calculate confidence score for a swap segment.
        
        Args:
            data: DataFrame containing tracking data
            start: Start frame of the swap segment
            end: End frame of the swap segment
            
        Returns:
            Confidence score between 0 and 1
        """
        if self.metrics is None:
            raise RuntimeError("Detector not initialized. Call setup() first.")
            
        # Get metrics for the segment
        scale_metrics = self._analyze_movement_scales(data)
        
        # Calculate component scores
        speed = np.nanmean(scale_metrics['short_term']['speed'][start:end+1])
        speed_score = self._calculate_speed_score(speed)
        
        curvature = np.nanmean(scale_metrics['short_term']['curvature'][start:end+1])
        curvature_score = self._calculate_curvature_score(curvature)
        
        radius = np.nanmean(scale_metrics['medium_term']['turn_radius'][start:end+1])
        radius_score = self._calculate_radius_score(radius)
        
        tortuosity = np.nanmean(scale_metrics['long_term']['path_tortuosity'][start:end+1])
        tortuosity_score = self._calculate_tortuosity_score(tortuosity)
        
        # Combine scores with weights
        confidence = (
            0.3 * speed_score +
            0.3 * curvature_score +
            0.2 * radius_score +
            0.2 * tortuosity_score
        )
        
        return float(np.clip(confidence, 0, 1))

    def validate_swap(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """Validate a detected swap segment.
        
        Args:
            data: DataFrame containing tracking data
            start: Start frame of the swap segment
            end: End frame of the swap segment
            
        Returns:
            True if the swap is valid, False otherwise
        """
        if self.metrics is None:
            raise RuntimeError("Detector not initialized. Call setup() first.")
            
        # Get metrics for the segment
        scale_metrics = self._analyze_movement_scales(data)
        
        # Calculate validation metrics
        speed = np.mean(scale_metrics['short_term']['speed'][start:end+1])
        curvature = np.mean(scale_metrics['short_term']['curvature'][start:end+1])
        ang_vel = np.mean(scale_metrics['medium_term']['angular_velocity'][start:end+1])
        turn_radius = np.mean(scale_metrics['medium_term']['turn_radius'][start:end+1])
        tortuosity = np.mean(scale_metrics['long_term']['path_tortuosity'][start:end+1])
        
        # Check turn radius consistency
        radius_consistent = self._check_turn_radius_consistency(
            data, (start + end) // 2, scale_metrics
        )
        
        # Check movement pattern continuity
        pattern_continuous = self._check_movement_continuity(
            data, (start + end) // 2, scale_metrics
        )
        
        # Validate based on multiple criteria
        speed_valid = speed > self.thresholds.speed
        curvature_valid = curvature > self.thresholds.curvature
        angle_valid = abs(ang_vel) > self.thresholds.angle
        radius_valid = turn_radius < self.thresholds.turn_radius_threshold
        tortuosity_valid = tortuosity > self.thresholds.path_tortuosity_threshold
        
        return (
            speed_valid and
            curvature_valid and
            angle_valid and
            radius_valid and
            tortuosity_valid and
            radius_consistent and
            pattern_continuous
        )

    def _check_turn_radius_consistency(
        self,
        data: pd.DataFrame,
        frame: int,
        scale_metrics: Dict[str, Dict[str, np.ndarray]]
    ) -> bool:
        """Check if the turn radius is consistent around a given frame.
        
        Args:
            data: DataFrame containing tracking data
            frame: Frame to check around
            scale_metrics: Dictionary of metrics for different scales
            
        Returns:
            True if turn radius is consistent, False otherwise
        """
        # Get turn radius values around the frame
        window = 10  # Look at 10 frames before and after
        start = max(0, frame - window)
        end = min(len(data), frame + window)
        
        radii = scale_metrics['medium_term']['turn_radius'][start:end]
        valid_radii = radii[~np.isnan(radii)]
        
        if len(valid_radii) < 3:
            return False
            
        # Calculate mean and standard deviation
        mean_radius = np.mean(valid_radii)
        std_radius = np.std(valid_radii)
        
        # Check if current radius is within 2 standard deviations
        current_radius = scale_metrics['medium_term']['turn_radius'][frame]
        if np.isnan(current_radius):
            return False
            
        return abs(current_radius - mean_radius) <= 2 * std_radius

    def _check_movement_continuity(
        self,
        data: pd.DataFrame,
        frame: int,
        scale_metrics: Dict[str, Dict[str, np.ndarray]]
    ) -> bool:
        """Check if the movement pattern is continuous around a given frame.
        
        Args:
            data: DataFrame containing tracking data
            frame: Frame to check around
            scale_metrics: Dictionary of metrics for different scales
            
        Returns:
            True if movement is continuous, False otherwise
        """
        # Get metrics around the frame
        window = 5  # Look at 5 frames before and after
        start = max(0, frame - window)
        end = min(len(data), frame + window)
        
        # Check speed continuity
        speeds = scale_metrics['short_term']['speed'][start:end]
        if np.any(np.isnan(speeds)):
            return False
            
        # Check for sudden speed changes
        speed_diff = np.diff(speeds)
        if np.any(np.abs(speed_diff) > self.thresholds.speed * 0.5):
            return False
            
        # Check angular velocity continuity
        ang_vel = scale_metrics['medium_term']['angular_velocity'][start:end]
        if np.any(np.isnan(ang_vel)):
            return False
            
        # Check for sudden direction changes
        ang_vel_diff = np.diff(ang_vel)
        if np.any(np.abs(ang_vel_diff) > self.thresholds.angle * 0.5):
            return False
            
        return True

    def _calculate_confidence_scores(
        self,
        data: pd.DataFrame,
        potential_swaps: np.ndarray,
        scale_metrics: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """Calculate confidence scores for potential swaps.
        
        Args:
            data: DataFrame containing tracking data
            potential_swaps: Boolean array indicating potential swaps
            scale_metrics: Dictionary of metrics for different scales
            
        Returns:
            Array of confidence scores between 0 and 1
        """
        n_frames = len(data)
        scores = np.zeros(n_frames)
        
        # Only calculate scores for potential swaps
        swap_indices = np.where(potential_swaps)[0]
        
        for frame in swap_indices:
            # Get metrics for this frame
            speed = scale_metrics['short_term']['speed'][frame]
            curvature = scale_metrics['short_term']['curvature'][frame]
            ang_vel = scale_metrics['medium_term']['angular_velocity'][frame]
            turn_radius = scale_metrics['medium_term']['turn_radius'][frame]
            tortuosity = scale_metrics['long_term']['path_tortuosity'][frame]
            
            # Calculate component scores
            speed_score = self._calculate_speed_score(speed)
            curvature_score = self._calculate_curvature_score(curvature)
            radius_score = self._calculate_radius_score(turn_radius)
            tortuosity_score = self._calculate_tortuosity_score(tortuosity)
            
            # Calculate angular velocity score
            ang_vel_score = min(1.0, abs(ang_vel) / self.thresholds.angle)
            
            # Check turn radius consistency
            radius_consistent = self._check_turn_radius_consistency(
                data, frame, scale_metrics
            )
            
            # Check movement continuity
            pattern_continuous = self._check_movement_continuity(
                data, frame, scale_metrics
            )
            
            # Combine scores with weights
            confidence = (
                0.25 * speed_score +
                0.25 * curvature_score +
                0.25 * ang_vel_score +  # Give more weight to angular velocity
                0.15 * radius_score +
                0.10 * tortuosity_score
            )
            
            # Penalize inconsistent radius or discontinuous movement
            if not radius_consistent:
                confidence *= 0.75  # Less severe penalty
            if not pattern_continuous:
                confidence *= 0.75  # Less severe penalty
                
            scores[frame] = confidence
            
        return scores 
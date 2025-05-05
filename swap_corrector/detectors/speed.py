"""Speed-based swap detector implementation."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict

from .base import SwapDetector
from ..metrics import MovementMetrics
from ..config import SwapConfig, SwapThresholds

class SpeedDetector(SwapDetector):
    """Detector for speed-related swap events.
    
    This detector identifies swaps that occur during high-speed movements by:
    1. Analyzing speed and acceleration patterns
    2. Using momentum-based prediction
    3. Validating movement consistency
    4. Scoring confidence based on multiple factors
    """
    
    def __init__(self, config: Optional[SwapConfig] = None):
        """Initialize the speed detector.
        
        Args:
            config: Optional configuration object. If None, will use defaults.
        """
        super().__init__(config)
        self.analysis_scales = {
            'short_term': {'window': 3, 'metrics': ['speed', 'acceleration']},
            'medium_term': {'window': 10, 'metrics': ['momentum', 'kinetic_energy']},
            'long_term': {'window': 20, 'metrics': ['avg_speed', 'speed_variability']}
        }
        self.thresholds = SwapThresholds(
            speed=5.0,  # mm/s - Lower threshold for better sensitivity
            proximity=2.0,  # mm - Threshold for head-tail distance
            angle=np.pi/6,  # radians - More sensitive to direction changes
            curvature=0.1,  # 1/mm - Threshold for path curvature
            body_length=0.7,  # ratio of mean length
            acceleration_threshold=20.0,  # mm/s^2 - Lower threshold for better sensitivity
            jerk_threshold=10.0,  # mm/s^3 - Threshold for jerk
            prediction_error_threshold=10.0,  # mm/s - Lower threshold for better sensitivity
            speed_ratio_threshold=1.5,  # Lower ratio threshold for better sensitivity
            path_tortuosity_threshold=1.2  # Lower threshold for better sensitivity
        )
        
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect swaps based on speed patterns.
        
        A swap is detected when:
        1. There is high speed or acceleration
        2. The momentum changes abruptly
        3. The movement pattern is inconsistent
        
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
        acceleration = short_term['acceleration']
        
        # Get medium-term metrics
        medium_term = scale_metrics['medium_term']
        momentum = medium_term['momentum']
        kinetic_energy = medium_term['kinetic_energy']
        
        # Calculate head-tail speed ratio
        head_speed = self.metrics.get_speed('Head')
        tail_speed = self.metrics.get_speed('Tail')
        speed_ratio = np.abs(head_speed - tail_speed) / (np.abs(head_speed + tail_speed) + 1e-6)
        
        # Calculate head-tail distance
        head_pos = np.column_stack([data['X-Head'], data['Y-Head']])
        tail_pos = np.column_stack([data['X-Tail'], data['Y-Tail']])
        head_tail_dist = np.linalg.norm(head_pos - tail_pos, axis=1)
        dist_mean = np.mean(head_tail_dist)
        dist_std = np.std(head_tail_dist)
        
        # Detect high speed events
        is_high_speed = speed > self.thresholds.speed
        
        # Detect rapid acceleration
        is_rapid_accel = np.abs(acceleration) > self.thresholds.acceleration_threshold
        
        # Detect momentum changes
        momentum_change = np.abs(np.diff(momentum, prepend=momentum[0]))
        is_momentum_change = momentum_change > self.thresholds.prediction_error_threshold
        
        # Detect energy changes
        energy_change = np.abs(np.diff(kinetic_energy, prepend=kinetic_energy[0]))
        is_energy_change = energy_change > self.thresholds.prediction_error_threshold
        
        # Detect speed ratio anomalies
        is_speed_ratio_anomaly = speed_ratio > self.thresholds.speed_ratio_threshold
        
        # Detect distance anomalies
        is_dist_anomaly = np.abs(head_tail_dist - dist_mean) > 2.0 * dist_std
        
        # Ensure all arrays have the same length
        n_frames = len(data)
        arrays_to_check = [
            is_high_speed, is_rapid_accel, is_momentum_change, is_energy_change,
            is_speed_ratio_anomaly, is_dist_anomaly
        ]
        for i, arr in enumerate(arrays_to_check):
            if len(arr) != n_frames:
                arrays_to_check[i] = np.pad(arr, (0, n_frames - len(arr)), mode='edge')
        
        is_high_speed, is_rapid_accel, is_momentum_change, is_energy_change, \
        is_speed_ratio_anomaly, is_dist_anomaly = arrays_to_check
        
        # Combine detections
        potential_swaps = (
            (is_high_speed & is_speed_ratio_anomaly) |  # High speed with abnormal head-tail ratio
            (is_rapid_accel & is_dist_anomaly) |  # Rapid acceleration with distance anomaly
            (is_momentum_change & is_speed_ratio_anomaly) |  # Momentum change with abnormal ratio
            (is_energy_change & is_dist_anomaly)  # Energy change with distance anomaly
        )
        
        # Validate detections
        validated_swaps = self._validate_with_momentum(data, potential_swaps)
        
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
                elif metric == 'acceleration':
                    metrics['acceleration'] = self.metrics.get_acceleration('Midpoint')
                elif metric == 'momentum':
                    speed = self.metrics.get_speed('Midpoint')
                    metrics['momentum'] = speed  # Simplified momentum (mass=1)
                elif metric == 'kinetic_energy':
                    speed = self.metrics.get_speed('Midpoint')
                    metrics['kinetic_energy'] = 0.5 * speed * speed  # Simplified KE (mass=1)
                elif metric == 'avg_speed':
                    speed = self.metrics.get_speed('Midpoint')
                    metrics['avg_speed'] = np.convolve(
                        speed,
                        np.ones(window)/window,
                        mode='same'
                    )
                elif metric == 'speed_variability':
                    speed = self.metrics.get_speed('Midpoint')
                    rolling_std = np.array([
                        np.std(speed[max(0, i-window):min(len(speed), i+window)])
                        for i in range(len(speed))
                    ])
                    rolling_mean = np.array([
                        np.mean(speed[max(0, i-window):min(len(speed), i+window)])
                        for i in range(len(speed))
                    ])
                    metrics['speed_variability'] = rolling_std / (rolling_mean + 1e-6)
            
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
        
        # Ensure all arrays have the same length
        n_frames = len(data)
        if len(potential_swaps) != n_frames:
            potential_swaps = np.pad(potential_swaps, (0, n_frames - len(potential_swaps)), mode='edge')
        if len(is_swap) != n_frames:
            is_swap = np.pad(is_swap, (0, n_frames - len(is_swap)), mode='edge')
        if len(is_sudden_vel_change) != n_frames:
            is_sudden_vel_change = np.pad(is_sudden_vel_change, (0, n_frames - len(is_sudden_vel_change)), mode='edge')
        if len(is_dist_anomaly) != n_frames:
            is_dist_anomaly = np.pad(is_dist_anomaly, (0, n_frames - len(is_dist_anomaly)), mode='edge')
        
        # Combine all validation criteria
        validated_swaps = (
            potential_swaps &
            (is_swap | (is_sudden_vel_change & is_dist_anomaly))  # Must have either swap pattern or both velocity change and distance anomaly
        )
        
        return validated_swaps
        
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
        speed_score = min(1.0, speed / (self.thresholds.speed + 1e-6))
        
        accel = np.nanmean(np.abs(scale_metrics['short_term']['acceleration'][start:end+1]))
        accel_score = min(1.0, accel / (self.thresholds.acceleration_threshold + 1e-6))
        
        momentum = np.nanmean(scale_metrics['medium_term']['momentum'][start:end+1])
        momentum_score = min(1.0, momentum / (self.thresholds.prediction_error_threshold + 1e-6))
        
        energy = np.nanmean(scale_metrics['medium_term']['kinetic_energy'][start:end+1])
        energy_score = min(1.0, energy / (self.thresholds.prediction_error_threshold**2 + 1e-6))
        
        # Combine scores with weights
        confidence = (
            0.3 * speed_score +
            0.3 * accel_score +
            0.2 * momentum_score +
            0.2 * energy_score
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
            
        # Get speed metrics for the segment
        head_speed = self.metrics.get_speed('Head')[start:end+1]
        tail_speed = self.metrics.get_speed('Tail')[start:end+1]
        
        # Calculate speed ratio
        speed_ratio = np.mean(np.abs(head_speed - tail_speed)) / np.mean(np.abs(head_speed + tail_speed))
        
        # Get acceleration metrics
        head_accel = self.metrics.get_acceleration('Head')[start:end+1]
        tail_accel = self.metrics.get_acceleration('Tail')[start:end+1]
        
        # Calculate acceleration ratio
        accel_ratio = np.mean(np.abs(head_accel - tail_accel)) / np.mean(np.abs(head_accel + tail_accel))
        
        # Validate based on momentum prediction
        potential_swaps = np.zeros(len(data), dtype=bool)
        potential_swaps[start:end+1] = True
        validated_swaps = self._validate_with_momentum(data, potential_swaps)
        
        # Combine validation criteria
        speed_valid = speed_ratio > self.thresholds.speed_ratio_threshold  # High speed difference
        accel_valid = accel_ratio > 0.3  # High acceleration difference
        momentum_valid = np.any(validated_swaps[start:end+1])  # At least one frame validated by momentum
        
        return speed_valid and accel_valid and momentum_valid 

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
        
        # Convert potential_swaps to numpy array if it's a pandas Series
        if isinstance(potential_swaps, pd.Series):
            potential_swaps = potential_swaps.values
        
        # Only calculate scores for potential swaps
        swap_indices = np.where(potential_swaps)[0]
        
        # Get all metrics at once to avoid repeated indexing
        speed = scale_metrics['short_term']['speed']
        acceleration = scale_metrics['short_term']['acceleration']
        momentum = scale_metrics['medium_term']['momentum']
        kinetic_energy = scale_metrics['medium_term']['kinetic_energy']
        
        # Calculate head-tail speed ratio
        head_speed = self.metrics.get_speed('Head')
        tail_speed = self.metrics.get_speed('Tail')
        speed_ratio = np.abs(head_speed - tail_speed) / (np.abs(head_speed + tail_speed) + 1e-6)
        
        # Calculate head-tail distance
        head_pos = np.column_stack([data['X-Head'], data['Y-Head']])
        tail_pos = np.column_stack([data['X-Tail'], data['Y-Tail']])
        head_tail_dist = np.linalg.norm(head_pos - tail_pos, axis=1)
        dist_mean = np.mean(head_tail_dist)
        dist_std = np.std(head_tail_dist)
        
        for frame in swap_indices:
            if frame >= len(speed):  # Skip if frame is out of bounds
                continue
                
            # Calculate component scores
            speed_score = min(1.0, speed[frame] / (self.thresholds.speed * 0.8 + 1e-6))
            accel_score = min(1.0, abs(acceleration[frame]) / (self.thresholds.acceleration_threshold * 0.8 + 1e-6))
            momentum_score = min(1.0, abs(momentum[frame]) / (self.thresholds.prediction_error_threshold * 0.8 + 1e-6))
            energy_score = min(1.0, kinetic_energy[frame] / (self.thresholds.prediction_error_threshold**2 * 0.8 + 1e-6))
            
            # Calculate additional scores
            speed_ratio_score = min(1.0, speed_ratio[frame] / (self.thresholds.speed_ratio_threshold * 0.8 + 1e-6))
            dist_score = min(1.0, abs(head_tail_dist[frame] - dist_mean) / (2.0 * dist_std + 1e-6))
            
            # Combine scores with adjusted weights
            confidence = (
                0.25 * speed_score +
                0.25 * accel_score +
                0.15 * momentum_score +
                0.15 * energy_score +
                0.10 * speed_ratio_score +
                0.10 * dist_score
            )
            
            # Boost confidence for high speed ratio or distance anomaly
            if speed_ratio_score > 0.8 or dist_score > 0.8:
                confidence = min(1.0, confidence * 1.2)
            
            scores[frame] = confidence
            
        return scores 
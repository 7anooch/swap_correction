"""
Speed-based swap detector.
"""

import numpy as np
import pandas as pd
from .base import SwapDetector
from ..config import SwapConfig
from typing import Dict, Any, List, Tuple

class SpeedDetector(SwapDetector):
    """Detector that identifies swaps based on speed patterns."""
    
    def __init__(self, config: SwapConfig):
        """Initialize speed detector.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.window_sizes = {
            'short_term': 3,
            'medium_term': 7,
            'long_term': 15,
            'outlier': 5
        }
        
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
        
        # Calculate velocities
        head_dx = np.gradient(data['X-Head'])
        head_dy = np.gradient(data['Y-Head'])
        tail_dx = np.gradient(data['X-Tail'])
        tail_dy = np.gradient(data['Y-Tail'])
        
        head_speed = np.sqrt(head_dx**2 + head_dy**2)
        tail_speed = np.sqrt(tail_dx**2 + tail_dy**2)
        metrics['head_speed'] = head_speed
        metrics['tail_speed'] = tail_speed
        
        # Calculate relative speeds with safe division
        metrics['relative_head_speed'] = np.where(
            mean_body_length > 0,
            head_speed / mean_body_length,
            0
        )
        metrics['relative_tail_speed'] = np.where(
            mean_body_length > 0,
            tail_speed / mean_body_length,
            0
        )
        
        # Calculate speed ratio with safe division
        max_speed = np.maximum(head_speed, tail_speed)
        min_speed = np.minimum(head_speed, tail_speed)
        metrics['speed_ratio'] = np.where(max_speed > 0, min_speed / max_speed, 1.0)
        
        # Calculate accelerations
        head_accel = np.abs(np.gradient(head_speed))
        tail_accel = np.abs(np.gradient(tail_speed))
        metrics['head_accel'] = head_accel
        metrics['tail_accel'] = tail_accel
        
        # Calculate acceleration ratio with safe division
        max_accel = np.maximum(head_accel, tail_accel)
        min_accel = np.minimum(head_accel, tail_accel)
        metrics['accel_ratio'] = np.where(max_accel > 0, min_accel / max_accel, 1.0)
        
        # Calculate jerk
        head_jerk = np.abs(np.gradient(head_accel))
        tail_jerk = np.abs(np.gradient(tail_accel))
        metrics['head_jerk'] = head_jerk
        metrics['tail_jerk'] = tail_jerk
        
        # Calculate relative accelerations with safe division
        metrics['relative_head_accel'] = np.where(
            mean_body_length > 0,
            head_accel / mean_body_length,
            0
        )
        metrics['relative_tail_accel'] = np.where(
            mean_body_length > 0,
            tail_accel / mean_body_length,
            0
        )
        
        # Calculate relative jerk with safe division
        metrics['relative_head_jerk'] = np.where(
            mean_body_length > 0,
            head_jerk / mean_body_length,
            0
        )
        metrics['relative_tail_jerk'] = np.where(
            mean_body_length > 0,
            tail_jerk / mean_body_length,
            0
        )
        
        return metrics
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect potential swaps based on speed.
        
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
        
        # Analyze movement at different scales
        scale_metrics = self._analyze_movement_scales(data)
        
        # Detect potential swaps
        potential_swaps = np.zeros(len(data), dtype=bool)
        
        # Pattern 1: Sudden speed changes
        pattern1 = (
            (metrics['relative_head_speed'] > 0.2) |  # Reduced threshold
            (metrics['relative_tail_speed'] > 0.2)
        )
        
        # Pattern 2: Speed ratio inversion
        pattern2 = metrics['speed_ratio'] < 0.3  # Reduced threshold
        
        # Pattern 3: High acceleration
        pattern3 = (
            (metrics['relative_head_accel'] > 0.15) |  # Reduced threshold
            (metrics['relative_tail_accel'] > 0.15)
        )
        
        # Pattern 4: High jerk
        pattern4 = (
            (metrics['relative_head_jerk'] > 0.1) |  # Reduced threshold
            (metrics['relative_tail_jerk'] > 0.1)
        )
        
        # Pattern 5: Multi-scale speed changes
        short_term = scale_metrics['short_term']
        medium_term = scale_metrics['medium_term']
        long_term = scale_metrics['long_term']
        
        pattern5 = (
            (short_term['speed_ratio'] < 0.4) &  # Reduced threshold
            (medium_term['speed_ratio'] < 0.5) &
            (long_term['speed_ratio'] < 0.6)
        )
        
        # Pattern 6: Multi-scale acceleration changes
        pattern6 = (
            (short_term['accel_ratio'] < 0.3) &  # Reduced threshold
            (medium_term['accel_ratio'] < 0.4) &
            (long_term['accel_ratio'] < 0.5)
        )
        
        # Combine patterns with weights
        weights = {
            'pattern1': 0.25,  # Increased weight for speed changes
            'pattern2': 0.2,
            'pattern3': 0.15,
            'pattern4': 0.1,
            'pattern5': 0.2,  # Increased weight for multi-scale patterns
            'pattern6': 0.1
        }
        
        weighted_sum = (
            pattern1 * weights['pattern1'] +
            pattern2 * weights['pattern2'] +
            pattern3 * weights['pattern3'] +
            pattern4 * weights['pattern4'] +
            pattern5 * weights['pattern5'] +
            pattern6 * weights['pattern6']
        )
        
        # Convert to binary with lower threshold
        potential_swaps = weighted_sum > 0.15  # Reduced threshold
        
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
        scale_metrics = self._analyze_movement_scales(data)
        
        # Calculate speed confidence
        speed_ratio = metrics['speed_ratio']
        speed_conf = 1 - np.clip(speed_ratio, 0, 1)
        
        # Calculate acceleration confidence
        accel_ratio = metrics['accel_ratio']
        accel_conf = 1 - np.clip(accel_ratio, 0, 1)
        
        # Calculate jerk confidence
        jerk_conf = np.clip(
            (metrics['relative_head_jerk'] + metrics['relative_tail_jerk']) / 2,
            0, 1
        )
        
        # Calculate multi-scale confidence
        short_term = scale_metrics['short_term']
        medium_term = scale_metrics['medium_term']
        long_term = scale_metrics['long_term']
        
        scale_conf = np.clip(
            (1 - short_term['speed_ratio']) * 0.5 +
            (1 - medium_term['speed_ratio']) * 0.3 +
            (1 - long_term['speed_ratio']) * 0.2,
            0, 1
        )
        
        # Combine confidence factors with weights
        weights = {
            'speed': 0.3,
            'acceleration': 0.2,
            'jerk': 0.2,
            'scale': 0.3
        }
        
        confidence = (
            speed_conf * weights['speed'] +
            accel_conf * weights['acceleration'] +
            jerk_conf * weights['jerk'] +
            scale_conf * weights['scale']
        )
        
        # Apply sigmoid function to boost mid-range confidences
        confidence = 1 / (1 + np.exp(-10 * (confidence - 0.5)))
        
        # Ensure confidence is between 0 and 1
        confidence = np.clip(confidence, 0, 1)
        
        # Handle NaN values
        confidence = np.nan_to_num(confidence, nan=0.5)
        
        return confidence
        
    def _analyze_movement_scales(self, data: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Analyze movement at different time scales.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Dictionary of metrics at different scales
        """
        # Calculate velocities
        head_dx = np.gradient(data['X-Head'])
        head_dy = np.gradient(data['Y-Head'])
        tail_dx = np.gradient(data['X-Tail'])
        tail_dy = np.gradient(data['Y-Tail'])
        
        head_speed = np.sqrt(head_dx**2 + head_dy**2)
        tail_speed = np.sqrt(tail_dx**2 + tail_dy**2)
        
        # Calculate accelerations
        head_accel = np.abs(np.gradient(head_speed))
        tail_accel = np.abs(np.gradient(tail_speed))
        
        # Short-term analysis (few frames)
        short_term = {}
        short_term['speed'] = np.maximum(head_speed, tail_speed)
        short_term['speed_ratio'] = np.where(
            short_term['speed'] > 0,
            np.minimum(head_speed, tail_speed) / short_term['speed'],
            1.0
        )
        short_term['acceleration'] = np.maximum(head_accel, tail_accel)
        short_term['accel_ratio'] = np.where(
            short_term['acceleration'] > 0,
            np.minimum(head_accel, tail_accel) / short_term['acceleration'],
            1.0
        )
        
        # Medium-term analysis (more frames)
        window = self.window_sizes['medium_term']
        head_speed_smooth = self._smooth_signal(head_speed, window)
        tail_speed_smooth = self._smooth_signal(tail_speed, window)
        head_accel_smooth = self._smooth_signal(head_accel, window)
        tail_accel_smooth = self._smooth_signal(tail_accel, window)
        
        medium_term = {}
        medium_term['speed'] = np.maximum(head_speed_smooth, tail_speed_smooth)
        medium_term['speed_ratio'] = np.where(
            medium_term['speed'] > 0,
            np.minimum(head_speed_smooth, tail_speed_smooth) / medium_term['speed'],
            1.0
        )
        medium_term['acceleration'] = np.maximum(head_accel_smooth, tail_accel_smooth)
        medium_term['accel_ratio'] = np.where(
            medium_term['acceleration'] > 0,
            np.minimum(head_accel_smooth, tail_accel_smooth) / medium_term['acceleration'],
            1.0
        )
        
        # Long-term analysis (many frames)
        window = self.window_sizes['long_term']
        head_speed_smooth = self._smooth_signal(head_speed, window)
        tail_speed_smooth = self._smooth_signal(tail_speed, window)
        head_accel_smooth = self._smooth_signal(head_accel, window)
        tail_accel_smooth = self._smooth_signal(tail_accel, window)
        
        long_term = {}
        long_term['speed'] = np.maximum(head_speed_smooth, tail_speed_smooth)
        long_term['speed_ratio'] = np.where(
            long_term['speed'] > 0,
            np.minimum(head_speed_smooth, tail_speed_smooth) / long_term['speed'],
            1.0
        )
        long_term['acceleration'] = np.maximum(head_accel_smooth, tail_accel_smooth)
        long_term['accel_ratio'] = np.where(
            long_term['acceleration'] > 0,
            np.minimum(head_accel_smooth, tail_accel_smooth) / long_term['acceleration'],
            1.0
        )
        
        return {
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term
        }
        
    def _smooth_signal(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Smooth a signal using a rolling window.
        
        Args:
            signal: Array of values to smooth
            window: Window size for smoothing
            
        Returns:
            Smoothed array
        """
        # Use pandas rolling mean with proper handling of NaN values
        smoothed = pd.Series(signal).rolling(window=window, center=True).mean()
        smoothed = smoothed.ffill().bfill()
        return smoothed.values
        
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
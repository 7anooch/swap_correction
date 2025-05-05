"""Proximity-based swap detector."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .base import SwapDetector
from ..config import SwapConfig
from ..metrics.metrics import MovementMetrics

class ProximityDetector(SwapDetector):
    """Detector for swaps based on head-tail proximity."""
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect swaps based on proximity between head and tail points.
        
        A swap is detected when:
        1. The distance between head and tail is below the proximity threshold
        2. There is a significant change in body length
        
        Args:
            data: DataFrame containing tracking data
            
        Returns:
            Boolean array indicating frames with potential swaps
        """
        if self.metrics is None:
            self.setup(data)
            
        # Get body length and its statistics
        body_length = self.metrics.get_body_length()
        mean_length = np.nanmean(body_length)
        std_length = np.nanstd(body_length)
        
        # Calculate head-tail distance
        head_pos = np.column_stack([data['X-Head'], data['Y-Head']])
        tail_pos = np.column_stack([data['X-Tail'], data['Y-Tail']])
        distance = np.linalg.norm(head_pos - tail_pos, axis=1)
        
        # Calculate body length change
        length_change = np.abs(np.gradient(body_length))
        
        # Ensure arrays have the same length
        if len(distance) != len(length_change):
            if len(distance) > len(length_change):
                length_change = np.pad(length_change, (0, len(distance) - len(length_change)), mode='edge')
            else:
                distance = np.pad(distance, (0, len(length_change) - len(distance)), mode='edge')
        
        # Detect potential swaps
        is_close = distance < mean_length * 0.5  # Less strict proximity threshold
        is_length_change = length_change > std_length * 1.5  # Less strict length change threshold
        
        # Combine detections
        potential_swaps = is_close & is_length_change
        
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
        
        # Ensure potential_swaps has the same length as is_swap
        if len(potential_swaps) != len(is_swap):
            potential_swaps = np.pad(potential_swaps, (0, len(is_swap) - len(potential_swaps)), mode='edge')
        
        # Combine with potential swaps
        validated_swaps = potential_swaps & is_swap
        
        return validated_swaps
    
    def validate_swap(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """Validate a detected swap segment.
        
        Validation checks:
        1. Body length after potential correction should be closer to mean
        2. Movement should be more continuous after correction
        3. Speed patterns should be more consistent after correction
        
        Args:
            data: DataFrame containing tracking data
            start: Start frame of the swap segment
            end: End frame of the swap segment
            
        Returns:
            True if the swap is valid, False otherwise
        """
        if self.metrics is None:
            self.setup(data)
        
        # Get original body length statistics
        body_length = self.metrics.get_body_length()
        mean_length = np.nanmean(body_length)
        std_length = np.nanstd(body_length)
        
        # Create corrected data by swapping head and tail
        corrected_data = data.copy()
        swap_slice = slice(start, end + 1)
        
        # Swap head and tail positions
        head_x = corrected_data.loc[swap_slice, 'X-Head'].copy()
        head_y = corrected_data.loc[swap_slice, 'Y-Head'].copy()
        corrected_data.loc[swap_slice, 'X-Head'] = corrected_data.loc[swap_slice, 'X-Tail']
        corrected_data.loc[swap_slice, 'Y-Head'] = corrected_data.loc[swap_slice, 'Y-Tail']
        corrected_data.loc[swap_slice, 'X-Tail'] = head_x
        corrected_data.loc[swap_slice, 'Y-Tail'] = head_y
        
        # Calculate metrics for corrected data
        corrected_metrics = MovementMetrics(corrected_data, self.config.fps)
        corrected_length = corrected_metrics.get_body_length()
        
        # Check if correction improves body length consistency
        window = 5  # Consider surrounding frames
        start_idx = max(0, start - window)
        end_idx = min(len(body_length), end + window + 1)
        
        # Calculate errors over the extended window
        orig_length_error = np.mean(np.abs(body_length[start_idx:end_idx] - mean_length))
        corr_length_error = np.mean(np.abs(corrected_length[start_idx:end_idx] - mean_length))
        
        # Check if correction significantly improves body length
        length_improvement = (orig_length_error - corr_length_error) / (orig_length_error + 1e-6)
        
        # Check speed consistency
        orig_speed = self.metrics.get_speed('Head')[start_idx:end_idx]
        corr_speed = corrected_metrics.get_speed('Head')[start_idx:end_idx]
        
        orig_speed_std = np.nanstd(orig_speed)
        corr_speed_std = np.nanstd(corr_speed)
        
        speed_improvement = (orig_speed_std - corr_speed_std) / (orig_speed_std + 1e-6)
        
        # Combined validation criteria
        length_valid = length_improvement > 0.1  # Lower threshold
        speed_valid = speed_improvement > 0.05  # Lower threshold
        
        return length_valid and speed_valid
    
    def get_confidence(self, data: pd.DataFrame, start: int, end: int) -> float:
        """Calculate confidence score for a swap segment.
        
        The confidence score is based on:
        1. How much the body length deviates from mean
        2. How sudden the change in body length is
        3. How much correction improves the metrics
        4. The consistency of movement patterns
        5. The spatial relationship between head and tail
        
        Args:
            data: DataFrame containing tracking data
            start: Start frame of the swap segment
            end: End frame of the swap segment
            
        Returns:
            Confidence score between 0 and 1
        """
        if self.metrics is None:
            self.setup(data)
            
        # Get body length statistics
        body_length = self.metrics.get_body_length()
        mean_length = np.nanmean(body_length)
        std_length = np.nanstd(body_length)
        
        # Calculate metrics for the swap segment
        segment_length = body_length[start:end+1]
        if len(segment_length) < 2:
            # For single-frame segments, use surrounding frames
            window = 2
            start_idx = max(0, start - window)
            end_idx = min(len(body_length), end + window + 1)
            segment_length = body_length[start_idx:end_idx]
        
        # Calculate confidence components
        length_deviation = np.abs(np.nanmean(segment_length) - mean_length) / (std_length + 1e-6)
        length_change = np.nanmean(np.abs(np.diff(segment_length))) / (mean_length + 1e-6)
        
        # Create corrected data
        corrected_data = data.copy()
        swap_slice = slice(start, end + 1)
        head_x = corrected_data.loc[swap_slice, 'X-Head'].copy()
        head_y = corrected_data.loc[swap_slice, 'Y-Head'].copy()
        corrected_data.loc[swap_slice, 'X-Head'] = corrected_data.loc[swap_slice, 'X-Tail']
        corrected_data.loc[swap_slice, 'Y-Head'] = corrected_data.loc[swap_slice, 'Y-Tail']
        corrected_data.loc[swap_slice, 'X-Tail'] = head_x
        corrected_data.loc[swap_slice, 'Y-Tail'] = head_y
        
        # Calculate improvement metrics
        corrected_metrics = MovementMetrics(corrected_data, self.config.fps)
        corrected_length = corrected_metrics.get_body_length()
        
        window = 5
        start_idx = max(0, start - window)
        end_idx = min(len(body_length), end + window + 1)
        orig_error = np.nanmean(np.abs(body_length[start_idx:end_idx] - mean_length))
        corr_error = np.nanmean(np.abs(corrected_length[start_idx:end_idx] - mean_length))
        improvement = (orig_error - corr_error) / (orig_error + 1e-6)
        
        # Calculate movement pattern consistency
        head_speed = self.metrics.get_speed('Head')
        tail_speed = self.metrics.get_speed('Tail')
        speed_ratio = np.nanmean(np.abs(head_speed[start:end+1] - tail_speed[start:end+1])) / \
                     (np.nanmean(np.abs(head_speed[start:end+1] + tail_speed[start:end+1])) + 1e-6)
        
        # Calculate spatial relationship score
        head_pos = np.column_stack([data.loc[swap_slice, 'X-Head'], data.loc[swap_slice, 'Y-Head']])
        tail_pos = np.column_stack([data.loc[swap_slice, 'X-Tail'], data.loc[swap_slice, 'Y-Tail']])
        spatial_score = np.nanmean(np.linalg.norm(head_pos - tail_pos, axis=1)) / (mean_length + 1e-6)
        
        # Calculate confidence scores with adjusted weights
        deviation_conf = np.exp(-length_deviation)  # High deviation -> low confidence
        change_conf = np.minimum(length_change * 3, 1)  # Sudden change -> high confidence
        improve_conf = np.clip(improvement * 2.5, 0, 1)  # High improvement -> high confidence
        pattern_conf = np.clip(speed_ratio * 2, 0, 1)  # Inconsistent patterns -> high confidence
        spatial_conf = np.exp(-spatial_score)  # Close proximity -> high confidence
        
        # Combine confidence scores with adjusted weights
        confidence = (0.25 * deviation_conf + 
                     0.25 * change_conf + 
                     0.2 * improve_conf +
                     0.15 * pattern_conf +
                     0.15 * spatial_conf)
        
        return float(np.clip(confidence, 0, 1)) 
"""Filtering module for trajectory smoothing and prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .kalman_filter import KalmanFilter
from .config import SwapConfig

class TrajectoryFilter:
    """Filter for smoothing and predicting trajectories."""
    
    def __init__(self, config: Optional[SwapConfig] = None):
        """Initialize the trajectory filter.
        
        Args:
            config: Configuration object with filter parameters
        """
        self.config = config or SwapConfig()
        
        # Initialize filters for head and tail
        self.head_filter = self._create_kalman_filter()
        self.tail_filter = self._create_kalman_filter()
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-processing filtering to raw data.
        
        This step helps reduce noise and fill small gaps before
        swap detection.
        
        Args:
            data: Raw tracking data
            
        Returns:
            Filtered data
        """
        filtered_data = data.copy()
        
        # Extract head and tail positions
        head_pos = np.column_stack([
            data['X-Head'].values,
            data['Y-Head'].values
        ])
        tail_pos = np.column_stack([
            data['X-Tail'].values,
            data['Y-Tail'].values
        ])
        
        # Filter head trajectory
        filtered_head = self.head_filter.filter(head_pos)
        filtered_data['X-Head'] = filtered_head[:, 0]
        filtered_data['Y-Head'] = filtered_head[:, 1]
        
        # Filter tail trajectory
        filtered_tail = self.tail_filter.filter(tail_pos)
        filtered_data['X-Tail'] = filtered_tail[:, 0]
        filtered_data['Y-Tail'] = filtered_tail[:, 1]
        
        # Update midpoints
        filtered_data['X-Midpoint'] = (
            filtered_data['X-Head'] + filtered_data['X-Tail']
        ) / 2
        filtered_data['Y-Midpoint'] = (
            filtered_data['Y-Head'] + filtered_data['Y-Tail']
        ) / 2
        
        return filtered_data
    
    def postprocess(
        self,
        data: pd.DataFrame,
        swap_segments: List[Tuple[int, int, str, float]]
    ) -> pd.DataFrame:
        """Apply post-processing filtering after swap correction.
        
        This step smooths transitions around swap points and
        ensures trajectory consistency.
        
        Args:
            data: Data with corrected swaps
            swap_segments: List of (start, end, detector, confidence) tuples
            
        Returns:
            Smoothed data
        """
        processed_data = data.copy()
        
        # Reset filters
        self.head_filter = self._create_kalman_filter()
        self.tail_filter = self._create_kalman_filter()
        
        # Process each segment separately
        current_pos = 0
        for start, end, _, _ in sorted(swap_segments):
            # Process segment before swap
            if start > current_pos:
                self._filter_segment(
                    processed_data,
                    current_pos,
                    start - 1
                )
            
            # Process swap segment with extra smoothing
            self._filter_swap_segment(
                processed_data,
                start,
                end
            )
            
            current_pos = end + 1
        
        # Process remaining data
        if current_pos < len(processed_data):
            self._filter_segment(
                processed_data,
                current_pos,
                len(processed_data) - 1
            )
        
        return processed_data
    
    def predict_positions(
        self,
        data: pd.DataFrame,
        frames_ahead: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict future positions for validation.
        
        Args:
            data: Current tracking data
            frames_ahead: Number of frames to predict
            
        Returns:
            Tuple of predicted (head_positions, tail_positions)
        """
        # Extract current positions
        head_pos = np.column_stack([
            data['X-Head'].values[-1],
            data['Y-Head'].values[-1]
        ])
        tail_pos = np.column_stack([
            data['X-Tail'].values[-1],
            data['Y-Tail'].values[-1]
        ])
        
        # Initialize prediction arrays
        head_pred = np.zeros((frames_ahead, 2))
        tail_pred = np.zeros((frames_ahead, 2))
        
        # Predict future positions
        for i in range(frames_ahead):
            head_pred[i] = self.head_filter.predict()
            tail_pred[i] = self.tail_filter.predict()
        
        return head_pred, tail_pred
    
    def _create_kalman_filter(self) -> KalmanFilter:
        """Create a Kalman filter for 2D position tracking."""
        # State: [x, y, vx, vy]
        # Measurement: [x, y]
        dt = 1.0 / self.config.fps
        
        # Create and initialize the filter with default parameters
        kf = KalmanFilter(
            dt=dt,
            ndim=2,  # 2D tracking
            derivatives=2,  # Track up to acceleration
            estimateCov=1.0,  # Initial state covariance
            measurementCov=1.0,  # Measurement noise
            processCov=0.1  # Process noise
        )
        
        return kf
    
    def _filter_segment(
        self,
        data: pd.DataFrame,
        start: int,
        end: int
    ) -> None:
        """Filter a segment of data in place.
        
        Args:
            data: Data to filter
            start: Start frame
            end: End frame
        """
        # Extract positions
        head_pos = np.column_stack([
            data.loc[start:end, 'X-Head'].values,
            data.loc[start:end, 'Y-Head'].values
        ])
        tail_pos = np.column_stack([
            data.loc[start:end, 'X-Tail'].values,
            data.loc[start:end, 'Y-Tail'].values
        ])
        
        # Apply filtering
        filtered_head = self.head_filter.filter(head_pos)
        filtered_tail = self.tail_filter.filter(tail_pos)
        
        # Update data
        data.loc[start:end, 'X-Head'] = filtered_head[:, 0]
        data.loc[start:end, 'Y-Head'] = filtered_head[:, 1]
        data.loc[start:end, 'X-Tail'] = filtered_tail[:, 0]
        data.loc[start:end, 'Y-Tail'] = filtered_tail[:, 1]
        
        # Update midpoints
        data.loc[start:end, 'X-Midpoint'] = (
            data.loc[start:end, 'X-Head'] +
            data.loc[start:end, 'X-Tail']
        ) / 2
        data.loc[start:end, 'Y-Midpoint'] = (
            data.loc[start:end, 'Y-Head'] +
            data.loc[start:end, 'Y-Tail']
        ) / 2
    
    def _filter_swap_segment(
        self,
        data: pd.DataFrame,
        start: int,
        end: int
    ) -> None:
        """Filter a swap segment with extra smoothing.
        
        Args:
            data: Data to filter
            start: Start frame
            end: End frame
        """
        # Add padding for better smoothing
        pad = 5
        start_pad = max(0, start - pad)
        end_pad = min(len(data), end + pad)
        
        # Filter with padding
        self._filter_segment(data, start_pad, end_pad)
        
        # Additional smoothing at transition points
        if start > 0:
            self._smooth_transition(data, start - 1, start + 1)
        if end < len(data) - 1:
            self._smooth_transition(data, end - 1, end + 1)
    
    def _smooth_transition(
        self,
        data: pd.DataFrame,
        start: int,
        end: int
    ) -> None:
        """Apply additional smoothing to transition points.
        
        Args:
            data: Data to smooth
            start: Start frame
            end: End frame
        """
        window = np.hanning(end - start + 1)
        
        for col in ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']:
            values = data.loc[start:end, col].values
            smoothed = values * window
            data.loc[start:end, col] = smoothed
        
        # Update midpoints
        data.loc[start:end, 'X-Midpoint'] = (
            data.loc[start:end, 'X-Head'] +
            data.loc[start:end, 'X-Tail']
        ) / 2
        data.loc[start:end, 'Y-Midpoint'] = (
            data.loc[start:end, 'Y-Head'] +
            data.loc[start:end, 'Y-Tail']
        ) / 2 
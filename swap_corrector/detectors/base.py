"""
Base class for swap detectors.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
from ..config import SwapConfig

class SwapDetector(ABC):
    """Abstract base class for swap detectors."""
    
    def __init__(self, config: SwapConfig):
        """Initialize detector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.data = None
        self.metrics = {}
        self._current_thresholds = None
        
    def setup(self, data: pd.DataFrame) -> None:
        """Set up detector with data.
        
        Args:
            data: DataFrame with trajectory data
        """
        self.data = data
        self.metrics = self._calculate_metrics(data)
        self._update_thresholds()
        
    def _update_thresholds(self) -> None:
        """Update detection thresholds based on data."""
        if self.data is None:
            raise RuntimeError("Detector must be set up with data before updating thresholds")
            
        # Get base thresholds from config
        self._current_thresholds = self.config.get_thresholds()
        
        # Adjust thresholds based on data statistics
        if len(self.data) > 0:
            # Calculate speed statistics
            speeds = np.sqrt(
                np.diff(self.data['X-Head'])**2 + 
                np.diff(self.data['Y-Head'])**2
            ) * self.config.fps
            
            # Adjust speed threshold based on data
            if len(speeds) > 0:
                speed_mean = np.mean(speeds)
                speed_std = np.std(speeds)
                self._current_thresholds['speed'] = max(
                    self._current_thresholds['speed'],
                    speed_mean + 2 * speed_std
                )
                
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect potential swaps.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Array of boolean values indicating potential swaps
        """
        raise NotImplementedError("Subclasses must implement detect()")
        
    def get_confidence(self, data: pd.DataFrame) -> np.ndarray:
        """Get confidence scores for each frame.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Array of confidence scores between 0 and 1
        """
        if self.data is None or len(self.data) != len(data) or not np.array_equal(self.data, data):
            self.setup(data)
            
        # Calculate confidence scores
        scores = np.zeros(len(data))
        
        # Calculate metrics for each frame
        for i in range(len(data)):
            # Get window around current frame
            start = max(0, i - 5)
            end = min(len(data), i + 6)
            segment = data.iloc[start:end]
            
            # Calculate metrics for segment
            metrics = self._calculate_segment_metrics(segment)
            
            # Calculate confidence score
            scores[i] = self._calculate_confidence(metrics)
            
        return scores
        
    def _calculate_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for detection.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement _calculate_metrics()")
        
    def _calculate_segment_metrics(self, segment: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics for a segment of data.
        
        Args:
            segment: DataFrame segment
            
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError("Subclasses must implement _calculate_segment_metrics()")
        
    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate confidence score based on metrics.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Confidence score between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement _calculate_confidence()")
        
    def validate_swap(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """Validate if a potential swap is valid.
        
        Args:
            data: DataFrame with trajectory data
            start_idx: Start index of potential swap
            end_idx: End index of potential swap
            
        Returns:
            Boolean indicating if swap is valid
        """
        if self.data is None:
            raise ValueError("Detector not setup with data")
            
        # Get segment
        segment = self.data.iloc[start_idx:end_idx+1]
        
        # Calculate metrics for segment
        segment_metrics = self._calculate_metrics(segment)
        
        # Validate metrics
        return self._validate_metrics(segment_metrics)
        
    def _validate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Validate metrics against thresholds.
        
        Args:
            metrics: Dictionary of metrics to validate
            
        Returns:
            Whether the metrics indicate a valid swap
        """
        if self._current_thresholds is None:
            return True
            
        # Check if any metric exceeds its threshold
        for key, threshold in self._current_thresholds.items():
            if key in metrics:
                metric_values = metrics[key]
                if isinstance(metric_values, (np.ndarray, pd.Series)):
                    if isinstance(metric_values, pd.Series):
                        metric_values = metric_values.values
                    if np.any(metric_values > threshold):
                        return True
                else:
                    if metric_values > threshold:
                        return True
                        
        return False 
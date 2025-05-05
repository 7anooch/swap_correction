"""Base class for swap detectors."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

from ..config import SwapConfig
from ..metrics.metrics import MovementMetrics

class SwapDetector:
    """Base class for all swap detectors."""
    
    def __init__(self, config: SwapConfig):
        """Initialize the detector.
        
        Args:
            config: Configuration object with detection parameters
        """
        self.config = config
        self.metrics: Optional[MovementMetrics] = None
        self._current_thresholds: Optional[Dict[str, float]] = None
    
    def setup(self, data: pd.DataFrame) -> None:
        """Set up the detector with data.
        
        Args:
            data: DataFrame containing tracking data
        """
        self.metrics = MovementMetrics(data, self.config.fps)
        self._update_thresholds()
    
    def _update_thresholds(self) -> None:
        """Update thresholds based on current metrics."""
        if self.metrics is None:
            raise RuntimeError("Metrics not initialized. Call setup() first.")
            
        # Calculate current metrics for threshold adjustment
        current_metrics = {
            'speed': np.mean(self.metrics.get_speed('Midpoint')),
            'acceleration': np.mean(self.metrics.get_acceleration('Midpoint'))
        }
        
        # Update thresholds
        self._current_thresholds = self.config.get_thresholds(current_metrics)
    
    def detect(self, data: pd.DataFrame) -> np.ndarray:
        """Detect potential swaps in the data.
        
        Args:
            data: DataFrame containing tracking data
            
        Returns:
            Boolean array indicating frames with potential swaps
        """
        raise NotImplementedError("Subclasses must implement detect()")
    
    def get_swap_segments(self, swap_frames: np.ndarray) -> List[Tuple[int, int]]:
        """Convert frame-by-frame swap detection to segments.
        
        Args:
            swap_frames: Boolean array indicating frames with swaps
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        if not np.any(swap_frames):
            return []
        
        # Find transitions
        transitions = np.diff(swap_frames.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends = np.where(transitions == -1)[0] + 1
        
        # Handle edge cases
        if swap_frames[0]:
            starts = np.concatenate([[0], starts])
        if swap_frames[-1]:
            ends = np.concatenate([ends, [len(swap_frames)]])
        
        return list(zip(starts, ends))
    
    def validate_swap(self, data: pd.DataFrame, start: int, end: int) -> bool:
        """Validate a detected swap segment.
        
        Args:
            data: DataFrame containing tracking data
            start: Start frame of the swap segment
            end: End frame of the swap segment
            
        Returns:
            True if the swap is valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_swap()")
    
    def get_confidence(self, data: pd.DataFrame, start: int, end: int) -> float:
        """Calculate confidence score for a swap segment.
        
        Args:
            data: DataFrame containing tracking data
            start: Start frame of the swap segment
            end: End frame of the swap segment
            
        Returns:
            Confidence score between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement get_confidence()") 
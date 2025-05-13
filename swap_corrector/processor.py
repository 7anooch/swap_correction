"""
Main processor for swap correction pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from .config import SwapConfig, SwapCorrectionConfig
from .detectors import SwapDetector, ProximityDetector, SpeedDetector, TurnDetector
from .logger import setup_logger
from . import utils

class SwapProcessor:
    """Class for processing and correcting tracking swaps."""
    
    def __init__(self, config: Optional[SwapCorrectionConfig] = None,
                 detector_config: Optional[SwapConfig] = None,
                 **kwargs):
        """Initialize processor.
        
        Args:
            config: Optional correction configuration object
            detector_config: Optional detector configuration object
            **kwargs: Additional keyword arguments to update config
        """
        self.config = config or SwapCorrectionConfig()
        self.detector_config = detector_config or SwapConfig()
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
        # Initialize logger
        self.logger = setup_logger(level=self.config.log_level)
        
        # Initialize detectors
        self.detectors = {
            'speed': SpeedDetector(self.detector_config),
            'proximity': ProximityDetector(self.detector_config),
            'turn': TurnDetector(self.detector_config)
        }
        
        # Initialize data and metrics
        self.data = None
        self.fps = None
        self.predictions = {}
        self.confidences = {}
        self.metrics = None
        
        # Ensure detector metrics are initialized as None
        for detector in self.detectors.values():
            detector.metrics = None
            detector._current_thresholds = None
        
    def setup(self, data: Union[pd.DataFrame, Tuple[pd.DataFrame, float]]) -> None:
        """Setup processor with data.
        
        Args:
            data: DataFrame with trajectory data or tuple of (DataFrame, fps)
        """
        # Unpack tuple if needed
        if isinstance(data, tuple):
            self.data = data[0]
            self.fps = data[1]
        else:
            self.data = data
            self.fps = self.config.fps
        # Error handling
        if self.data is None or len(self.data) == 0:
            raise ValueError("Empty data")
        required_cols = {'X-Head', 'Y-Head', 'X-Tail', 'Y-Tail'}
        if not required_cols.issubset(self.data.columns):
            raise ValueError("Missing required columns")
        # Setup detectors
        for detector in self.detectors.values():
            detector.setup(self.data)
            
    def _run_detectors(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Run all detectors on data.
        
        Args:
            data: DataFrame with trajectory data
            
        Returns:
            Dictionary mapping detector names to boolean arrays of detections
        """
        # Setup processor with data if not already done
        if self.data is None or len(self.data) != len(data) or not np.array_equal(self.data, data):
            self.setup(data)
        
        # Run each detector
        detector_results = {}
        for name, detector in self.detectors.items():
            # Get predictions
            predictions = detector.detect(self.data)
            detector_results[name] = predictions
            
            # Get confidences
            confidences = detector.get_confidence(self.data)
            self.confidences[name] = confidences
            
            # Apply confidence thresholding
            if self.config.thresholds.confidence_threshold > 0:
                predictions = predictions & (confidences > self.config.thresholds.confidence_threshold)
                detector_results[name] = predictions
        
        return detector_results
    
    def _boolean_array_to_segments(self, arr: np.ndarray) -> list:
        """Convert a boolean array to a list of (start, end) segments (inclusive)."""
        segments = []
        start = None
        for i, val in enumerate(arr):
            if val:
                if start is None:
                    start = i
            elif start is not None:
                if i - start > 1:
                    segments.append((start, i-1))
                start = None
        if start is not None and len(arr) - start > 1:
            segments.append((start, len(arr)-1))
        return segments

    def _combine_predictions(self, detector_results: Dict[str, np.ndarray]) -> list:
        """Combine predictions from multiple detectors and return segments."""
        # Get weights for each detector
        weights = {
            'speed': 0.4,
            'proximity': 0.4,
            'turn': 0.2
        }
        weighted_sum = np.zeros(len(self.data))
        for name, predictions in detector_results.items():
            weighted_sum += predictions.astype(float) * weights[name]
        combined = weighted_sum > self.config.thresholds.detector_agreement
        combined = self._post_process_predictions(combined)
        # Convert to segments
        return self._boolean_array_to_segments(combined)

    def process(self, data: Union[pd.DataFrame, Tuple[pd.DataFrame, float]]) -> pd.DataFrame:
        """Process data and correct swaps."""
        self.setup(data)
        detector_results = self._run_detectors(self.data)
        self.predictions = detector_results
        combined_segments = self._combine_predictions(detector_results)
        corrected_data = self._apply_corrections(self.data, combined_segments)
        corrected_data['X-Midpoint'] = (corrected_data['X-Head'] + corrected_data['X-Tail']) / 2
        corrected_data['Y-Midpoint'] = (corrected_data['Y-Head'] + corrected_data['Y-Tail']) / 2
        return corrected_data

    def _post_process_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Post-process combined predictions.
        
        Args:
            predictions: Boolean array of predictions
            
        Returns:
            Post-processed boolean array
        """
        # Remove isolated detections
        min_duration = self.config.thresholds.min_swap_duration
        
        # Initialize result array
        result = predictions.copy()
        
        # Remove short segments
        start_idx = None
        for i in range(len(predictions)):
            if predictions[i]:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                if i - start_idx < min_duration:
                    result[start_idx:i] = False
                start_idx = None
        
        # Handle case where last segment extends to end
        if start_idx is not None and len(predictions) - start_idx < min_duration:
            result[start_idx:] = False
        
        # Fill gaps
        gap_threshold = self.config.thresholds.max_gap
        for i in range(1, len(predictions) - 1):
            if not predictions[i]:
                # Check if surrounded by predictions
                left_idx = i - 1
                while left_idx >= 0 and not predictions[left_idx]:
                    left_idx -= 1
                right_idx = i + 1
                while right_idx < len(predictions) and not predictions[right_idx]:
                    right_idx += 1
                
                # Fill gap if within threshold
                if (left_idx >= 0 and right_idx < len(predictions) and 
                    predictions[left_idx] and predictions[right_idx] and
                    right_idx - left_idx <= gap_threshold):
                    result[left_idx:right_idx+1] = True
        
        return result
        
    def _apply_corrections(self, data: pd.DataFrame, segments: list) -> pd.DataFrame:
        """Apply corrections to data using a list of (start, end) segments."""
        corrected = data.copy()
        swapped_indices = set()
        for seg in segments:
            if isinstance(seg, tuple) and len(seg) >= 2:
                start, end = seg[:2]
            else:
                continue
            # Only swap rows that haven't been swapped yet
            indices_to_swap = [i for i in range(start, end+1) if i not in swapped_indices]
            if not indices_to_swap:
                continue
            valid = self._validate_swap(data, start, end)
            print(f"[DEBUG] Applying correction: {start}-{end}, valid={valid}")
            if not valid:
                continue
            cols_to_swap = ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']
            temp = corrected.loc[indices_to_swap, cols_to_swap[:2]].copy()
            corrected.loc[indices_to_swap, cols_to_swap[:2]] = corrected.loc[indices_to_swap, cols_to_swap[2:]].values
            corrected.loc[indices_to_swap, cols_to_swap[2:]] = temp.values
            swapped_indices.update(indices_to_swap)
        return corrected
        
    def _get_swap_segments(self, predictions: np.ndarray) -> List[Tuple[int, int]]:
        """Get segments of consecutive swap predictions.
        
        Args:
            predictions: Array of swap predictions
            
        Returns:
            List of (start, end) indices for swap segments
        """
        segments = []
        start_idx = None
        
        for i in range(len(predictions)):
            if predictions[i]:
                if start_idx is None:
                    start_idx = i
            elif start_idx is not None:
                segments.append((start_idx, i-1))
                start_idx = None
                
        # Handle case where last segment extends to end
        if start_idx is not None:
            segments.append((start_idx, len(predictions)-1))
            
        return segments
        
    def _validate_swap(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """Validate if a potential swap is valid."""
        # For testing/debug, allow all swaps
        if getattr(self.config, 'debug', False):
            return True
        # Check if any detector validates the swap
        for detector in self.detectors.values():
            valid = detector.validate_swap(data, start_idx, end_idx)
            if valid:
                return True
        return False

    def _combine_detections(self, detector_results: Dict[str, np.ndarray]) -> List[Tuple[int, int, str, float]]:
        """Combine detections from all detectors into segments with detector and confidence info."""
        segments = []
        conf_thresh = getattr(self.config.thresholds, 'confidence_threshold', 0.0)
        for name, detections in detector_results.items():
            confidences = self.confidences.get(name, None)
            if confidences is None:
                confidences = np.ones(len(detections))
            start_idx = None
            for i, flag in enumerate(detections):
                if flag:
                    if start_idx is None:
                        start_idx = i
                elif start_idx is not None:
                    if i - start_idx > 1:
                        segment_conf = float(np.mean(confidences[start_idx:i]))
                        print(f"[DEBUG] Segment {name}: {start_idx}-{i-1}, conf={segment_conf:.3f}, thresh={conf_thresh}")
                        if segment_conf >= conf_thresh:
                            segments.append((start_idx, i-1, name, segment_conf))
                    start_idx = None
            if start_idx is not None and len(detections) - start_idx > 1:
                segment_conf = float(np.mean(confidences[start_idx:]))
                print(f"[DEBUG] Segment {name}: {start_idx}-{len(detections)-1}, conf={segment_conf:.3f}, thresh={conf_thresh}")
                if segment_conf >= conf_thresh:
                    segments.append((start_idx, len(detections)-1, name, segment_conf))
        segments.sort(key=lambda x: (x[0], x[1]))
        print(f"[DEBUG] Total segments after filtering: {len(segments)}")
        return segments 

    def get_detector_segments(self, data: pd.DataFrame = None) -> Dict[str, list]:
        """Return a dict mapping detector names to segment lists for the given data (or self.data)."""
        if data is None:
            data = self.data
        segments_dict = {}
        for name, detector in self.detectors.items():
            detections = detector.detect(data)
            segments = self._boolean_array_to_segments(detections)
            segments_dict[name] = segments
        return segments_dict 
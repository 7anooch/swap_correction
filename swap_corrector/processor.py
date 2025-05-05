"""Swap detection and correction processor."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from .config import SwapConfig, SwapCorrectionConfig
from .detectors.base import SwapDetector
from .detectors.proximity import ProximityDetector
from .detectors.speed import SpeedDetector
from .detectors.turn import TurnDetector
from .metrics import MovementMetrics
from .filtering import TrajectoryFilter
from .visualization import SwapVisualizer
from . import logger

class SwapProcessor:
    """Main processor for swap detection and correction.
    
    This class integrates multiple specialized detectors to identify and
    correct head-tail swaps in animal tracking data. It manages the complete
    pipeline from detection through validation and correction.
    """
    
    def __init__(
        self,
        config: Optional[SwapConfig] = None,
        correction_config: Optional[SwapCorrectionConfig] = None
    ):
        """Initialize the swap processor.
        
        Args:
            config: Configuration for swap detection
            correction_config: Configuration for correction pipeline
        """
        self.config = config or SwapConfig()
        self.correction_config = correction_config or SwapCorrectionConfig()
        
        # Initialize components
        self.detectors: Dict[str, SwapDetector] = {
            'proximity': ProximityDetector(self.config),
            'speed': SpeedDetector(self.config),
            'turn': TurnDetector(self.config)
        }
        self.metrics: Optional[MovementMetrics] = None
        self.filter = TrajectoryFilter(self.config)
        self.visualizer = SwapVisualizer(self.correction_config)
        
        # Set up logging
        self.logger = logger.setup_logger(
            self.correction_config,
            name="swap_processor"
        )
    
    def process(
        self,
        data: pd.DataFrame,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """Process tracking data to detect and correct swaps.
        
        Args:
            data: DataFrame containing tracking data with columns:
                - X-Head, Y-Head: Head coordinates
                - X-Tail, Y-Tail: Tail coordinates
                - X-Midpoint, Y-Midpoint: Midpoint coordinates
            output_dir: Directory to save diagnostic plots and reports
                
        Returns:
            DataFrame with corrected tracking data
        """
        self.logger.info("Starting swap detection and correction")
        
        # Pre-process data with Kalman filtering
        filtered_data = self.filter.preprocess(data)
        
        # Initialize metrics with filtered data
        self.metrics = MovementMetrics(filtered_data, self.config.fps)
        
        # Set up detectors with filtered data
        for detector in self.detectors.values():
            detector.setup(filtered_data)
            detector.metrics = self.metrics  # Share metrics instance
        
        # Detect swaps using all detectors
        detector_results = self._run_detectors(filtered_data)
        
        # Combine and validate detections
        swap_segments = self._combine_detections(detector_results)
        
        # Apply corrections
        corrected_data = self._apply_corrections(filtered_data, swap_segments)
        
        # Validate corrections if enabled
        if self.correction_config.validate:
            self.logger.info("Validating corrections")
            corrected_data = self._validate_corrections(corrected_data, swap_segments)
        
        # Post-process with Kalman filtering
        final_data = self.filter.postprocess(corrected_data, swap_segments)
        
        # Generate diagnostic report if output directory is provided
        if output_dir and self.correction_config.diagnostic_plots:
            self._generate_diagnostics(
                data,
                final_data,
                swap_segments,
                output_dir
            )
        
        self.logger.info("Completed swap detection and correction")
        return final_data
    
    def _run_detectors(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Tuple[np.ndarray, List[Tuple[int, int]]]]:
        """Run all detectors on the data.
        
        Args:
            data: DataFrame containing tracking data
            
        Returns:
            Dictionary mapping detector names to (detections, segments) tuples
        """
        results = {}
        
        for name, detector in self.detectors.items():
            self.logger.debug(f"Running {name} detector")
            
            # Get frame-by-frame detections
            detections = detector.detect(data)
            
            # Convert to segments
            segments = detector.get_swap_segments(detections)
            
            results[name] = (detections, segments)
            
            self.logger.debug(
                f"{name} detector found {len(segments)} swap segments"
            )
        
        return results
    
    def _combine_detections(
        self,
        detector_results: Dict[str, Tuple[np.ndarray, List[Tuple[int, int]]]]
    ) -> List[Tuple[int, int, str, float]]:
        """Combine and validate detections from all detectors.
        
        Args:
            detector_results: Dictionary of detector results
            
        Returns:
            List of (start, end, detector, confidence) tuples
        """
        combined_segments = []
        
        for detector_name, (detections, segments) in detector_results.items():
            detector = self.detectors[detector_name]
            
            for start, end in segments:
                # Calculate confidence score
                confidence = detector.get_confidence(
                    self.metrics.data,
                    start,
                    end
                )
                
                # Only include if confidence exceeds threshold
                if confidence > self.config.thresholds.confidence_threshold:
                    combined_segments.append((start, end, detector_name, confidence))
        
        # Sort by confidence (highest first)
        combined_segments.sort(key=lambda x: x[3], reverse=True)
        
        return combined_segments
    
    def _apply_corrections(
        self,
        data: pd.DataFrame,
        swap_segments: List[Tuple[int, int, str, float]]
    ) -> pd.DataFrame:
        """Apply corrections for detected swaps.
        
        Args:
            data: Original tracking data
            swap_segments: List of validated swap segments
            
        Returns:
            DataFrame with corrected data
        """
        corrected_data = data.copy()
        
        for start, end, detector_name, confidence in swap_segments:
            self.logger.debug(
                f"Applying correction for {detector_name} swap "
                f"(frames {start}-{end}, confidence: {confidence:.2f})"
            )
            
            # Predict positions to validate swap
            head_pred, tail_pred = self.filter.predict_positions(
                corrected_data.iloc[:start]
            )
            
            # Only apply correction if predictions support it
            if self._validate_with_predictions(
                corrected_data.iloc[start:end+1],
                head_pred,
                tail_pred
            ):
                # Swap head and tail positions
                swap_slice = slice(start, end)
                
                # Store original head positions
                head_x = corrected_data.loc[swap_slice, 'X-Head'].copy()
                head_y = corrected_data.loc[swap_slice, 'Y-Head'].copy()
                
                # Apply swap
                corrected_data.loc[swap_slice, 'X-Head'] = corrected_data.loc[swap_slice, 'X-Tail']
                corrected_data.loc[swap_slice, 'Y-Head'] = corrected_data.loc[swap_slice, 'Y-Tail']
                corrected_data.loc[swap_slice, 'X-Tail'] = head_x
                corrected_data.loc[swap_slice, 'Y-Tail'] = head_y
                
                # Update midpoint
                corrected_data.loc[swap_slice, 'X-Midpoint'] = (
                    corrected_data.loc[swap_slice, ['X-Head', 'X-Tail']].mean(axis=1)
                )
                corrected_data.loc[swap_slice, 'Y-Midpoint'] = (
                    corrected_data.loc[swap_slice, ['Y-Head', 'Y-Tail']].mean(axis=1)
                )
            else:
                self.logger.warning(
                    f"Skipping correction at frames {start}-{end} "
                    "due to prediction mismatch"
                )
        
        return corrected_data
    
    def _validate_corrections(
        self,
        corrected_data: pd.DataFrame,
        swap_segments: List[Tuple[int, int, str, float]]
    ) -> pd.DataFrame:
        """Validate and potentially refine corrections.
        
        Args:
            corrected_data: Data with initial corrections
            swap_segments: List of applied corrections
            
        Returns:
            DataFrame with validated corrections
        """
        self.logger.info("Validating corrections")
        
        # Update metrics with corrected data
        self.metrics = MovementMetrics(corrected_data, self.config.fps)
        
        # Validate each correction
        for start, end, detector_name, confidence in swap_segments:
            detector = self.detectors[detector_name]
            
            # Check if correction is valid
            is_valid = detector.validate_swap(corrected_data, start, end)
            
            if not is_valid:
                self.logger.warning(
                    f"Invalid correction detected for {detector_name} swap "
                    f"(frames {start}-{end}). Reverting correction."
                )
                
                # Revert the swap
                swap_slice = slice(start, end)
                
                # Store corrected head positions
                head_x = corrected_data.loc[swap_slice, 'X-Head'].copy()
                head_y = corrected_data.loc[swap_slice, 'Y-Head'].copy()
                
                # Revert swap
                corrected_data.loc[swap_slice, 'X-Head'] = corrected_data.loc[swap_slice, 'X-Tail']
                corrected_data.loc[swap_slice, 'Y-Head'] = corrected_data.loc[swap_slice, 'Y-Tail']
                corrected_data.loc[swap_slice, 'X-Tail'] = head_x
                corrected_data.loc[swap_slice, 'Y-Tail'] = head_y
                
                # Update midpoint
                corrected_data.loc[swap_slice, 'X-Midpoint'] = (
                    corrected_data.loc[swap_slice, ['X-Head', 'X-Tail']].mean(axis=1)
                )
                corrected_data.loc[swap_slice, 'Y-Midpoint'] = (
                    corrected_data.loc[swap_slice, ['Y-Head', 'Y-Tail']].mean(axis=1)
                )
        
        return corrected_data
    
    def _validate_with_predictions(
        self,
        data: pd.DataFrame,
        head_pred: np.ndarray,
        tail_pred: np.ndarray
    ) -> bool:
        """Validate swap using predicted positions.
        
        Args:
            data: Data segment to validate
            head_pred: Predicted head positions
            tail_pred: Predicted tail positions
            
        Returns:
            True if predictions support the swap
        """
        # Calculate distances to predictions
        head_dist_orig = np.mean(np.sqrt(
            (data['X-Head'].values - head_pred[0, 0])**2 +
            (data['Y-Head'].values - head_pred[0, 1])**2
        ))
        head_dist_swap = np.mean(np.sqrt(
            (data['X-Tail'].values - head_pred[0, 0])**2 +
            (data['Y-Tail'].values - head_pred[0, 1])**2
        ))
        
        # Check if swapped positions better match predictions
        return head_dist_swap < head_dist_orig
    
    def _generate_diagnostics(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        swap_segments: List[Tuple[int, int, str, float]],
        output_dir: Path
    ) -> None:
        """Generate diagnostic visualizations and reports.
        
        Args:
            raw_data: Original tracking data
            processed_data: Corrected tracking data
            swap_segments: List of detected and corrected swaps
            output_dir: Directory to save diagnostics
        """
        # Calculate metrics for visualization
        metrics = {
            'Speed': self.metrics.get_speed('Midpoint'),
            'Angular Velocity': self.metrics.get_angular_velocity(),
            'Curvature': self.metrics.get_curvature()
        }
        
        # Calculate performance metrics
        results = {
            'Total Swaps': len(swap_segments),
            'Average Confidence': np.mean([conf for _, _, _, conf in swap_segments])
        }
        
        # Generate diagnostic report
        self.visualizer.create_diagnostic_report(
            raw_data,
            processed_data,
            swap_segments,
            metrics,
            results,
            output_dir
        ) 
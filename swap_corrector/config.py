"""
Configuration classes for swap correction pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, ItemsView
from pathlib import Path
import numpy as np
import os
import pandas as pd

@dataclass
class SwapThresholds:
    """Thresholds for swap detection and correction."""
    
    # Detector thresholds
    speed: float = 3.0  # Speed threshold for swap detection
    accel: float = 50.0  # Acceleration threshold for swap detection
    proximity: float = 3.0  # Proximity threshold for swap detection
    turn: float = 0.5  # Turn threshold for swap detection
    angle: float = np.pi/4  # Angle threshold for swap detection
    curvature: float = 0.5  # Curvature threshold for swap detection
    body_length: float = 30.0  # Body length threshold for swap detection
    speed_factor: float = 2.0  # Speed factor for threshold adjustments
    accel_factor: float = 1.5  # Acceleration factor for threshold adjustments
    
    # Confidence thresholds
    confidence_threshold: float = 0.6  # Minimum confidence for a detection
    detector_agreement: float = 0.3  # Minimum agreement between detectors
    
    # Post-processing thresholds
    min_swap_duration: int = 2  # Minimum duration of a swap in frames
    max_gap: int = 3  # Maximum gap between swaps to fill
    
    # Validation thresholds
    max_body_length_deviation: float = 3.0  # Maximum standard deviations from mean body length
    max_angle_change: float = 1.0  # Maximum angle change in radians
    
    def __post_init__(self):
        """Validate thresholds after initialization."""
        if self.speed <= 0:
            raise ValueError("Speed threshold must be positive")
        if self.accel <= 0:
            raise ValueError("Acceleration threshold must be positive")
        if self.proximity <= 0:
            raise ValueError("Proximity threshold must be positive")
        if self.turn <= 0:
            raise ValueError("Turn threshold must be positive")
        if self.angle <= 0:
            raise ValueError("Angle threshold must be positive")
        if self.curvature <= 0:
            raise ValueError("Curvature threshold must be positive")
        if self.body_length <= 0:
            raise ValueError("Body length threshold must be positive")
        if self.speed_factor <= 0:
            raise ValueError("Speed factor must be positive")
        if self.accel_factor <= 0:
            raise ValueError("Acceleration factor must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if not 0 <= self.detector_agreement <= 1:
            raise ValueError("Detector agreement must be between 0 and 1")
        if self.min_swap_duration < 1:
            raise ValueError("Minimum swap duration must be at least 1 frame")
        if self.max_gap < 0:
            raise ValueError("Maximum gap must be non-negative")
        if self.max_body_length_deviation <= 0:
            raise ValueError("Maximum body length deviation must be positive")
        if self.max_angle_change <= 0:
            raise ValueError("Maximum angle change must be positive")
            
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SwapThresholds':
        """Create thresholds from dictionary.
        
        Args:
            data: Dictionary of threshold values
            
        Returns:
            SwapThresholds instance
        """
        return cls(
            speed=data.get('speed', 3.0),
            accel=data.get('accel', 50.0),
            proximity=data.get('proximity', 3.0),
            turn=data.get('turn', 0.5),
            angle=data.get('angle', np.pi/4),
            curvature=data.get('curvature', 0.5),
            body_length=data.get('body_length', 30.0),
            speed_factor=data.get('speed_factor', 2.0),
            accel_factor=data.get('accel_factor', 1.5),
            confidence_threshold=data.get('confidence_threshold', 0.6),
            detector_agreement=data.get('detector_agreement', 0.3),
            min_swap_duration=data.get('min_swap_duration', 2),
            max_gap=data.get('max_gap', 3),
            max_body_length_deviation=data.get('max_body_length_deviation', 3.0),
            max_angle_change=data.get('max_angle_change', 1.0)
        )
        
    def to_dict(self) -> Dict[str, float]:
        """Convert thresholds to dictionary.
        
        Returns:
            Dictionary of threshold values
        """
        return {
            'speed': self.speed,
            'accel': self.accel,
            'proximity': self.proximity,
            'turn': self.turn,
            'angle': self.angle,
            'curvature': self.curvature,
            'body_length': self.body_length,
            'speed_factor': self.speed_factor,
            'accel_factor': self.accel_factor,
            'confidence_threshold': self.confidence_threshold,
            'detector_agreement': self.detector_agreement,
            'min_swap_duration': self.min_swap_duration,
            'max_gap': self.max_gap,
            'max_body_length_deviation': self.max_body_length_deviation,
            'max_angle_change': self.max_angle_change
        }
        
    def adjust_for_movement(self, metrics_data: pd.DataFrame) -> None:
        """Adjust thresholds based on movement metrics.
        
        Args:
            metrics_data: DataFrame containing movement metrics
        """
        if metrics_data is None or metrics_data.empty:
            return
            
        # Calculate mean and std of speed and acceleration
        mean_speed = metrics_data['Speed'].mean()
        std_speed = metrics_data['Speed'].std()
        mean_accel = metrics_data['Acceleration'].mean()
        std_accel = metrics_data['Acceleration'].std()
        
        # Handle NaN values from empty or single-value DataFrames
        if np.isnan(mean_speed) or np.isnan(std_speed):
            mean_speed = 0.0
            std_speed = 0.0
        if np.isnan(mean_accel) or np.isnan(std_accel):
            mean_accel = 0.0
            std_accel = 0.0
            
        # Calculate new thresholds
        new_speed = mean_speed + std_speed * self.speed_factor
        new_accel = mean_accel + std_accel * self.accel_factor
        
        # Adjust thresholds based on movement statistics
        self.speed = max(new_speed, self.speed)
        self.accel = max(new_accel, self.accel)

@dataclass
class SwapCorrectionConfig:
    """Configuration for the swap correction pipeline."""
    
    debug: bool = True
    diagnostic_plots: bool = True
    show_plots: bool = False
    log_level: str = "DEBUG"
    log_file: str = "swap_corrector.log"
    output_dir: str = "output"
    filtered_data_filename: str = "filtered_data.csv"
    fix_swaps: bool = True
    remove_errors: bool = True
    interpolate: bool = True
    validate: bool = True
    filter_data: bool = True
    fps: float = 30.0
    min_swap_duration: float = 0.1
    max_swap_duration: float = 5.0
    confidence_threshold: float = 0.8
    speed_threshold: float = 3.0
    accel_threshold: float = 50.0
    proximity_threshold: float = 3.0
    turn_threshold: float = 0.5
    times: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values."""
        # Check boolean values
        for attr in ['debug', 'diagnostic_plots', 'show_plots', 'fix_swaps', 
                    'remove_errors', 'interpolate', 'validate', 'filter_data']:
            if not isinstance(getattr(self, attr), bool):
                raise ValueError(f"{attr} must be a boolean")
        
        # Check string values
        for attr in ['log_level', 'log_file', 'output_dir', 'filtered_data_filename']:
            if not isinstance(getattr(self, attr), str):
                raise ValueError(f"{attr} must be a string")
        
        # Check numeric values
        if not isinstance(self.fps, (int, float)) or self.fps <= 0:
            raise ValueError("fps must be a positive number")
        
        if not isinstance(self.min_swap_duration, (int, float)) or self.min_swap_duration <= 0:
            raise ValueError("min_swap_duration must be a positive number")
        
        if not isinstance(self.max_swap_duration, (int, float)) or self.max_swap_duration <= self.min_swap_duration:
            raise ValueError("max_swap_duration must be greater than min_swap_duration")
        
        for attr in ['confidence_threshold', 'speed_threshold', 'accel_threshold', 
                    'proximity_threshold', 'turn_threshold']:
            if not isinstance(getattr(self, attr), (int, float)) or getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be a positive number")
        
        # Check times
        if self.times is not None:
            if not isinstance(self.times, tuple) or len(self.times) != 2:
                raise ValueError("times must be a tuple of two numbers")
            if not all(isinstance(t, (int, float)) for t in self.times):
                raise ValueError("times must be numeric values")
            if self.times[0] >= self.times[1]:
                raise ValueError("start time must be less than end time")
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get detection thresholds.
        
        Returns:
            Dictionary of threshold values
        """
        return {
            'speed': self.speed_threshold,
            'acceleration': self.accel_threshold,
            'proximity': self.proximity_threshold,
            'turn': self.turn_threshold,
            'confidence': self.confidence_threshold
        }

class SwapConfig:
    """Configuration for swap detection."""
    
    def __init__(self, fps: float = 30.0, speed_threshold: float = 3.0,
                 accel_threshold: float = 50.0, proximity_threshold: float = 3.0,
                 turn_threshold: float = 0.5, confidence_threshold: float = 0.8,
                 log_level: str = "INFO", window_sizes: Optional[Dict[str, int]] = None,
                 thresholds: Optional[SwapThresholds] = None):
        """Initialize configuration.
        
        Args:
            fps: Frames per second
            speed_threshold: Speed threshold for swap detection
            accel_threshold: Acceleration threshold for swap detection
            proximity_threshold: Proximity threshold for swap detection
            turn_threshold: Turn threshold for swap detection
            confidence_threshold: Confidence threshold for swap validation
            log_level: Logging level
            window_sizes: Optional dictionary of window sizes
            thresholds: Optional SwapThresholds object
        """
        self.fps = fps
        self.speed_threshold = speed_threshold
        self.accel_threshold = accel_threshold
        self.proximity_threshold = proximity_threshold
        self.turn_threshold = turn_threshold
        self.confidence_threshold = confidence_threshold
        self.log_level = log_level
        
        # Set default window sizes if not provided
        if window_sizes is None:
            self.window_sizes = {
                'speed': 5,
                'acceleration': 3,
                'curvature': 5,
                'outlier': 5
            }
        else:
            self.window_sizes = window_sizes
            
        # Set thresholds
        self.thresholds = thresholds or SwapThresholds(
            speed=speed_threshold,
            accel=accel_threshold,
            proximity=proximity_threshold,
            turn=turn_threshold
        )
        
    def validate(self) -> bool:
        """Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        return (
            self.fps > 0 and
            self.speed_threshold > 0 and
            self.accel_threshold > 0 and
            self.proximity_threshold > 0 and
            self.turn_threshold > 0 and
            self.confidence_threshold > 0 and
            self.confidence_threshold <= 1 and
            self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] and
            all(size > 0 for size in self.window_sizes.values()) and
            self.thresholds.validate()
        )
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SwapConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            
        Returns:
            SwapConfig object
        """
        # Handle thresholds separately
        thresholds_dict = config_dict.pop('thresholds', None)
        if thresholds_dict is not None:
            config_dict['thresholds'] = SwapThresholds.from_dict(thresholds_dict)
            
        return cls(**config_dict)
        
    def get_thresholds(self, metrics_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Get detection thresholds.
        
        Args:
            metrics_data: Optional DataFrame containing movement metrics
            
        Returns:
            Dictionary of threshold values
        """
        if metrics_data is not None:
            self.thresholds.adjust_for_movement(metrics_data)
            
        return {
            'speed': self.thresholds.speed,
            'acceleration': self.thresholds.accel,
            'proximity': self.thresholds.proximity,
            'turn': self.thresholds.turn,
            'angle': self.thresholds.angle,
            'curvature': self.thresholds.curvature,
            'body_length': self.thresholds.body_length,
            'confidence': self.confidence_threshold
        } 
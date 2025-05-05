"""
Configuration settings for the swap correction pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import numpy as np

@dataclass
class SwapThresholds:
    """Thresholds for swap detection."""
    
    # Base thresholds
    proximity: float = 2.0      # pixels
    speed: float = 10.0        # pixels/frame
    angle: float = np.pi/4     # radians
    curvature: float = 0.1     # 1/pixel
    body_length: float = 0.7   # ratio of mean length
    
    # High-speed detection thresholds
    acceleration_threshold: float = 5.0  # pixels/frame²
    jerk_threshold: float = 10.0        # pixels/frame³
    prediction_error_threshold: float = 3.0  # pixels
    speed_ratio_threshold: float = 2.0  # ratio of head/tail speed
    
    # Turn detection thresholds
    turn_radius_threshold: float = 5.0  # pixels
    path_tortuosity_threshold: float = 1.5  # ratio
    confidence_threshold: float = 0.7  # confidence score
    
    # Adaptive factors
    speed_factor: float = 0.1  # How much to adjust thresholds based on speed
    accel_factor: float = 0.05 # How much to adjust thresholds based on acceleration
    jerk_factor: float = 0.02  # How much to adjust thresholds based on jerk
    turn_factor: float = 0.1   # How much to adjust thresholds based on turn radius

    def adjust_for_movement(self, speed: float, acceleration: float, jerk: float = 0.0, turn_radius: float = np.inf) -> Dict[str, float]:
        """Adjust thresholds based on movement context.
        
        Args:
            speed: Current speed in pixels/frame
            acceleration: Current acceleration in pixels/frame²
            jerk: Current jerk in pixels/frame³
            turn_radius: Current turn radius in pixels
            
        Returns:
            Dictionary of adjusted thresholds
        """
        speed_adjustment = 1.0 + self.speed_factor * speed
        accel_adjustment = 1.0 + self.accel_factor * abs(acceleration)
        jerk_adjustment = 1.0 + self.jerk_factor * abs(jerk)
        turn_adjustment = 1.0 + self.turn_factor * (1.0 / (1.0 + turn_radius))
        
        return {
            'proximity': self.proximity * speed_adjustment,
            'speed': self.speed * speed_adjustment,
            'angle': self.angle,  # Angle threshold remains constant
            'curvature': self.curvature * accel_adjustment,
            'body_length': self.body_length,  # Body length ratio remains constant
            'acceleration': self.acceleration_threshold * accel_adjustment,
            'jerk': self.jerk_threshold * jerk_adjustment,
            'prediction_error': self.prediction_error_threshold * speed_adjustment,
            'speed_ratio': self.speed_ratio_threshold,
            'turn_radius': self.turn_radius_threshold * turn_adjustment,
            'path_tortuosity': self.path_tortuosity_threshold,
            'confidence': self.confidence_threshold
        }

def default_window_sizes() -> Dict[str, int]:
    """Default window sizes for various calculations."""
    return {
        'speed': 5,        # frames for speed smoothing
        'acceleration': 7,  # frames for acceleration smoothing
        'curvature': 5,    # frames for curvature calculation
        'outlier': 5       # frames for outlier detection
    }

@dataclass
class SwapCorrectionConfig:
    """Configuration settings for swap correction pipeline."""
    
    # File settings
    filtered_data_filename: str = "filtered_data.csv"
    
    # Processing settings
    fix_swaps: bool = True
    validate: bool = False
    remove_errors: bool = True
    interpolate: bool = True
    filter_data: bool = False
    
    # Debug settings
    debug: bool = False
    diagnostic_plots: bool = True
    show_plots: bool = False
    
    # Plot settings
    times: Optional[Tuple[float, float]] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

# Default configuration
default_config = SwapCorrectionConfig()

@dataclass
class SwapConfig:
    """Configuration for swap correction."""
    
    # FPS of the recording
    fps: int = 30
    
    # Detection thresholds
    thresholds: SwapThresholds = field(default_factory=SwapThresholds)
    
    # Window sizes for various calculations
    window_sizes: Dict[str, int] = field(default_factory=default_window_sizes)
    
    def get_thresholds(self, metrics_data: Dict[str, float]) -> Dict[str, float]:
        """Get adjusted thresholds based on current metrics.
        
        Args:
            metrics_data: Dictionary containing current speed and acceleration
            
        Returns:
            Dictionary of adjusted thresholds
        """
        return self.thresholds.adjust_for_movement(
            speed=metrics_data.get('speed', 0.0),
            acceleration=metrics_data.get('acceleration', 0.0),
            jerk=metrics_data.get('jerk', 0.0)
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SwapConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            SwapConfig instance
        """
        thresholds = SwapThresholds(**config_dict.get('thresholds', {}))
        return cls(
            fps=config_dict.get('fps', 30),
            thresholds=thresholds,
            window_sizes=config_dict.get('window_sizes', default_window_sizes())
        ) 
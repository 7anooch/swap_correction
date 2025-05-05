import numpy as np
import pandas as pd
from typing import Tuple, Optional

class MovementMetrics:
    """Class for calculating movement-related metrics from tracking data."""
    
    def __init__(self, data: pd.DataFrame, fps: int):
        """Initialize the metrics calculator.
        
        Args:
            data: DataFrame with columns ['X-Head', 'Y-Head', 'X-Tail', 'Y-Tail']
            fps: Frames per second of the recording
        """
        self.data = data
        self.fps = fps
        self._validate_data()
    
    def _validate_data(self):
        """Validate that the data contains required columns."""
        required_columns = [
            'X-Head', 'Y-Head', 'X-Tail', 'Y-Tail',
            'X-Midpoint', 'Y-Midpoint'
        ]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_position(self, point: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get x, y coordinates for a given point.
        
        Args:
            point: One of 'Head', 'Tail', or 'Midpoint'
            
        Returns:
            Tuple of (x_coordinates, y_coordinates)
        """
        if point not in ['Head', 'Tail', 'Midpoint']:
            raise ValueError("point must be one of: Head, Tail, Midpoint")
            
        return (
            self.data[f'X-{point}'].values,
            self.data[f'Y-{point}'].values
        )
    
    def get_speed(self, point: str) -> np.ndarray:
        """Calculate speed for head/tail/midpoint.
        
        Args:
            point: One of 'Head', 'Tail', or 'Midpoint'
            
        Returns:
            Array of speeds in pixels/second
        """
        x, y = self.get_position(point)
        dx = np.diff(x)
        dy = np.diff(y)
        speed = np.sqrt(dx**2 + dy**2) * self.fps
        # Repeat last value to match array length
        return np.concatenate([speed, [speed[-1]]])
    
    def get_acceleration(self, point: str) -> np.ndarray:
        """Calculate acceleration for a given point.
        
        Args:
            point: One of 'Head', 'Tail', or 'Midpoint'
            
        Returns:
            Array of accelerations in pixels/second²
        """
        speed = self.get_speed(point)
        return np.gradient(speed) * self.fps
    
    def get_angular_velocity(self) -> np.ndarray:
        """Calculate angular velocity between head and tail.
        
        Returns:
            Array of angular velocities in radians/second
        """
        head_x, head_y = self.get_position('Head')
        tail_x, tail_y = self.get_position('Tail')
        
        # Calculate angles
        angles = np.arctan2(head_y - tail_y, head_x - tail_x)
        
        # Unwrap angles to handle discontinuities
        angles = np.unwrap(angles)
        
        # Calculate angular velocity
        angular_velocity = np.gradient(angles) * self.fps
        
        # Handle potential outliers
        median_vel = np.median(angular_velocity)
        mad = np.median(np.abs(angular_velocity - median_vel))
        outliers = np.abs(angular_velocity - median_vel) > 3 * mad
        
        if np.any(outliers):
            # Replace outliers with local median
            window = 5
            for i in np.where(outliers)[0]:
                start = max(0, i - window)
                end = min(len(angular_velocity), i + window + 1)
                angular_velocity[i] = np.median(angular_velocity[start:end])
        
        return angular_velocity
    
    def get_curvature(self) -> np.ndarray:
        """Calculate path curvature for turn detection.
        
        Returns:
            Array of curvature values (1/pixel)
        """
        x, y = self.get_position('Midpoint')
        
        # First derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx * dx + dy * dy) ** 1.5
        
        # Handle potential division by zero
        curvature = np.zeros_like(numerator)
        nonzero = denominator > 1e-10
        curvature[nonzero] = numerator[nonzero] / denominator[nonzero]
        
        return curvature
    
    def get_body_length(self) -> np.ndarray:
        """Calculate body length (distance between head and tail).
        
        Returns:
            Array of body lengths in pixels
        """
        head_x, head_y = self.get_position('Head')
        tail_x, tail_y = self.get_position('Tail')
        
        return np.sqrt(
            (head_x - tail_x)**2 + 
            (head_y - tail_y)**2
        ) 
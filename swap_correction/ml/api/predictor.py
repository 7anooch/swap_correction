"""
Swap predictor API for easy model usage.

Provides a clean interface for making predictions on tracking data.
"""

import os
import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple, List
from swap_correction.ml.api.model_loader import load_model
from swap_correction.ml.features import extract_all_frame_features_optimized
from swap_correction import pivr_loader


class SwapPredictor:
    """
    High-level interface for swap detection predictions.
    
    This class provides an easy-to-use interface for detecting swaps
    in animal tracking data using trained ML models.
    
    Example:
    --------
    >>> from swap_correction.ml.api import SwapPredictor
    >>> predictor = SwapPredictor(model_type='level1')
    >>> predictions = predictor.predict(trial_data, fps=30)
    >>> segments = predictor.predict_segments(trial_data, fps=30)
    """
    
    def __init__(self, model_type: Literal['level1', 'raw', 'raw_data'] = 'level1',
                 filter_sigma: float = 4.6):
        """
        Initialize swap predictor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use: 'level1' or 'raw'/'raw_data'
        filter_sigma : float
            Gaussian filter sigma for feature extraction (default: 4.6)
        """
        self.model_type = model_type
        self.filter_sigma = filter_sigma
        
        # Load model and preprocessors
        self.model, self.scaler, self.imputer, self.feature_names = load_model(model_type)
        
    def predict(self, trial_data: pd.DataFrame, fps: int = 30,
                return_probabilities: bool = False, threshold: float = 0.5) -> np.ndarray:
        """
        Predict swapped frames for a trial.
        
        Parameters:
        -----------
        trial_data : pd.DataFrame
            Tracking data (level1.csv or raw _data.csv depending on model_type)
        fps : int
            Frame rate (default: 30)
        return_probabilities : bool
            If True, return probabilities instead of binary predictions
        threshold : float
            Classification threshold (default: 0.5). Only used if return_probabilities=False.
            Use threshold optimization to find the best value for your metric.
            
        Returns:
        --------
        np.ndarray
            Binary predictions (0=no swap, 1=swap) or probabilities if return_probabilities=True
        """
        # Extract features
        features = extract_all_frame_features_optimized(
            trial_data, fps=fps, apply_filtering=True, filter_sigma=self.filter_sigma
        )
        
        # Preprocess
        X = self.imputer.transform(features.values)
        X = self.scaler.transform(X)
        
        # Predict
        if return_probabilities:
            return self.model.predict_proba(X)[:, 1]
        else:
            # Use custom threshold instead of default 0.5
            probabilities = self.model.predict_proba(X)[:, 1]
            return (probabilities >= threshold).astype(int)
    
    def predict_proba(self, trial_data: pd.DataFrame, fps: int = 30) -> np.ndarray:
        """
        Get swap probabilities for each frame.
        
        Parameters:
        -----------
        trial_data : pd.DataFrame
            Tracking data
        fps : int
            Frame rate
            
        Returns:
        --------
        np.ndarray
            Probability of swap for each frame (0-1)
        """
        return self.predict(trial_data, fps, return_probabilities=True)
    
    def predict_segments(self, trial_data: pd.DataFrame, fps: int = 30,
                        min_segment_length: int = 1) -> List[Tuple[int, int]]:
        """
        Predict swap segments (contiguous regions of swapped frames).
        
        Parameters:
        -----------
        trial_data : pd.DataFrame
            Tracking data
        fps : int
            Frame rate
        min_segment_length : int
            Minimum length of a segment to include (default: 1)
            
        Returns:
        --------
        list of tuples
            List of (start_frame, end_frame) tuples for each swap segment
        """
        predictions = self.predict(trial_data, fps)
        
        # Find contiguous segments
        segments = []
        in_segment = False
        segment_start = None
        
        for i, pred in enumerate(predictions):
            if pred == 1:  # Swapped
                if not in_segment:
                    segment_start = i
                    in_segment = True
            else:  # Not swapped
                if in_segment:
                    segment_end = i - 1
                    if segment_end - segment_start + 1 >= min_segment_length:
                        segments.append((segment_start, segment_end))
                    in_segment = False
        
        # Handle segment that extends to end
        if in_segment:
            segment_end = len(predictions) - 1
            if segment_end - segment_start + 1 >= min_segment_length:
                segments.append((segment_start, segment_end))
        
        return segments
    
    def predict_from_file(self, trial_dir: str, data_file: Optional[str] = None,
                         fps: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict swaps from a trial directory.
        
        Parameters:
        -----------
        trial_dir : str
            Directory containing trial data
        data_file : str, optional
            Specific data file to use (if None, auto-detect based on model_type)
        fps : int, optional
            Frame rate (if None, load from experiment_settings.json)
            
        Returns:
        --------
        tuple
            (predictions, trial_data)
            - predictions: Binary predictions array
            - trial_data: Loaded tracking data
        """
        # Auto-detect data file if not provided
        if data_file is None:
            if self.model_type == 'level1':
                # Look for _level1.csv or _data_level1.csv
                csv_files = [f for f in os.listdir(trial_dir) 
                           if f.endswith('_level1.csv') or f.endswith('_data_level1.csv')]
            else:
                # Look for raw _data.csv (but not _data_level1.csv or _data_level2.csv)
                csv_files = [f for f in os.listdir(trial_dir) 
                           if f.endswith('_data.csv') and '_level' not in f]
            
            if not csv_files:
                raise FileNotFoundError(f"No appropriate data file found in {trial_dir}")
            data_file = csv_files[0]
        
        # Load data
        trial_data = pivr_loader.load_raw_data(trial_dir, data_file, px2mm=True)
        
        # Get fps if not provided
        if fps is None:
            try:
                fps = pivr_loader.get_all_settings(trial_dir)['Framerate']
            except:
                fps = 30  # Default
        
        # Predict
        predictions = self.predict(trial_data, fps)
        
        return predictions, trial_data
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
        --------
        dict
            Model metadata and performance information
        """
        from swap_correction.ml.api.model_loader import get_model_info
        return get_model_info(self.model_type)


"""
Batch processing for multiple trials.

Provides utilities for processing multiple trials and evaluating models on datasets.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Literal
import pandas as pd
import numpy as np
from swap_correction.ml.api.predictor import SwapPredictor
from swap_correction import pivr_loader, error_analysis


class BatchProcessor:
    """
    Process multiple trials in batch.
    
    Example:
    --------
    >>> from swap_correction.ml.api import BatchProcessor
    >>> processor = BatchProcessor(model_type='level1')
    >>> results = processor.process_trials(trial_dirs)
    """
    
    def __init__(self, model_type: Literal['level1', 'raw', 'raw_data'] = 'level1',
                 filter_sigma: float = 4.6):
        """
        Initialize batch processor.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use
        filter_sigma : float
            Gaussian filter sigma for feature extraction
        """
        self.predictor = SwapPredictor(model_type=model_type, filter_sigma=filter_sigma)
        self.model_type = model_type
    
    def process_trials(self, trial_dirs: List[str],
                     data_file: Optional[str] = None,
                     fps: Optional[int] = None) -> Dict[str, Dict]:
        """
        Process multiple trials and return predictions.
        
        Parameters:
        -----------
        trial_dirs : list of str
            List of trial directory paths
        data_file : str, optional
            Specific data file to use (if None, auto-detect)
        fps : int, optional
            Frame rate (if None, load from settings)
            
        Returns:
        --------
        dict
            Dictionary mapping trial names to results:
            {
                'trial_name': {
                    'predictions': np.ndarray,
                    'probabilities': np.ndarray,
                    'segments': list of tuples,
                    'n_frames': int,
                    'n_swapped': int,
                    'swap_rate': float
                }
            }
        """
        results = {}
        
        for trial_dir in trial_dirs:
            trial_name = os.path.basename(trial_dir)
            
            try:
                predictions, trial_data = self.predictor.predict_from_file(
                    trial_dir, data_file=data_file, fps=fps
                )
                
                probabilities = self.predictor.predict_proba(trial_data, fps or 30)
                segments = self.predictor.predict_segments(trial_data, fps or 30)
                
                results[trial_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'segments': segments,
                    'n_frames': len(predictions),
                    'n_swapped': int(predictions.sum()),
                    'swap_rate': float(predictions.mean()),
                    'trial_data': trial_data
                }
                
            except Exception as e:
                results[trial_name] = {
                    'error': str(e)
                }
        
        return results
    
    def evaluate_on_dataset(self, data_dir: str,
                          ground_truth_level: str = 'level2',
                          data_file_pattern: Optional[str] = None) -> Dict:
        """
        Evaluate model performance on a dataset with ground truth.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing trial subdirectories
        ground_truth_level : str
            Ground truth level to compare against ('level1' or 'level2')
        data_file_pattern : str, optional
            Pattern for data files (e.g., '*_level1.csv')
            
        Returns:
        --------
        dict
            Evaluation results with metrics for each trial and summary statistics
        """
        # Get all trial directories
        trial_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))]
        
        trial_results = []
        
        for trial_dir in trial_dirs:
            trial_name = os.path.basename(trial_dir)
            
            try:
                # Determine input data file based on model type
                if self.model_type == 'level1':
                    input_files = [f for f in os.listdir(trial_dir) if f.endswith('_level1.csv')]
                else:
                    input_files = [f for f in os.listdir(trial_dir) if f.endswith('_data.csv')]
                
                if not input_files:
                    continue
                
                input_file = input_files[0]
                input_data = pivr_loader.load_raw_data(trial_dir, input_file, px2mm=True)
                
                # Load ground truth
                gt_files = [f for f in os.listdir(trial_dir) 
                           if f.endswith(f'_{ground_truth_level}.csv')]
                if not gt_files:
                    continue
                
                gt_file = gt_files[0]
                gt_data = pivr_loader.load_raw_data(trial_dir, gt_file, px2mm=True)
                
                # Get fps
                try:
                    fps = pivr_loader.get_all_settings(trial_dir)['Framerate']
                except:
                    fps = 30
                
                # Predict
                predictions = self.predictor.predict(input_data, fps)
                
                # Get ground truth swaps
                swapped_frames = error_analysis.identify_swapped_frames(input_data, gt_data)
                gt_predictions = np.zeros(len(input_data), dtype=int)
                if len(swapped_frames) > 0:
                    gt_predictions[swapped_frames] = 1
                
                # Calculate metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                precision = precision_score(gt_predictions, predictions, zero_division=0)
                recall = recall_score(gt_predictions, predictions, zero_division=0)
                f1 = f1_score(gt_predictions, predictions, zero_division=0)
                
                # Confusion matrix
                tp = np.sum((gt_predictions == 1) & (predictions == 1))
                fp = np.sum((gt_predictions == 0) & (predictions == 1))
                fn = np.sum((gt_predictions == 1) & (predictions == 0))
                tn = np.sum((gt_predictions == 0) & (predictions == 0))
                
                trial_results.append({
                    'trial': trial_name,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'tp': int(tp),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tn': int(tn),
                    'n_frames': len(predictions),
                    'n_swapped_gt': int(gt_predictions.sum()),
                    'n_swapped_pred': int(predictions.sum()),
                })
                
            except Exception as e:
                trial_results.append({
                    'trial': trial_name,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        valid_results = [r for r in trial_results if 'error' not in r]
        
        if valid_results:
            summary = {
                'n_trials': len(trial_results),
                'n_valid': len(valid_results),
                'mean_precision': float(np.mean([r['precision'] for r in valid_results])),
                'mean_recall': float(np.mean([r['recall'] for r in valid_results])),
                'mean_f1': float(np.mean([r['f1'] for r in valid_results])),
                'std_precision': float(np.std([r['precision'] for r in valid_results])),
                'std_recall': float(np.std([r['recall'] for r in valid_results])),
                'std_f1': float(np.std([r['f1'] for r in valid_results])),
            }
        else:
            summary = {
                'n_trials': len(trial_results),
                'n_valid': 0,
            }
        
        return {
            'summary': summary,
            'trial_results': trial_results
        }


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
                    # Look for _level1.csv or _data_level1.csv
                    input_files = [f for f in os.listdir(trial_dir) 
                                 if f.endswith('_level1.csv') or f.endswith('_data_level1.csv')]
                else:
                    # Look for raw _data.csv (but not _data_level1.csv or _data_level2.csv)
                    input_files = [f for f in os.listdir(trial_dir) 
                                 if f.endswith('_data.csv') and '_level' not in f]
                
                if not input_files:
                    continue
                
                input_file = input_files[0]
                input_data = pivr_loader.load_raw_data(trial_dir, input_file, px2mm=True)
                
                # Load ground truth
                # Look for _level2.csv or _data_level2.csv
                gt_files = [f for f in os.listdir(trial_dir) 
                           if f.endswith(f'_{ground_truth_level}.csv') or 
                              f.endswith(f'_data_{ground_truth_level}.csv')]
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
                
                # Confusion matrix
                tp = np.sum((gt_predictions == 1) & (predictions == 1))
                fp = np.sum((gt_predictions == 0) & (predictions == 1))
                fn = np.sum((gt_predictions == 1) & (predictions == 0))
                tn = np.sum((gt_predictions == 0) & (predictions == 0))
                
                # Calculate metrics with special handling for edge cases
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                # Calculate specificity and sensitivity (sensitivity = recall)
                # Specificity = TN / (TN + FP)
                if (tn + fp) > 0:
                    specificity = tn / (tn + fp)
                else:
                    specificity = 1.0 if fp == 0 else 0.0
                
                # Sensitivity = Recall = TP / (TP + FN)
                if (tp + fn) > 0:
                    sensitivity = tp / (tp + fn)
                else:
                    sensitivity = 1.0 if fn == 0 else 0.0
                
                # Calculate additional metrics (before special case handling)
                total_frames = len(predictions)
                n_swapped_gt = int(gt_predictions.sum())
                n_swapped_pred = int(predictions.sum())
                
                # Percentage of swaps resolved = TP / (TP + FN) = sensitivity = recall
                # This is the percentage of ground truth swaps that were correctly detected
                if (tp + fn) > 0:
                    pct_swaps_resolved = (tp / (tp + fn)) * 100.0
                else:
                    pct_swaps_resolved = 100.0 if fn == 0 else 0.0
                
                # Percentage of frames clean pre-correction = (frames not swapped in GT) / total frames
                # This is the percentage of frames that are NOT swapped in the raw/level1 data
                pct_frames_clean_pre = ((total_frames - n_swapped_gt) / total_frames) * 100.0 if total_frames > 0 else 0.0
                
                # Percentage of frames clean post-correction = (TP + TN) / total frames
                # This is the accuracy: percentage of frames correctly classified
                pct_frames_clean_post = ((tp + tn) / total_frames) * 100.0 if total_frames > 0 else 0.0
                
                # Special case: if no swaps in ground truth and no predictions, perfect match
                if gt_predictions.sum() == 0 and predictions.sum() == 0:
                    precision = 1.0  # Perfect precision (no false positives)
                    recall = 1.0    # Perfect recall (no false negatives)
                    f1 = 1.0
                    sensitivity = 1.0  # No swaps to detect, perfect sensitivity
                    specificity = 1.0  # No false positives, perfect specificity
                    pct_swaps_resolved = 100.0  # No swaps to resolve, perfect
                    pct_frames_clean_post = 100.0  # All frames correctly classified
                # Special case: if no swaps in ground truth but model predicts swaps, precision=0
                elif gt_predictions.sum() == 0 and predictions.sum() > 0:
                    precision = 0.0  # All predictions are false positives
                    recall = 1.0     # No false negatives (no swaps to miss)
                    f1 = 0.0
                    sensitivity = 1.0  # No swaps to detect, perfect sensitivity
                    specificity = 0.0  # All predictions are false positives
                    pct_swaps_resolved = 100.0  # No swaps to resolve, perfect
                    pct_frames_clean_post = (tn / total_frames) * 100.0 if total_frames > 0 else 0.0
                # Special case: if swaps in ground truth but model predicts none, recall=0
                elif gt_predictions.sum() > 0 and predictions.sum() == 0:
                    precision = 1.0  # No false positives
                    recall = 0.0     # All swaps missed
                    f1 = 0.0
                    sensitivity = 0.0  # All swaps missed
                    specificity = 1.0  # No false positives, perfect specificity
                    pct_swaps_resolved = 0.0  # No swaps resolved
                    pct_frames_clean_post = (tn / total_frames) * 100.0 if total_frames > 0 else 0.0
                # Normal case: use standard metrics
                else:
                    precision = precision_score(gt_predictions, predictions, zero_division=0)
                    recall = recall_score(gt_predictions, predictions, zero_division=0)
                    f1 = f1_score(gt_predictions, predictions, zero_division=0)
                    # Sensitivity = recall, but recalculate for consistency
                    sensitivity = recall
                    # Specificity already calculated above
                    # pct_swaps_resolved, pct_frames_clean_pre, pct_frames_clean_post already calculated above
                
                trial_results.append({
                    'trial': trial_name,
                    'precision': float(precision),
                    'recall': float(recall),
                    'sensitivity': float(sensitivity),
                    'specificity': float(specificity),
                    'f1': float(f1),
                    'pct_swaps_resolved': float(pct_swaps_resolved),
                    'pct_frames_clean_pre': float(pct_frames_clean_pre),
                    'pct_frames_clean_post': float(pct_frames_clean_post),
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
                'mean_sensitivity': float(np.mean([r['sensitivity'] for r in valid_results])),
                'mean_specificity': float(np.mean([r['specificity'] for r in valid_results])),
                'mean_f1': float(np.mean([r['f1'] for r in valid_results])),
                'mean_pct_swaps_resolved': float(np.mean([r['pct_swaps_resolved'] for r in valid_results])),
                'mean_pct_frames_clean_pre': float(np.mean([r['pct_frames_clean_pre'] for r in valid_results])),
                'mean_pct_frames_clean_post': float(np.mean([r['pct_frames_clean_post'] for r in valid_results])),
                'std_precision': float(np.std([r['precision'] for r in valid_results])),
                'std_recall': float(np.std([r['recall'] for r in valid_results])),
                'std_sensitivity': float(np.std([r['sensitivity'] for r in valid_results])),
                'std_specificity': float(np.std([r['specificity'] for r in valid_results])),
                'std_f1': float(np.std([r['f1'] for r in valid_results])),
                'std_pct_swaps_resolved': float(np.std([r['pct_swaps_resolved'] for r in valid_results])),
                'std_pct_frames_clean_pre': float(np.std([r['pct_frames_clean_pre'] for r in valid_results])),
                'std_pct_frames_clean_post': float(np.std([r['pct_frames_clean_post'] for r in valid_results])),
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


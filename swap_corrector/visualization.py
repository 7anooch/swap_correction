"""Visualization tools for swap detection and correction."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns

from .config import SwapCorrectionConfig
from . import logger

class SwapVisualizer:
    """Visualization tools for swap detection and correction."""
    
    def __init__(self, config: Optional[SwapCorrectionConfig] = None):
        """Initialize visualizer.
        
        Args:
            config: Configuration for visualization settings
        """
        self.config = config or SwapCorrectionConfig()
        self.logger = logger.setup_logger(self.config, name="swap_visualizer")
        
        # Set up style
        plt.style.use('default')  # Use default style instead of seaborn
        sns.set_theme()  # Initialize seaborn with default theme
        sns.set_palette("husl")
    
    def plot_trajectories(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        swap_segments: List[Tuple[int, int, str, float]],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot raw and processed trajectories with swap segments highlighted.
        
        Args:
            raw_data: Original tracking data
            processed_data: Corrected tracking data
            swap_segments: List of (start, end, detector, confidence) tuples
            save_path: Path to save the plot, if None will display
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot raw data
        ax1.plot(raw_data['X-Head'], raw_data['Y-Head'], 'b-', label='Head', alpha=0.5)
        ax1.plot(raw_data['X-Tail'], raw_data['Y-Tail'], 'r-', label='Tail', alpha=0.5)
        ax1.set_title('Raw Trajectories')
        ax1.legend()
        
        # Plot processed data
        ax2.plot(processed_data['X-Head'], processed_data['Y-Head'], 'b-', label='Head', alpha=0.5)
        ax2.plot(processed_data['X-Tail'], processed_data['Y-Tail'], 'r-', label='Tail', alpha=0.5)
        
        # Highlight swap segments
        for start, end, detector, conf in swap_segments:
            color = self._get_detector_color(detector)
            x = processed_data.loc[start:end, 'X-Head']
            y = processed_data.loc[start:end, 'Y-Head']
            ax2.plot(x, y, color=color, linewidth=2, alpha=0.8,
                    label=f'{detector} ({conf:.2f})')
        
        ax2.set_title('Processed Trajectories')
        ax2.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_metrics(
        self,
        data: pd.DataFrame,
        metrics: Dict[str, np.ndarray],
        swap_segments: List[Tuple[int, int, str, float]],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot movement metrics with swap segments highlighted.
        
        Args:
            data: Tracking data
            metrics: Dictionary of metric arrays
            swap_segments: List of (start, end, detector, confidence) tuples
            save_path: Path to save the plot, if None will display
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 4*n_metrics))
        
        for (metric_name, metric_values), ax in zip(metrics.items(), axes):
            # Plot metric
            ax.plot(metric_values, 'k-', alpha=0.5, label=metric_name)
            
            # Highlight swap segments
            for start, end, detector, conf in swap_segments:
                color = self._get_detector_color(detector)
                ax.axvspan(start, end, color=color, alpha=0.3)
            
            ax.set_title(f'{metric_name} over time')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_confidence_distribution(
        self,
        swap_segments: List[Tuple[int, int, str, float]],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot distribution of confidence scores by detector.
        
        Args:
            swap_segments: List of (start, end, detector, confidence) tuples
            save_path: Path to save the plot, if None will display
        """
        # Extract confidences by detector
        confidences = {}
        for _, _, detector, conf in swap_segments:
            if detector not in confidences:
                confidences[detector] = []
            confidences[detector].append(conf)
        
        # Create violin plot
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        for detector, conf_list in confidences.items():
            data.append(conf_list)
            labels.append(detector)
        
        plt.violinplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel('Confidence Score')
        plt.title('Confidence Score Distribution by Detector')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_performance_metrics(
        self,
        results: Dict[str, float],
        save_path: Optional[Path] = None
    ) -> None:
        """Plot performance metrics.
        
        Args:
            results: Dictionary of performance metrics
            save_path: Path to save the plot, if None will display
        """
        # Create bar plot
        plt.figure(figsize=(10, 6))
        metrics = list(results.keys())
        values = list(results.values())
        
        plt.bar(metrics, values)
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        plt.title('Performance Metrics')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def create_diagnostic_report(
        self,
        raw_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        swap_segments: List[Tuple[int, int, str, float]],
        metrics: Dict[str, np.ndarray],
        results: Dict[str, float],
        output_dir: Path
    ) -> None:
        """Create comprehensive diagnostic report with all plots.
        
        Args:
            raw_data: Original tracking data
            processed_data: Corrected tracking data
            swap_segments: List of (start, end, detector, confidence) tuples
            metrics: Dictionary of metric arrays
            results: Dictionary of performance metrics
            output_dir: Directory to save report
        """
        output_dir.mkdir(exist_ok=True)
        
        # Plot trajectories
        self.plot_trajectories(
            raw_data,
            processed_data,
            swap_segments,
            output_dir / 'trajectories.png'
        )
        
        # Plot metrics
        self.plot_metrics(
            processed_data,
            metrics,
            swap_segments,
            output_dir / 'metrics.png'
        )
        
        # Plot confidence distribution
        self.plot_confidence_distribution(
            swap_segments,
            output_dir / 'confidence_distribution.png'
        )
        
        # Plot performance metrics
        self.plot_performance_metrics(
            results,
            output_dir / 'performance_metrics.png'
        )
        
        # Create summary report
        self._create_summary_report(
            swap_segments,
            results,
            output_dir / 'summary.txt'
        )
    
    def _get_detector_color(self, detector: str) -> str:
        """Get color for detector visualization."""
        colors = {
            'proximity': 'red',
            'speed': 'green',
            'turn': 'blue'
        }
        return colors.get(detector, 'gray')
    
    def _create_summary_report(
        self,
        swap_segments: List[Tuple[int, int, str, float]],
        results: Dict[str, float],
        output_path: Path
    ) -> None:
        """Create text summary report.
        
        Args:
            swap_segments: List of (start, end, detector, confidence) tuples
            results: Dictionary of performance metrics
            output_path: Path to save report
        """
        # Count swaps by detector
        detector_counts = {}
        for _, _, detector, _ in swap_segments:
            detector_counts[detector] = detector_counts.get(detector, 0) + 1
        
        # Calculate average confidence by detector
        detector_confidences = {}
        for _, _, detector, conf in swap_segments:
            if detector not in detector_confidences:
                detector_confidences[detector] = []
            detector_confidences[detector].append(conf)
        
        avg_confidences = {
            det: np.mean(confs)
            for det, confs in detector_confidences.items()
        }
        
        # Write report
        with open(output_path, 'w') as f:
            f.write("Swap Detection and Correction Summary\n")
            f.write("===================================\n\n")
            
            f.write("Detector Statistics:\n")
            f.write("-----------------\n")
            for detector in detector_counts:
                f.write(f"{detector}:\n")
                f.write(f"  - Swaps detected: {detector_counts[detector]}\n")
                f.write(f"  - Average confidence: {avg_confidences[detector]:.3f}\n")
            f.write("\n")
            
            f.write("Performance Metrics:\n")
            f.write("------------------\n")
            for metric, value in results.items():
                f.write(f"{metric}: {value:.3f}\n")
            f.write("\n")
            
            f.write("Swap Segments:\n")
            f.write("-------------\n")
            for start, end, detector, conf in swap_segments:
                f.write(f"Frames {start}-{end}: {detector} (confidence: {conf:.3f})\n") 
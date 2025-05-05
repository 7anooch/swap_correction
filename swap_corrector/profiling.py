"""Performance profiling and optimization tools."""

import cProfile
import pstats
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from functools import wraps
import line_profiler
import memory_profiler

from .config import SwapConfig, SwapCorrectionConfig
from .processor import SwapProcessor

class PerformanceProfiler:
    """Tools for profiling and optimizing performance."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize profiler.
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = output_dir or Path("profiling_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize profilers
        self.profiler = cProfile.Profile()
        self.line_profiler = line_profiler.LineProfiler()
        
        # Store timing results
        self.timing_results: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
    
    def profile_pipeline(
        self,
        data: pd.DataFrame,
        config: Optional[SwapConfig] = None,
        correction_config: Optional[SwapCorrectionConfig] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Profile the complete processing pipeline.
        
        Args:
            data: Input tracking data
            config: Configuration for swap detection
            correction_config: Configuration for correction pipeline
            
        Returns:
            Tuple of (processed_data, profiling_results)
        """
        # Create processor
        processor = SwapProcessor(config, correction_config)
        
        # Profile complete pipeline
        self.profiler.enable()
        start_time = time.time()
        start_memory = memory_profiler.memory_usage()[0]
        
        processed_data = processor.process(data)
        
        end_time = time.time()
        end_memory = memory_profiler.memory_usage()[0]
        self.profiler.disable()
        
        # Save profiling results
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats(self.output_dir / 'profile.stats')
        
        # Calculate metrics
        total_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        results = {
            'timing': {
                'total_time': total_time,
                'per_frame': total_time / len(data)
            },
            'memory': {
                'peak_usage': end_memory,
                'total_allocated': memory_used
            }
        }
        
        return processed_data, results
    
    def profile_component(
        self,
        component_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Tuple[any, Dict[str, float]]:
        """Profile a specific component or function.
        
        Args:
            component_name: Name of the component
            func: Function to profile
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (function_result, profiling_metrics)
        """
        # Add line profiler
        profiled_func = self.line_profiler(func)
        
        # Profile execution
        start_time = time.time()
        start_memory = memory_profiler.memory_usage()[0]
        
        result = profiled_func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = memory_profiler.memory_usage()[0]
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Store results
        if component_name not in self.timing_results:
            self.timing_results[component_name] = []
        if component_name not in self.memory_usage:
            self.memory_usage[component_name] = []
            
        self.timing_results[component_name].append(execution_time)
        self.memory_usage[component_name].append(memory_used)
        
        # Save line profiling results
        self.line_profiler.print_stats(output_unit=1e-3)  # milliseconds
        
        metrics = {
            'execution_time': execution_time,
            'memory_used': memory_used,
            'avg_execution_time': np.mean(self.timing_results[component_name]),
            'avg_memory_used': np.mean(self.memory_usage[component_name])
        }
        
        return result, metrics
    
    def analyze_bottlenecks(self) -> Dict[str, Dict[str, float]]:
        """Analyze profiling results to identify bottlenecks.
        
        Returns:
            Dictionary of bottleneck analysis results
        """
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        # Get top time-consuming functions
        time_bottlenecks = []
        for func_stats in stats.stats.items():
            ((file, line, name), (cc, nc, tt, ct, callers)) = func_stats
            time_bottlenecks.append({
                'function': name,
                'calls': cc,
                'time': ct,
                'time_per_call': ct/cc if cc > 0 else 0
            })
        
        # Sort by cumulative time
        time_bottlenecks.sort(key=lambda x: x['time'], reverse=True)
        
        # Analyze memory usage patterns
        memory_patterns = {}
        for component, usages in self.memory_usage.items():
            memory_patterns[component] = {
                'avg_usage': np.mean(usages),
                'peak_usage': np.max(usages),
                'usage_trend': np.polyfit(range(len(usages)), usages, 1)[0]
            }
        
        return {
            'time_bottlenecks': time_bottlenecks[:10],  # Top 10 bottlenecks
            'memory_patterns': memory_patterns
        }
    
    def generate_optimization_report(
        self,
        bottleneck_analysis: Dict[str, Dict[str, float]]
    ) -> None:
        """Generate a report with optimization recommendations.
        
        Args:
            bottleneck_analysis: Results from analyze_bottlenecks
        """
        report_path = self.output_dir / 'optimization_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Performance Optimization Report\n")
            f.write("============================\n\n")
            
            # Time bottlenecks
            f.write("Time Bottlenecks:\n")
            f.write("-----------------\n")
            for bottleneck in bottleneck_analysis['time_bottlenecks']:
                f.write(f"\nFunction: {bottleneck['function']}\n")
                f.write(f"  - Total time: {bottleneck['time']:.3f}s\n")
                f.write(f"  - Calls: {bottleneck['calls']}\n")
                f.write(f"  - Time per call: {bottleneck['time_per_call']:.3f}s\n")
            f.write("\n")
            
            # Memory patterns
            f.write("Memory Usage Patterns:\n")
            f.write("---------------------\n")
            for component, metrics in bottleneck_analysis['memory_patterns'].items():
                f.write(f"\nComponent: {component}\n")
                f.write(f"  - Average usage: {metrics['avg_usage']:.2f}MB\n")
                f.write(f"  - Peak usage: {metrics['peak_usage']:.2f}MB\n")
                f.write(f"  - Usage trend: {'Increasing' if metrics['usage_trend'] > 0 else 'Decreasing'}\n")
            f.write("\n")
            
            # Optimization recommendations
            f.write("Optimization Recommendations:\n")
            f.write("--------------------------\n")
            
            # Time optimizations
            f.write("\n1. Time Optimizations:\n")
            for bottleneck in bottleneck_analysis['time_bottlenecks'][:3]:
                f.write(f"\n   {bottleneck['function']}:\n")
                if bottleneck['calls'] > 1000:
                    f.write("   - Consider caching results\n")
                if bottleneck['time_per_call'] > 0.1:
                    f.write("   - Look for algorithmic improvements\n")
                if bottleneck['time'] > 1.0:
                    f.write("   - Consider parallelization\n")
            
            # Memory optimizations
            f.write("\n2. Memory Optimizations:\n")
            for component, metrics in bottleneck_analysis['memory_patterns'].items():
                if metrics['peak_usage'] > 1000:  # More than 1GB
                    f.write(f"\n   {component}:\n")
                    f.write("   - Consider batch processing\n")
                    f.write("   - Look for memory leaks\n")
                if metrics['usage_trend'] > 0:
                    f.write("   - Investigate growing memory usage\n")
            
            # General recommendations
            f.write("\n3. General Recommendations:\n")
            f.write("   - Use numpy operations instead of loops where possible\n")
            f.write("   - Consider using numba for compute-intensive functions\n")
            f.write("   - Profile with larger datasets to verify scaling\n")

def profile_function(component_name: str):
    """Decorator for profiling individual functions.
    
    Args:
        component_name: Name of the component being profiled
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            result, metrics = profiler.profile_component(
                component_name,
                func,
                *args,
                **kwargs
            )
            return result
        return wrapper
    return decorator 
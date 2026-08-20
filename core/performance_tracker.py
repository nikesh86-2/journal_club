"""
performance_tracker.py

Performance tracking utilities for Journal Club pipeline.

Provides context managers for tracking execution time and memory usage.
"""

import time
import tracemalloc
import logging
from typing import Optional

log = logging.getLogger("journal_club.performance")


class PerformanceTracker:
    """Track execution time and memory usage for operations."""
    
    def __init__(self, operation_name: str, enable_tracking: bool = True):
        """
        Initialize performance tracker.
        
        Args:
            operation_name: Name of the operation being tracked
            enable_tracking: Whether to enable tracking (can be disabled for production)
        """
        self.operation_name = operation_name
        self.enable_tracking = enable_tracking
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
    
    def __enter__(self):
        """Start tracking when entering context."""
        if not self.enable_tracking:
            return self
        
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = tracemalloc.get_traced_memory()[0]
        return self
    
    def __exit__(self, *args):
        """Stop tracking and log results when exiting context."""
        if not self.enable_tracking:
            return
        
        elapsed = time.time() - self.start_time
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        memory_delta = (current_memory - self.start_memory) / (1024**2)  # MB
        peak_memory_mb = peak_memory / (1024**2)
        tracemalloc.stop()
        
        log.info(
            f"{self.operation_name}: {elapsed:.2f}s, "
            f"+{memory_delta:.2f}MB memory, peak: {peak_memory_mb:.2f}MB"
        )
    
    def get_elapsed(self) -> Optional[float]:
        """Get elapsed time if tracking is active."""
        if self.start_time and self.enable_tracking:
            return time.time() - self.start_time
        return None


def get_optimal_batch_size(base_size: int = 20) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Args:
        base_size: Base batch size to adjust from
    
    Returns:
        Optimal batch size for current system
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb < 8:
            return max(5, base_size // 4)
        elif available_gb < 16:
            return max(10, base_size // 2)
        elif available_gb < 32:
            return base_size
        else:
            return min(base_size * 2, 50)
    except ImportError:
        log.warning("psutil not available, using default batch size")
        return base_size
    except Exception as e:
        log.warning("Failed to calculate optimal batch size: %s", e)
        return base_size

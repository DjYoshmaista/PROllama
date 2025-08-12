# core/memory.py
import gc
import torch
import psutil
import logging
import threading
import time
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class MemoryManager:
    """Centralized memory management and monitoring"""
    
    def __init__(self, cleanup_threshold: float = 85.0):
        self.cleanup_threshold = cleanup_threshold
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and clear GPU cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
        logger.debug("Memory cleanup performed")
    
    @staticmethod
    def get_memory_info() -> dict:
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        info = {
            'percent': memory.percent,
            'used_mb': memory.used / (1024 ** 2),
            'available_mb': memory.available / (1024 ** 2),
            'total_mb': memory.total / (1024 ** 2)
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            info['gpu_allocated_mb'] = gpu_memory.get('allocated_bytes.all.current', 0) / (1024 ** 2)
            info['gpu_reserved_mb'] = gpu_memory.get('reserved_bytes.all.current', 0) / (1024 ** 2)
        
        return info
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is above threshold"""
        return psutil.virtual_memory().percent > self.cleanup_threshold
    
    def force_cleanup_if_needed(self):
        """Force cleanup if memory pressure is high"""
        if self.check_memory_pressure():
            mem_before = psutil.virtual_memory()
            self.cleanup_memory()
            mem_after = psutil.virtual_memory()
            freed_mb = (mem_before.used - mem_after.used) / (1024 ** 2)
            logger.info(f"High memory pressure detected. Freed {freed_mb:.1f}MB")
    
    def start_monitoring(self, interval: int = 10, callback: Optional[Callable] = None):
        """Start memory monitoring in background thread"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._stop_event.clear()
        
        def monitor():
            while not self._stop_event.wait(interval):
                try:
                    memory_info = self.get_memory_info()
                    cpu_percent = psutil.cpu_percent()
                    load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
                    
                    log_msg = (
                        f"RESOURCE MONITOR | CPU: {cpu_percent}% | "
                        f"Memory: {memory_info['percent']:.1f}% | "
                        f"Used: {memory_info['used_mb']:.1f}MB | "
                        f"Load: {load_avg[0]:.1f}, {load_avg[1]:.1f}, {load_avg[2]:.1f}"
                    )
                    
                    if 'gpu_allocated_mb' in memory_info:
                        log_msg += f" | GPU: {memory_info['gpu_allocated_mb']:.1f}MB"
                    
                    logger.info(log_msg)
                    
                    # Auto cleanup if needed
                    if memory_info['percent'] > self.cleanup_threshold:
                        self.force_cleanup_if_needed()
                    
                    # Call custom callback if provided
                    if callback:
                        callback(memory_info)
                        
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        logger.info("Memory monitoring stopped")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup and stop monitoring"""
        self.stop_monitoring()
        self.cleanup_memory()

# Global memory manager instance
memory_manager = MemoryManager()
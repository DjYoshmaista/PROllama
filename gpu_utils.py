
# gpu_utils.py
import gc
import logging
import psutil
import torch

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Context manager for automatic GPU memory cleanup"""
    
    def __init__(self, cleanup_threshold_mb=100):
        self.cleanup_threshold = cleanup_threshold_mb * 1024 * 1024  # Convert to bytes
        self.initial_mem = 0
        
    def __enter__(self):
        if torch.cuda.is_available():
            self.initial_mem = torch.cuda.memory_allocated()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated()
            mem_diff = current_mem - self.initial_mem
            
            if mem_diff > self.cleanup_threshold:
                torch.cuda.empty_cache()
                logger.debug(f"GPU memory cleaned up: {mem_diff / 1024 / 1024:.1f}MB freed")

def cleanup_memory():
    """Comprehensive memory cleanup function"""
    try:
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        # PyTorch GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared PyTorch CUDA cache")
        
        # Log memory status after cleanup
        memory = psutil.virtual_memory()
        logger.info(f"Memory after cleanup: {memory.percent}% used ({memory.used / (1024**3):.1f}GB)")
        
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")

def get_memory_info():
    """Get current memory information"""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'free_gb': memory.free / (1024**3),
            'percent_used': memory.percent
        }
    except Exception as e:
        logger.error(f"Failed to get memory info: {e}")
        return None

def get_gpu_memory_info():
    """Get GPU memory information if available"""
    if not torch.cuda.is_available():
        return None
    
    try:
        gpu_memory = {}
        for i in range(torch.cuda.device_count()):
            gpu_memory[f'gpu_{i}'] = {
                'allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'cached_gb': torch.cuda.memory_reserved(i) / (1024**3),
                'total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
            }
        return gpu_memory
    except Exception as e:
        logger.error(f"Failed to get GPU memory info: {e}")
        return None
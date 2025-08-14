# gpu_utils.py
import torch
import gc
import logging

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

def cleanup_gpu_objects(objects):
    """Cleanup specific GPU objects and memory"""
    for obj in objects:
        if hasattr(obj, 'cpu'):
            obj.cpu()
        del obj
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def cleanup_memory():
    """Force garbage collection and clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("Memory cleanup performed")
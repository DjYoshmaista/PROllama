# performance_optimizer.py - Performance optimization utilities
import asyncio
import gc
import logging
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union
import numpy as np
import torch
from config import Config

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages system resources and performance optimization"""
    
    def __init__(self):
        self.memory_threshold = Config.MEMORY_CLEANUP_THRESHOLD
        self.last_cleanup = time.time()
        self.cleanup_interval = 30  # seconds
        
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def should_cleanup_memory(self) -> bool:
        """Determine if memory cleanup is needed"""
        memory_info = self.check_memory_usage()
        time_elapsed = time.time() - self.last_cleanup
        
        return (memory_info['percent'] > self.memory_threshold or 
                time_elapsed > self.cleanup_interval)
    
    def cleanup_memory(self, force: bool = False) -> bool:
        """Perform memory cleanup"""
        if not force and not self.should_cleanup_memory():
            return False
        
        logger.debug("Performing memory cleanup")
        
        # Python garbage collection
        collected = gc.collect()
        
        # GPU memory cleanup if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.last_cleanup = time.time()
        logger.debug(f"Memory cleanup completed, collected {collected} objects")
        return True
    
    def optimize_for_workload(self, workload_type: str):
        """Optimize system settings for specific workload"""
        if workload_type == "heavy_processing":
            # Optimize for CPU-intensive tasks
            gc.set_threshold(700, 10, 10)  # Less frequent GC
        elif workload_type == "memory_intensive":
            # Optimize for memory-heavy tasks
            gc.set_threshold(100, 5, 5)  # More frequent GC
        else:
            # Default settings
            gc.set_threshold(700, 10, 10)

class BatchProcessor:
    """Optimized batch processing utilities"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.resource_manager = ResourceManager()
    
    async def process_in_parallel_batches(
        self,
        items: List[Any],
        processor_func: Callable,
        batch_size: int,
        max_concurrent_batches: int = 3
    ) -> List[Any]:
        """Process items in parallel batches with resource management"""
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        async def process_batch(batch: List[Any]) -> List[Any]:
            async with semaphore:
                # Check memory before processing batch
                if self.resource_manager.should_cleanup_memory():
                    self.resource_manager.cleanup_memory()
                
                # Process batch
                batch_results = []
                for item in batch:
                    try:
                        result = await processor_func(item)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing item in batch: {e}")
                        batch_results.append(None)
                
                return batch_results
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches concurrently
        tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch processing failed: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    def process_cpu_intensive_parallel(
        self,
        items: List[Any],
        processor_func: Callable,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Process CPU-intensive tasks in parallel using thread pool"""
        
        chunk_size = chunk_size or max(1, len(items) // self.max_workers)
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_chunk = {}
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                future = executor.submit(self._process_chunk, chunk, processor_func)
                future_to_chunk[future] = chunk
            
            # Collect results
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
            
            # Periodic memory cleanup
            if self.resource_manager.should_cleanup_memory():
                self.resource_manager.cleanup_memory()
        
        return results
    
    def _process_chunk(self, chunk: List[Any], processor_func: Callable) -> List[Any]:
        """Process a chunk of items"""
        results = []
        for item in chunk:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk item: {e}")
                results.append(None)
        return results

class VectorOptimizer:
    """Optimizations for vector operations"""
    
    @staticmethod
    def optimize_batch_cosine_similarity(
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        batch_size: int = 10000,
        use_gpu: bool = None
    ) -> np.ndarray:
        """Optimized batch cosine similarity calculation"""
        
        # Auto-detect GPU usage based on data size
        if use_gpu is None:
            use_gpu = torch.cuda.is_available() and len(embeddings) > 1000
        
        if use_gpu:
            return VectorOptimizer._gpu_batch_similarity(query_embedding, embeddings, batch_size)
        else:
            return VectorOptimizer._cpu_batch_similarity(query_embedding, embeddings)
    
    @staticmethod
    def _gpu_batch_similarity(
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        """GPU-optimized similarity calculation"""
        device = torch.cuda.current_device()
        
        # Convert to tensors
        query_tensor = torch.from_numpy(query_embedding).float().to(device)
        
        all_similarities = []
        
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_tensor = torch.from_numpy(batch_embeddings).float().to(device)
            
            # Normalize vectors
            query_norm = torch.nn.functional.normalize(query_tensor.unsqueeze(0), p=2, dim=1)
            batch_norm = torch.nn.functional.normalize(batch_tensor, p=2, dim=1)
            
            # Compute similarities
            similarities = torch.mm(query_norm, batch_norm.t()).squeeze()
            
            # Move to CPU and store
            all_similarities.append(similarities.cpu().numpy())
            
            # Clear GPU memory
            del batch_tensor, similarities
            
        torch.cuda.empty_cache()
        return np.concatenate(all_similarities)
    
    @staticmethod
    def _cpu_batch_similarity(
        query_embedding: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """CPU-optimized similarity calculation using NumPy"""
        
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute dot product (cosine similarity for normalized vectors)
        return np.dot(embeddings_norm, query_norm)
    
    @staticmethod
    def optimize_embeddings_storage(embeddings: List[List[float]]) -> np.ndarray:
        """Optimize embedding storage format"""
        # Convert to optimized numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Ensure contiguous memory layout for better cache performance
        if not embeddings_array.flags['C_CONTIGUOUS']:
            embeddings_array = np.ascontiguousarray(embeddings_array)
        
        return embeddings_array

class DatabaseOptimizer:
    """Database performance optimization utilities"""
    
    @staticmethod
    def estimate_optimal_batch_size(record_size_bytes: int, available_memory_gb: float = None) -> int:
        """Estimate optimal batch size based on memory constraints"""
        if available_memory_gb is None:
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
        
        # Use 10% of available memory for batching
        memory_budget_bytes = available_memory_gb * 0.1 * (1024**3)
        
        # Calculate batch size
        batch_size = max(10, int(memory_budget_bytes // record_size_bytes))
        
        # Cap at reasonable limits
        return min(batch_size, 10000)
    
    @staticmethod
    def optimize_connection_pool_size() -> Dict[str, int]:
        """Calculate optimal connection pool sizes"""
        cpu_count = psutil.cpu_count() or 1
        
        return {
            'min_size': max(2, cpu_count // 2),
            'max_size': min(20, cpu_count * 2),
            'overflow': 5
        }
    
    @staticmethod
    def get_vacuum_recommendations(table_stats: Dict[str, Any]) -> List[str]:
        """Get vacuum/analyze recommendations based on table stats"""
        recommendations = []
        
        if table_stats.get('dead_tuples', 0) > 1000:
            recommendations.append("VACUUM ANALYZE recommended - high dead tuple count")
        
        if table_stats.get('last_vacuum') and time.time() - table_stats['last_vacuum'] > 86400:
            recommendations.append("VACUUM recommended - last vacuum > 24h ago")
        
        if table_stats.get('last_analyze') and time.time() - table_stats['last_analyze'] > 86400:
            recommendations.append("ANALYZE recommended - statistics outdated")
        
        return recommendations

class ConcurrencyOptimizer:
    """Concurrency and parallelism optimization"""
    
    @staticmethod
    def get_optimal_worker_count(workload_type: str) -> int:
        """Get optimal worker count for different workload types"""
        cpu_count = psutil.cpu_count() or 1
        
        if workload_type == "cpu_bound":
            return cpu_count
        elif workload_type == "io_bound":
            return min(50, cpu_count * 5)  # More workers for I/O
        elif workload_type == "mixed":
            return cpu_count * 2
        else:
            return cpu_count
    
    @staticmethod
    def calculate_chunk_size(total_items: int, worker_count: int, min_chunk_size: int = 10) -> int:
        """Calculate optimal chunk size for parallel processing"""
        base_chunk_size = max(min_chunk_size, total_items // worker_count)
        
        # Ensure chunk size is reasonable
        return min(base_chunk_size, 1000)  # Cap at 1000 items per chunk

# Performance monitoring
class PerformanceProfiler:
    """Profile and monitor performance"""
    
    def __init__(self):
        self.operation_times = {}
        self.resource_usage = []
        self.lock = threading.Lock()
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations"""
        return PerformanceContext(self, operation_name)
    
    def record_operation(self, operation: str, duration: float, **metadata):
        """Record operation performance"""
        with self.lock:
            if operation not in self.operation_times:
                self.operation_times[operation] = []
            
            record = {
                'duration': duration,
                'timestamp': time.time(),
                **metadata
            }
            
            self.operation_times[operation].append(record)
            
            # Keep only last 100 records per operation
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation].pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        
        with self.lock:
            for operation, records in self.operation_times.items():
                if not records:
                    continue
                
                durations = [r['duration'] for r in records]
                summary[operation] = {
                    'count': len(durations),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'total_duration': sum(durations)
                }
        
        return summary

class PerformanceContext:
    """Context manager for performance profiling"""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.profiler.record_operation(
                self.operation_name,
                duration,
                had_exception=exc_type is not None
            )

# Global instances
resource_manager = ResourceManager()
batch_processor = BatchProcessor()
vector_optimizer = VectorOptimizer()
db_optimizer = DatabaseOptimizer()
concurrency_optimizer = ConcurrencyOptimizer()
performance_profiler = PerformanceProfiler()
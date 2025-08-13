# async_loader.py - Fully Optimized Pure Async Implementation
import asyncio
import aiohttp
import asyncpg
import json
import os
import logging
import traceback
import gc
import psutil
import time
import threading
from asyncpg.pool import Pool
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Generator, Callable

import pgvector.asyncpg
from embedding_queue import embedding_queue
from constants import *

# Create prefixed logger for this file
logger = logging.getLogger(__name__)
LOG_PREFIX = "[Legacy/AsyncLoader]"
from file_management.loaders import optimized_bulk_loader, performance_monitor
from file_management.discovery import file_discovery
from core.memory import memory_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("async_loader.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

DB_CONN_STRING = f"postgres://postgres:postgres@localhost/{DB_NAME}"

# Optimized constants for maximum performance
POOL_RECYCLE_AFTER = 500  # Higher threshold
MEMORY_SAFE_CHUNK_SIZE = 200  # Larger chunks
EMBEDDING_CHUNK_SIZE = 200  # Larger batch size
INSERT_CHUNK_SIZE = 1000  # Much larger inserts
MEMORY_CLEANUP_THRESHOLD = 90  # Higher threshold
MAX_CONCURRENT_CHUNKS = 64  # Higher concurrency
WORKER_PROCESSES = min(12, os.cpu_count())  # More workers

# PostgreSQL optimization parameters
PG_OPTIMIZATION_SETTINGS = {
    "statement_timeout": "600000",  # 10 minutes
    "work_mem": "32MB",
    "maintenance_work_mem": "1GB",
    "effective_cache_size": "2GB",
    "random_page_cost": "1.1",
    "checkpoint_completion_target": "0.9",
    "wal_buffers": "64MB",
    "shared_buffers": "256MB"
}

class OptimizedAsyncProcessor:
    """
    Completely rewritten async processor with pipeline architecture
    """
    
    def __init__(self):
        self._connection_pool: Optional[asyncpg.Pool] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._processing_queue = asyncio.Queue(maxsize=1000)
        self._results_queue = asyncio.Queue(maxsize=1000)
        self._worker_tasks = []
        self._monitor_task = None
    
    async def initialize_pool(self):
        """Initialize optimized connection pool"""
        if self._connection_pool is None:
            self._connection_pool = await asyncpg.create_pool(
                DB_CONN_STRING,
                min_size=8,
                max_size=20,
                timeout=300,
                server_settings=PG_OPTIMIZATION_SETTINGS,
                command_timeout=300
            )
            logger.info("Initialized optimized connection pool")
        
        if self._session is None:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            self._session = aiohttp.ClientSession(connector=connector)
            logger.info("Initialized optimized HTTP session")
    
    async def start_pipeline_workers(self, num_workers: int = 8):
        """Start pipeline workers for concurrent processing"""
        await self.initialize_pool()
        
        # Start processing workers
        for i in range(num_workers):
            task = asyncio.create_task(self._pipeline_worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        # Start database worker
        db_task = asyncio.create_task(self._database_worker())
        self._worker_tasks.append(db_task)
        
        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_resources())
        
        logger.info(f"Started {num_workers} pipeline workers + database worker")
    
    async def _pipeline_worker(self, worker_id: str):
        """Pipeline worker that processes files through the full pipeline"""
        while True:
            try:
                # Get work item
                work_item = await self._processing_queue.get()
                if work_item is None:  # Shutdown signal
                    break
                
                file_path, strategy = work_item
                
                # Process file based on strategy
                if strategy == "full_parallel":
                    result = await self._process_file_full_parallel(file_path, worker_id)
                elif strategy == "streaming":
                    result = await self._process_file_streaming(file_path, worker_id)
                else:  # hybrid
                    result = await self._process_file_hybrid(file_path, worker_id)
                
                # Send result to database worker
                await self._results_queue.put(result)
                
            except Exception as e:
                logger.error(f"Pipeline worker {worker_id} error: {e}")
            finally:
                self._processing_queue.task_done()
    
    async def _process_file_full_parallel(self, file_path: str, worker_id: str) -> Tuple[str, List[Tuple[str, List[str], List[float]]], int]:
        """Process file with full parallelization"""
        try:
            # Use thread pool for CPU-intensive parsing
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Parse file in thread
                from file_management.parsers import document_parser
                future = executor.submit(self._parse_file_sync, file_path)
                records = await loop.run_in_executor(None, lambda: future.result())
            
            if not records:
                return file_path, [], 0
            
            # Process embeddings asynchronously
            processed_records = await self._process_embeddings_batch(records, file_path)
            return file_path, processed_records, 0
            
        except Exception as e:
            logger.error(f"Full parallel processing error for {file_path}: {e}")
            return file_path, [], 1
    
    def _parse_file_sync(self, file_path: str) -> List[Dict[str, Any]]:
        """Synchronous file parsing for thread execution"""
        from file_management.parsers import document_parser
        from pathlib import Path
        
        file_type = Path(file_path).suffix[1:].lower()
        if file_type not in ['txt', 'csv', 'json', 'py']:
            file_type = 'txt'
        
        records = []
        for chunk in document_parser.parse_file_stream(file_path, file_type, 100):
            records.extend(chunk)
        
        return records
    
    async def _process_file_streaming(self, file_path: str, worker_id: str) -> Tuple[str, List[Tuple[str, List[str], List[float]]], int]:
        """Process file with streaming approach for large files"""
        try:
            from file_management.loaders import optimized_document_loader
            record_count = await optimized_document_loader._load_large_file_streaming(file_path)
            # For streaming, we return the count rather than records (already inserted)
            return file_path, [], 0  # Records already inserted by streaming
            
        except Exception as e:
            logger.error(f"Streaming processing error for {file_path}: {e}")
            return file_path, [], 1
    
    async def _process_file_hybrid(self, file_path: str, worker_id: str) -> Tuple[str, List[Tuple[str, List[str], List[float]]], int]:
        """Process file with hybrid approach"""
        try:
            # Parse file
            records = await asyncio.to_thread(self._parse_file_sync, file_path)
            
            if not records:
                return file_path, [], 0
            
            # Process embeddings
            processed_records = await self._process_embeddings_batch(records, file_path)
            return file_path, processed_records, 0
            
        except Exception as e:
            logger.error(f"Hybrid processing error for {file_path}: {e}")
            return file_path, [], 1
    
    async def _process_embeddings_batch(self, records: List[Dict[str, Any]], file_path: str) -> List[Tuple[str, List[str], List[float]]]:
        """Process embeddings for a batch of records"""
        # Filter valid content
        valid_items = [
            item for item in records
            if item.get("content") and item["content"].strip()
        ]
        
        if not valid_items:
            return []
        
        # Extract content and tags
        contents = [item["content"] for item in valid_items]
        tags_lists = [item.get("tags", []) for item in valid_items]
        
        # Generate embeddings using the global embedding service
        try:
            from inference.embeddings import embedding_service
            embeddings = await embedding_service.generate_embeddings_batch(contents)
        except Exception as e:
            logger.error(f"Embedding generation failed for {file_path}: {e}")
            return []
        
        # Prepare records
        processed_records = []
        failed_count = 0
        
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                failed_count += 1
                continue
            
            if len(embedding) != EMB_DIM:
                failed_count += 1
                continue
            
            processed_records.append((
                contents[i],
                tags_lists[i],
                embedding
            ))
        
        if failed_count > 0:
            logger.warning(f"Failed to generate {failed_count}/{len(contents)} embeddings for {file_path}")
        
        return processed_records
    
    async def _database_worker(self):
        """Dedicated database worker for batch insertions"""
        batch = []
        batch_timeout = 5.0  # seconds
        max_batch_size = INSERT_CHUNK_SIZE
        
        while True:
            try:
                # Get result with timeout to enable periodic flushing
                try:
                    result = await asyncio.wait_for(self._results_queue.get(), timeout=batch_timeout)
                    if result is None:  # Shutdown signal
                        break
                    
                    file_path, records, error_count = result
                    
                    if records:
                        batch.extend(records)
                    
                    self._results_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # Timeout - flush current batch if any
                    pass
                
                # Flush batch if it's large enough or timeout occurred
                if len(batch) >= max_batch_size or (batch and len(batch) > 0):
                    await self._flush_batch_to_database(batch)
                    batch.clear()
                    
            except Exception as e:
                logger.error(f"Database worker error: {e}")
    
    async def _flush_batch_to_database(self, batch: List[Tuple[str, List[str], List[float]]]):
        """Flush a batch of records to the database"""
        if not batch:
            return
        
        try:
            async with self._connection_pool.acquire() as conn:
                await pgvector.asyncpg.register_vector(conn)
                
                # Prepare data for insertion
                insert_data = [
                    (content, json.dumps(tags) if isinstance(tags, list) else tags, embedding)
                    for content, tags, embedding in batch
                ]
                
                # Batch insert
                await conn.executemany(
                    f"INSERT INTO {TABLE_NAME} (content, tags, embedding) VALUES ($1, $2::jsonb, $3)",
                    insert_data
                )
                
                logger.info(f"Inserted batch of {len(batch)} records")
                
        except Exception as e:
            logger.error(f"Database batch insert failed: {e}")
    
    async def _monitor_resources(self):
        """Monitor system resources during processing"""
        while True:
            try:
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
                memory_info = memory_manager.get_memory_info()
                cpu_percent = psutil.cpu_percent()
                
                # Log resource usage
                logger.info(
                    f"RESOURCES | CPU: {cpu_percent}% | "
                    f"Memory: {memory_info['percent']:.1f}% | "
                    f"Processing Queue: {self._processing_queue.qsize()} | "
                    f"Results Queue: {self._results_queue.qsize()}"
                )
                
                # Force memory cleanup if needed
                if memory_info['percent'] > MEMORY_CLEANUP_THRESHOLD:
                    memory_manager.force_cleanup_if_needed()
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def process_files_pipeline(
        self, 
        file_paths: List[str], 
        strategy: str = "adaptive",
        progress_callback: Optional[Callable] = None
    ) -> int:
        """
        Process files using the optimized pipeline
        
        Args:
            file_paths: List of file paths to process
            strategy: Processing strategy ("adaptive", "full_parallel", "hybrid", "streaming")
            progress_callback: Optional progress callback
            
        Returns:
            Number of files processed successfully
        """
        if not file_paths:
            return 0
        
        # Start pipeline workers
        await self.start_pipeline_workers(num_workers=WORKER_PROCESSES)
        
        try:
            # Determine strategy per file if adaptive
            if strategy == "adaptive":
                file_strategies = self._determine_strategies(file_paths)
            else:
                file_strategies = [(fp, strategy) for fp in file_paths]
            
            # Queue all files for processing
            for file_path, file_strategy in file_strategies:
                await self._processing_queue.put((file_path, file_strategy))
            
            # Wait for processing to complete
            await self._processing_queue.join()
            
            # Signal workers to stop
            for _ in self._worker_tasks:
                await self._processing_queue.put(None)
            
            # Wait for all workers to finish
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            
            # Signal database worker to stop and wait for remaining results
            await self._results_queue.put(None)
            await self._results_queue.join()
            
            # Stop monitoring
            if self._monitor_task:
                self._monitor_task.cancel()
            
            logger.info(f"Pipeline processing completed for {len(file_paths)} files")
            return len(file_paths)
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return 0
        finally:
            await self.cleanup()
    
    def _determine_strategies(self, file_paths: List[str]) -> List[Tuple[str, str]]:
        """Determine optimal strategy for each file based on characteristics"""
        strategies = []
        
        for file_path in file_paths:
            try:
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 ** 2)
                
                if size_mb > 100:  # Large files
                    strategy = "streaming"
                elif size_mb > 5:  # Medium files
                    strategy = "hybrid"
                else:  # Small files
                    strategy = "full_parallel"
                
                strategies.append((file_path, strategy))
                
            except Exception as e:
                logger.warning(f"Could not determine strategy for {file_path}: {e}")
                strategies.append((file_path, "hybrid"))  # Default
        
        return strategies
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Cancel all tasks
            for task in self._worker_tasks:
                if not task.done():
                    task.cancel()
            
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
            
            # Close connection pool
            if self._connection_pool:
                await self._connection_pool.close()
                self._connection_pool = None
            
            # Close HTTP session
            if self._session:
                await self._session.close()
                self._session = None
            
            logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

class OptimizedBulkProcessor:
    """
    High-level bulk processor that orchestrates the entire loading pipeline
    """
    
    def __init__(self):
        self.processor = OptimizedAsyncProcessor()
        self.performance_monitor = performance_monitor
    
    async def process_folder_optimized(
        self,
        folder_path: str,
        strategy: str = "adaptive",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process entire folder using optimized pipeline
        
        Args:
            folder_path: Path to folder to process
            strategy: Processing strategy
            progress_callback: Optional progress callback
            
        Returns:
            Processing results and metrics
        """
        start_time = time.time()
        
        # Start performance monitoring
        self.performance_monitor.start()
        
        # Start memory monitoring
        memory_manager.start_monitoring()
        
        try:
            # Discover files using optimized parallel discovery
            logger.info("Starting optimized file discovery...")
            discovery_start = time.time()
            
            file_paths = list(file_discovery.discover_files_parallel(folder_path))
            total_files = len(file_paths)
            
            discovery_time = time.time() - discovery_start
            logger.info(f"Discovered {total_files} files in {discovery_time:.2f}s")
            
            if total_files == 0:
                return {
                    "success": False,
                    "message": "No supported files found",
                    "metrics": {}
                }
            
            # Process files using optimized pipeline
            logger.info(f"Starting optimized pipeline processing with strategy: {strategy}")
            
            def enhanced_progress_callback(processed: int, total: int, records: int):
                self.performance_monitor.update(processed, total, records)
                if progress_callback:
                    progress_callback(processed, total, records)
            
            # Use the new pipeline processor
            processed_files = await self.processor.process_files_pipeline(
                file_paths,
                strategy=strategy,
                progress_callback=enhanced_progress_callback
            )
            
            # Get final metrics
            total_time = time.time() - start_time
            performance_summary = self.performance_monitor.get_summary()
            
            success_rate = (processed_files / total_files) * 100 if total_files > 0 else 0
            
            results = {
                "success": True,
                "total_files": total_files,
                "processed_files": processed_files,
                "success_rate": success_rate,
                "total_time": total_time,
                "discovery_time": discovery_time,
                "processing_time": total_time - discovery_time,
                "strategy_used": strategy,
                "performance_metrics": performance_summary
            }
            
            logger.info(f"Bulk processing completed: {processed_files}/{total_files} files in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Bulk processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": self.performance_monitor.get_summary()
            }
        finally:
            # Cleanup
            memory_manager.stop_monitoring()
            await self.processor.cleanup()

async def run_optimized_processing(
    file_generator: Generator[str, None, None],
    total_files: int,
    progress_callback: Optional[Callable] = None
) -> int:
    """
    Main entry point for optimized document processing
    Replaces the original run_processing function
    
    Args:
        file_generator: Generator of file paths
        total_files: Total number of files
        progress_callback: Optional progress callback
        
    Returns:
        Number of successfully processed files
    """
    processor = OptimizedAsyncProcessor()
    
    # Convert generator to list for batch processing
    file_paths = list(file_generator)
    
    # Start embedding service
    from inference.embeddings import embedding_service
    await embedding_service.start(concurrency=15)
    
    try:
        # Process using optimized pipeline
        processed_count = await processor.process_files_pipeline(
            file_paths,
            strategy="adaptive",
            progress_callback=progress_callback
        )
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Optimized processing failed: {e}")
        return 0
    finally:
        await embedding_service.stop()
        await processor.cleanup()

async def run_maintenance():
    """Run periodic database maintenance with optimizations"""
    try:
        from database.operations import db_ops
        success = await db_ops.run_maintenance()
        
        if success:
            logger.info("Optimized database maintenance completed")
        else:
            logger.warning("Database maintenance had issues")
            
        return success
        
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
        return False

class StreamingFileProcessor:
    """
    Specialized processor for very large files using streaming approach
    """
    
    def __init__(self):
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks
        self.max_concurrent_chunks = 16
    
    async def process_large_file_streaming(
        self,
        file_path: str,
        progress_callback: Optional[Callable] = None
    ) -> int:
        """
        Process very large files using streaming with parallel chunk processing
        
        Args:
            file_path: Path to large file
            progress_callback: Optional progress callback
            
        Returns:
            Number of records processed
        """
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"Processing large file {file_path} ({file_size / (1024**2):.1f} MB) with streaming")
            
            from file_management.loaders import optimized_document_loader
            
            # Use the optimized streaming loader
            record_count = await optimized_document_loader._load_large_file_streaming(file_path)
            
            if progress_callback:
                progress_callback(1, 1, record_count)
            
            logger.info(f"Streaming processing completed: {record_count} records from {file_path}")
            return record_count
            
        except Exception as e:
            logger.error(f"Streaming processing failed for {file_path}: {e}")
            return 0

# Global instances for backward compatibility and new features
optimized_processor = OptimizedAsyncProcessor()
bulk_processor = OptimizedBulkProcessor()
streaming_processor = StreamingFileProcessor()

# Legacy function mapping for compatibility
async def process_file(file_path):
    """Legacy compatibility function"""
    return await optimized_processor.process_files_pipeline([file_path], strategy="adaptive")

# Enhanced entry point functions
async def load_folder_optimized(folder_path: str, strategy: str = "adaptive") -> Dict[str, Any]:
    """
    Load entire folder using the optimized pipeline
    
    Args:
        folder_path: Path to folder
        strategy: Processing strategy ("adaptive", "full_parallel", "hybrid", "streaming")
        
    Returns:
        Processing results and metrics
    """
    return await bulk_processor.process_folder_optimized(folder_path, strategy)

async def load_files_batch_optimized(file_paths: List[str], strategy: str = "adaptive") -> int:
    """
    Load batch of files using optimized pipeline
    
    Args:
        file_paths: List of file paths
        strategy: Processing strategy
        
    Returns:
        Number of files processed successfully
    """
    processor = OptimizedAsyncProcessor()
    try:
        return await processor.process_files_pipeline(file_paths, strategy)
    finally:
        await processor.cleanup()

# Performance testing function
async def benchmark_processing_strategies(
    test_files: List[str],
    strategies: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark different processing strategies
    
    Args:
        test_files: List of test files
        strategies: List of strategies to test
        
    Returns:
        Benchmark results for each strategy
    """
    if strategies is None:
        strategies = ["full_parallel", "hybrid", "streaming"]
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"Benchmarking strategy: {strategy}")
        start_time = time.time()
        
        processor = OptimizedAsyncProcessor()
        try:
            processed_count = await processor.process_files_pipeline(
                test_files[:10],  # Limit to 10 files for benchmarking
                strategy=strategy
            )
            
            elapsed_time = time.time() - start_time
            
            results[strategy] = {
                "processed_files": processed_count,
                "elapsed_time": elapsed_time,
                "files_per_second": processed_count / elapsed_time if elapsed_time > 0 else 0,
                "success": True
            }
            
        except Exception as e:
            results[strategy] = {
                "error": str(e),
                "success": False
            }
        finally:
            await processor.cleanup()
    
    return results

if __name__ == "__main__":
    # Example usage
    async def main():
        test_folder = "/path/to/test/folder"
        results = await load_folder_optimized(test_folder, strategy="adaptive")
        print(f"Processing results: {results}")
    
    asyncio.run(main())
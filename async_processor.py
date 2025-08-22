# async_processor.py - Unified Asynchronous Processing Pipeline
import asyncio
import logging
import psutil
import time
import threading
import json
import os
from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass
from core_config import config
from database_manager import db_manager
from document_processor import file_processor, file_tracker
from embedding_manager import embedding_queue
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Comprehensive processing statistics"""
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_records: int = 0
    total_processing_time: float = 0.0
    start_time: float = 0.0
    current_file: str = ""
    stage: str = "initializing"
    files_skipped: int = 0
    queue_stats: Dict[str, Any] = None
    system_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.queue_stats is None:
            self.queue_stats = {}
        if self.system_stats is None:
            self.system_stats = {}

class ProcessingStateManager:
    """Manages processing state for resumable operations"""
    
    def __init__(self, state_file: str = "processing_state.json"):
        self.state_file = state_file
        self.current_state = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': [],
            'current_batch': 0,
            'last_processed_file': None,
            'processing_start_time': None,
            'last_checkpoint': None,
            'session_id': int(time.time())
        }
        self.load_state()
    
    def load_state(self):
        """Load processing state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    saved_state = json.load(f)
                    self.current_state.update(saved_state)
                logger.info(f"Loaded processing state: {self.current_state['processed_files']}/{self.current_state['total_files']} files")
        except Exception as e:
            logger.warning(f"Could not load processing state: {e}")
    
    def save_state(self):
        """Save current processing state to disk"""
        try:
            self.current_state['last_checkpoint'] = time.time()
            temp_file = f"{self.state_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.current_state, f, indent=2)
            os.replace(temp_file, self.state_file)
        except Exception as e:
            logger.error(f"Failed to save processing state: {e}")
    
    def start_processing(self, total_files: int):
        """Initialize processing state"""
        self.current_state.update({
            'total_files': total_files,
            'processed_files': 0,
            'failed_files': [],
            'current_batch': 0,
            'processing_start_time': time.time(),
            'session_id': int(time.time())
        })
        self.save_state()
    
    def update_progress(self, processed_files: int, current_file: str = None):
        """Update processing progress"""
        self.current_state['processed_files'] = processed_files
        if current_file:
            self.current_state['last_processed_file'] = current_file
        
        # Save state periodically
        if processed_files % 10 == 0:
            self.save_state()
    
    def add_failed_file(self, file_path: str, error: str):
        """Record a failed file"""
        self.current_state['failed_files'].append({
            'file': file_path,
            'error': error,
            'timestamp': time.time()
        })
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        elapsed = time.time() - (self.current_state['processing_start_time'] or time.time())
        remaining = self.current_state['total_files'] - self.current_state['processed_files']
        
        return {
            'progress_percent': (self.current_state['processed_files'] / max(1, self.current_state['total_files'])) * 100,
            'files_remaining': remaining,
            'failed_count': len(self.current_state['failed_files']),
            'elapsed_time': elapsed,
            'estimated_remaining': self._estimate_remaining_time()
        }
    
    def _estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining processing time"""
        if self.current_state['processed_files'] == 0:
            return None
        
        elapsed = time.time() - (self.current_state['processing_start_time'] or time.time())
        rate = self.current_state['processed_files'] / elapsed
        remaining_files = self.current_state['total_files'] - self.current_state['processed_files']
        
        return remaining_files / rate if rate > 0 else None

class ResourceMonitor:
    """Monitor system resources during processing"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.stats_history = []
        self.cleanup_threshold = config.PERFORMANCE_CONFIG['memory_cleanup_threshold']
    
    def start_monitoring(self, callback: Optional[Callable] = None):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(callback,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, callback: Optional[Callable]):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                stats = self._collect_stats()
                self.stats_history.append(stats)
                
                # Keep only last 60 entries (10 minutes at 10s intervals)
                if len(self.stats_history) > 60:
                    self.stats_history.pop(0)
                
                # Log periodically
                if len(self.stats_history) % 6 == 0:  # Every minute
                    self._log_stats(stats)
                
                # Trigger cleanup if needed
                if stats['memory_percent'] >= self.cleanup_threshold:
                    logger.warning(f"High memory usage: {stats['memory_percent']:.1f}%")
                    self._cleanup_memory()
                
                # Call callback if provided
                if callback:
                    try:
                        callback(stats)
                    except Exception as e:
                        logger.warning(f"Resource monitor callback failed: {e}")
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)
    
    def _collect_stats(self) -> Dict[str, Any]:
        """Collect current system statistics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        try:
            load_avg = os.getloadavg()
        except (AttributeError, OSError):
            load_avg = (0, 0, 0)  # Windows fallback
        
        return {
            'timestamp': time.time(),
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu_percent,
            'load_avg_1m': load_avg[0],
            'load_avg_5m': load_avg[1],
            'load_avg_15m': load_avg[2]
        }
    
    def _log_stats(self, stats: Dict[str, Any]):
        """Log system statistics"""
        logger.info(
            f"SYSTEM | CPU: {stats['cpu_percent']:.1f}% | "
            f"Memory: {stats['memory_percent']:.1f}% ({stats['memory_used_gb']:.1f}GB) | "
            f"Load: {stats['load_avg_1m']:.2f}"
        )
    
    def _cleanup_memory(self):
        """Trigger memory cleanup"""
        try:
            # Python garbage collection
            collected = gc.collect()
            logger.info(f"Memory cleanup: collected {collected} objects")
            
            # GPU cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("GPU memory cache cleared")
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return self._collect_stats()
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of recent statistics"""
        if not self.stats_history:
            return {}
        
        recent_stats = self.stats_history[-10:]  # Last 10 entries
        
        return {
            'avg_memory_percent': sum(s['memory_percent'] for s in recent_stats) / len(recent_stats),
            'max_memory_percent': max(s['memory_percent'] for s in recent_stats),
            'avg_cpu_percent': sum(s['cpu_percent'] for s in recent_stats) / len(recent_stats),
            'max_cpu_percent': max(s['cpu_percent'] for s in recent_stats),
            'current_load': recent_stats[-1]['load_avg_1m'] if recent_stats else 0,
            'samples': len(recent_stats)
        }

class AsyncProcessor:
    """Unified asynchronous processing pipeline"""
    
    def __init__(self):
        self.state_manager = ProcessingStateManager()
        self.resource_monitor = ResourceMonitor()
        self.processing_active = False
        self.current_stats = ProcessingStats()
    
    async def process_files_enhanced(self, 
                                   file_paths: List[str],
                                   progress_manager: Optional[Callable] = None,
                                   insert_callback: Optional[Callable] = None) -> int:
        """
        Enhanced file processing with queue system and comprehensive tracking
        
        Args:
            file_paths: List of file paths to process or generator
            progress_manager: Optional progress tracking callback
            insert_callback: Database insertion callback
            
        Returns:
            Number of successfully processed files
        """
        # Convert generator to list if needed
        if hasattr(file_paths, '__iter__') and not isinstance(file_paths, (list, tuple)):
            file_paths = list(file_paths)
        
        total_files = len(file_paths)
        logger.info(f"Starting enhanced processing of {total_files} files")
        
        # Initialize processing state
        self.state_manager.start_processing(total_files)
        self.current_stats = ProcessingStats(
            total_files=total_files,
            start_time=time.time()
        )
        
        # Filter files that need processing
        logger.info("Filtering files for processing...")
        files_to_process, filter_stats = file_tracker.batch_filter_files(file_paths)
        
        logger.info(f"File filtering complete:")
        logger.info(f"  Total files: {filter_stats.total_files}")
        logger.info(f"  Files to process: {filter_stats.files_to_process}")
        logger.info(f"  Files skipped: {filter_stats.files_skipped}")
        logger.info(f"  New files: {filter_stats.new_files}")
        logger.info(f"  Changed files: {filter_stats.size_changed + filter_stats.time_changed + filter_stats.content_changed}")
        
        if not files_to_process:
            logger.info("No files need processing")
            return 0
        
        self.current_stats.total_files = len(files_to_process)
        self.current_stats.files_skipped = filter_stats.files_skipped
        
        # Prepare database insertion callback
        if not insert_callback:
            insert_callback = self._default_database_callback
        
        # Start embedding queue workers
        logger.info("Starting embedding queue workers...")
        try:
            worker_count = config.PERFORMANCE_CONFIG['worker_processes']
            await embedding_queue.start_workers(
                concurrency=worker_count,
                insert_callback=insert_callback
            )
            
            if not embedding_queue.started:
                raise RuntimeError("Embedding queue failed to start")
            
            logger.info(f"✓ Embedding queue started with {worker_count} workers")
            
        except Exception as e:
            logger.error(f"Failed to start embedding queue: {e}")
            return 0
        
        # Start resource monitoring
        def resource_callback(stats):
            self.current_stats.system_stats = stats
            if progress_manager and hasattr(progress_manager, 'update_system_stats'):
                try:
                    progress_manager.update_system_stats(
                        stats['cpu_percent'],
                        stats['memory_percent'],
                        stats['load_avg_1m']
                    )
                except Exception as e:
                    logger.debug(f"Progress manager system stats update failed: {e}")
        
        self.resource_monitor.start_monitoring(resource_callback)
        self.processing_active = True
        
        try:
            # Initialize progress manager
            if progress_manager and hasattr(progress_manager, 'start_processing'):
                try:
                    progress_manager.start_processing(len(files_to_process))
                except Exception as e:
                    logger.warning(f"Progress manager start failed: {e}")
            
            # Process files
            success_count = await self._process_file_batch(
                files_to_process,
                progress_manager
            )
            
            # Wait for embedding queue to complete
            logger.info("Waiting for embedding queue to complete...")
            await self._wait_for_queue_completion()
            
            # Update final statistics
            self.current_stats.successful_files = success_count
            self.current_stats.failed_files = len(files_to_process) - success_count
            self.current_stats.total_processing_time = time.time() - self.current_stats.start_time
            
            logger.info(f"Processing complete!")
            logger.info(f"  Files processed: {success_count}/{len(files_to_process)}")
            logger.info(f"  Total records: {self.current_stats.total_records}")
            logger.info(f"  Processing time: {self.current_stats.total_processing_time:.2f}s")
            logger.info(f"  Final queue stats: {embedding_queue.stats}")
            
            return success_count
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self.state_manager.add_failed_file("SYSTEM_ERROR", str(e))
            return 0
            
        finally:
            self.processing_active = False
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            # Stop embedding queue
            logger.info("Stopping embedding queue workers...")
            await embedding_queue.stop_workers()
            
            # Save final state
            file_tracker.save_tracker()
            self.state_manager.save_state()
            
            # Finalize progress manager
            if progress_manager and hasattr(progress_manager, 'finish_processing'):
                try:
                    progress_manager.finish_processing(
                        self.current_stats.successful_files,
                        self.current_stats.failed_files,
                        self.current_stats.total_records
                    )
                except Exception as e:
                    logger.warning(f"Progress manager finish failed: {e}")
    
    async def _process_file_batch(self, 
                                 files_to_process: List[str],
                                 progress_manager: Optional[Callable] = None) -> int:
        """Process batch of files with progress tracking"""
        batch_size = 50  # Process files in smaller batches
        success_count = 0
        processed_count = 0
        
        # Progress callback for individual files
        def file_progress_callback(file_path, stage, current, total, message):
            try:
                if progress_manager and hasattr(progress_manager, 'update_file_progress'):
                    filename = os.path.basename(file_path)
                    status_message = f"[{stage}] {message}" if message else f"[{stage}]"
                    progress_manager.update_file_progress(
                        filename=filename,
                        stage=stage,
                        current=current,
                        total=total,
                        message=status_message
                    )
            except Exception as e:
                logger.debug(f"File progress callback failed: {e}")
        
        # Process files in batches
        with tqdm(total=len(files_to_process), desc="Processing files", unit="file") as pbar:
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
                
                # Process files sequentially to maintain order and resource control
                for file_path in batch:
                    try:
                        self.current_stats.current_file = file_path
                        self.current_stats.stage = "processing"
                        
                        # Process single file
                        result = await self._process_single_file(file_path, file_progress_callback)
                        file_path, success, records_count, processing_time = result
                        
                        if success:
                            success_count += 1
                            self.current_stats.total_records += records_count
                            logger.debug(f"File success: {file_path}, records: {records_count}")
                        else:
                            self.state_manager.add_failed_file(file_path, "Processing failed")
                            logger.warning(f"File failed: {file_path}")
                        
                        processed_count += 1
                        self.current_stats.processed_files = processed_count
                        self.state_manager.update_progress(processed_count, file_path)
                        
                        # Update progress display
                        pbar.update(1)
                        queue_stats = embedding_queue.stats
                        pbar.set_postfix(
                            success=success_count,
                            failed=processed_count - success_count,
                            queued=queue_stats['queue_size'],
                            records=self.current_stats.total_records,
                            mem=f"{psutil.virtual_memory().percent:.1f}%"
                        )
                        
                        # Update overall progress manager
                        if progress_manager and hasattr(progress_manager, 'update_overall_progress'):
                            try:
                                progress_manager.update_overall_progress(
                                    processed_count,
                                    len(files_to_process),
                                    success_count,
                                    processed_count - success_count,
                                    self.current_stats.total_records
                                )
                            except Exception as e:
                                logger.debug(f"Overall progress update failed: {e}")
                        
                        # Update queue stats in progress manager
                        if progress_manager and hasattr(progress_manager, 'update_queue_stats'):
                            try:
                                progress_manager.update_queue_stats(
                                    queue_stats['queue_size'],
                                    len([w for w in embedding_queue.workers if not w.done()]) if embedding_queue.workers else 0
                                )
                            except Exception as e:
                                logger.debug(f"Queue stats update failed: {e}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        self.state_manager.add_failed_file(file_path, str(e))
                        processed_count += 1
                        pbar.update(1)
                
                # Save state after each batch
                file_tracker.save_tracker()
                self.state_manager.save_state()
                
                # Memory cleanup for large batches
                if processed_count % (batch_size * 2) == 0:
                    memory = psutil.virtual_memory()
                    if memory.percent > 80:
                        logger.warning("High memory usage, running cleanup")
                        self.resource_monitor._cleanup_memory()
                
                # Brief pause between batches
                await asyncio.sleep(0.1)
        
        return success_count
    
    async def _process_single_file(self, file_path: str, progress_callback: Callable) -> tuple:
        """Process a single file using the document processor and queue system"""
        start_time = time.time()
        
        try:
            # Process file to generate parsed records
            result = await file_processor.process_file(file_path, progress_callback)
            file_path, success, records_count, processing_time = result
            
            if not success:
                return result
            
            # If records were generated, they were already queued by the processor
            # The embedding queue will handle the actual embedding generation and database insertion
            
            return (file_path, True, records_count, processing_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing file {file_path}: {e}")
            return (file_path, False, 0, processing_time)
    
    async def _wait_for_queue_completion(self):
        """Wait for embedding queue to complete all pending items"""
        logger.info("Waiting for embedding queue to complete...")
        wait_count = 0
        last_queue_size = -1
        stall_count = 0
        
        while True:
            queue_stats = embedding_queue.stats
            queue_size = queue_stats['queue_size']
            
            if queue_size == 0:
                logger.info("Embedding queue is empty")
                break
            
            # Check for stalled queue
            if queue_size == last_queue_size:
                stall_count += 1
                if stall_count > 30:  # 30 seconds of no progress
                    logger.warning(f"Queue appears stalled with {queue_size} items")
                    break
            else:
                stall_count = 0
            
            last_queue_size = queue_size
            wait_count += 1
            
            # Log progress every 10 seconds
            if wait_count % 10 == 0:
                logger.info(f"Queue items remaining: {queue_size}, "
                           f"processed: {queue_stats['processed_items']}, "
                           f"failed: {queue_stats['failed_items']}")
            
            # Safety timeout (10 minutes)
            if wait_count > 600:
                logger.warning("Queue completion timeout reached")
                break
            
            await asyncio.sleep(1)
    
    async def _default_database_callback(self, records: List[Dict[str, Any]]):
        """Default database insertion callback"""
        if not records:
            return
        
        try:
            inserted_count = await db_manager.insert_records_batch(records)
            logger.debug(f"Inserted {inserted_count} records into database")
            
        except Exception as e:
            logger.error(f"Database insertion failed: {e}", exc_info=True)
            raise
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get comprehensive processing status"""
        return {
            'processing_state': self.state_manager.get_progress_info(),
            'current_stats': {
                'total_files': self.current_stats.total_files,
                'processed_files': self.current_stats.processed_files,
                'successful_files': self.current_stats.successful_files,
                'failed_files': self.current_stats.failed_files,
                'total_records': self.current_stats.total_records,
                'current_file': self.current_stats.current_file,
                'stage': self.current_stats.stage,
                'processing_time': time.time() - self.current_stats.start_time if self.processing_active else self.current_stats.total_processing_time
            },
            'queue_stats': embedding_queue.stats,
            'system_stats': self.resource_monitor.get_current_stats(),
            'file_tracker_stats': file_tracker.get_processed_files_stats(),
            'resource_summary': self.resource_monitor.get_stats_summary()
        }
    
    async def run_maintenance(self):
        """Run system maintenance operations"""
        logger.info("Running system maintenance...")
        
        try:
            # Database maintenance
            maintenance_success = await db_manager.run_maintenance()
            if maintenance_success:
                logger.info("Database maintenance completed successfully")
            else:
                logger.warning("Database maintenance failed")
            
            # File tracker cleanup
            missing_files = file_tracker.cleanup_missing_files()
            if missing_files:
                logger.info(f"Cleaned up {len(missing_files)} missing file records")
            
            # Memory cleanup
            self.resource_monitor._cleanup_memory()
            
            # Save states
            file_tracker.save_tracker(force=True)
            self.state_manager.save_state()
            
            logger.info("System maintenance completed")
            return True
            
        except Exception as e:
            logger.error(f"System maintenance failed: {e}")
            return False

# Global processor instance
async_processor = AsyncProcessor()

# Legacy compatibility functions
async def run_processing_with_queue_and_tracking(file_generator, total_files, progress_manager=None):
    """Legacy compatibility function"""
    # Convert generator to list
    if hasattr(file_generator, '__iter__') and not isinstance(file_generator, (list, tuple)):
        file_list = list(file_generator)
    else:
        file_list = file_generator
    
    return await async_processor.process_files_enhanced(
        file_list,
        progress_manager
    )

async def run_processing(file_generator, total_files):
    """Legacy compatibility function"""
    return await run_processing_with_queue_and_tracking(file_generator, total_files)

def get_processing_status():
    """Legacy compatibility function"""
    return async_processor.get_processing_status()

async def run_maintenance():
    """Legacy compatibility function"""
    return await async_processor.run_maintenance()
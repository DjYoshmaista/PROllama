# async_loader.py - Fixed with comprehensive DEBUG logging and critical fixes
import multiprocessing
import asyncio
import aiohttp
import asyncpg
import json
import os
import logging
from aiomultiprocess import Pool as AsyncPool
from tqdm import tqdm
import gc
import psutil
import time
import threading
import load_documents
import pgvector.asyncpg
from embedding_queue import embedding_queue
from config import Config
from db import db_manager
from gpu_utils import cleanup_memory
from file_tracker import file_tracker
from security_utils import input_validator, SecurityError

# Set up logger for this module
logger = logging.getLogger(__name__)

DB_CONN_STRING = Config.get_db_connection_string()

# Connection pool per worker process
_worker_pool = None
_processing_stats = {
    'files_processed': 0,
    'records_created': 0,
    'embeddings_failed': 0,
    'last_update': time.time()
}

class ProcessingStateManager:
    """Manages processing state for resumable operations"""
    
    def __init__(self):
        self.state_file = "processing_state.json"
        self.current_state = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': [],
            'current_batch': 0,
            'last_processed_file': None,
            'processing_start_time': None,
            'last_checkpoint': None
        }
        self.load_state()
    
    def load_state(self):
        """Load processing state from disk"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    saved_state = json.load(f)
                    self.current_state.update(saved_state)
                logger.info(f"Loaded processing state: {self.current_state['processed_files']}/{self.current_state['total_files']} files processed")
        except Exception as e:
            logger.warning(f"Could not load processing state: {e}")
    
    def save_state(self):
        """Save current processing state to disk"""
        try:
            self.current_state['last_checkpoint'] = time.time()
            with open(self.state_file, 'w') as f:
                json.dump(self.current_state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processing state: {e}")
    
    def start_processing(self, total_files: int):
        """Initialize processing state"""
        self.current_state.update({
            'total_files': total_files,
            'processed_files': 0,
            'failed_files': [],
            'current_batch': 0,
            'processing_start_time': time.time()
        })
        self.save_state()
    
    def update_progress(self, processed_files: int, current_file: str = None):
        """Update processing progress"""
        self.current_state['processed_files'] = processed_files
        if current_file:
            self.current_state['last_processed_file'] = current_file
        
        # Save state every 10 files
        if processed_files % 10 == 0:
            self.save_state()
    
    def add_failed_file(self, file_path: str, error: str):
        """Record a failed file"""
        self.current_state['failed_files'].append({
            'file': file_path,
            'error': error,
            'timestamp': time.time()
        })
    
    def get_progress_info(self):
        """Get current progress information"""
        return {
            'progress_percent': (self.current_state['processed_files'] / max(1, self.current_state['total_files'])) * 100,
            'files_remaining': self.current_state['total_files'] - self.current_state['processed_files'],
            'failed_count': len(self.current_state['failed_files']),
            'elapsed_time': time.time() - (self.current_state['processing_start_time'] or time.time())
        }

# Global state manager
processing_state = ProcessingStateManager()

async def initialize_worker_pool():
    """Initialize connection pool for worker using centralized DB manager"""
    global _worker_pool
    logger.debug("initialize_worker_pool() called")
    if _worker_pool is None:
        logger.debug("Creating new worker pool")
        _worker_pool = await db_manager.get_async_pool()
        logger.debug("Worker pool created")
    return _worker_pool

async def database_insert_callback(records: list):
    """Callback function for inserting records into database"""
    logger.debug(f"database_insert_callback() called with {len(records) if records else 0} records")
    
    if not records:
        logger.debug("No records to insert")
        return
    
    try:
        logger.debug("Getting database pool")
        pool = await initialize_worker_pool()
        
        logger.debug("Acquiring database connection")
        async with pool.acquire() as conn:
            logger.debug("Registering vector extension")
            await pgvector.asyncpg.register_vector(conn)
            
            # Insert records in batches
            logger.debug(f"Inserting {len(records)} records into {Config.TABLE_NAME}")
            
            # Log first record for debugging
            if records:
                logger.debug(f"Sample record - content length: {len(records[0]['content'])}, "
                           f"tags: {records[0]['tags']}, embedding dim: {len(records[0]['embedding'])}")
            
            await conn.executemany(
                f"INSERT INTO {Config.TABLE_NAME} (content, tags, embedding) VALUES ($1, $2::text[], $3)",
                [(r['content'], r['tags'], r['embedding']) for r in records]
            )
            
            logger.info(f"Successfully inserted {len(records)} records into database")
            
    except Exception as e:
        logger.error(f"Database insert failed: {e}", exc_info=True)
        raise

async def process_file_with_queue(file_path, progress_callback=None):
    """
    Process a single file using the embedding queue system
    FIXED: Properly handle sync generators from parse_documents
    """
    logger.debug(f"process_file_with_queue() called for: {file_path}")
    
    file_type = os.path.splitext(file_path)[1][1:].lower()
    processed_records = 0

    def report_progress(stage, current=0, total=1, message=""):
        """Helper function to report progress if callback is provided"""
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(file_path, stage, current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback failed for {file_path}: {e}")

    try:
        # Check if file should be processed (using file tracker)
        logger.debug(f"Checking if {file_path} should be processed")
        should_process, reason = file_tracker.should_process_file(file_path)
        if not should_process:
            report_progress("skipped", 1, 1, f"Skipped: {reason}")
            logger.debug(f"Skipping {file_path}: {reason}")
            return (file_path, True, 0)  # Return success but 0 records

        report_progress("initializing", 0, 1, "Starting file processing")
        logger.debug(f"Processing file: {file_path}")

        # Parse file and enqueue chunks
        chunk_count = 0
        total_chunks_queued = 0
        
        import parse_documents
        
        logger.debug(f"Starting to parse {file_path} with type {file_type}")
        
        # FIXED: Use regular for loop since stream_parse_file returns a sync generator
        for chunk in parse_documents.stream_parse_file(file_path, file_type, Config.CHUNK_SIZES['file_processing']):
            if not chunk:
                logger.debug("Empty chunk, skipping")
                continue
                
            chunk_count += 1
            logger.debug(f"Processing chunk {chunk_count} with {len(chunk)} items")
            report_progress("parsing", chunk_count, chunk_count + 1, f"Processing chunk {chunk_count}")
            
            # Process each item in the chunk
            for item_idx, item in enumerate(chunk):
                content = item.get('content', '')
                tags = item.get('tags', [])
                
                logger.debug(f"Processing item {item_idx}: content length {len(content)}, tags: {tags}")
                
                if not content.strip():
                    logger.debug("Empty content, skipping item")
                    continue
                
                # Wait if queue is getting full
                while embedding_queue.wait_for_space():
                    logger.debug("Queue space full, waiting...")
                    report_progress("waiting_queue", 0, 1, "Waiting for queue space...")
                    await asyncio.sleep(0.1)
                
                # CRITICAL FIX: Check if embedding queue is started
                if not embedding_queue.started:
                    logger.error("Embedding queue is not started! This is the main issue!")
                    return (file_path, False, 0)
                
                # Enqueue item for embedding processing
                logger.debug(f"Enqueueing item {item_idx} from chunk {chunk_count}")
                queued = await embedding_queue.enqueue_for_embedding(
                    content=content,
                    tags=tags,
                    file_path=file_path,
                    chunk_index=chunk_count * 1000 + item_idx  # Unique chunk index
                )
                
                if queued:
                    total_chunks_queued += 1
                    logger.debug(f"Successfully queued item {item_idx}")
                else:
                    logger.warning(f"Failed to queue item from {file_path}, chunk {chunk_count}, item {item_idx}")
        
        # Mark file as processed in tracker
        logger.debug(f"Marking file as processed: {file_path}")
        file_tracker.mark_file_processed(file_path, total_chunks_queued)
        
        report_progress("queued", 1, 1, f"Queued {total_chunks_queued} items for processing")
        logger.info(f"Successfully queued {total_chunks_queued} items from {file_path}")
        
        return (file_path, True, total_chunks_queued)
        
    except Exception as e:
        report_progress("error", 0, 1, f"Error: {str(e)[:100]}")
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        processing_state.add_failed_file(file_path, str(e))
        return (file_path, False, 0)

async def log_batch_metrics(conn):
    """Log metrics for monitoring embedding quality"""
    try:
        metrics = await conn.fetch(f"""
            SELECT 
                AVG(vector_norm(embedding)) AS avg_norm,
                COUNT(*) FILTER (WHERE vector_norm(embedding) = 0) AS zero_vectors,
                COUNT(*) AS total_vectors
            FROM {Config.TABLE_NAME}
            WHERE embedding IS NOT NULL
        """)
        
        if metrics:
            metric_data = dict(metrics[0])
            logger.info(f"Embedding metrics: avg_norm={metric_data.get('avg_norm', 0):.4f}, "
                       f"zero_vectors={metric_data.get('zero_vectors', 0)}, "
                       f"total={metric_data.get('total_vectors', 0)}")
    except Exception as e:
        logger.warning(f"Could not log batch metrics: {e}")

async def run_processing_with_queue_and_tracking(file_generator, total_files, progress_manager=None):
    """
    Enhanced processing pipeline with queue system and file tracking
    CRITICAL FIX: Properly initialize embedding queue with workers
    """
    logger.info(f"run_processing_with_queue_and_tracking() called with {total_files} total files")
    
    # Initialize processing state
    processing_state.start_processing(total_files)
    
    # Filter files that need processing using file tracker
    logger.info("Filtering files that need processing...")
    files_to_process, filter_stats = file_tracker.batch_filter_files(list(file_generator))
    
    logger.info(f"File filtering complete:")
    logger.info(f"  Total files: {filter_stats.total_files}")
    logger.info(f"  Files to process: {filter_stats.files_to_process}")
    logger.info(f"  Files skipped: {filter_stats.files_skipped}")
    logger.info(f"  New files: {filter_stats.new_files}")
    logger.info(f"  Changed files: {filter_stats.size_changed + filter_stats.time_changed + filter_stats.content_changed}")
    logger.info(f"  Already processed: {filter_stats.already_processed}")
    
    if not files_to_process:
        logger.info("No files need processing")
        return 0
    
    # CRITICAL FIX: Start embedding queue with database callback BEFORE processing files
    logger.info("Starting embedding queue workers...")
    try:
        await embedding_queue.start_workers(
            concurrency=Config.WORKER_PROCESSES,
            insert_callback=database_insert_callback
        )
        logger.info(f"Embedding queue started with {Config.WORKER_PROCESSES} workers")
        
        # Verify the queue is actually started
        if not embedding_queue.started:
            logger.error("CRITICAL: Embedding queue failed to start!")
            return 0
        else:
            logger.info("✓ Embedding queue confirmed started")
            
    except Exception as e:
        logger.error(f"Failed to start embedding queue: {e}", exc_info=True)
        return 0
    
    # Progress callback function
    def progress_callback(file_path, stage, current, total, message):
        logger.debug(f"Progress: {file_path} - {stage} - {message}")
        if progress_manager and hasattr(progress_manager, 'update_file_progress'):
            try:
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
                logger.warning(f"Progress manager update failed: {e}")
    
    # Resource monitoring
    processing_active = threading.Event()
    processing_active.set()
    
    def monitor_resources():
        while processing_active.is_set():
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            load = os.getloadavg()
            queue_stats = embedding_queue.stats
            
            logger.info(
                f"SYSTEM | CPU: {cpu}% | Memory: {mem.percent}% ({mem.used/(1024**3):.1f}GB) | "
                f"Load: {load[0]:.1f} | Queue: {queue_stats['queue_size']} items "
                f"({queue_stats['current_memory_mb']:.1f}MB) | Processed: {queue_stats['processed_items']} | Failed: {queue_stats['failed_items']}"
            )
            
            # Additional queue debugging
            logger.debug(f"Queue detailed stats: {queue_stats}")
            
            # Update progress manager with system stats if available
            if progress_manager:
                try:
                    if hasattr(progress_manager, 'update_system_stats'):
                        progress_manager.update_system_stats(
                            cpu_percent=cpu,
                            memory_percent=mem.percent,
                            load_avg=load[0]
                        )
                    if hasattr(progress_manager, 'update_queue_stats'):
                        progress_manager.update_queue_stats(
                            queue_size=queue_stats['queue_size'],
                            active_workers=len([w for w in embedding_queue.workers if not w.done()]) if embedding_queue.workers else 0
                        )
                except Exception as e:
                    logger.warning(f"Progress manager stats update failed: {e}")
            
            # Memory cleanup if needed
            if mem.percent >= Config.MEMORY_CLEANUP_THRESHOLD:
                logger.warning(f"High memory usage ({mem.percent}%). Running cleanup")
                cleanup_memory()
                
            time.sleep(10)

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    success_count = 0
    total_items_queued = 0
    failed_files = []
    
    try:
        batch_size = 50  # Process files in smaller batches
        processed_count = 0

        # Initialize progress manager if available
        if progress_manager and hasattr(progress_manager, 'start_processing'):
            try:
                progress_manager.start_processing(len(files_to_process))
            except Exception as e:
                logger.warning(f"Progress manager start failed: {e}")

        with tqdm(total=len(files_to_process), desc="Processing files", unit="file") as pbar:
            # Process files in batches
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
                
                # Process batch sequentially to maintain file tracker order
                for file_path in batch:
                    logger.debug(f"Processing file: {file_path}")
                    result = await process_file_with_queue(file_path, progress_callback)
                    file_path, success, items_queued = result
                    
                    if success:
                        success_count += 1
                        total_items_queued += items_queued
                        logger.debug(f"File success: {file_path}, queued: {items_queued}")
                    else:
                        failed_files.append(file_path)
                        logger.warning(f"File failed: {file_path}")
                    
                    processed_count += 1
                    processing_state.update_progress(processed_count, file_path)
                    
                    pbar.update(1)
                    pbar.set_postfix(
                        success=success_count,
                        failed=len(failed_files),
                        queued=total_items_queued,
                        mem=f"{psutil.virtual_memory().percent}%",
                        qsize=embedding_queue.stats['queue_size']
                    )
                    
                    # Update overall progress in progress manager
                    if progress_manager and hasattr(progress_manager, 'update_overall_progress'):
                        try:
                            progress_manager.update_overall_progress(
                                processed_count,
                                len(files_to_process),
                                success_count,
                                len(failed_files),
                                total_items_queued
                            )
                        except Exception as e:
                            logger.warning(f"Overall progress update failed: {e}")
                
                # Save file tracker state after each batch
                file_tracker.save_tracker()
                processing_state.save_state()
                
                # Run maintenance if memory pressure is high
                mem = psutil.virtual_memory()
                if mem.percent > 80:
                    logger.warning("High memory pressure, running maintenance")
                    await run_maintenance()
                
                gc.collect()

        # Wait for queue to empty
        logger.info("Waiting for embedding queue to complete processing...")
        wait_count = 0
        while embedding_queue.stats['queue_size'] > 0:
            await asyncio.sleep(1)
            wait_count += 1
            queue_size = embedding_queue.stats['queue_size']
            if wait_count % 10 == 0:  # Log every 10 seconds
                logger.info(f"Queue items remaining: {queue_size}, processed: {embedding_queue.stats['processed_items']}")
            
            # Safety timeout
            if wait_count > 300:  # 5 minutes timeout
                logger.warning("Queue processing timeout reached")
                break

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        processing_state.add_failed_file("SYSTEM_ERROR", str(e))
        if progress_manager and hasattr(progress_manager, 'report_error'):
            try:
                progress_manager.report_error(str(e))
            except Exception as pm_e:
                logger.warning(f"Progress manager error reporting failed: {pm_e}")
    finally:
        processing_active.clear()
        monitor_thread.join(timeout=1.0)
        
        # Stop embedding queue workers
        logger.info("Stopping embedding queue workers...")
        await embedding_queue.stop_workers()
        
        # Final save of state
        file_tracker.save_tracker()
        processing_state.save_state()
        
        # Finalize progress manager
        if progress_manager and hasattr(progress_manager, 'finish_processing'):
            try:
                progress_manager.finish_processing(success_count, len(failed_files), total_items_queued)
            except Exception as e:
                logger.warning(f"Progress manager finish failed: {e}")
        
        logger.info(f"Processing complete!")
        logger.info(f"  Files processed: {success_count}/{len(files_to_process)}")
        logger.info(f"  Items queued for embedding: {total_items_queued}")
        logger.info(f"  Failed files: {len(failed_files)}")
        logger.info(f"  Final queue stats: {embedding_queue.stats}")
        
        if failed_files:
            logger.warning(f"Failed files saved to processing_state.json")
        
        return success_count

# Legacy function for backwards compatibility
async def run_processing(file_generator, total_files):
    """Legacy processing function - redirects to enhanced version"""
    return await run_processing_with_queue_and_tracking(file_generator, total_files)

async def run_maintenance():
    """Run periodic database maintenance"""
    logger.debug("Running database maintenance")
    try:
        pool = await initialize_worker_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"ANALYZE {Config.TABLE_NAME}")
            await conn.execute(f"VACUUM (VERBOSE, ANALYZE) {Config.TABLE_NAME}")
            logger.info("Database maintenance completed")
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")

def get_processing_status():
    """Get current processing status"""
    return {
        'processing_state': processing_state.get_progress_info(),
        'file_tracker_stats': file_tracker.get_processed_files_stats(),
        'queue_stats': embedding_queue.stats,
        'system_memory': psutil.virtual_memory().percent
    }
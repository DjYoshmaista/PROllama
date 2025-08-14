# async_loader.py
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

logger = logging.getLogger(__name__)
DB_CONN_STRING = Config.get_db_connection_string()

# Shared counters for progress tracking (multiprocessing-safe)
shared_counters = None

# Connection pool per worker process
_worker_pool = None
_processing_stats = {
    'files_processed': 0,
    'records_created': 0,
    'embeddings_failed': 0,
    'last_update': time.time()
}

async def initialize_worker_pool():
    """Initialize connection pool for worker using centralized DB manager"""
    global _worker_pool
    if _worker_pool is None:
        # Use the centralized database manager instead of creating separate pools
        _worker_pool = await db_manager.get_async_pool()
    return _worker_pool

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

def init_shared_counters():
    """Initialize shared counters for multiprocessing"""
    global shared_counters
    manager = multiprocessing.Manager()
    shared_counters = manager.dict({
        'files_processed': 0,
        'data_pieces_created': 0,
        'embeddings_created': 0,
        'records_inserted': 0
    })
    return shared_counters

def update_shared_counter(counter_name, increment=1):
    """Update shared counter (multiprocessing-safe)"""
    global shared_counters
    if shared_counters is not None:
        shared_counters[counter_name] = shared_counters.get(counter_name, 0) + increment

async def process_file(file_path, progress_callback=None):
    """
    Process a single file with optional progress callback
    
    Args:
        file_path: Path to the file to process
        progress_callback: Optional callable that receives progress updates
                          Signature: progress_callback(file_path, stage, current, total, message)
    """
    global _worker_pool
    file_type = os.path.splitext(file_path)[1][1:].lower()
    processed_records = 0

    def report_progress(stage, current=0, total=1, message=""):
        """Helper function to report progress if callback is provided"""
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(file_path, stage, current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback failed for {file_path}: {e}")

    # Report initialization stage
    report_progress("initializing", 0, 1, "Setting up database connection")

    if _worker_pool is None:
        _worker_pool = await asyncpg.create_pool(
            DB_CONN_STRING, 
            min_size=2,
            max_size=8,
            timeout=180,
            server_settings=Config.PG_OPTIMIZATION_SETTINGS
        )
    
    try:
        async with _worker_pool.acquire() as conn:
            await pgvector.asyncpg.register_vector(conn)
            await conn.execute("SET jit = off")
            
            # Report connection established
            report_progress("connected", 1, 1, "Database connection established")
            
            connector = aiohttp.TCPConnector(limit=50)
            async with aiohttp.ClientSession(connector=connector) as session:
                # Report file loading stage
                report_progress("loading", 0, 1, "Loading and chunking file")
                
                chunk_generator = load_documents.load_file_chunked(
                    file_path, file_type, session, Config.CHUNK_SIZES['embedding_batch']
                )
                batch_counter = 0
                failed_embeddings = 0
                total_chunks_processed = 0
                
                # Report processing start
                report_progress("processing", 0, 1, "Starting chunk processing")

                chunk_index = 0
                async for records in chunk_generator:
                    if not records:
                        continue
                    
                    # Report chunk processing progress
                    report_progress("processing_chunk", chunk_index, chunk_index + 1, 
                                  f"Processing chunk {chunk_index + 1} ({len(records)} records)")
                    
                    chunk_index += 1
                    
                    for i in range(0, len(records), Config.CHUNK_SIZES['insert_batch']):
                        chunk = records[i:i+Config.CHUNK_SIZES['insert_batch']]
                        
                        # Report batch insertion progress
                        batch_num = i // Config.CHUNK_SIZES['insert_batch'] + 1
                        total_batches = (len(records) + Config.CHUNK_SIZES['insert_batch'] - 1) // Config.CHUNK_SIZES['insert_batch']
                        report_progress("inserting_batch", batch_num, total_batches, 
                                      f"Inserting batch {batch_num}/{total_batches} ({len(chunk)} records)")
                        
                        try:
                            # Insert tags as JSONB directly
                            await conn.executemany(
                                f"INSERT INTO {Config.TABLE_NAME} (content, tags, embedding) VALUES ($1, $2::text[], $3)",
                                [(r['content'], r['tags'] if isinstance(r['tags'], list) else r['tags'], r['embedding']) for r in chunk]
                            )
                            processed_records += len(chunk)
                            
                            # Report successful batch insertion
                            report_progress("batch_inserted", batch_num, total_batches, 
                                          f"Successfully inserted batch {batch_num} ({processed_records} total records)")
                            
                        except asyncpg.exceptions.UndefinedColumnError:
                            # Handle missing tags column
                            report_progress("schema_migration", 0, 1, "Adding missing tags column")
                            await conn.execute(f"ALTER TABLE {Config.TABLE_NAME} ADD COLUMN tags TEXT[] DEFAULT '[]'::text[]")
                            
                            await conn.executemany(
                                f"INSERT INTO {Config.TABLE_NAME} (content, tags, embedding) VALUES ($1, $2, $3)",
                                [(r['content'], r['tags'], r['embedding']) for r in chunk]
                            )
                            processed_records += len(chunk)
                            
                            # Report successful migration and insertion
                            report_progress("migration_complete", 1, 1, f"Schema migrated and batch inserted ({processed_records} total records)")
                            
                        except Exception as e:
                            failed_embeddings += len(chunk)
                            logger.error(f"Insert failed: {e}")
                            
                            # Report batch failure
                            report_progress("batch_failed", batch_num, total_batches, 
                                          f"Batch {batch_num} failed: {str(e)[:100]}")
                            
                    total_chunks_processed += 1
                    
                    # Report chunk completion
                    report_progress("chunk_complete", total_chunks_processed, total_chunks_processed, 
                                  f"Completed chunk {total_chunks_processed} ({processed_records} total records)")
                            
                    if batch_counter % 100 == 0:
                        # Report metrics logging
                        report_progress("logging_metrics", batch_counter, batch_counter + 1, "Logging batch metrics")
                        await log_batch_metrics(conn)
                    batch_counter += 1
                
                # Report final processing results
                if processed_records > 0 or failed_embeddings > 0:
                    success_message = f"Processed {processed_records} records"
                    if failed_embeddings > 0:
                        success_message += f", failed {failed_embeddings} embeddings"
                    
                    report_progress("processing_complete", 1, 1, success_message)
                    
                    logger.info(f"Processed {processed_records} records, "
                                f"failed {failed_embeddings} embeddings from {file_path}")
                else:
                    report_progress("no_records", 1, 1, "No records to process")

        # Report final file completion
        report_progress("file_complete", 1, 1, f"File processing complete: {processed_records} records")
        
        logger.info(f"Processed {processed_records} records from {file_path}")
        return (file_path, processed_records > 0, processed_records)
        
    except Exception as e:
        # Report error
        error_message = f"Error processing file: {str(e)[:100]}"
        report_progress("error", 0, 1, error_message)
        
        logger.error(f"Error processing {file_path}: {e}")
        return (file_path, False, 0)
    
async def log_batch_metrics(conn):
    metrics = await conn.fetch(f"""
        SELECT 
            AVG(vector_norm(embedding)) AS avg_norm,
            COUNT(*) FILTER (WHERE vector_norm(embedding) = 0) AS zero_vectors
        FROM {Config.TABLE_NAME}
    """)
    logger.info(f"Embedding batch metrics: {dict(metrics[0])}")

    del metrics
    gc.collect()

async def run_processing(file_generator, total_files):
    """Process files from generator with resource monitoring"""
    # Start the embedding queue workers
    await embedding_queue.start_workers(concurrency=10)
    
    processing_active = threading.Event()
    processing_active.set()
    
    def monitor_resources():
        while processing_active.is_set():
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            load = os.getloadavg()
            
            logger.info(
                f"RESOURCE MONITOR | CPU: {cpu}% | "
                f"Memory: {mem.percent}% | Used: {mem.used/(1024**2):.1f}MB | "
                f"Load: {load[0]:.1f}, {load[1]:.1f}, {load[2]:.1f}"
            )
            
            # Add embedding queue stats
            if embedding_queue.started:
                qsize = embedding_queue.queue.qsize()
                active_workers = sum(1 for w in embedding_queue.workers if not w.done())
                logger.info(f"EMBEDDING QUEUE | Size: {qsize} | Active Workers: {active_workers}")
            
            if mem.percent >= Config.MEMORY_CLEANUP_THRESHOLD:
                logger.warning(f"High memory usage ({mem.percent}%). Forcing garbage collection")
                cleanup_memory()
                mem_after = psutil.virtual_memory()
                logger.info(f"GC freed {((mem.used - mem_after.used)/(1024**2)):.1f}MB")
            
            time.sleep(10)

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    success_count = 0
    total_embeddings = 0
    failed_files = []
    
    try:
        batch_size = 100
        processed_count = 0
        batch_num = 1

        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            while processed_count < total_files:
                file_batch = []
                for _ in range(min(batch_size, total_files - processed_count)):
                    try:
                        file_path = next(file_generator)
                        file_batch.append(file_path)
                    except StopIteration:
                        break
                
                if not file_batch:
                    break
                
                num_processes = Config.WORKER_PROCESSES
                async with AsyncPool(processes=num_processes) as pool:
                    tasks = [
                        pool.apply(process_file, (file_path,))
                        for file_path in file_batch
                    ]
                    
                    for future in asyncio.as_completed(tasks):
                        result = await future
                        file_path, success, embeddings_count = result
                        if success:
                            success_count += 1
                            total_embeddings += embeddings_count
                        else:
                            failed_files.append(file_path)
                        
                        pbar.update(1)
                        processed_count += 1
                        pbar.set_postfix(
                            success=success_count, 
                            failed=len(failed_files),
                            embeddings=total_embeddings,
                            mem=f"{psutil.virtual_memory().percent}%",
                            qsize=f"{embedding_queue.queue.qsize()}"
                        )
                
                del file_batch
                gc.collect()
                
                mem = psutil.virtual_memory()
                if mem.percent > 80:
                    logger.warning("High memory pressure, running VACUUM")
                    await run_maintenance()
                
                batch_num += 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        processing_active.clear()
        monitor_thread.join(timeout=1.0)
        
        # Stop workers when processing completes
        await embedding_queue.stop_workers()
        
        logger.info(f"Processed {success_count}/{total_files} files successfully")
        logger.info(f"Generated {total_embeddings} embeddings")
        if failed_files:
            logger.warning(f"{len(failed_files)} files failed processing")
            with open("failed_files.json", "w") as f:
                json.dump(failed_files, f)
        
        return success_count

async def run_processing_with_progress_manager(file_generator, total_files, progress_manager=None):
    """
    Enhanced version of run_processing that integrates with a progress manager
    
    Args:
        file_generator: Generator yielding file paths
        total_files: Total number of files to process
        progress_manager: Progress manager instance with update methods
    """
    # Define progress callback function that integrates with progress manager
    def progress_callback(file_path, stage, current, total, message):
        """Progress callback that forwards updates to the progress manager"""
        if progress_manager and hasattr(progress_manager, 'update_file_progress'):
            try:
                # Extract filename for cleaner display
                filename = os.path.basename(file_path)
                
                # Create detailed status message
                status_message = f"[{stage}] {message}" if message else f"[{stage}]"
                
                # Forward to progress manager
                progress_manager.update_file_progress(
                    filename=filename,
                    stage=stage,
                    current=current,
                    total=total,
                    message=status_message
                )
            except Exception as e:
                logger.warning(f"Progress manager update failed: {e}")
    
    # Start the embedding queue workers
    await embedding_queue.start_workers(concurrency=10)
    
    processing_active = threading.Event()
    processing_active.set()
    
    def monitor_resources():
        while processing_active.is_set():
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            load = os.getloadavg()
            
            logger.info(
                f"RESOURCE MONITOR | CPU: {cpu}% | "
                f"Memory: {mem.percent}% | Used: {mem.used/(1024**2):.1f}MB | "
                f"Load: {load[0]:.1f}, {load[1]:.1f}, {load[2]:.1f}"
            )
            
            # Update progress manager with system stats if available
            if progress_manager and hasattr(progress_manager, 'update_system_stats'):
                try:
                    progress_manager.update_system_stats(
                        cpu_percent=cpu,
                        memory_percent=mem.percent,
                        load_avg=load[0]
                    )
                except Exception as e:
                    logger.warning(f"System stats update failed: {e}")
            
            # Add embedding queue stats
            if embedding_queue.started:
                qsize = embedding_queue.queue.qsize()
                active_workers = sum(1 for w in embedding_queue.workers if not w.done())
                logger.info(f"EMBEDDING QUEUE | Size: {qsize} | Active Workers: {active_workers}")
                
                # Update progress manager with queue stats if available
                if progress_manager and hasattr(progress_manager, 'update_queue_stats'):
                    try:
                        progress_manager.update_queue_stats(
                            queue_size=qsize,
                            active_workers=active_workers
                        )
                    except Exception as e:
                        logger.warning(f"Queue stats update failed: {e}")
            
            if mem.percent >= Config.MEMORY_CLEANUP_THRESHOLD:
                logger.warning(f"High memory usage ({mem.percent}%). Forcing garbage collection")
                cleanup_memory()
                mem_after = psutil.virtual_memory()
                logger.info(f"GC freed {((mem.used - mem_after.used)/(1024**2)):.1f}MB")
            
            time.sleep(10)

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    success_count = 0
    total_embeddings = 0
    failed_files = []
    
    try:
        batch_size = 100
        processed_count = 0
        batch_num = 1

        # Initialize progress manager if available
        if progress_manager and hasattr(progress_manager, 'start_processing'):
            try:
                progress_manager.start_processing(total_files)
            except Exception as e:
                logger.warning(f"Progress manager start failed: {e}")

        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            while processed_count < total_files:
                file_batch = []
                for _ in range(min(batch_size, total_files - processed_count)):
                    try:
                        file_path = next(file_generator)
                        file_batch.append(file_path)
                    except StopIteration:
                        break
                
                if not file_batch:
                    break
                
                num_processes = Config.WORKER_PROCESSES
                async with AsyncPool(processes=num_processes) as pool:
                    # Create tasks with progress callback
                    tasks = [
                        pool.apply(process_file, (file_path, progress_callback))
                        for file_path in file_batch
                    ]
                    
                    for future in asyncio.as_completed(tasks):
                        result = await future
                        file_path, success, embeddings_count = result
                        if success:
                            success_count += 1
                            total_embeddings += embeddings_count
                        else:
                            failed_files.append(file_path)
                        
                        pbar.update(1)
                        processed_count += 1
                        pbar.set_postfix(
                            success=success_count, 
                            failed=len(failed_files),
                            embeddings=total_embeddings,
                            mem=f"{psutil.virtual_memory().percent}%",
                            qsize=f"{embedding_queue.queue.qsize()}"
                        )
                        
                        # Update overall progress in progress manager
                        if progress_manager and hasattr(progress_manager, 'update_overall_progress'):
                            try:
                                progress_manager.update_overall_progress(
                                    processed_count, 
                                    total_files, 
                                    success_count, 
                                    len(failed_files),
                                    total_embeddings
                                )
                            except Exception as e:
                                logger.warning(f"Overall progress update failed: {e}")
                
                del file_batch
                gc.collect()
                
                mem = psutil.virtual_memory()
                if mem.percent > 80:
                    logger.warning("High memory pressure, running VACUUM")
                    await run_maintenance()
                
                batch_num += 1
                
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if progress_manager and hasattr(progress_manager, 'report_error'):
            try:
                progress_manager.report_error(str(e))
            except Exception as pm_e:
                logger.warning(f"Progress manager error reporting failed: {pm_e}")
    finally:
        processing_active.clear()
        monitor_thread.join(timeout=1.0)
        
        # Stop workers when processing completes
        await embedding_queue.stop_workers()
        
        # Finalize progress manager
        if progress_manager and hasattr(progress_manager, 'finish_processing'):
            try:
                progress_manager.finish_processing(success_count, len(failed_files), total_embeddings)
            except Exception as e:
                logger.warning(f"Progress manager finish failed: {e}")
        
        logger.info(f"Processed {success_count}/{total_files} files successfully")
        logger.info(f"Generated {total_embeddings} embeddings")
        if failed_files:
            logger.warning(f"{len(failed_files)} files failed processing")
            with open("failed_files.json", "w") as f:
                json.dump(failed_files, f)
        
        return success_count

async def run_maintenance():
    """Run periodic database maintenance"""
    conn = None
    try:
        conn = await asyncpg.connect(DB_CONN_STRING)
        # Analyze to update statistics
        await conn.execute(f"ANALYZE {Config.TABLE_NAME}")
        # Vacuum to reclaim space
        await conn.execute(f"VACUUM (VERBOSE, ANALYZE) {Config.TABLE_NAME}")
        logger.info("Database maintenance completed")
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
    finally:
        if conn:
            await conn.close()
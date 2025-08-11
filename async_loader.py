# async_loader.py
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
from aiomultiprocess import Pool as AsyncPool
from tqdm import tqdm
import load_documents
import pgvector.asyncpg
from embedding_queue import embedding_queue
from constants import *

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

# New constants for memory optimization
POOL_RECYCLE_AFTER = 200  # Increased recycle threshold
MEMORY_SAFE_CHUNK_SIZE = 100  # Increased for efficiency
EMBEDDING_CHUNK_SIZE = 100  # Increased batch size
INSERT_CHUNK_SIZE = 500  # Larger inserts
MEMORY_CLEANUP_THRESHOLD = 85  # Higher threshold
MAX_CONCURRENT_CHUNKS = 32  # Increased concurrency
WORKER_PROCESSES = min(8, os.cpu_count())  # More worker processes

# New PostgreSQL optimization parameters
PG_OPTIMIZATION_SETTINGS = {
    "statement_timeout": "300000",  # 5 minutes
    "work_mem": "16MB",
    "maintenance_work_mem": "512MB"
}

# Connection pool per worker process
_worker_pool = None

async def process_file(file_path):
    global _worker_pool
    file_type = os.path.splitext(file_path)[1][1:].lower()
    processed_records = 0

    if _worker_pool is None:
        _worker_pool = await asyncpg.create_pool(
            DB_CONN_STRING, 
            min_size=2,
            max_size=8,
            timeout=180,
            server_settings=PG_OPTIMIZATION_SETTINGS
        )
    
    try:
        async with _worker_pool.acquire() as conn:
            await pgvector.asyncpg.register_vector(conn)
            await conn.execute("SET jit = off")
            
            connector = aiohttp.TCPConnector(limit=50)
            async with aiohttp.ClientSession(connector=connector) as session:
                chunk_generator = load_documents.load_file_chunked(
                    file_path, file_type, session, EMBEDDING_CHUNK_SIZE
                )
                batch_counter = 0
                failed_embeddings = 0   # Track embedding failures

                async for records in chunk_generator:
                    if not records:
                        continue
                    
                    for i in range(0, len(records), INSERT_CHUNK_SIZE):
                        chunk = records[i:i+INSERT_CHUNK_SIZE]
                        try:
                            # Insert tags as JSONB directly
                            await conn.executemany(
                                f"INSERT INTO {TABLE_NAME} (content, tags, embedding) VALUES ($1, $2::jsonb, $3)",
                                [(r[0], json.dumps(r[1]) if isinstance(r[1], list) else r[1], r[2]) for r in chunk]
                            )
                            processed_records += len(chunk)
                        except asyncpg.exceptions.UndefinedColumnError:
                            # Handle missing tags column
                            await conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN tags JSONB DEFAULT '[]'::jsonb")
                            await conn.executemany(
                                f"INSERT INTO {TABLE_NAME} (content, tags, embedding) VALUES ($1, $2, $3)",
                                [(r[0], r[1], r[2]) for r in chunk]
                            )
                            processed_records += len(chunk)
                        except Exception as e:
                            failed_embeddings += len(chunk)
                            logger.error(f"Insert failed: {e}")
                            
                    if batch_counter % 100 == 0:
                        await log_batch_metrics(conn)
                    batch_counter += 1
                
                # Log results with failure count
                if processed_records > 0 or failed_embeddings > 0:
                    logger.info(f"Processed {processed_records} recrods, "
                                f"failed {failed_embeddings} embeddings from {file_path}")

        logger.info(f"Processed {processed_records} records from {file_path}")
        return (file_path, processed_records > 0, processed_records)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return (file_path, False, 0)
    
async def log_batch_metrics(conn):
    metrics = await conn.fetch(f"""
        SELECT 
            AVG(vector_norm(embedding)) AS avg_norm,
            COUNT(*) FILTER (WHERE vector_norm(embedding) = 0) AS zero_vectors
        FROM {TABLE_NAME}
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
            
            if mem.percent >= MEMORY_CLEANUP_THRESHOLD:
                logger.warning(f"High memory usage ({mem.percent}%). Forcing garbage collection")
                gc.collect()
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
                
                num_processes = WORKER_PROCESSES
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
                            qsize=f"{embedding_queue.queue.qsize()}"  # Show queue size
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

async def run_maintenance():
    """Run periodic database maintenance"""
    conn = None
    try:
        conn = await asyncpg.connect(DB_CONN_STRING)
        # Analyze to update statistics
        await conn.execute(f"ANALYZE {TABLE_NAME}")
        # Vacuum to reclaim space
        await conn.execute(f"VACUUM (VERBOSE, ANALYZE) {TABLE_NAME}")
        logger.info("Database maintenance completed")
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")
    finally:
        if conn:
            await conn.close()

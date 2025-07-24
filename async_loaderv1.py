#!/usr/bin/env python3
# async_loader.py
import asyncio
import aiohttp
import asyncpg
import json
import os
import logging
import traceback
from aiomultiprocess import Pool
from parse_documents import parse_file
from utils import get_embeddings_batch
from tqdm import tqdm
import psutil
import time
import re
import csv
import hashlib

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("async_loader.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Global resources
global DB_POOL, PROGRESS_BAR, FAILED_FILES, SEMAPHORE, EMBEDDING_FAILURES
DB_POOL = None
PROGRESS_BAR = None
FAILED_FILES = []
SEMAPHORE = asyncio.Semaphore(100)  # Concurrency limiter
EMBEDDING_FAILURES = 0

async def init_db_pool():
    """Initialize PostgreSQL connection pool with optimized settings"""
    global DB_POOL
    DB_POOL = await asyncpg.create_pool(
        dsn="postgres://postgres:postgres@localhost/rag_db",
        min_size=4,
        max_size=16,
        command_timeout=300,
        server_settings={
            'jit': 'off',
            'max_parallel_workers_per_gather': '0',
            'idle_in_transaction_session_timeout': '60000'
        }
    )
    logger.info("Database pool initialized")

async def process_file(file_path):
    """Process a single file with comprehensive error handling"""
    file_type = os.path.splitext(file_path)[1][1:].lower()
    if not file_type:
        logger.warning(f"Skipping file without extension: {file_path}")
        return 0
        
    async with SEMAPHORE:
        try:
            # 1. Parse file with content validation
            parsed_data = parse_file(file_path, file_type)
            if not parsed_data:
                logger.debug(f"Skipped empty/unparseable file: {file_path}")
                return 0
                
            # 2. Prepare data with content sanitization
            contents = []
            for item in parsed_data:
                content = item.get("content", "")
                # Skip empty or invalid content
                if not content or not isinstance(content, str) or len(content.strip()) < 5:
                    continue
                contents.append(content[:1000000])  # Limit to 1MB per document
            
            if not contents:
                logger.warning(f"No valid content in {file_path}")
                return 0
                
            # 3. Get embeddings with automatic retry
            embeddings = []
            retry_count = 0
            while retry_count < 3 and not embeddings:
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http_session:
                        embeddings = await get_embeddings_batch(http_session, contents)
                except Exception as e:
                    logger.warning(f"Embedding attempt {retry_count+1} failed for {file_path}: {str(e)}")
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    retry_count += 1
            
            # 4. Handle embedding failures
            if not embeddings:
                global EMBEDDING_FAILURES
                EMBEDDING_FAILURES += 1
                logger.error(f"All embedding attempts failed for {file_path}")
                return 0
                
            # 5. Prepare tags with fallback serialization
            tags_list = []
            for item in parsed_data:
                tags = item.get("tags", [])
                try:
                    tags_json = json.dumps(tags, ensure_ascii=False)
                except TypeError:
                    try:
                        tags_json = json.dumps([str(t) for t in tags], ensure_ascii=False)
                    except Exception:
                        tags_json = "[]"
                tags_list.append(tags_json)
            
            # 6. Prepare records with length validation
            records = []
            min_length = min(len(contents), len(embeddings), len(tags_list))
            for i in range(min_length):
                if embeddings[i] is not None:
                    records.append((contents[i], tags_list[i], embeddings[i]))
            
            if not records:
                logger.warning(f"No valid records created for {file_path}")
                return 0
                
            # 7. Bulk insert with connection management
            async with DB_POOL.acquire() as conn:
                async with conn.transaction():
                    await conn.copy_records(
                        "documents",
                        records=records,
                        columns=["content", "tags", "embedding"],
                        timeout=60
                    )
            
            # 8. Update progress
            global PROGRESS_BAR
            if PROGRESS_BAR:
                PROGRESS_BAR.update(len(records))
                
            return len(records)
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"{error_type} processing {file_path}: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Add to failed files list
            global FAILED_FILES
            FAILED_FILES.append({
                "path": file_path,
                "error": str(e),
                "type": error_type,
                "traceback": traceback.format_exc()
            })
            return 0

def safe_parse_csv(file_path):
    """Robust CSV parsing with multi-layer fallbacks"""
    try:
        # Attempt 1: Standard parser
        return parse_file(file_path, 'csv')
    except Exception as e:
        logger.warning(f"Standard CSV parse failed for {file_path}: {str(e)}")
    
    try:
        # Attempt 2: Manual parsing with error recovery
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Try to detect CSV dialect
            sample = f.read(10240)
            f.seek(0)
            
            # Skip problematic characters
            sample = sample.replace('\0', '')
            
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                has_header = sniffer.has_header(sample)
            except:
                dialect = 'excel'
                has_header = False
            
            reader = csv.reader(f, dialect)
            headers = []
            parsed_data = []
            
            if has_header:
                try:
                    headers = next(reader)
                except StopIteration:
                    headers = []
            
            for row in reader:
                # Clean row data
                cleaned_row = [cell.replace('\0', '').strip() for cell in row]
                content = " ".join(cleaned_row)
                parsed_data.append({
                    "content": content,
                    "tags": headers.copy()
                })
                
            return parsed_data
    except Exception as e:
        logger.warning(f"Manual CSV parse failed for {file_path}: {str(e)}")
    
    try:
        # Attempt 3: Read as plain text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(1000000)  # Limit to 1MB
            return [{"content": content, "tags": ["text_fallback"]}]
    except Exception as e:
        logger.error(f"Complete failure reading {file_path}: {str(e)}")
        return []

async def process_file_batch(file_paths):
    """Process a batch of files with enhanced failure handling"""
    global PROGRESS_BAR
    PROGRESS_BAR = tqdm(
        total=len(file_paths),
        desc="Processing files",
        unit="file",
        mininterval=10,
        position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        postfix=f"Failures: {EMBEDDING_FAILURES}"
    )
    
    # Process files in parallel
    async with Pool(processes=min(200, os.cpu_count() * 4)) as pool:
        results = await pool.map(process_file, file_paths)
        total_loaded = sum(results)
        PROGRESS_BAR.close()
        
        # Log completion stats
        success_rate = (total_loaded / len(file_paths)) * 100 if file_paths else 0
        logger.info(
            f"Batch completed: {total_loaded}/{len(file_paths)} files "
            f"({success_rate:.2f}% success, {EMBEDDING_FAILURES} embedding failures)"
        )
        return total_loaded

def monitor_resources():
    """Comprehensive resource monitoring with PostgreSQL checks"""
    import psycopg2
    while True:
        try:
            # System resources
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=10)
            disk = psutil.disk_usage('/')
            load = os.getloadavg()
            
            # PostgreSQL connection stats
            pg_status = ""
            try:
                conn = psycopg2.connect(
                    dbname="rag_db",
                    user="postgres",
                    password="postgres",
                    host="localhost",
                    connect_timeout=2
                )
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM pg_stat_activity")
                active_conns = cur.fetchone()[0]
                cur.execute("SHOW max_connections")
                max_conns = cur.fetchone()[0]
                pg_status = f"PG: {active_conns}/{max_conns} conns | "
                cur.close()
                conn.close()
            except Exception as e:
                pg_status = f"PG: Error {str(e)} | "
            
            status = (
                f"RESOURCE MONITOR | {pg_status}"
                f"CPU: {cpu}% | "
                f"Memory: {mem.percent}% | "
                f"Disk: {disk.percent}% | "
                f"Load: {load[0]:.1f},{load[1]:.1f},{load[2]:.1f}"
            )
            
            logger.info(status)
            
            # Dynamic concurrency adjustment
            global SEMAPHORE
            if cpu > 90 or mem.percent > 90:
                new_value = max(50, int(SEMAPHORE._value * 0.7))
                SEMAPHORE = asyncio.Semaphore(new_value)
                logger.warning(f"High load! Reduced concurrency to {new_value}")
            elif cpu < 70 and mem.percent < 80:
                new_value = min(1000, int(SEMAPHORE._value * 1.2))
                if new_value > SEMAPHORE._value:
                    SEMAPHORE = asyncio.Semaphore(new_value)
            
            time.sleep(20)
        except Exception as e:
            logger.error(f"Monitor error: {str(e)}")
            time.sleep(30)

async def save_failed_files():
    """Periodically save failed files list with compression"""
    while True:
        await asyncio.sleep(300)
        global FAILED_FILES
        if FAILED_FILES:
            try:
                # Compress similar errors
                error_summary = {}
                for entry in FAILED_FILES:
                    error_key = hashlib.md5(entry["error"].encode()).hexdigest()[:8]
                    if error_key not in error_summary:
                        error_summary[error_key] = {
                            "error": entry["error"],
                            "type": entry["type"],
                            "count": 0,
                            "example": entry["path"],
                            "first_occurrence": time.time()
                        }
                    error_summary[error_key]["count"] += 1
                
                # Save summary
                with open("error_summary.json", "w") as f:
                    json.dump(error_summary, f, indent=2)
                    
                # Save detailed failures
                if len(FAILED_FILES) > 1000:
                    rotated_file = f"failed_files_{int(time.time())}.json.gz"
                    import gzip
                    with gzip.open(rotated_file, "wt") as f:
                        json.dump(FAILED_FILES, f)
                    FAILED_FILES = []
                    logger.info(f"Rotated failure log to {rotated_file}")
                else:
                    with open("failed_files.json", "w") as f:
                        json.dump(FAILED_FILES, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save error log: {str(e)}")
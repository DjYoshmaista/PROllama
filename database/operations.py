# database/operations.py - Optimized Database Operations
import json
import logging
import asyncio
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from database.connection import db_connection
from core.config import config
from file_management.chunking import TextChunk
from inference.summarization import ChunkSummary

# Create prefixed logger for this file
logger = logging.getLogger(__name__)
LOG_PREFIX = "[Database]"

class DatabaseOperations:
    """Optimized database operations with batch processing"""
    
    def __init__(self):
        self.table_name = config.database.table_name
        self.chunks_table = f"{self.table_name}_chunks"
        self.summaries_table = f"{self.table_name}_summaries"  
        self.checksums_table = f"{self.table_name}_checksums"
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._insert_buffer = []
        self._buffer_lock = asyncio.Lock()
        self._buffer_size = 1000  # Batch size for inserts
    
    def initialize_schema(self):
        """Initialize enhanced database schema with deduplication support"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                # Original documents table (now for chunks)
                create_chunks_table = f"""
                    CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                        id SERIAL PRIMARY KEY,
                        chunk_id VARCHAR(255) UNIQUE NOT NULL,
                        content_hash VARCHAR(64) UNIQUE NOT NULL,
                        source_file TEXT NOT NULL,
                        content TEXT NOT NULL,
                        tags JSONB DEFAULT '[]'::jsonb,
                        chunk_index INTEGER NOT NULL,
                        total_chunks INTEGER NOT NULL,
                        start_token INTEGER,
                        end_token INTEGER,
                        overlap_before TEXT DEFAULT '',
                        overlap_after TEXT DEFAULT '',
                        parent_chunks JSONB DEFAULT '[]'::jsonb,
                        child_chunks JSONB DEFAULT '[]'::jsonb,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        embedding VECTOR(1024)
                    )
                """
                cur.execute(create_chunks_table)
                
                # Summaries table for faster retrieval
                create_summaries_table = f"""
                    CREATE TABLE IF NOT EXISTS {self.summaries_table} (
                        id SERIAL PRIMARY KEY,
                        chunk_id VARCHAR(255) NOT NULL,
                        summary_hash VARCHAR(64) UNIQUE NOT NULL,
                        source_file TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        key_topics JSONB DEFAULT '[]'::jsonb,
                        chunk_indices JSONB DEFAULT '[]'::jsonb,
                        related_chunk_ids JSONB DEFAULT '[]'::jsonb,
                        original_length INTEGER,
                        summary_length INTEGER,
                        importance_score FLOAT DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        embedding VECTOR(1024)
                    )
                """
                cur.execute(create_summaries_table)
                
                # File checksums table for preventing reprocessing
                create_checksums_table = f"""
                    CREATE TABLE IF NOT EXISTS {self.checksums_table} (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        file_checksum VARCHAR(32) UNIQUE NOT NULL,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        chunks_created INTEGER DEFAULT 0,
                        summaries_created INTEGER DEFAULT 0
                    )
                """
                cur.execute(create_checksums_table)
                
                # Keep original table for backward compatibility (rename it)
                try:
                    cur.execute(f"ALTER TABLE {self.table_name} RENAME TO {self.table_name}_legacy")
                    logger.info("Legacy table renamed for backward compatibility")
                except Exception:
                    # Table might not exist or already renamed
                    pass
                
                # Create view that combines chunks and legacy data
                create_view = f"""
                    CREATE OR REPLACE VIEW {self.table_name} AS
                    SELECT id, content, tags, created_at, embedding 
                    FROM {self.chunks_table}
                    UNION ALL
                    SELECT id, content, tags, created_at, embedding 
                    FROM {self.table_name}_legacy
                    WHERE EXISTS (SELECT 1 FROM {self.table_name}_legacy LIMIT 1)
                """
                try:
                    cur.execute(create_view)
                except Exception as view_e:
                    logger.info(f"View creation skipped (legacy table may not exist): {view_e}")
                
                # Create indexes for performance
                indexes = [
                    f"CREATE INDEX IF NOT EXISTS {self.chunks_table}_embedding_idx ON {self.chunks_table} USING hnsw (embedding vector_cosine_ops)",
                    f"CREATE INDEX IF NOT EXISTS {self.summaries_table}_embedding_idx ON {self.summaries_table} USING hnsw (embedding vector_cosine_ops)",
                    f"CREATE INDEX IF NOT EXISTS {self.chunks_table}_source_idx ON {self.chunks_table} (source_file)",
                    f"CREATE INDEX IF NOT EXISTS {self.chunks_table}_chunk_id_idx ON {self.chunks_table} (chunk_id)",
                    f"CREATE INDEX IF NOT EXISTS {self.chunks_table}_content_hash_idx ON {self.chunks_table} (content_hash)",
                    f"CREATE INDEX IF NOT EXISTS {self.summaries_table}_summary_hash_idx ON {self.summaries_table} (summary_hash)",
                    f"CREATE INDEX IF NOT EXISTS {self.summaries_table}_importance_idx ON {self.summaries_table} (importance_score DESC)",
                    f"CREATE INDEX IF NOT EXISTS {self.checksums_table}_checksum_idx ON {self.checksums_table} (file_checksum)",
                    f"CREATE INDEX IF NOT EXISTS {self.checksums_table}_path_idx ON {self.checksums_table} (file_path)",
                ]
                
                for index_query in indexes:
                    try:
                        cur.execute(index_query)
                    except Exception as idx_e:
                        logger.warning(f"Index creation failed: {idx_e}")
                
                conn.commit()
                logger.info("Enhanced database schema with deduplication initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for deduplication"""
        # Normalize content for consistent hashing
        normalized = ' '.join(content.strip().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def get_existing_content_hashes(self) -> Set[str]:
        """Get all existing content hashes"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(f"SELECT content_hash FROM {self.chunks_table}")
                results = cur.fetchall()
                return {row[0] for row in results}
        except Exception as e:
            logger.error(f"Failed to get existing content hashes: {e}")
            return set()
    
    def get_existing_summary_hashes(self) -> Set[str]:
        """Get all existing summary hashes"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(f"SELECT summary_hash FROM {self.summaries_table}")
                results = cur.fetchall()
                return {row[0] for row in results}
        except Exception as e:
            logger.error(f"Failed to get existing summary hashes: {e}")
            return set()
    
    def get_existing_file_checksums(self) -> Set[str]:
        """Get all existing file checksums"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(f"SELECT file_checksum FROM {self.checksums_table}")
                results = cur.fetchall()
                return {row[0] for row in results}
        except Exception as e:
            logger.error(f"Failed to get existing file checksums: {e}")
            return set()
    
    def is_file_already_processed(self, file_path: str) -> bool:
        """Check if file was already processed"""
        checksum = self.calculate_file_checksum(file_path)
        if not checksum:
            return False
        
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(f"SELECT 1 FROM {self.checksums_table} WHERE file_checksum = %s", (checksum,))
                return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check if file processed: {e}")
            return False
    
    def record_file_processed(self, file_path: str, chunks_created: int = 0, summaries_created: int = 0):
        """Record that a file has been processed"""
        checksum = self.calculate_file_checksum(file_path)
        if not checksum:
            return
        
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(
                    f"""
                    INSERT INTO {self.checksums_table} 
                    (file_path, file_checksum, chunks_created, summaries_created)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (file_checksum) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    processed_at = CURRENT_TIMESTAMP,
                    chunks_created = EXCLUDED.chunks_created,
                    summaries_created = EXCLUDED.summaries_created
                    """,
                    (file_path, checksum, chunks_created, summaries_created)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record file as processed: {e}")

    def get_embeddings_for_cache(self, limit: int = 10000) -> Optional[Tuple[List[int], List[List[float]]]]:
        """
        Get embeddings for caching - optimized query
        """
        try:
            # Check total count first
            total_docs = self.get_document_count()
            if total_docs > limit:
                logger.info(f"Database too large for caching ({total_docs} > {limit})")
                return None
            
            with db_connection.get_sync_connection() as (conn, cur):
                # Use server-side cursor for memory efficiency
                cur_name = f"embedding_cursor_{id(self)}"
                with conn.cursor(name=cur_name) as server_cur:
                    server_cur.itersize = 1000  # Fetch 1000 rows at a time
                    server_cur.execute(f"SELECT id, embedding FROM {self.table_name} ORDER BY id LIMIT %s", (limit,))
                    
                    ids = []
                    embeddings = []
                    
                    for row in server_cur:
                        doc_id, embedding = row
                        if embedding is None:
                            continue
                        
                        # Handle different embedding formats
                        try:
                            if isinstance(embedding, str):
                                if embedding.startswith('[') and embedding.endswith(']'):
                                    embedding = [float(x) for x in embedding[1:-1].split(',')]
                                else:
                                    embedding = json.loads(embedding)
                            elif isinstance(embedding, (list, tuple)):
                                embedding = list(embedding)
                            else:
                                embedding = list(embedding)
                            
                            ids.append(doc_id)
                            embeddings.append(embedding)
                            
                        except Exception as e:
                            logger.warning(f"Failed to parse embedding for ID {doc_id}: {e}")
                            continue
                
                logger.info(f"Loaded {len(ids)} embeddings for caching")
                return ids, embeddings
                
        except Exception as e:
            logger.error(f"Failed to get embeddings for cache: {e}")
            return None
        
    async def insert_chunks_batch(
        self, 
        chunks: List[TextChunk], 
        embeddings: List[List[float]]
    ) -> int:
        """Insert text chunks with embeddings and deduplication"""
        if not chunks or len(chunks) != len(embeddings):
            return 0
        
        def _sync_insert_chunks():
            try:
                with db_connection.get_sync_connection() as (conn, cur):
                    insert_data = []
                    for chunk, embedding in zip(chunks, embeddings):
                        content_hash = self.calculate_content_hash(chunk.content)
                        insert_data.append((
                            chunk.chunk_id,
                            content_hash,
                            chunk.source_file,
                            chunk.content,
                            json.dumps([]),  # tags
                            chunk.chunk_index,
                            chunk.total_chunks,
                            chunk.start_token,
                            chunk.end_token,
                            chunk.overlap_before,
                            chunk.overlap_after,
                            json.dumps(chunk.parent_chunks),
                            json.dumps(chunk.child_chunks),
                            embedding
                        ))
                    
                    query = f"""
                        INSERT INTO {self.chunks_table} 
                        (chunk_id, content_hash, source_file, content, tags, chunk_index, total_chunks,
                         start_token, end_token, overlap_before, overlap_after,
                         parent_chunks, child_chunks, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (content_hash) DO NOTHING
                    """
                    
                    cur.executemany(query, insert_data)
                    
                    # Count actual insertions
                    cur.execute("SELECT ROW_COUNT()")
                    inserted_count = cur.fetchone()[0] if cur.rowcount > 0 else len(insert_data)
                    
                    conn.commit()
                    return inserted_count
            except Exception as e:
                logger.error(f"Chunk batch insert failed: {e}")
                return 0
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _sync_insert_chunks)
    
    async def insert_summaries_batch(
        self, 
        summaries: List[ChunkSummary], 
        embeddings: List[List[float]]
    ) -> int:
        """Insert summaries with embeddings and deduplication"""
        if not summaries or len(summaries) != len(embeddings):
            return 0
        
        def _sync_insert_summaries():
            try:
                with db_connection.get_sync_connection() as (conn, cur):
                    insert_data = []
                    for summary, embedding in zip(summaries, embeddings):
                        summary_hash = self.calculate_content_hash(summary.summary)
                        insert_data.append((
                            summary.chunk_id,
                            summary_hash,
                            summary.source_file,
                            summary.summary,
                            json.dumps(summary.key_topics),
                            json.dumps(summary.chunk_indices),
                            json.dumps(summary.related_chunk_ids),
                            summary.original_length,
                            summary.summary_length,
                            summary.importance_score,
                            embedding
                        ))
                    
                    query = f"""
                        INSERT INTO {self.summaries_table} 
                        (chunk_id, summary_hash, source_file, summary, key_topics, chunk_indices,
                         related_chunk_ids, original_length, summary_length, 
                         importance_score, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (summary_hash) DO UPDATE SET
                        chunk_id = EXCLUDED.chunk_id,
                        key_topics = EXCLUDED.key_topics,
                        importance_score = EXCLUDED.importance_score,
                        embedding = EXCLUDED.embedding
                    """
                    
                    cur.executemany(query, insert_data)
                    
                    # Count actual insertions/updates
                    cur.execute("SELECT ROW_COUNT()")
                    affected_count = cur.fetchone()[0] if cur.rowcount > 0 else len(insert_data)
                    
                    conn.commit()
                    return affected_count
            except Exception as e:
                logger.error(f"Summary batch insert failed: {e}")
                return 0
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _sync_insert_summaries)
    
    def similarity_search_summaries(
        self, 
        query_embedding: List[float], 
        top_k: int = 10, 
        threshold: float = 0.3,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search summaries for high-level matches"""
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                query = f"""
                    SELECT 
                        chunk_id,
                        source_file,
                        summary,
                        key_topics,
                        chunk_indices,
                        related_chunk_ids,
                        importance_score,
                        1 - (embedding <=> '{embedding_str}'::vector) as cosine_similarity
                    FROM {self.summaries_table}
                    WHERE 1 - (embedding <=> '{embedding_str}'::vector) >= %s
                    AND importance_score >= %s
                    ORDER BY importance_score DESC, embedding <=> '{embedding_str}'::vector
                    LIMIT %s
                """
                
                cur.execute(query, (threshold, min_importance, top_k))
                return [dict(row) for row in cur.fetchall()]
                
        except Exception as e:
            logger.error(f"Summary similarity search failed: {e}")
            return []
    
    def similarity_search_chunks(
        self, 
        query_embedding: List[float], 
        top_k: int = 25, 
        threshold: float = 0.3,
        source_files: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search chunks with optional file filtering"""
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                where_clause = f"1 - (embedding <=> '{embedding_str}'::vector) >= %s"
                params = [threshold]
                
                if source_files:
                    placeholders = ','.join(['%s'] * len(source_files))
                    where_clause += f" AND source_file IN ({placeholders})"
                    params.extend(source_files)
                
                query = f"""
                    SELECT 
                        chunk_id,
                        source_file,
                        content,
                        chunk_index,
                        total_chunks,
                        parent_chunks,
                        child_chunks,
                        1 - (embedding <=> '{embedding_str}'::vector) as cosine_similarity
                    FROM {self.chunks_table}
                    WHERE {where_clause}
                    ORDER BY embedding <=> '{embedding_str}'::vector
                    LIMIT %s
                """
                
                params.append(top_k)
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]
                
        except Exception as e:
            logger.error(f"Chunk similarity search failed: {e}")
            return []
    
    def get_related_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get related chunks for context reconstruction"""
        if not chunk_ids:
            return []
        
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                placeholders = ','.join(['%s'] * len(chunk_ids))
                query = f"""
                    SELECT chunk_id, source_file, content, chunk_index, 
                           parent_chunks, child_chunks
                    FROM {self.chunks_table}
                    WHERE chunk_id IN ({placeholders})
                    ORDER BY source_file, chunk_index
                """
                cur.execute(query, chunk_ids)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get related chunks: {e}")
            return []
    
    def reconstruct_context_from_chunks(
        self, 
        primary_chunks: List[Dict[str, Any]], 
        expand_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Reconstruct context by expanding to related chunks"""
        if not primary_chunks:
            return []
        
        if not expand_context:
            return primary_chunks
        
        # Collect all related chunk IDs
        related_ids = set()
        for chunk in primary_chunks:
            related_ids.add(chunk['chunk_id'])
            
            # Add parent/child chunks
            parent_chunks = json.loads(chunk.get('parent_chunks', '[]'))
            child_chunks = json.loads(chunk.get('child_chunks', '[]'))
            
            related_ids.update(parent_chunks)
            related_ids.update(child_chunks)
        
        # Get all related chunks
        all_chunks = self.get_related_chunks(list(related_ids))
        
        # Sort by file and chunk index for proper ordering
        all_chunks.sort(key=lambda x: (x['source_file'], x['chunk_index']))
        
        return all_chunks
    
    def get_deduplication_stats(self) -> Dict[str, int]:
        """Get deduplication statistics"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                # Get chunk stats
                cur.execute(f"SELECT COUNT(*) FROM {self.chunks_table}")
                total_chunks = cur.fetchone()[0]
                
                # Get summary stats
                cur.execute(f"SELECT COUNT(*) FROM {self.summaries_table}")
                total_summaries = cur.fetchone()[0]
                
                # Get file stats
                cur.execute(f"SELECT COUNT(*), SUM(chunks_created), SUM(summaries_created) FROM {self.checksums_table}")
                file_result = cur.fetchone()
                
                return {
                    'total_chunks': total_chunks,
                    'total_summaries': total_summaries,
                    'processed_files': file_result[0] if file_result else 0,
                    'total_chunks_created': file_result[1] if file_result and file_result[1] else 0,
                    'total_summaries_created': file_result[2] if file_result and file_result[2] else 0
                }
        except Exception as e:
            logger.error(f"Failed to get deduplication stats: {e}")
            return {}
    
    async def buffered_insert(self, records: List[Tuple[str, List[str], List[float]]]) -> int:
        """
        Buffer inserts with parallel flushing for better performance
        """
        async with self._buffer_lock:
            self._insert_buffer.extend(records)
            
            # Check if we need to flush multiple buffers
            if len(self._insert_buffer) >= self._buffer_size * 2:
                # Extract multiple batches for parallel processing
                batches_to_insert = []
                while len(self._insert_buffer) >= self._buffer_size:
                    batch = self._insert_buffer[:self._buffer_size]
                    self._insert_buffer = self._insert_buffer[self._buffer_size:]
                    batches_to_insert.append(batch)
                
                # Process batches in parallel without holding lock
                if batches_to_insert:
                    tasks = [self.insert_documents_batch(batch) for batch in batches_to_insert]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    total_inserted = 0
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Parallel buffered insert failed: {result}")
                        elif isinstance(result, int):
                            total_inserted += result
                    
                    return total_inserted
            
            elif len(self._insert_buffer) >= self._buffer_size:
                # Single buffer flush
                to_insert = self._insert_buffer[:self._buffer_size]
                self._insert_buffer = self._insert_buffer[self._buffer_size:]
                
                # Insert without holding lock
                inserted = await self.insert_documents_batch(to_insert)
                return inserted
        
        return 0
    
    async def flush_buffer(self) -> int:
        """Force flush any remaining buffered inserts"""
        async with self._buffer_lock:
            if not self._insert_buffer:
                return 0
            
            to_insert = self._insert_buffer
            self._insert_buffer = []
        
        return await self.insert_documents_batch(to_insert)
        
    def insert_document(self, content: str, tags: List[str], embedding: List[float]) -> bool:
        """Legacy method - inserts as single chunk"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                content_hash = self.calculate_content_hash(content)
                chunk_id = f"legacy_{content_hash[:12]}"
                
                query = f"""
                    INSERT INTO {self.chunks_table} 
                    (chunk_id, content_hash, source_file, content, tags, chunk_index, total_chunks, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                """
                
                cur.execute(query, (
                    chunk_id, content_hash, "legacy_document", content, json.dumps(tags),
                    0, 1, embedding
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            return False
    
    async def insert_documents_batch(self, records: List[Tuple[str, List[str], List[float]]]) -> int:
        """
        Insert multiple documents efficiently using parallel COPY operations
        """
        if not records:
            return 0
        
        # For very large batches, split into parallel chunks
        if len(records) > 5000:
            return await self._insert_documents_parallel_chunks(records)
        else:
            return await self._insert_documents_single_batch(records)
    
    async def _insert_documents_parallel_chunks(self, records: List[Tuple[str, List[str], List[float]]]) -> int:
        """Insert large batches using parallel chunked processing"""
        logger.info(f"{LOG_PREFIX} Starting parallel chunked insert for {len(records)} records")
        parallel_start = time.time()
        
        chunk_size = 2500  # Optimal chunk size for parallel processing
        chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        
        print(f"{LOG_PREFIX} Split {len(records)} records into {len(chunks)} chunks of size {chunk_size}")
        logger.info(f"{LOG_PREFIX} Created {len(chunks)} parallel chunks (size: {chunk_size})")
        
        # Use semaphore to limit concurrent database connections
        max_concurrent = min(4, self._executor._max_workers)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"{LOG_PREFIX} Using semaphore with {max_concurrent} max concurrent connections")
        
        chunk_times = []
        
        async def insert_chunk_with_semaphore(chunk_index, chunk):
            chunk_start = time.time()
            async with semaphore:
                logger.debug(f"{LOG_PREFIX} Starting chunk {chunk_index+1}/{len(chunks)} ({len(chunk)} records)")
                result = await self._insert_documents_single_batch(chunk)
                chunk_time = time.time() - chunk_start
                chunk_times.append(chunk_time)
                
                logger.debug(f"{LOG_PREFIX} Chunk {chunk_index+1} completed: {result} records in {chunk_time:.2f}s")
                print(f"{LOG_PREFIX} Chunk {chunk_index+1}/{len(chunks)}: {result} records in {chunk_time:.2f}s")
                
                return result
        
        # Process chunks in parallel
        logger.info(f"{LOG_PREFIX} Starting parallel execution of {len(chunks)} chunks")
        tasks = [insert_chunk_with_semaphore(i, chunk) for i, chunk in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        total_inserted = 0
        successful_chunks = 0
        failed_chunks = 0
        
        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                failed_chunks += 1
                logger.error(f"{LOG_PREFIX} Chunk {i+1} insertion failed: {result}")
            elif isinstance(result, int):
                successful_chunks += 1
                total_inserted += result
        
        parallel_time = time.time() - parallel_start
        avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
        records_per_sec = total_inserted / parallel_time if parallel_time > 0 else 0
        
        print(f"{LOG_PREFIX} === PARALLEL INSERT SUMMARY ===")
        print(f"{LOG_PREFIX} Total time: {parallel_time:.2f}s")
        print(f"{LOG_PREFIX} Records inserted: {total_inserted}/{len(records)}")
        print(f"{LOG_PREFIX} Successful chunks: {successful_chunks}/{len(chunks)}")
        print(f"{LOG_PREFIX} Average chunk time: {avg_chunk_time:.2f}s")
        print(f"{LOG_PREFIX} Throughput: {records_per_sec:.1f} records/s")
        
        logger.info(f"{LOG_PREFIX} Parallel batch insert completed: {total_inserted}/{len(records)} records, "
                   f"{successful_chunks}/{len(chunks)} chunks successful, {parallel_time:.2f}s, "
                   f"{records_per_sec:.1f} records/s")
        
        if failed_chunks > 0:
            logger.warning(f"{LOG_PREFIX} {failed_chunks}/{len(chunks)} chunks failed")
            
        return total_inserted
    
    async def _insert_documents_single_batch(self, records: List[Tuple[str, List[str], List[float]]]) -> int:
        """Insert a single batch using optimized COPY operation"""
        logger.debug(f"{LOG_PREFIX} Starting single batch insert for {len(records)} records")
        batch_start = time.time()
        
        def _sync_batch_insert(records_batch):
            sync_start = time.time()
            
            try:
                logger.debug(f"{LOG_PREFIX} Attempting COPY-based batch insert")
                with db_connection.get_sync_connection() as (conn, cur):
                    # Use COPY for fastest insertion
                    import io
                    import csv
                    
                    # Prepare data for COPY
                    prep_start = time.time()
                    output = io.StringIO()
                    writer = csv.writer(output, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    
                    for content, tags, embedding in records_batch:
                        # Format data for COPY
                        tags_json = json.dumps(tags) if isinstance(tags, list) else tags
                        # Convert embedding to PostgreSQL array format
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        writer.writerow([content, tags_json, embedding_str])
                    
                    output.seek(0)
                    prep_time = time.time() - prep_start
                    logger.debug(f"{LOG_PREFIX} Data preparation completed in {prep_time:.3f}s")
                    
                    # Use COPY for bulk insert
                    copy_start = time.time()
                    cur.copy_expert(
                        f"""COPY {self.table_name} (content, tags, embedding) 
                        FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t', QUOTE '"')""",
                        output
                    )
                    copy_time = time.time() - copy_start
                    
                    commit_start = time.time()
                    conn.commit()
                    commit_time = time.time() - commit_start
                    
                    total_sync_time = time.time() - sync_start
                    records_per_sec = len(records_batch) / total_sync_time if total_sync_time > 0 else 0
                    
                    logger.debug(f"{LOG_PREFIX} COPY insert successful: {len(records_batch)} records, "
                               f"prep: {prep_time:.3f}s, copy: {copy_time:.3f}s, commit: {commit_time:.3f}s, "
                               f"total: {total_sync_time:.3f}s, {records_per_sec:.1f} records/s")
                    
                    return len(records_batch)
                    
            except Exception as e:
                fallback_start = time.time()
                logger.warning(f"{LOG_PREFIX} COPY batch insert failed, trying executemany fallback: {e}")
                
                # Fallback to executemany
                try:
                    with db_connection.get_sync_connection() as (conn, cur):
                        insert_data = [
                            (content, json.dumps(tags) if isinstance(tags, list) else tags, embedding)
                            for content, tags, embedding in records_batch
                        ]
                        
                        query = f"INSERT INTO {self.table_name} (content, tags, embedding) VALUES (%s, %s, %s)"
                        cur.executemany(query, insert_data)
                        conn.commit()
                        
                        fallback_time = time.time() - fallback_start
                        records_per_sec = len(records_batch) / fallback_time if fallback_time > 0 else 0
                        
                        logger.info(f"{LOG_PREFIX} Fallback executemany successful: {len(records_batch)} records "
                                   f"in {fallback_time:.3f}s, {records_per_sec:.1f} records/s")
                        
                        return len(records_batch)
                except Exception as fallback_e:
                    logger.error(f"{LOG_PREFIX} Fallback batch insert also failed: {fallback_e}")
                    return 0
        
        # Run sync operation in thread pool
        logger.debug(f"{LOG_PREFIX} Submitting batch insert to thread pool executor")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, _sync_batch_insert, records)
        
        batch_time = time.time() - batch_start
        logger.debug(f"{LOG_PREFIX} Single batch insert completed: {result} records in {batch_time:.3f}s")
        
        return result
    
    def get_document_count(self) -> int:
        """Get total number of chunks"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(f"SELECT COUNT(*) FROM {self.chunks_table}")
                result = cur.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def get_summary_count(self) -> int:
        """Get total number of summaries"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(f"SELECT COUNT(*) FROM {self.summaries_table}")
                result = cur.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get summary count: {e}")
            return 0
    
    # Keep existing methods for backward compatibility
    def similarity_search_database(self, query_embedding, top_k=25, threshold=0.3):
        """Legacy compatibility - searches chunks"""
        return self.similarity_search_chunks(query_embedding, top_k, threshold)
    
    async def similarity_search_database_async(self, query_embedding, top_k=25, threshold=0.3):
        """Legacy compatibility - async wrapper"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.similarity_search_chunks, 
            query_embedding, top_k, threshold
        )
    
    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get chunks by IDs (updated for new schema)"""
        return self.get_related_chunks(chunk_ids)
    
    def get_documents_page(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get documents with pagination"""
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                query = f"""
                    SELECT id, content, tags, created_at
                    FROM {self.table_name}
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (limit, offset))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get documents page: {e}")
            return []
    
    def list_all_documents(self, preview_length: int = 200) -> List[Dict[str, Any]]:
        """List all chunks with content preview"""
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                query = f"""
                    SELECT id, LEFT(content, %s) as content_preview, source_file
                    FROM {self.chunks_table} 
                    ORDER BY source_file, chunk_index
                """
                cur.execute(query, (preview_length,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def get_batch_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including deduplication stats"""
        def _sync_get_metrics():
            try:
                with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                    # Chunks metrics
                    cur.execute(f"""
                        SELECT 
                            COUNT(*) as total_chunks,
                            COUNT(DISTINCT source_file) as unique_files,
                            AVG(total_chunks) as avg_chunks_per_file
                        FROM {self.chunks_table}
                    """)
                    chunks_metrics = dict(cur.fetchone() or {})
                    
                    # Summaries metrics
                    cur.execute(f"""
                        SELECT 
                            COUNT(*) as total_summaries,
                            AVG(importance_score) as avg_importance,
                            AVG(summary_length) as avg_summary_length
                        FROM {self.summaries_table}
                    """)
                    summaries_metrics = dict(cur.fetchone() or {})
                    
                    # Deduplication metrics
                    dedup_stats = self.get_deduplication_stats()
                    
                    return {**chunks_metrics, **summaries_metrics, **dedup_stats}
            except Exception as e:
                logger.error(f"Failed to get batch metrics: {e}")
                return {}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _sync_get_metrics)
    
    async def run_maintenance(self):
        """Enhanced maintenance for all tables including cleanup of duplicates"""
        def _sync_maintenance():
            try:
                with db_connection.get_sync_connection() as (conn, cur):
                    # Analyze all tables
                    cur.execute(f"ANALYZE {self.chunks_table}")
                    cur.execute(f"ANALYZE {self.summaries_table}")
                    cur.execute(f"ANALYZE {self.checksums_table}")
                    
                    # Vacuum all tables
                    cur.execute(f"VACUUM ANALYZE {self.chunks_table}")
                    cur.execute(f"VACUUM ANALYZE {self.summaries_table}")
                    cur.execute(f"VACUUM ANALYZE {self.checksums_table}")
                    
                    conn.commit()
                    logger.info("Enhanced database maintenance with deduplication completed")
                    return True
            except Exception as e:
                logger.error(f"Database maintenance failed: {e}")
                return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _sync_maintenance)

    def optimize_configuration(self, apply_settings: bool = False):
        """Apply PostgreSQL configuration optimizations"""
        optimizations = {
            "shared_buffers": "256MB",
            "work_mem": "16MB",
            "maintenance_work_mem": "512MB",
            "effective_cache_size": "1GB",
            "random_page_cost": "1.1",
            "effective_io_concurrency": "200",
            "max_parallel_workers_per_gather": "4",
            "max_parallel_workers": "8",
            "max_parallel_maintenance_workers": "4",
            "jit": "off"
        }
        
        if not apply_settings:
            logger.info("Recommended PostgreSQL optimizations:")
            for setting, value in optimizations.items():
                logger.info(f"  {setting} = {value}")
            return
        
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                for setting, value in optimizations.items():
                    try:
                        cur.execute(f"ALTER SYSTEM SET {setting} = %s", (value,))
                        logger.info(f"Set {setting} = {value}")
                    except Exception as e:
                        logger.warning(f"Couldn't set {setting}: {e}")
                
                conn.commit()
                logger.info("PostgreSQL configuration applied. Please restart PostgreSQL.")
                    
        except Exception as e:
            logger.error(f"Configuration optimization failed: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        # Flush any remaining buffer
        if hasattr(self, '_insert_buffer') and self._insert_buffer:
            logger.warning(f"Unflushed buffer with {len(self._insert_buffer)} records")
        
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

# Global enhanced database operations instance
db_ops = DatabaseOperations()
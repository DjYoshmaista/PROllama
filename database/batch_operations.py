# database/batch_operations.py - Enhanced Batch Database Operations
import asyncio
import logging
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time

from database.connection import db_connection
from core.config import config
from file_management.chunking import TextChunk
from inference.summarization import ChunkSummary

logger = logging.getLogger(__name__)

class BatchDatabaseOperations:
    """High-performance batch database operations"""
    
    def __init__(self):
        self.table_name = config.database.table_name
        self.summaries_table = f"{self.table_name}_summaries"
        self.chunks_table = f"{self.table_name}_chunks"
        self.checksums_table = f"{self.table_name}_checksums"
        self._executor = ThreadPoolExecutor(max_workers=6)
        
        # Batch configuration
        self.max_batch_size = 100
        self.batch_timeout = 2.0  # seconds
        
        # Connection pooling
        self._connection_pool = []
        self._pool_size = 4
        self._pool_lock = asyncio.Lock()
    
    async def initialize_connection_pool(self):
        """Initialize connection pool for batch operations"""
        async with self._pool_lock:
            if self._connection_pool:
                return
            
            for _ in range(self._pool_size):
                try:
                    conn = psycopg2.connect(config.database.connection_string)
                    conn.autocommit = False  # Use transactions for batch operations
                    self._connection_pool.append(conn)
                except Exception as e:
                    logger.error(f"Failed to create pooled connection: {e}")
            
            logger.info(f"Initialized database connection pool with {len(self._connection_pool)} connections")
    
    async def get_pooled_connection(self):
        """Get a connection from the pool"""
        async with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()
            else:
                # Create new connection if pool is empty
                try:
                    conn = psycopg2.connect(config.database.connection_string)
                    conn.autocommit = False
                    return conn
                except Exception as e:
                    logger.error(f"Failed to create new connection: {e}")
                    return None
    
    async def return_pooled_connection(self, conn):
        """Return a connection to the pool"""
        if conn and not conn.closed:
            async with self._pool_lock:
                if len(self._connection_pool) < self._pool_size:
                    self._connection_pool.append(conn)
                else:
                    conn.close()
    
    async def close_connection_pool(self):
        """Close all pooled connections"""
        async with self._pool_lock:
            for conn in self._connection_pool:
                try:
                    conn.close()
                except:
                    pass
            self._connection_pool.clear()
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash for content deduplication"""
        normalized = ' '.join(content.strip().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    async def insert_chunks_batch(
        self, 
        chunks: List[TextChunk], 
        embeddings: List[List[float]]
    ) -> int:
        """Insert multiple chunks with embeddings in a single batch transaction"""
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return 0
        
        conn = await self.get_pooled_connection()
        if not conn:
            logger.error("No database connection available")
            return 0
        
        try:
            cur = conn.cursor()
            
            # Prepare batch data
            batch_data = []
            content_hashes = set()
            
            for chunk, embedding in zip(chunks, embeddings):
                content_hash = self._calculate_content_hash(chunk.content)
                
                # Skip duplicates within the batch
                if content_hash in content_hashes:
                    continue
                content_hashes.add(content_hash)
                
                batch_data.append((
                    chunk.chunk_id,
                    content_hash,
                    chunk.source_file,
                    chunk.content,
                    json.dumps(getattr(chunk, 'tags', [])),
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
            
            if not batch_data:
                return 0
            
            # Use ON CONFLICT to handle duplicates
            insert_query = f"""
                INSERT INTO {self.chunks_table} (
                    chunk_id, content_hash, source_file, content, tags,
                    chunk_index, total_chunks, start_token, end_token,
                    overlap_before, overlap_after, parent_chunks, child_chunks,
                    embedding
                ) VALUES %s
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
            """
            
            # Execute batch insert
            start_time = time.time()
            psycopg2.extras.execute_values(
                cur, insert_query, batch_data,
                template=None, page_size=self.max_batch_size
            )
            
            inserted_count = cur.rowcount
            conn.commit()
            
            elapsed = time.time() - start_time
            logger.info(f"Batch inserted {inserted_count}/{len(batch_data)} chunks in {elapsed:.2f}s")
            
            return inserted_count
        
        except Exception as e:
            logger.error(f"Batch chunk insert failed: {e}")
            conn.rollback()
            return 0
        
        finally:
            cur.close()
            await self.return_pooled_connection(conn)
    
    async def insert_summaries_batch(
        self,
        summaries: List[ChunkSummary],
        embeddings: List[List[float]]
    ) -> int:
        """Insert multiple summaries with embeddings in a single batch transaction"""
        if not summaries or not embeddings or len(summaries) != len(embeddings):
            return 0
        
        conn = await self.get_pooled_connection()
        if not conn:
            logger.error("No database connection available")
            return 0
        
        try:
            cur = conn.cursor()
            
            # Prepare batch data
            batch_data = []
            summary_hashes = set()
            
            for summary, embedding in zip(summaries, embeddings):
                summary_hash = self._calculate_content_hash(summary.summary)
                
                # Skip duplicates within the batch
                if summary_hash in summary_hashes:
                    continue
                summary_hashes.add(summary_hash)
                
                batch_data.append((
                    summary.chunk_id,
                    summary_hash,
                    summary.source_file,
                    summary.summary,
                    json.dumps(getattr(summary, 'key_topics', [])),
                    json.dumps(getattr(summary, 'chunk_indices', [])),
                    json.dumps(getattr(summary, 'related_chunk_ids', [])),
                    getattr(summary, 'original_length', 0),
                    len(summary.summary),
                    getattr(summary, 'importance_score', 0.0),
                    embedding
                ))
            
            if not batch_data:
                return 0
            
            # Use ON CONFLICT to handle duplicates
            insert_query = f"""
                INSERT INTO {self.summaries_table} (
                    chunk_id, summary_hash, source_file, summary, key_topics,
                    chunk_indices, related_chunk_ids, original_length,
                    summary_length, importance_score, embedding
                ) VALUES %s
                ON CONFLICT (summary_hash) DO NOTHING
                RETURNING id
            """
            
            # Execute batch insert
            start_time = time.time()
            psycopg2.extras.execute_values(
                cur, insert_query, batch_data,
                template=None, page_size=self.max_batch_size
            )
            
            inserted_count = cur.rowcount
            conn.commit()
            
            elapsed = time.time() - start_time
            logger.info(f"Batch inserted {inserted_count}/{len(batch_data)} summaries in {elapsed:.2f}s")
            
            return inserted_count
        
        except Exception as e:
            logger.error(f"Batch summary insert failed: {e}")
            conn.rollback()
            return 0
        
        finally:
            cur.close()
            await self.return_pooled_connection(conn)
    
    async def get_existing_content_hashes(self) -> set:
        """Get all existing content hashes for deduplication"""
        conn = await self.get_pooled_connection()
        if not conn:
            return set()
        
        try:
            cur = conn.cursor()
            
            # Get chunk hashes
            cur.execute(f"SELECT content_hash FROM {self.chunks_table}")
            chunk_hashes = {row[0] for row in cur.fetchall()}
            
            # Get summary hashes
            cur.execute(f"SELECT summary_hash FROM {self.summaries_table}")
            summary_hashes = {row[0] for row in cur.fetchall()}
            
            logger.info(f"Retrieved {len(chunk_hashes)} chunk hashes and {len(summary_hashes)} summary hashes")
            
            return chunk_hashes | summary_hashes
        
        except Exception as e:
            logger.error(f"Failed to get existing hashes: {e}")
            return set()
        
        finally:
            cur.close()
            await self.return_pooled_connection(conn)
    
    async def get_existing_file_checksums(self) -> set:
        """Get checksums of already processed files"""
        conn = await self.get_pooled_connection()
        if not conn:
            return set()
        
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT file_checksum FROM {self.checksums_table}")
            checksums = {row[0] for row in cur.fetchall()}
            
            logger.info(f"Retrieved {len(checksums)} file checksums")
            return checksums
        
        except Exception as e:
            logger.error(f"Failed to get file checksums: {e}")
            return set()
        
        finally:
            cur.close()
            await self.return_pooled_connection(conn)
    
    async def record_file_processed(
        self,
        file_path: str,
        file_checksum: str,
        chunks_created: int,
        summaries_created: int
    ) -> bool:
        """Record that a file has been processed"""
        conn = await self.get_pooled_connection()
        if not conn:
            return False
        
        try:
            cur = conn.cursor()
            
            insert_query = f"""
                INSERT INTO {self.checksums_table} 
                (file_path, file_checksum, chunks_created, summaries_created)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (file_checksum) DO UPDATE SET
                    processed_at = CURRENT_TIMESTAMP,
                    chunks_created = EXCLUDED.chunks_created,
                    summaries_created = EXCLUDED.summaries_created
            """
            
            cur.execute(insert_query, (file_path, file_checksum, chunks_created, summaries_created))
            conn.commit()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to record file processing: {e}")
            conn.rollback()
            return False
        
        finally:
            cur.close()
            await self.return_pooled_connection(conn)
    
    async def bulk_semantic_search(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 25,
        similarity_threshold: float = 0.3
    ) -> List[List[Dict[str, Any]]]:
        """Perform bulk semantic search for multiple query embeddings"""
        if not query_embeddings:
            return []
        
        conn = await self.get_pooled_connection()
        if not conn:
            return []
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            all_results = []
            
            for query_embedding in query_embeddings:
                # Search in summaries first (faster)
                summary_query = f"""
                    SELECT *, (1 - (embedding <=> %s::vector)) as similarity
                    FROM {self.summaries_table}
                    WHERE (1 - (embedding <=> %s::vector)) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """
                
                cur.execute(summary_query, [query_embedding, query_embedding, similarity_threshold, top_k // 2])
                summary_results = cur.fetchall()
                
                # Search in chunks for detailed results
                chunk_query = f"""
                    SELECT *, (1 - (embedding <=> %s::vector)) as similarity
                    FROM {self.chunks_table}
                    WHERE (1 - (embedding <=> %s::vector)) > %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """
                
                cur.execute(chunk_query, [query_embedding, query_embedding, similarity_threshold, top_k])
                chunk_results = cur.fetchall()
                
                # Combine and sort results
                combined_results = list(summary_results) + list(chunk_results)
                combined_results.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Convert to list of dicts and limit
                results = [dict(row) for row in combined_results[:top_k]]
                all_results.append(results)
            
            return all_results
        
        except Exception as e:
            logger.error(f"Bulk semantic search failed: {e}")
            return [[] for _ in query_embeddings]
        
        finally:
            cur.close()
            await self.return_pooled_connection(conn)
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        conn = await self.get_pooled_connection()
        if not conn:
            return {
                'total_chunks': 0,
                'total_summaries': 0,
                'processed_files': 0,
                'unique_files_chunks': 0,
                'unique_files_summaries': 0,
                'total_chunks_created': 0,
                'total_summaries_created': 0,
                'avg_chunks_per_file': 0.0,
                'avg_summaries_per_file': 0.0,
                'chunks_table_size': 'Unknown',
                'summaries_table_size': 'Unknown',
                'checksums_table_size': 'Unknown'
            }
        
        try:
            # Ensure we start with a clean transaction
            conn.rollback()
            cur = conn.cursor()
            
            stats = {}
            
            # Safely check if tables exist first
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN (%s, %s, %s)
            """, (self.chunks_table, self.summaries_table, self.checksums_table))
            
            existing_tables = {row[0] for row in cur.fetchall()}
            
            # Chunks statistics
            if self.chunks_table in existing_tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {self.chunks_table}")
                    stats['total_chunks'] = cur.fetchone()[0]
                    
                    cur.execute(f"SELECT COUNT(DISTINCT source_file) FROM {self.chunks_table}")
                    stats['unique_files_chunks'] = cur.fetchone()[0]
                except Exception as e:
                    logger.warning(f"Failed to get chunks stats: {e}")
                    stats['total_chunks'] = 0
                    stats['unique_files_chunks'] = 0
            else:
                stats['total_chunks'] = 0
                stats['unique_files_chunks'] = 0
            
            # Summaries statistics
            if self.summaries_table in existing_tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {self.summaries_table}")
                    stats['total_summaries'] = cur.fetchone()[0]
                    
                    cur.execute(f"SELECT COUNT(DISTINCT source_file) FROM {self.summaries_table}")
                    stats['unique_files_summaries'] = cur.fetchone()[0]
                except Exception as e:
                    logger.warning(f"Failed to get summaries stats: {e}")
                    stats['total_summaries'] = 0
                    stats['unique_files_summaries'] = 0
            else:
                stats['total_summaries'] = 0
                stats['unique_files_summaries'] = 0
            
            # File processing statistics
            if self.checksums_table in existing_tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {self.checksums_table}")
                    stats['processed_files'] = cur.fetchone()[0]
                    
                    cur.execute(f"""
                        SELECT 
                            COALESCE(SUM(chunks_created), 0) as total_chunks_created,
                            COALESCE(SUM(summaries_created), 0) as total_summaries_created,
                            COALESCE(AVG(chunks_created), 0) as avg_chunks_per_file,
                            COALESCE(AVG(summaries_created), 0) as avg_summaries_per_file
                        FROM {self.checksums_table}
                    """)
                    result = cur.fetchone()
                    if result:
                        stats.update({
                            'total_chunks_created': result[0] or 0,
                            'total_summaries_created': result[1] or 0,
                            'avg_chunks_per_file': float(result[2] or 0),
                            'avg_summaries_per_file': float(result[3] or 0)
                        })
                except Exception as e:
                    logger.warning(f"Failed to get file processing stats: {e}")
                    stats.update({
                        'total_chunks_created': 0,
                        'total_summaries_created': 0,
                        'avg_chunks_per_file': 0.0,
                        'avg_summaries_per_file': 0.0
                    })
                    stats['processed_files'] = 0
            else:
                stats['processed_files'] = 0
                stats.update({
                    'total_chunks_created': 0,
                    'total_summaries_created': 0,
                    'avg_chunks_per_file': 0.0,
                    'avg_summaries_per_file': 0.0
                })
            
            # Database size information (safe fallback)
            try:
                if existing_tables:
                    size_queries = []
                    for table in [self.chunks_table, self.summaries_table, self.checksums_table]:
                        if table in existing_tables:
                            size_queries.append(f"pg_size_pretty(pg_total_relation_size('{table}'))")
                        else:
                            size_queries.append("'0 bytes'")
                    
                    cur.execute(f"SELECT {', '.join(size_queries)}")
                    size_result = cur.fetchone()
                    if size_result:
                        stats.update({
                            'chunks_table_size': size_result[0] if len(size_result) > 0 else '0 bytes',
                            'summaries_table_size': size_result[1] if len(size_result) > 1 else '0 bytes',
                            'checksums_table_size': size_result[2] if len(size_result) > 2 else '0 bytes'
                        })
                else:
                    stats.update({
                        'chunks_table_size': '0 bytes',
                        'summaries_table_size': '0 bytes',
                        'checksums_table_size': '0 bytes'
                    })
            except Exception as e:
                logger.warning(f"Failed to get table sizes: {e}")
                stats.update({
                    'chunks_table_size': 'Unknown',
                    'summaries_table_size': 'Unknown',
                    'checksums_table_size': 'Unknown'
                })
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            # Return safe defaults
            return {
                'total_chunks': 0,
                'total_summaries': 0,
                'processed_files': 0,
                'unique_files_chunks': 0,
                'unique_files_summaries': 0,
                'total_chunks_created': 0,
                'total_summaries_created': 0,
                'avg_chunks_per_file': 0.0,
                'avg_summaries_per_file': 0.0,
                'chunks_table_size': 'Unknown',
                'summaries_table_size': 'Unknown',
                'checksums_table_size': 'Unknown'
            }
        
        finally:
            try:
                if 'cur' in locals():
                    cur.close()
                # Ensure clean state before returning connection
                if conn and not conn.closed:
                    conn.rollback()
                await self.return_pooled_connection(conn)
            except Exception as e:
                logger.warning(f"Error in database stats cleanup: {e}")
    
    async def insert_documents_parallel(self, documents_data: List[Dict[str, Any]]) -> int:
        """
        Compatible method for chunked processor system
        Converts document data to chunk/summary format and inserts
        """
        if not documents_data:
            return 0
        
        logger.info(f"Processing {len(documents_data)} documents for parallel insertion")
        
        try:
            # Convert documents to appropriate format for existing batch methods
            from file_management.chunking import TextChunk
            from inference.summarization import ChunkSummary
            
            chunks = []
            chunk_embeddings = []
            summaries = []
            summary_embeddings = []
            
            for doc_data in documents_data:
                # Create TextChunk object
                chunk_metadata = doc_data.get('metadata', {})
                chunk = TextChunk(
                    content=doc_data['content'],
                    source_file=doc_data['file_path'],
                    chunk_index=chunk_metadata.get('chunk_index', 0),
                    total_chunks=chunk_metadata.get('total_chunks', 1),
                    start_token=chunk_metadata.get('start_token', 0),
                    end_token=chunk_metadata.get('end_token', len(doc_data['content'])),
                    overlap_before=chunk_metadata.get('overlap_before', []),
                    overlap_after=chunk_metadata.get('overlap_after', []),
                    parent_chunks=chunk_metadata.get('parent_chunks', []),
                    child_chunks=chunk_metadata.get('child_chunks', [])
                )
                
                chunks.append(chunk)
                chunk_embeddings.append(doc_data['embedding'])
            
            # Insert chunks using existing batch method
            inserted_count = await self.insert_chunks_batch(chunks, chunk_embeddings)
            
            logger.info(f"Successfully inserted {inserted_count} documents via parallel batch processing")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Parallel document insertion failed: {e}")
            return 0
    
    async def optimize_database(self) -> bool:
        """Optimize database performance with VACUUM and ANALYZE"""
        conn = await self.get_pooled_connection()
        if not conn:
            return False
        
        try:
            # Enable autocommit for VACUUM operations
            conn.autocommit = True
            cur = conn.cursor()
            
            tables = [self.chunks_table, self.summaries_table, self.checksums_table]
            
            for table in tables:
                try:
                    logger.info(f"Optimizing table {table}...")
                    
                    # VACUUM to reclaim space
                    cur.execute(f"VACUUM {table}")
                    
                    # ANALYZE to update statistics
                    cur.execute(f"ANALYZE {table}")
                    
                    logger.info(f"Optimized table {table}")
                
                except Exception as e:
                    logger.error(f"Failed to optimize table {table}: {e}")
            
            return True
        
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
        
        finally:
            conn.autocommit = False  # Reset autocommit
            cur.close()
            await self.return_pooled_connection(conn)

# Global batch database operations instance
batch_db_ops = BatchDatabaseOperations()
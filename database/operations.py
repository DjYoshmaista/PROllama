# database/operations.py - Hybrid Sync/Async Operations
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from database.connection import db_connection
from core.config import config

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """Hybrid database operations - sync for vector queries, async for bulk operations"""
    
    def __init__(self):
        self.table_name = config.database.table_name
        self._executor = ThreadPoolExecutor(max_workers=4)  # For running sync operations in async context
    
    def initialize_schema(self):
        """Initialize database schema with migration support"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                # Create documents table with JSONB tags
                create_table_query = f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        tags JSONB DEFAULT '[]'::jsonb,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        embedding VECTOR(1024)
                    )
                """
                cur.execute(create_table_query)
                
                # Check if tags column exists and migrate if needed
                cur.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{self.table_name}' 
                    AND column_name = 'tags'
                """)
                
                if not cur.fetchone():
                    logger.info("Migrating tags column to JSONB")
                    cur.execute(f"ALTER TABLE {self.table_name} ADD COLUMN tags JSONB DEFAULT '[]'::jsonb")
                
                # Create index for vector similarity search
                index_query = f"""
                    CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                    ON {self.table_name}
                    USING hnsw (embedding vector_cosine_ops)
                """
                cur.execute(index_query)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def insert_document(self, content: str, tags: List[str], embedding: List[float]) -> bool:
        """Insert a single document"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                query = f"INSERT INTO {self.table_name} (content, tags, embedding) VALUES (%s, %s, %s)"
                cur.execute(query, (content, json.dumps(tags), embedding))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            return False
    
    async def insert_documents_batch(self, records: List[Tuple[str, List[str], List[float]]]) -> int:
        """Insert multiple documents efficiently using sync operations in thread pool"""
        if not records:
            return 0
        
        def _sync_batch_insert(records_batch):
            try:
                with db_connection.get_sync_connection() as (conn, cur):
                    # Prepare data for insertion
                    insert_data = [
                        (content, json.dumps(tags) if isinstance(tags, list) else tags, embedding)
                        for content, tags, embedding in records_batch
                    ]
                    
                    # Use executemany for batch insert
                    query = f"INSERT INTO {self.table_name} (content, tags, embedding) VALUES (%s, %s, %s)"
                    cur.executemany(query, insert_data)
                    conn.commit()
                    return len(records_batch)
            except Exception as e:
                logger.error(f"Sync batch insert failed: {e}")
                return 0
        
        # Run sync operation in thread pool to maintain async interface
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _sync_batch_insert, records)
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                result = cur.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def similarity_search_database(
        self, 
        query_embedding: List[float], 
        top_k: int = 25, 
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search directly in the database using vector operations
        This uses sync connections which work reliably with pgvector
        """
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                # Convert embedding to string format for PostgreSQL vector type
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # Use PostgreSQL's vector similarity directly
                # This leverages the HNSW index for fast similarity search
                query = f"""
                    SELECT 
                        id,
                        content,
                        tags,
                        1 - (embedding <=> '{embedding_str}'::vector) as cosine_similarity
                    FROM {self.table_name}
                    WHERE 1 - (embedding <=> '{embedding_str}'::vector) >= %s
                    ORDER BY embedding <=> '{embedding_str}'::vector
                    LIMIT %s
                """
                
                cur.execute(query, (threshold, top_k))
                results = cur.fetchall()
                
                # Convert to list of dictionaries
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Database similarity search failed: {e}")
            return []
    
    async def similarity_search_database_async(
        self, 
        query_embedding: List[float], 
        top_k: int = 25, 
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Async wrapper for similarity search - runs sync operation in thread pool
        This avoids asyncpg/pgvector compatibility issues
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.similarity_search_database, 
            query_embedding, 
            top_k, 
            threshold
        )
    
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
    
    def get_embeddings_for_cache(self, limit: int = 10000) -> Optional[Tuple[List[int], List[List[float]]]]:
        """
        Get embeddings for caching - only load if database is small enough
        Returns None if database is too large for caching
        """
        try:
            # Check total count first
            total_docs = self.get_document_count()
            if total_docs > limit:
                logger.info(f"Database too large for caching ({total_docs} > {limit})")
                return None
            
            with db_connection.get_sync_connection() as (conn, cur):
                query = f"SELECT id, embedding FROM {self.table_name} ORDER BY id LIMIT %s"
                cur.execute(query, (limit,))
                
                ids = []
                embeddings = []
                
                for row in cur.fetchall():
                    doc_id, embedding = row
                    if embedding is None:
                        continue
                    
                    # Handle different embedding formats from pgvector
                    try:
                        if isinstance(embedding, str):
                            # Parse string representation of vector
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
    
    def get_documents_by_ids(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """Get documents by their IDs"""
        if not doc_ids:
            return []
        
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                placeholders = ','.join(['%s'] * len(doc_ids))
                query = f"SELECT id, content, tags FROM {self.table_name} WHERE id IN ({placeholders})"
                cur.execute(query, doc_ids)
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return []
    
    def list_all_documents(self, preview_length: int = 200) -> List[Dict[str, Any]]:
        """List all documents with content preview"""
        try:
            with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                query = f"SELECT id, LEFT(content, %s) as content_preview FROM {self.table_name} ORDER BY id"
                cur.execute(query, (preview_length,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def get_batch_metrics(self) -> Dict[str, Any]:
        """Get embedding quality metrics using sync in thread pool"""
        def _sync_get_metrics():
            try:
                with db_connection.get_sync_connection(dict_cursor=True) as (conn, cur):
                    cur.execute(f"""
                        SELECT 
                            COUNT(*) as total_vectors,
                            COUNT(*) FILTER (WHERE embedding IS NULL) as null_embeddings
                        FROM {self.table_name}
                    """)
                    result = cur.fetchone()
                    return dict(result) if result else {}
            except Exception as e:
                logger.error(f"Failed to get batch metrics: {e}")
                return {}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _sync_get_metrics)
    
    async def run_maintenance(self):
        """Run database maintenance operations using sync in thread pool"""
        def _sync_maintenance():
            try:
                with db_connection.get_sync_connection() as (conn, cur):
                    # Analyze to update statistics
                    cur.execute(f"ANALYZE {self.table_name}")
                    # Vacuum to reclaim space  
                    cur.execute(f"VACUUM ANALYZE {self.table_name}")
                    conn.commit()
                    logger.info("Database maintenance completed")
                    return True
            except Exception as e:
                logger.error(f"Database maintenance failed: {e}")
                return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _sync_maintenance)
    
    def optimize_configuration(self, postgresql_conf_path: str = "postgresql.conf"):
        """Apply PostgreSQL configuration optimizations"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                try:
                    with open(postgresql_conf_path, "r") as conf_file:
                        for line in conf_file:
                            if line.strip() and not line.startswith("#"):
                                parts = line.split("=", 1)
                                if len(parts) == 2:
                                    setting = parts[0].strip()
                                    value = parts[1].strip().replace("'", "")
                                    try:
                                        cur.execute(f"ALTER SYSTEM SET {setting} = %s", (value,))
                                    except Exception as e:
                                        logger.warning(f"Couldn't set {setting}: {e}")
                    
                    conn.commit()
                    logger.info("PostgreSQL configuration applied. Please restart PostgreSQL.")
                    
                except FileNotFoundError:
                    logger.warning(f"Configuration file {postgresql_conf_path} not found")
                    
        except Exception as e:
            logger.error(f"Configuration optimization failed: {e}")

# Global database operations instance
db_ops = DatabaseOperations()
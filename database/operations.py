# database/operations.py
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from database.connection import db_connection
from core.config import config

logger = logging.getLogger(__name__)

class DatabaseOperations:
    """Centralized database operations"""
    
    def __init__(self):
        self.table_name = config.database.table_name
    
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
        """Insert multiple documents efficiently"""
        if not records:
            return 0
        
        try:
            async with db_connection.get_async_connection() as conn:
                # Prepare data for insertion
                insert_data = [
                    (content, json.dumps(tags) if isinstance(tags, list) else tags, embedding)
                    for content, tags, embedding in records
                ]
                
                try:
                    # Try with tags column
                    await conn.executemany(
                        f"INSERT INTO {self.table_name} (content, tags, embedding) VALUES ($1, $2::jsonb, $3)",
                        insert_data
                    )
                except Exception:
                    # Fallback for missing tags column
                    await conn.execute(f"ALTER TABLE {self.table_name} ADD COLUMN tags JSONB DEFAULT '[]'::jsonb")
                    await conn.executemany(
                        f"INSERT INTO {self.table_name} (content, tags, embedding) VALUES ($1, $2::jsonb, $3)",
                        insert_data
                    )
                
                return len(records)
                
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return 0
    
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
    
    def get_embeddings_page(self, limit: int = 1000, offset: int = 0) -> List[Tuple[int, List[float]]]:
        """Get embeddings with pagination for similarity search"""
        try:
            with db_connection.get_sync_connection() as (conn, cur):
                query = f"""
                    SELECT id, embedding
                    FROM {self.table_name}
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """
                cur.execute(query, (limit, offset))
                results = []
                
                for row in cur.fetchall():
                    doc_id, embedding = row
                    if embedding is None:
                        continue
                    
                    # Handle different embedding formats from pgvector
                    if isinstance(embedding, str):
                        try:
                            if embedding.startswith('[') and embedding.endswith(']'):
                                embedding = [float(x) for x in embedding[1:-1].split(',')]
                            else:
                                embedding = json.loads(embedding)
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.error(f"Failed to parse embedding for ID {doc_id}: {e}")
                            continue
                    elif isinstance(embedding, (list, tuple)):
                        embedding = list(embedding)
                    else:
                        try:
                            embedding = list(embedding)
                        except Exception as e:
                            logger.error(f"Cannot convert embedding to list for ID {doc_id}: {e}")
                            continue
                    
                    results.append((doc_id, embedding))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get embeddings page: {e}")
            return []
    
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
        """Get embedding quality metrics"""
        try:
            async with db_connection.get_async_connection() as conn:
                metrics = await conn.fetch(f"""
                    SELECT 
                        AVG(vector_norm(embedding)) AS avg_norm,
                        COUNT(*) FILTER (WHERE vector_norm(embedding) = 0) AS zero_vectors,
                        COUNT(*) AS total_vectors
                    FROM {self.table_name}
                """)
                return dict(metrics[0]) if metrics else {}
        except Exception as e:
            logger.error(f"Failed to get batch metrics: {e}")
            return {}
    
    async def run_maintenance(self):
        """Run database maintenance operations"""
        try:
            async with db_connection.get_async_connection() as conn:
                # Analyze to update statistics
                await conn.execute(f"ANALYZE {self.table_name}")
                # Vacuum to reclaim space
                await conn.execute(f"VACUUM (VERBOSE, ANALYZE) {self.table_name}")
                logger.info("Database maintenance completed")
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
    
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
# database/schema_init.py - Database Schema Initialization
import logging
import psycopg2
from core.config import config

logger = logging.getLogger(__name__)

def initialize_database_schema():
    """Initialize the database schema with proper error handling"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            dbname=config.database.name,
            user=config.database.user,
            password=config.database.password,
            host=config.database.host
        )
        
        cur = conn.cursor()
        
        # Ensure pgvector extension is available
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension verified/created")
        except Exception as e:
            logger.warning(f"Could not create vector extension: {e}")
        
        # Create chunks table
        chunks_table = f"{config.database.table_name}_chunks"
        create_chunks_sql = f"""
            CREATE TABLE IF NOT EXISTS {chunks_table} (
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
        cur.execute(create_chunks_sql)
        
        # Create summaries table
        summaries_table = f"{config.database.table_name}_summaries"
        create_summaries_sql = f"""
            CREATE TABLE IF NOT EXISTS {summaries_table} (
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
        cur.execute(create_summaries_sql)
        
        # Create checksums table
        checksums_table = f"{config.database.table_name}_checksums"
        create_checksums_sql = f"""
            CREATE TABLE IF NOT EXISTS {checksums_table} (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_checksum VARCHAR(32) UNIQUE NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunks_created INTEGER DEFAULT 0,
                summaries_created INTEGER DEFAULT 0
            )
        """
        cur.execute(create_checksums_sql)
        
        # Create indexes for performance
        try:
            cur.execute(f"CREATE INDEX IF NOT EXISTS {chunks_table}_embedding_idx ON {chunks_table} USING hnsw (embedding vector_cosine_ops)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {summaries_table}_embedding_idx ON {summaries_table} USING hnsw (embedding vector_cosine_ops)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {chunks_table}_source_file_idx ON {chunks_table} (source_file)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {summaries_table}_source_file_idx ON {summaries_table} (source_file)")
        except Exception as e:
            logger.warning(f"Could not create some indexes: {e}")
        
        # Commit changes
        conn.commit()
        logger.info("Database schema initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        if conn:
            conn.rollback()
        return False
    
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    initialize_database_schema()
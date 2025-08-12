# database/connection.py
import asyncio
import psycopg2
import asyncpg
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, Dict, Any
from psycopg2.extras import RealDictCursor
import pgvector.asyncpg
from core.config import config

logger = logging.getLogger(__name__)

# PostgreSQL optimization settings
PG_OPTIMIZATION_SETTINGS = {
    "statement_timeout": "300000",  # 5 minutes
    "work_mem": "16MB",
    "maintenance_work_mem": "512MB",
    "jit": "off"
}

class DatabaseConnection:
    """Centralized database connection management"""
    
    def __init__(self):
        self._sync_conn_params = {
            'dbname': config.database.name,
            'user': config.database.user,
            'password': config.database.password,
            'host': config.database.host,
            'connect_timeout': 30
        }
        self._async_pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()
    
    @contextmanager
    def get_sync_connection(self, dict_cursor=False):
        """Get synchronous database connection"""
        conn = None
        try:
            conn = psycopg2.connect(**self._sync_conn_params)
            cursor_factory = RealDictCursor if dict_cursor else None
            cur = conn.cursor(cursor_factory=cursor_factory)
            yield conn, cur
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    async def get_async_pool(self, min_size=2, max_size=8) -> asyncpg.Pool:
        """Get or create async connection pool"""
        if self._async_pool is None or self._async_pool.is_closing():
            async with self._pool_lock:
                if self._async_pool is None or self._async_pool.is_closing():
                    self._async_pool = await asyncpg.create_pool(
                        config.database.connection_string,
                        min_size=min_size,
                        max_size=max_size,
                        timeout=180,
                        server_settings=PG_OPTIMIZATION_SETTINGS
                    )
                    logger.info(f"Created async connection pool (size: {min_size}-{max_size})")
        
        return self._async_pool
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async database connection from pool"""
        pool = await self.get_async_pool()
        conn = None
        try:
            conn = await pool.acquire()
            # Register vector type with asyncpg - this is crucial for pgvector
            try:
                await pgvector.asyncpg.register_vector(conn)
            except Exception as vector_e:
                logger.warning(f"Could not register vector type: {vector_e}")
                # If vector registration fails, this connection won't work for vector ops
                raise Exception(f"Vector type registration failed: {vector_e}")
            
            # Apply optimization settings one by one to avoid parameter issues
            try:
                await conn.execute("SET statement_timeout = '300000'")
                await conn.execute("SET work_mem = '16MB'") 
                await conn.execute("SET maintenance_work_mem = '512MB'")
                await conn.execute("SET jit = off")
            except Exception as setting_e:
                logger.debug(f"Could not apply all settings: {setting_e}")
            
            yield conn
        except Exception as e:
            logger.error(f"Async connection error: {e}")
            raise
        finally:
            if conn:
                await pool.release(conn)
    
    async def close_async_pool(self):
        """Close async connection pool"""
        if self._async_pool:
            await self._async_pool.close()
            self._async_pool = None
            logger.info("Async connection pool closed")
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_sync_connection() as (conn, cur):
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def test_async_connection(self) -> bool:
        """Test async database connectivity"""
        try:
            async with self.get_async_connection() as conn:
                # Use a simple query without parameters to avoid asyncpg issues
                result = await conn.fetchval("SELECT 1 as test")
                return result == 1
        except Exception as e:
            logger.error(f"Async connection test failed: {e}")
            return False

# Global database connection manager
db_connection = DatabaseConnection()
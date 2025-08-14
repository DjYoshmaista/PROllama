# db.py
import psycopg2
import asyncpg
import pgvector.asyncpg
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor
from config import Config
import logging

logger = logging.getLogger(__name__)

# Connection string for async connections
DB_CONN_STRING = f"postgres://{Config.DB_CONFIG['user']}:{Config.DB_CONFIG['password']}@{Config.DB_CONFIG['host']}/{Config.DB_CONFIG['dbname']}"

class DatabaseManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.async_pool = None
            self._initialized = True
    
    async def get_async_pool(self):
        """Get or create async connection pool"""
        if self.async_pool is None:
            self.async_pool = await asyncpg.create_pool(
                DB_CONN_STRING,
                min_size=2,
                max_size=10,
                timeout=180,
                server_settings=Config.PG_OPTIMIZATION_SETTINGS
            )
            # Register vector extension for all connections in pool
            async with self.async_pool.acquire() as conn:
                await pgvector.asyncpg.register_vector(conn)
        return self.async_pool
    
    async def get_async_connection(self):
        """Get a single async connection"""
        pool = await self.get_async_pool()
        return pool.acquire()
    
    @staticmethod
    @contextmanager
    def get_sync_cursor():
        """Get synchronous cursor with proper cleanup"""
        conn = cur = None
        try:
            conn = psycopg2.connect(
                dbname=Config.DB_CONFIG['dbname'],
                user=Config.DB_CONFIG['user'],
                password=Config.DB_CONFIG['password'],
                host=Config.DB_CONFIG['host'],
                connect_timeout=30
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)
            yield (conn, cur)
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    
    async def close_pools(self):
        """Close all connection pools"""
        if self.async_pool:
            await self.async_pool.close()
            self.async_pool = None

# Global database manager instance
db_manager = DatabaseManager()
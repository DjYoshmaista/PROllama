# db.py
import psycopg2
import asyncpg
import pgvector.asyncpg
from contextlib import contextmanager, asynccontextmanager
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
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async connection as context manager"""
        pool = await self.get_async_pool()
        async with pool.acquire() as conn:
            yield conn
    
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
            conn.commit()  # Auto-commit on success
        except Exception as e:
            if conn:
                conn.rollback()
            raise
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

    def test_connection(self):
        """Test if we can connect to the database"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.close()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

# This is the missing function your code needs!
@contextmanager
def db_cursor():
    """Convenience function that your existing code expects"""
    with db_manager.get_sync_cursor() as (conn, cur):
        yield conn, cur

def test_db_connection():
    """Test database connectivity"""
    return db_manager.test_connection()

# Database initialization functions
def init_db():
    """Initialize database with proper error handling"""
    try:
        # First test the connection
        if not test_db_connection():
            raise Exception("Cannot connect to database")
        
        logger.info("Initializing database schema...")
        
        with db_cursor() as (conn, cur):
            table_name = Config.TABLE_NAME
            
            # Enable vector extension
            logger.info("Enabling pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table with vector column
            logger.info(f"Creating table {table_name}...")
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    tags TEXT[] DEFAULT '[]'::text[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1024)
                )
            """
            cur.execute(create_table_query)
            
            # Check and add tags column if it doesn't exist (migration)
            logger.info("Checking for schema migrations...")
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name = 'tags'
            """, (table_name,))
            
            if not cur.fetchone():
                logger.info("Adding tags column...")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN tags TEXT[] DEFAULT '[]'::text[]")
            
            # Create vector index
            logger.info("Creating vector index...")
            index_query = f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                ON {table_name}
                USING hnsw (embedding vector_cosine_ops)
            """
            cur.execute(index_query)
            
            # Check table was created successfully
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_name = %s
            """, (table_name,))
            
            result = cur.fetchone()
            if result and result['count'] > 0:
                logger.info(f"Database initialized successfully. Table {table_name} is ready.")
                return True
            else:
                raise Exception(f"Table {table_name} was not created properly")
                
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL error during initialization: {e}")
        logger.error(f"Error code: {e.pgcode if hasattr(e, 'pgcode') else 'Unknown'}")
        raise Exception(f"Database initialization failed: {e}")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise Exception(f"Database initialization failed: {e}")

def get_db_stats():
    """Get basic database statistics for debugging"""
    try:
        with db_cursor() as (conn, cur):
            table_name = Config.TABLE_NAME
            
            # Check if table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                return {"table_exists": False, "record_count": 0}
            
            # Get record count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cur.fetchone()[0]
            
            # Get table size
            cur.execute(f"""
                SELECT pg_size_pretty(pg_total_relation_size('{table_name}')) as size
            """)
            size = cur.fetchone()[0]
            
            return {
                "table_exists": True,
                "record_count": count,
                "table_size": size,
                "table_name": table_name
            }
            
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {"error": str(e)}

def diagnose_db_issues():
    """Comprehensive database diagnostics"""
    print("\n" + "="*60)
    print("DATABASE DIAGNOSTICS")
    print("="*60)
    
    # Test basic connection
    print("1. Testing basic connection...")
    if test_db_connection():
        print("   ✅ Basic connection: SUCCESS")
    else:
        print("   ❌ Basic connection: FAILED")
        print("   Check your database configuration in config.py")
        return False
    
    # Test vector extension
    print("2. Testing pgvector extension...")
    try:
        with db_cursor() as (conn, cur):
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone():
                print("   ✅ pgvector extension: INSTALLED")
            else:
                print("   ⚠️  pgvector extension: NOT INSTALLED")
                print("   Run: CREATE EXTENSION vector; in your database")
    except Exception as e:
        print(f"   ❌ pgvector check failed: {e}")
    
    # Get database stats
    print("3. Getting database statistics...")
    stats = get_db_stats()
    if "error" in stats:
        print(f"   ❌ Stats failed: {stats['error']}")
    else:
        print(f"   📊 Table exists: {stats['table_exists']}")
        if stats['table_exists']:
            print(f"   📊 Record count: {stats['record_count']}")
            print(f"   📊 Table size: {stats['table_size']}")
    
    # Test permissions
    print("4. Testing database permissions...")
    try:
        with db_cursor() as (conn, cur):
            # Test CREATE permission
            cur.execute("CREATE TABLE IF NOT EXISTS test_permissions (id SERIAL)")
            cur.execute("DROP TABLE test_permissions")
            print("   ✅ CREATE/DROP permissions: OK")
            
            # Test INSERT permission on main table if it exists
            if stats.get('table_exists'):
                table_name = Config.TABLE_NAME
                print(f"   ✅ Table {table_name}: ACCESSIBLE")
            
    except Exception as e:
        print(f"   ❌ Permission test failed: {e}")
    
    print("="*60)
    return True
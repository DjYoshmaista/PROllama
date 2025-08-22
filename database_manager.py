# database_manager.py - Unified Database Operations
import psycopg2
import asyncpg
import pgvector.asyncpg
import logging
import time
import json
import os
import sys
import traceback
from contextlib import contextmanager, asynccontextmanager
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, List, Optional, Tuple, Union

# CRITICAL: Set up immediate diagnostic logging
logger = logging.getLogger(__name__)
logger.info("🚀 database_manager.py module loading...")

# Check if config is available
try:
    logger.info("📊 Attempting to import config...")
    from core_config import config
    logger.info("✅ Config imported successfully")
    logger.info(f"   Config type: {type(config)}")
    logger.info(f"   Config id: {id(config)}")
except Exception as config_import_error:
    logger.error(f"❌ Config import failed: {config_import_error}")
    logger.debug(f"Config import traceback: {traceback.format_exc()}")
    raise

# Immediate environment diagnostic at module load
logger.info("🔍 IMMEDIATE ENVIRONMENT DIAGNOSTIC:")
env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
for var in env_vars:
    value = os.environ.get(var)
    logger.info(f"   {var}: {value if value else 'NOT_SET'}")

# Test config access immediately
try:
    logger.info("🔍 IMMEDIATE CONFIG ACCESS TEST:")
    logger.info(f"   config.DB_NAME: {getattr(config, 'DB_NAME', 'NOT_FOUND')}")
    logger.info(f"   config.DB_USER: {getattr(config, 'DB_USER', 'NOT_FOUND')}")
    logger.info(f"   config.DB_HOST: {getattr(config, 'DB_HOST', 'NOT_FOUND')}")
    logger.info(f"   config.DB_PORT: {getattr(config, 'DB_PORT', 'NOT_FOUND')}")
    
    # Test DB_CONFIG property
    try:
        test_db_config = config.DB_CONFIG
        logger.info(f"   config.DB_CONFIG: {test_db_config}")
    except Exception as db_config_error:
        logger.error(f"   config.DB_CONFIG failed: {db_config_error}")
        
except Exception as config_test_error:
    logger.error(f"❌ Immediate config test failed: {config_test_error}")
    logger.debug(f"Config test traceback: {traceback.format_exc()}")

# Enable debug logging for psycopg2
logging.getLogger('psycopg2').setLevel(logging.DEBUG)

class DatabaseManager:
    """Unified database manager with both sync and async support"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            logger.debug("🔧 Creating new DatabaseManager instance")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            logger.debug("✅ DatabaseManager instance created")
        else:
            logger.debug("♻️ Returning existing DatabaseManager instance")
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("🚀 DatabaseManager.__init__() called - Starting initialization")
            logger.info(f"   Process ID: {os.getpid()}")
            logger.info(f"   Current working directory: {os.getcwd()}")
            logger.info(f"   Python version: {sys.version}")
            
            self.async_pool = None
            
            # CRITICAL: Check if we're getting the same config instance
            logger.info("🔍 CONFIG INSTANCE VERIFICATION:")
            logger.info(f"   Global config id: {id(config)}")
            logger.info(f"   Global config type: {type(config)}")
            logger.info(f"   Config _initialized: {getattr(config, '_initialized', 'NOT_FOUND')}")
            
            # Step 1: Environment variable diagnostic
            logger.info("📊 STEP 1: Environment Variable Diagnostic")
            env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'TABLE_NAME']
            for var in env_vars:
                value = os.environ.get(var)
                logger.info(f"   🔍 {var}: {value if value else 'NOT_SET'} (type: {type(value)})")
                
            # Step 1.5: Force reload environment variables
            logger.info("📊 STEP 1.5: Force reload environment variables")
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                logger.info("   ✅ Environment variables reloaded")
                
                # Check again after reload
                for var in env_vars:
                    value = os.environ.get(var)
                    logger.info(f"   🔄 {var} after reload: {value if value else 'NOT_SET'}")
                    
            except Exception as reload_error:
                logger.error(f"   ❌ Environment reload failed: {reload_error}")
            
            # Step 2: Config object diagnostic with method testing
            logger.info("📊 STEP 2: Config Object Diagnostic")
            try:
                logger.debug(f"   🔍 Config object type: {type(config)}")
                logger.debug(f"   🔍 Config object id: {id(config)}")
                
                # Test if config methods work
                logger.info("   📊 SUB-STEP 2.1: Testing config methods")
                
                # Test individual attributes
                attrs_to_test = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
                for attr in attrs_to_test:
                    try:
                        value = getattr(config, attr, 'NOT_FOUND')
                        logger.info(f"      🔍 config.{attr}: {value} (type: {type(value)})")
                    except Exception as attr_error:
                        logger.error(f"      ❌ config.{attr} failed: {attr_error}")
                
                # Test DB_CONFIG property specifically
                logger.info("   📊 SUB-STEP 2.2: Testing DB_CONFIG property")
                try:
                    db_config_test = config.DB_CONFIG
                    logger.info(f"      ✅ config.DB_CONFIG accessible")
                    logger.info(f"      📋 DB_CONFIG content: {db_config_test}")
                    
                    # Verify each key in DB_CONFIG
                    expected_keys = ['dbname', 'user', 'password', 'host', 'port']
                    for key in expected_keys:
                        if key in db_config_test:
                            value = db_config_test[key]
                            if key == 'password':
                                logger.info(f"         🔑 {key}: ***HIDDEN*** (type: {type(value)}, length: {len(str(value)) if value else 0})")
                            else:
                                logger.info(f"         🔑 {key}: {value} (type: {type(value)})")
                        else:
                            logger.error(f"         ❌ Missing key: {key}")
                            
                except Exception as db_config_error:
                    logger.error(f"      ❌ config.DB_CONFIG failed: {db_config_error}")
                    logger.debug(f"      📋 DB_CONFIG error traceback: {traceback.format_exc()}")
                    
                    # Try to diagnose the DB_CONFIG issue
                    logger.error("      🚨 DB_CONFIG DIAGNOSTIC:")
                    try:
                        logger.error(f"         config.get method exists: {hasattr(config, 'get')}")
                        logger.error(f"         config._db_config exists: {hasattr(config, '_db_config')}")
                        logger.error(f"         config.runtime_config exists: {hasattr(config, 'runtime_config')}")
                        
                        # Try to get individual components
                        logger.error(f"         Individual DB vars from config:")
                        logger.error(f"            DB_NAME: {getattr(config, 'DB_NAME', 'NOT_FOUND')}")
                        logger.error(f"            DB_USER: {getattr(config, 'DB_USER', 'NOT_FOUND')}")
                        logger.error(f"            DB_HOST: {getattr(config, 'DB_HOST', 'NOT_FOUND')}")
                        logger.error(f"            DB_PORT: {getattr(config, 'DB_PORT', 'NOT_FOUND')}")
                        
                    except Exception as diag_error:
                        logger.error(f"         Diagnostic failed: {diag_error}")
                    
                    raise
                
            except Exception as e:
                logger.error(f"   ❌ Config diagnostic failed: {e}")
                logger.debug(f"   📋 Config diagnostic traceback: {traceback.format_exc()}")
                raise
            
            # Step 3: Manual DB_CONFIG construction as fallback
            logger.info("📊 STEP 3: Manual DB_CONFIG construction test")
            try:
                manual_db_config = {
                    'dbname': getattr(config, 'DB_NAME', os.environ.get('DB_NAME', 'rag_db')),
                    'user': getattr(config, 'DB_USER', os.environ.get('DB_USER', 'postgres')),
                    'password': getattr(config, 'DB_PASSWORD', os.environ.get('DB_PASSWORD', 'postgres')),
                    'host': getattr(config, 'DB_HOST', os.environ.get('DB_HOST', 'localhost')),
                    'port': getattr(config, 'DB_PORT', os.environ.get('DB_PORT', '5432'))
                }
                
                logger.info(f"   ✅ Manual DB_CONFIG constructed: {dict(manual_db_config, password='***HIDDEN***')}")
                
                # Validate manual config
                for key, value in manual_db_config.items():
                    if value is None or str(value).strip() == '':
                        logger.error(f"      ❌ Manual config {key} is empty: {value}")
                    else:
                        logger.info(f"      ✅ Manual config {key} is valid")
                        
            except Exception as manual_error:
                logger.error(f"   ❌ Manual DB_CONFIG construction failed: {manual_error}")
            
            # Step 4: Connection string creation
            logger.info("📊 STEP 4: Connection String Creation")
            try:
                logger.debug("   🔄 Calling config.get_db_connection_string()...")
                self._connection_string = config.get_db_connection_string()
                logger.info(f"   ✅ Connection string created: {self._connection_string}")
                
            except Exception as e:
                logger.error(f"   ❌ Connection string creation failed: {e}")
                logger.debug(f"   📋 Connection string traceback: {traceback.format_exc()}")
                
                # Try manual connection string construction
                try:
                    logger.info("   🔄 Attempting manual connection string construction...")
                    manual_conn_str = f"postgresql://{manual_db_config['user']}:{manual_db_config['password']}@{manual_db_config['host']}:{manual_db_config['port']}/{manual_db_config['dbname']}"
                    self._connection_string = manual_conn_str
                    logger.info(f"   ✅ Manual connection string: {manual_conn_str}")
                except Exception as manual_conn_error:
                    logger.error(f"   ❌ Manual connection string failed: {manual_conn_error}")
                    raise
            
            # Step 5: DB_CONFIG retrieval with fallback
            logger.info("📊 STEP 5: DB_CONFIG Retrieval with Fallback")
            try:
                logger.debug("   🔄 Calling config.DB_CONFIG...")
                self._db_config = config.DB_CONFIG
                logger.info(f"   ✅ DB_CONFIG retrieved successfully")
                
            except Exception as e:
                logger.error(f"   ❌ DB_CONFIG retrieval failed: {e}")
                logger.debug(f"   📋 DB_CONFIG traceback: {traceback.format_exc()}")
                
                # Use manual config as fallback
                logger.info("   🔄 Using manual DB_CONFIG as fallback...")
                self._db_config = manual_db_config
                logger.info(f"   ✅ Using fallback DB_CONFIG")
            
            # Final validation
            logger.info("📊 STEP 6: Final Configuration Validation")
            logger.info(f"   📋 Final DB_CONFIG contents:")
            for key, value in self._db_config.items():
                if key == 'password':
                    display_value = '***HIDDEN***' if value else 'NOT_SET'
                else:
                    display_value = value
                logger.info(f"      🔑 {key}: {display_value} (type: {type(value)})")
            
            self._initialized = True
            logger.info("🎉 DatabaseManager initialization completed successfully")
        else:
            logger.debug("♻️ DatabaseManager already initialized, skipping")
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for psycopg2 with comprehensive step-by-step logging"""
        logger.debug("🔧 connection_params() called")
        
        try:
            # Step 1: Extract raw parameters
            logger.debug("   📊 STEP 1: Extracting raw parameters from DB_CONFIG")
            logger.debug(f"      🔍 self._db_config type: {type(self._db_config)}")
            logger.debug(f"      🔍 self._db_config id: {id(self._db_config)}")
            
            raw_params = {}
            for key in ['dbname', 'user', 'password', 'host', 'port']:
                try:
                    value = self._db_config.get(key)
                    raw_params[key] = value
                    logger.debug(f"      🔑 Raw {key}: {value} (type: {type(value)})")
                except Exception as e:
                    logger.error(f"      ❌ Failed to get {key}: {e}")
                    raw_params[key] = None
            
            # Step 2: Validate required parameters exist
            logger.debug("   📊 STEP 2: Validating required parameters exist")
            missing_params = []
            for key, value in raw_params.items():
                if value is None:
                    logger.error(f"      ❌ Missing parameter: {key}")
                    missing_params.append(key)
                else:
                    logger.debug(f"      ✅ Parameter {key} exists")
            
            if missing_params:
                error_msg = f"Missing required database parameters: {missing_params}"
                logger.error(f"   ❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Step 3: Type conversion and validation
            logger.debug("   📊 STEP 3: Type conversion and validation")
            params = {}
            
            # Handle string parameters
            string_params = ['dbname', 'user', 'password', 'host']
            for key in string_params:
                logger.debug(f"      🔄 Processing string parameter: {key}")
                value = raw_params[key]
                
                if value is None:
                    logger.error(f"         ❌ {key} is None")
                    raise ValueError(f"Parameter {key} cannot be None")
                
                str_value = str(value).strip()
                if not str_value:
                    logger.error(f"         ❌ {key} is empty after string conversion")
                    raise ValueError(f"Parameter {key} cannot be empty")
                
                params[key] = str_value
                logger.debug(f"         ✅ {key} = '{str_value}' (length: {len(str_value)})")
            
            # Handle port parameter specially
            logger.debug(f"      🔄 Processing port parameter")
            port_value = raw_params['port']
            logger.debug(f"         🔍 Raw port value: {port_value} (type: {type(port_value)})")
            
            try:
                if isinstance(port_value, int):
                    port_int = port_value
                    logger.debug(f"         📝 Port already integer: {port_int}")
                elif isinstance(port_value, str):
                    port_int = int(port_value)
                    logger.debug(f"         📝 Port converted from string: {port_int}")
                else:
                    port_str = str(port_value)
                    port_int = int(port_str)
                    logger.debug(f"         📝 Port converted via string: {port_str} -> {port_int}")
                
                if port_int <= 0 or port_int > 65535:
                    logger.error(f"         ❌ Port out of range: {port_int}")
                    raise ValueError(f"Port {port_int} is out of valid range (1-65535)")
                
                params['port'] = port_int
                logger.debug(f"         ✅ Port validated: {port_int}")
                
            except (ValueError, TypeError) as e:
                logger.error(f"         ❌ Port conversion failed: {e}")
                logger.error(f"         📋 Original port value: {port_value} (type: {type(port_value)})")
                raise ValueError(f"Invalid port number: {port_value}")
            
            # Step 4: Add additional connection parameters
            logger.debug("   📊 STEP 4: Adding additional connection parameters")
            params['connect_timeout'] = 30
            logger.debug(f"      ✅ Added connect_timeout: 30")
            
            # Step 5: Final validation
            logger.debug("   📊 STEP 5: Final parameter validation")
            logger.debug("      🔍 Final parameters (password hidden):")
            for key, value in params.items():
                if key == 'password':
                    logger.debug(f"         🔑 {key}: ***HIDDEN*** (length: {len(value)})")
                else:
                    logger.debug(f"         🔑 {key}: {value} (type: {type(value)})")
            
            # Test parameter completeness
            required_final = ['dbname', 'user', 'password', 'host', 'port', 'connect_timeout']
            for key in required_final:
                if key not in params:
                    logger.error(f"      ❌ Missing final parameter: {key}")
                    raise ValueError(f"Missing final parameter: {key}")
                else:
                    logger.debug(f"      ✅ Final parameter {key} present")
            
            logger.debug("   🎉 Connection parameters successfully prepared")
            return params
            
        except Exception as e:
            logger.error(f"❌ connection_params() failed: {e}")
            logger.debug(f"📋 Full traceback: {traceback.format_exc()}")
            
            # Emergency diagnostic dump
            try:
                logger.error("🚨 EMERGENCY DIAGNOSTIC DUMP:")
                logger.error(f"   self._db_config: {self._db_config}")
                logger.error(f"   Environment DB_* vars:")
                for var in ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']:
                    logger.error(f"      {var}: {os.environ.get(var, 'NOT_SET')}")
            except Exception as dump_error:
                logger.error(f"   Emergency dump failed: {dump_error}")
            
            raise
    
    async def get_async_pool(self):
        """Get or create async connection pool with enhanced error handling"""
        if self.async_pool is None:
            try:
                logger.debug("Creating async connection pool...")
                pool_config = config.PERFORMANCE_CONFIG
                
                logger.debug(f"Pool configuration: min_size={max(2, pool_config['worker_processes'] // 2)}, max_size={min(20, pool_config['worker_processes'] * 2)}")
                
                self.async_pool = await asyncpg.create_pool(
                    self._connection_string,
                    min_size=max(2, pool_config['worker_processes'] // 2),
                    max_size=min(20, pool_config['worker_processes'] * 2),
                    timeout=180,
                    server_settings=config.PG_OPTIMIZATION_SETTINGS
                )
                
                # Register vector extension for all connections in pool
                async with self.async_pool.acquire() as conn:
                    await pgvector.asyncpg.register_vector(conn)
                    logger.debug("Vector extension registered for async pool")
                
                logger.info("✅ Async connection pool created successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to create async pool: {e}")
                logger.debug(f"Async pool creation error details: {e}", exc_info=True)
                raise
        return self.async_pool
    
    async def test_async_connection(self) -> bool:
        """Test async database connection with enhanced diagnostics"""
        logger.info("Testing async database connection...")
        try:
            async with self.get_async_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    logger.info("✅ Async database connection test successful")
                    
                    # Get version info
                    version = await conn.fetchval("SELECT version()")
                    logger.info(f"Async connection to: {version}")
                    
                    return True
                else:
                    logger.error(f"❌ Async test query returned unexpected result: {result}")
                    return False
        except Exception as e:
            logger.error(f"❌ Async database connection test failed: {e}")
            logger.debug(f"Async connection test error details: {e}", exc_info=True)
            return False
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Get async connection as context manager"""
        pool = await self.get_async_pool()
        async with pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Async connection error: {e}")
                raise
    
    @contextmanager
    def get_sync_cursor(self):
        """Get synchronous cursor with comprehensive connection tracing"""
        logger.info("🔌 get_sync_cursor() called - Starting connection process")
        conn = cur = None
        
        try:
            # Step 1: Get connection parameters
            logger.info("   📊 STEP 1: Getting connection parameters")
            try:
                conn_params = self.connection_params
                logger.info("      ✅ Connection parameters retrieved successfully")
            except Exception as e:
                logger.error(f"      ❌ Failed to get connection parameters: {e}")
                logger.debug(f"      📋 Parameters error traceback: {traceback.format_exc()}")
                raise
            
            # Step 2: Pre-connection diagnostic
            logger.info("   📊 STEP 2: Pre-connection diagnostic")
            logger.info(f"      🎯 Target: {conn_params['user']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}")
            logger.info(f"      ⏱️ Timeout: {conn_params['connect_timeout']} seconds")
            logger.info(f"      🔐 Password length: {len(conn_params['password'])} characters")
            
            # Step 3: Network connectivity check
            logger.info("   📊 STEP 3: Network connectivity check")
            try:
                import socket
                logger.debug(f"      🔄 Testing TCP connection to {conn_params['host']}:{conn_params['port']}")
                sock = socket.socket(socket.AF_INET, socket.AF_INET)
                sock.settimeout(5)
                result = sock.connect_ex((conn_params['host'], conn_params['port']))
                sock.close()
                
                if result == 0:
                    logger.info(f"      ✅ TCP connection successful to {conn_params['host']}:{conn_params['port']}")
                else:
                    logger.warning(f"      ⚠️ TCP connection failed to {conn_params['host']}:{conn_params['port']} (code: {result})")
                    logger.warning(f"         💡 This might indicate PostgreSQL is not running or not accessible")
                    
            except Exception as net_error:
                logger.warning(f"      ⚠️ Network check failed: {net_error}")
            
            # Step 4: PostgreSQL connection attempt
            logger.info("   📊 STEP 4: PostgreSQL connection attempt")
            logger.debug(f"      🔄 Calling psycopg2.connect() with params...")
            
            # Log the exact parameters being passed (hide password)
            safe_params = dict(conn_params)
            safe_params['password'] = '***HIDDEN***'
            logger.debug(f"      📋 psycopg2.connect parameters: {safe_params}")
            
            try:
                logger.debug("      ⏳ Attempting connection...")
                start_time = time.time()
                
                conn = psycopg2.connect(**conn_params)
                
                connect_time = time.time() - start_time
                logger.info(f"      ✅ PostgreSQL connection established in {connect_time:.3f}s")
                
                # Test the connection
                logger.debug("      🔄 Testing connection with basic query...")
                test_cur = conn.cursor()
                test_cur.execute("SELECT 1")
                test_result = test_cur.fetchone()
                test_cur.close()
                
                if test_result and test_result[0] == 1:
                    logger.info("      ✅ Connection test query successful")
                else:
                    logger.error(f"      ❌ Connection test query failed: {test_result}")
                
            except psycopg2.OperationalError as op_error:
                logger.error(f"      ❌ PostgreSQL OperationalError: {op_error}")
                logger.debug(f"      📋 OperationalError traceback: {traceback.format_exc()}")
                
                # Detailed error analysis
                error_str = str(op_error).lower()
                logger.error("      🔍 Error Analysis:")
                if "could not connect to server" in error_str:
                    logger.error("         💡 PostgreSQL server is not running or not accessible")
                    logger.error("         💡 Try: sudo systemctl start postgresql")
                elif "authentication failed" in error_str:
                    logger.error(f"         💡 Authentication failed for user '{conn_params['user']}'")
                    logger.error("         💡 Check username/password in .env file")
                elif "does not exist" in error_str and "database" in error_str:
                    logger.error(f"         💡 Database '{conn_params['dbname']}' does not exist")
                    logger.error("         💡 Create the database or check DB_NAME in .env")
                elif "role" in error_str and "does not exist" in error_str:
                    logger.error(f"         💡 User '{conn_params['user']}' does not exist")
                    logger.error("         💡 Create the user or check DB_USER in .env")
                elif "password authentication failed" in error_str:
                    logger.error("         💡 Password authentication failed")
                    logger.error("         💡 Check DB_PASSWORD in .env file")
                elif "connection refused" in error_str:
                    logger.error("         💡 Connection refused - PostgreSQL may not be running")
                    logger.error("         💡 Check if PostgreSQL service is started")
                else:
                    logger.error(f"         💡 Unknown PostgreSQL error: {error_str}")
                
                raise
                
            except psycopg2.Error as pg_error:
                logger.error(f"      ❌ PostgreSQL Error: {pg_error}")
                logger.debug(f"      📋 PostgreSQL error traceback: {traceback.format_exc()}")
                raise
                
            except Exception as unexpected_error:
                logger.error(f"      ❌ Unexpected connection error: {unexpected_error}")
                logger.debug(f"      📋 Unexpected error traceback: {traceback.format_exc()}")
                raise
            
            # Step 5: Create cursor
            logger.info("   📊 STEP 5: Creating database cursor")
            try:
                logger.debug("      🔄 Creating RealDictCursor...")
                cur = conn.cursor(cursor_factory=RealDictCursor)
                logger.info("      ✅ Database cursor created successfully")
            except Exception as cursor_error:
                logger.error(f"      ❌ Failed to create cursor: {cursor_error}")
                logger.debug(f"      📋 Cursor error traceback: {traceback.format_exc()}")
                raise
            
            # Step 6: Yield connection and cursor
            logger.info("   📊 STEP 6: Yielding connection and cursor to caller")
            yield (conn, cur)
            
            # Step 7: Commit transaction
            logger.info("   📊 STEP 7: Committing transaction")
            try:
                conn.commit()
                logger.info("      ✅ Transaction committed successfully")
            except Exception as commit_error:
                logger.error(f"      ❌ Failed to commit transaction: {commit_error}")
                logger.debug(f"      📋 Commit error traceback: {traceback.format_exc()}")
                raise
            
        except Exception as e:
            logger.error(f"❌ get_sync_cursor() failed: {e}")
            logger.debug(f"📋 Full get_sync_cursor traceback: {traceback.format_exc()}")
            
            # Rollback if possible
            if conn:
                try:
                    logger.info("   🔄 Rolling back transaction due to error...")
                    conn.rollback()
                    logger.info("   ✅ Transaction rolled back")
                except Exception as rollback_error:
                    logger.error(f"   ❌ Failed to rollback transaction: {rollback_error}")
            
            raise
            
        finally:
            # Cleanup
            logger.info("   📊 CLEANUP: Closing database resources")
            if cur:
                try:
                    cur.close()
                    logger.debug("      ✅ Cursor closed")
                except Exception as e:
                    logger.warning(f"      ⚠️ Error closing cursor: {e}")
            if conn:
                try:
                    conn.close()
                    logger.debug("      ✅ Connection closed")
                except Exception as e:
                    logger.warning(f"      ⚠️ Error closing connection: {e}")
            
            logger.info("🔌 get_sync_cursor() completed")
    
    def test_connection(self) -> bool:
        """Test database connection with comprehensive step-by-step diagnostics"""
        logger.info("🧪 test_connection() called - Starting comprehensive connection test")
        
        try:
            # Step 1: Configuration validation
            logger.info("   📊 STEP 1: Configuration validation")
            try:
                conn_params = self.connection_params
                logger.info("      ✅ Configuration validation passed")
                logger.info(f"      🎯 Testing connection to: {conn_params['user']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}")
            except Exception as config_error:
                logger.error(f"      ❌ Configuration validation failed: {config_error}")
                logger.debug(f"      📋 Config validation traceback: {traceback.format_exc()}")
                return False
            
            # Step 2: Connection test
            logger.info("   📊 STEP 2: Database connection test")
            try:
                with self.get_sync_cursor() as (conn, cur):
                    logger.info("      ✅ Connection established successfully")
                    
                    # Step 3: Basic query test
                    logger.info("   📊 STEP 3: Basic query test")
                    logger.debug("      🔄 Executing: SELECT 1")
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    
                    logger.debug(f"      🔍 Query result: {result} (type: {type(result)})")
                    
                    # Handle RealDictCursor result properly
                    if result:
                        # RealDictCursor returns a RealDictRow, access by column name or convert to list
                        try:
                            # Try accessing as a list first
                            test_value = list(result.values())[0] if hasattr(result, 'values') else result[0]
                            logger.debug(f"      🔍 Test value extracted: {test_value}")
                        except (KeyError, IndexError, TypeError):
                            # Fallback: try to get the first value differently
                            try:
                                test_value = result['?column?']  # PostgreSQL default column name for SELECT 1
                                logger.debug(f"      🔍 Test value via column name: {test_value}")
                            except (KeyError, TypeError):
                                # Final fallback: convert to tuple and access by index
                                try:
                                    result_tuple = tuple(result)
                                    test_value = result_tuple[0]
                                    logger.debug(f"      🔍 Test value via tuple conversion: {test_value}")
                                except (TypeError, IndexError):
                                    logger.error(f"      ❌ Could not extract value from result: {result}")
                                    return False
                        
                        if test_value == 1:
                            logger.info("      ✅ Basic query test passed")
                        else:
                            logger.error(f"      ❌ Basic query test failed: expected 1, got {test_value}")
                            return False
                    else:
                        logger.error(f"      ❌ Basic query test failed: no result returned")
                        return False
                    
                    # Step 4: Database version check
                    logger.info("   📊 STEP 4: Database version check")
                    try:
                        logger.debug("      🔄 Executing: SELECT version()")
                        cur.execute("SELECT version()")
                        version_result = cur.fetchone()
                        
                        if version_result:
                            # Handle RealDictCursor result
                            try:
                                version = list(version_result.values())[0] if hasattr(version_result, 'values') else version_result[0]
                            except (KeyError, IndexError, TypeError):
                                try:
                                    version = version_result['version']
                                except (KeyError, TypeError):
                                    version = tuple(version_result)[0]
                            
                            logger.info(f"      ✅ Database version: {version}")
                        else:
                            logger.warning("      ⚠️ Could not get database version: no result")
                            
                    except Exception as version_error:
                        logger.warning(f"      ⚠️ Could not get database version: {version_error}")
                    
                    # Step 5: Permissions test
                    logger.info("   📊 STEP 5: Basic permissions test")
                    try:
                        logger.debug("      🔄 Testing SELECT permissions on pg_database")
                        cur.execute("SELECT datname FROM pg_database LIMIT 1")
                        db_result = cur.fetchone()
                        
                        if db_result:
                            # Handle RealDictCursor result
                            try:
                                db_name = list(db_result.values())[0] if hasattr(db_result, 'values') else db_result[0]
                            except (KeyError, IndexError, TypeError):
                                try:
                                    db_name = db_result['datname']
                                except (KeyError, TypeError):
                                    db_name = tuple(db_result)[0]
                            
                            logger.info(f"      ✅ Basic SELECT permissions confirmed (found database: {db_name})")
                        else:
                            logger.warning("      ⚠️ Permissions test: no databases found")
                            
                    except Exception as perm_error:
                        logger.warning(f"      ⚠️ Limited permissions detected: {perm_error}")
                    
            except Exception as connection_error:
                logger.error(f"      ❌ Database connection failed: {connection_error}")
                logger.debug(f"      📋 Connection error traceback: {traceback.format_exc()}")
                return False
            
            # Step 6: Test completion and success confirmation
            logger.info("   📊 STEP 6: Test completion confirmation")
            logger.info("   🎉 All connection tests passed successfully")
            
            return True  # ✅ CRITICAL FIX: Explicitly return True on success
                
        except Exception as e:
            logger.error(f"❌ test_connection() failed: {e}")
            logger.debug(f"📋 Full test_connection traceback: {traceback.format_exc()}")
            
            # Emergency diagnostic information
            logger.error("🚨 EMERGENCY DIAGNOSTIC DUMP:")
            try:
                logger.error(f"   Python version: {sys.version}")
                logger.error(f"   psycopg2 version: {psycopg2.__version__}")
                logger.error(f"   Working directory: {os.getcwd()}")
                logger.error(f"   .env file exists: {os.path.exists('.env')}")
                
                if os.path.exists('.env'):
                    with open('.env', 'r') as f:
                        env_content = f.read()
                        logger.error(f"   .env file content length: {len(env_content)} chars")
                        
            except Exception as dump_error:
                logger.error(f"   Emergency dump failed: {dump_error}")
            
            return False
        
    async def test_async_connection(self) -> bool:
        """Test async database connection"""
        try:
            async with self.get_async_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    logger.info("Async database connection test successful")
                    return True
            return False
        except Exception as e:
            logger.error(f"Async database connection test failed: {e}")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize database with comprehensive error handling and migration support"""
        logger.info("🏗️ initialize_database() called - Starting database initialization")
        
        try:
            # Step 1: Test basic connection
            logger.info("   📊 STEP 1: Testing basic database connection")
            connection_test_result = self.test_connection()  # ✅ Store the result
            if not connection_test_result:
                logger.error("      ❌ Basic connection test failed")
                raise Exception("Cannot connect to database")
            
            logger.info("      ✅ Basic connection confirmed")
            
            # Step 2: Schema initialization
            logger.info("   📊 STEP 2: Database schema initialization")
            
            with self.get_sync_cursor() as (conn, cur):
                table_name = config.TABLE_NAME
                logger.info(f"      🎯 Target table: {table_name}")
                
                # Enable vector extension
                logger.info("      📊 SUB-STEP 2.1: Enabling pgvector extension")
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    logger.info("      ✅ pgvector extension enabled")
                except Exception as ext_error:
                    logger.warning(f"      ⚠️ pgvector extension issue: {ext_error}")
                
                # Create table with vector column - FIXED ARRAY SYNTAX
                logger.info("      📊 SUB-STEP 2.2: Creating main table")
                create_table_query = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        embedding VECTOR({config.EMB_DIM}),
                        file_path TEXT,
                        chunk_index INTEGER DEFAULT 0,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    )
                """
                logger.debug(f"      📋 Table creation query: {create_table_query}")
                cur.execute(create_table_query)
                logger.info(f"      ✅ Table {table_name} created/verified")
                
                # Create indexes for better performance
                logger.info("      📊 SUB-STEP 2.3: Creating performance indexes")
                
                # Vector similarity index (HNSW for fast similarity search)
                try:
                    vector_index_query = f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_embedding_hnsw_idx 
                        ON {table_name} 
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                    """
                    cur.execute(vector_index_query)
                    logger.info(f"      ✅ Vector HNSW index created for {table_name}")
                except Exception as idx_error:
                    logger.warning(f"      ⚠️ Vector index creation issue: {idx_error}")
                    # Fallback to basic index if HNSW fails
                    try:
                        basic_vector_index = f"""
                            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                            ON {table_name} 
                            USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100)
                        """
                        cur.execute(basic_vector_index)
                        logger.info(f"      ✅ Vector IVFFlat index created for {table_name}")
                    except Exception as basic_idx_error:
                        logger.warning(f"      ⚠️ Basic vector index also failed: {basic_idx_error}")
                
                # Text search index (GIN for full-text search)
                try:
                    text_index_query = f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_content_gin_idx 
                        ON {table_name} 
                        USING gin(to_tsvector('english', content))
                    """
                    cur.execute(text_index_query)
                    logger.info(f"      ✅ Full-text search index created for {table_name}")
                except Exception as text_idx_error:
                    logger.warning(f"      ⚠️ Text search index creation issue: {text_idx_error}")
                
                # Tags array index (GIN for array operations)
                try:
                    tags_index_query = f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_tags_gin_idx 
                        ON {table_name} 
                        USING gin(tags)
                    """
                    cur.execute(tags_index_query)
                    logger.info(f"      ✅ Tags array index created for {table_name}")
                except Exception as tags_idx_error:
                    logger.warning(f"      ⚠️ Tags index creation issue: {tags_idx_error}")
                
                # File path index (B-tree for file-based queries)
                try:
                    file_index_query = f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_file_path_idx 
                        ON {table_name} (file_path)
                    """
                    cur.execute(file_index_query)
                    logger.info(f"      ✅ File path index created for {table_name}")
                except Exception as file_idx_error:
                    logger.warning(f"      ⚠️ File path index creation issue: {file_idx_error}")
                
                # Metadata JSONB index (GIN for JSON queries)
                try:
                    metadata_index_query = f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_metadata_gin_idx 
                        ON {table_name} 
                        USING gin(metadata)
                    """
                    cur.execute(metadata_index_query)
                    logger.info(f"      ✅ Metadata JSONB index created for {table_name}")
                except Exception as meta_idx_error:
                    logger.warning(f"      ⚠️ Metadata index creation issue: {meta_idx_error}")
                
                # Composite index for file_path + chunk_index (for document reconstruction)
                try:
                    composite_index_query = f"""
                        CREATE INDEX IF NOT EXISTS {table_name}_file_chunk_idx 
                        ON {table_name} (file_path, chunk_index)
                    """
                    cur.execute(composite_index_query)
                    logger.info(f"      ✅ File+chunk composite index created for {table_name}")
                except Exception as comp_idx_error:
                    logger.warning(f"      ⚠️ Composite index creation issue: {comp_idx_error}")
                
                logger.info("      📊 SUB-STEP 2.4: Database schema setup complete")
                
                # Verify table creation
                logger.info("      📊 SUB-STEP 2.5: Verifying table structure")
                try:
                    cur.execute(f"""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}'
                        ORDER BY ordinal_position
                    """)
                    columns = cur.fetchall()
                    
                    logger.info(f"      ✅ Table {table_name} structure verified:")
                    for col in columns:
                        col_name = col['column_name']
                        col_type = col['data_type']
                        nullable = col['is_nullable']
                        default = col['column_default']
                        logger.info(f"        - {col_name}: {col_type} (nullable: {nullable}, default: {default})")
                        
                except Exception as verify_error:
                    logger.warning(f"      ⚠️ Table verification failed: {verify_error}")
                
            logger.info("🎉 Database initialization completed successfully")
            return True
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            logger.debug(f"📋 Initialization error traceback: {traceback.format_exc()}")
            raise Exception(f"Database initialization failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            with self.get_sync_cursor() as (conn, cur):
                table_name = config.TABLE_NAME
                stats = {}
                
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table_name,))
                
                table_exists = cur.fetchone()[0]
                stats['table_exists'] = table_exists
                
                if not table_exists:
                    stats['record_count'] = 0
                    stats['table_size'] = '0 bytes'
                    return stats
                
                # Get record count
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                stats['record_count'] = cur.fetchone()[0]
                
                # Get table size
                cur.execute(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{table_name}')) as size,
                           pg_total_relation_size('{table_name}') as size_bytes
                """)
                size_result = cur.fetchone()
                stats['table_size'] = size_result[0]
                stats['table_size_bytes'] = size_result[1]
                
                # Get index information
                cur.execute(f"""
                    SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) as size
                    FROM pg_indexes 
                    WHERE tablename = '{table_name}'
                """)
                stats['indexes'] = [{'name': row[0], 'size': row[1]} for row in cur.fetchall()]
                
                # Get column statistics
                cur.execute(f"""
                    SELECT 
                        COUNT(*) FILTER (WHERE embedding IS NOT NULL) as embeddings_count,
                        COUNT(*) FILTER (WHERE tags IS NOT NULL AND array_length(tags, 1) > 0) as tagged_count,
                        COUNT(*) FILTER (WHERE file_path IS NOT NULL) as files_with_path,
                        AVG(LENGTH(content)) as avg_content_length
                    FROM {table_name}
                """)
                column_stats = cur.fetchone()
                stats.update({
                    'embeddings_count': column_stats[0] or 0,
                    'tagged_records': column_stats[1] or 0,
                    'files_with_path': column_stats[2] or 0,
                    'avg_content_length': float(column_stats[3] or 0)
                })
                
                # Get recent activity
                cur.execute(f"""
                    SELECT COUNT(*) as recent_count
                    FROM {table_name}
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                stats['recent_24h'] = cur.fetchone()[0] or 0
                
                stats['table_name'] = table_name
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    async def insert_records_batch(self, records: List[Dict[str, Any]]) -> int:
        """Insert multiple records efficiently using async connection"""
        if not records:
            return 0
        
        try:
            async with self.get_async_connection() as conn:
                await pgvector.asyncpg.register_vector(conn)
                
                table_name = config.TABLE_NAME
                
                # Prepare the insert query based on available fields
                sample_record = records[0]
                fields = []
                placeholders = []
                field_idx = 1
                
                if 'content' in sample_record:
                    fields.append('content')
                    placeholders.append(f'${field_idx}')
                    field_idx += 1
                
                if 'tags' in sample_record:
                    fields.append('tags')
                    placeholders.append(f'${field_idx}::text[]')
                    field_idx += 1
                
                if 'embedding' in sample_record:
                    fields.append('embedding')
                    placeholders.append(f'${field_idx}')
                    field_idx += 1
                
                if 'file_path' in sample_record:
                    fields.append('file_path')
                    placeholders.append(f'${field_idx}')
                    field_idx += 1
                
                if 'chunk_index' in sample_record:
                    fields.append('chunk_index')
                    placeholders.append(f'${field_idx}')
                    field_idx += 1
                
                if 'metadata' in sample_record:
                    fields.append('metadata')
                    placeholders.append(f'${field_idx}::jsonb')
                    field_idx += 1
                
                query = f"""
                    INSERT INTO {table_name} ({', '.join(fields)})
                    VALUES ({', '.join(placeholders)})
                """
                
                # Prepare record tuples
                record_tuples = []
                for record in records:
                    record_tuple = []
                    for field in fields:
                        value = record.get(field)
                        if field == 'metadata' and isinstance(value, dict):
                            value = json.dumps(value)
                        record_tuple.append(value)
                    record_tuples.append(tuple(record_tuple))
                
                # Execute batch insert
                await conn.executemany(query, record_tuples)
                
                logger.info(f"Successfully inserted {len(records)} records")
                return len(records)
                
        except Exception as e:
            logger.error(f"Batch insert failed: {e}", exc_info=True)
            raise
    
    def search_by_text(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search records by text content using full-text search"""
        try:
            with self.get_sync_cursor() as (conn, cur):
                table_name = config.TABLE_NAME
                
                search_query = f"""
                    SELECT id, content, tags, file_path, 
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as rank
                    FROM {table_name}
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """
                
                cur.execute(search_query, (query, query, limit))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    def search_by_tags(self, tags: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search records by tags"""
        try:
            with self.get_sync_cursor() as (conn, cur):
                table_name = config.TABLE_NAME
                
                search_query = f"""
                    SELECT id, content, tags, file_path
                    FROM {table_name}
                    WHERE tags && %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """
                
                cur.execute(search_query, (tags, limit))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Tag search failed: {e}")
            return []
    
    async def get_embeddings_chunk(self, offset: int, limit: int) -> Tuple[List[int], List[List[float]]]:
        """Get a chunk of embeddings for similarity search"""
        try:
            async with self.get_async_connection() as conn:
                table_name = config.TABLE_NAME
                
                query = f"""
                    SELECT id, embedding 
                    FROM {table_name} 
                    WHERE embedding IS NOT NULL
                    ORDER BY id 
                    LIMIT $1 OFFSET $2
                """
                
                rows = await conn.fetch(query, limit, offset)
                
                ids = []
                embeddings = []
                
                for row in rows:
                    if row['embedding']:
                        ids.append(row['id'])
                        embeddings.append(list(row['embedding']))
                
                return ids, embeddings
                
        except Exception as e:
            logger.error(f"Failed to get embeddings chunk: {e}")
            return [], []
    
    async def get_documents_by_ids(self, doc_ids: List[int]) -> List[Dict[str, Any]]:
        """Get document contents by IDs"""
        try:
            async with self.get_async_connection() as conn:
                table_name = config.TABLE_NAME
                
                query = f"""
                    SELECT id, content, tags, file_path, chunk_index, metadata
                    FROM {table_name}
                    WHERE id = ANY($1)
                """
                
                rows = await conn.fetch(query, doc_ids)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return []
    
    async def run_maintenance(self) -> bool:
        """Run database maintenance operations"""
        try:
            async with self.get_async_connection() as conn:
                table_name = config.TABLE_NAME
                
                logger.info("Running database maintenance...")
                
                # Update statistics
                await conn.execute(f"ANALYZE {table_name}")
                
                # Vacuum with analyze
                await conn.execute(f"VACUUM ANALYZE {table_name}")
                
                logger.info("Database maintenance completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            return False
    
    def apply_optimizations(self) -> bool:
        """Apply PostgreSQL optimizations"""
        try:
            with self.get_sync_cursor() as (conn, cur):
                logger.info("Applying PostgreSQL optimizations...")
                
                # Apply settings from config
                pg_settings = config.PG_OPTIMIZATION_SETTINGS
                applied_count = 0
                
                for setting, value in pg_settings.items():
                    try:
                        cur.execute(f"ALTER SYSTEM SET {setting} = %s", (value,))
                        applied_count += 1
                        logger.info(f"Applied setting: {setting} = {value}")
                    except Exception as e:
                        logger.warning(f"Failed to apply {setting}: {e}")
                
                if applied_count > 0:
                    logger.info(f"Applied {applied_count} PostgreSQL optimizations")
                    logger.warning("Please restart PostgreSQL for changes to take effect")
                    return True
                else:
                    logger.warning("No optimizations were applied")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            return False
    
    def diagnose_issues(self) -> Dict[str, Any]:
        """Comprehensive database diagnostics"""
        diagnosis = {
            'connection': False,
            'table_exists': False,
            'pgvector_available': False,
            'indexes_present': [],
            'record_count': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Test basic connection
            if self.test_connection():
                diagnosis['connection'] = True
            else:
                diagnosis['issues'].append("Cannot connect to database")
                return diagnosis
            
            with self.get_sync_cursor() as (conn, cur):
                table_name = config.TABLE_NAME
                
                # Check pgvector extension
                cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
                if cur.fetchone():
                    diagnosis['pgvector_available'] = True
                else:
                    diagnosis['issues'].append("pgvector extension not installed")
                    diagnosis['recommendations'].append("Install pgvector: CREATE EXTENSION vector;")
                
                # Check table existence
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table_name,))
                
                if cur.fetchone()[0]:
                    diagnosis['table_exists'] = True
                    
                    # Get record count
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    diagnosis['record_count'] = cur.fetchone()[0]
                    
                    # Check indexes
                    cur.execute(f"""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE tablename = '{table_name}'
                    """)
                    diagnosis['indexes_present'] = [row[0] for row in cur.fetchall()]
                    
                    # Check for required indexes
                    required_indexes = [
                        f"{table_name}_embedding_idx",
                        f"{table_name}_content_idx",
                        f"{table_name}_tags_idx"
                    ]
                    
                    missing_indexes = [idx for idx in required_indexes 
                                     if idx not in diagnosis['indexes_present']]
                    
                    if missing_indexes:
                        diagnosis['issues'].append(f"Missing indexes: {missing_indexes}")
                        diagnosis['recommendations'].append("Recreate missing indexes")
                
                else:
                    diagnosis['issues'].append(f"Table {table_name} does not exist")
                    diagnosis['recommendations'].append("Initialize database schema")
                
                # Check permissions
                try:
                    cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                    diagnosis['permissions'] = 'read_ok'
                except Exception:
                    diagnosis['issues'].append("Insufficient read permissions")
                
                try:
                    # Test write permissions with a dry run
                    cur.execute("BEGIN")
                    cur.execute(f"INSERT INTO {table_name} (content) VALUES ('test') RETURNING id")
                    test_id = cur.fetchone()[0]
                    cur.execute(f"DELETE FROM {table_name} WHERE id = %s", (test_id,))
                    cur.execute("ROLLBACK")
                    diagnosis['permissions'] = 'read_write_ok'
                except Exception:
                    diagnosis['issues'].append("Insufficient write permissions")
                    diagnosis['permissions'] = 'read_only'
                
        except Exception as e:
            diagnosis['issues'].append(f"Diagnostic error: {e}")
        
        return diagnosis
    
    async def close_pools(self):
        """Close all connection pools"""
        if self.async_pool:
            await self.async_pool.close()
            self.async_pool = None
            logger.info("Closed async connection pool")

# Global database manager instance
db_manager = DatabaseManager()

# Legacy compatibility functions
@contextmanager
def db_cursor():
    """Legacy compatibility function"""
    with db_manager.get_sync_cursor() as (conn, cur):
        yield conn, cur

def test_db_connection():
    """Legacy compatibility function"""
    return db_manager.test_connection()

def init_db():
    """Legacy compatibility function"""
    return db_manager.initialize_database()

def get_db_stats():
    """Legacy compatibility function"""
    return db_manager.get_stats()

def diagnose_db_issues():
    """Legacy compatibility function with enhanced output"""
    print("\n" + "="*60)
    print("DATABASE DIAGNOSTICS")
    print("="*60)
    
    diagnosis = db_manager.diagnose_issues()
    
    # Display results
    if diagnosis['connection']:
        print("   ✅ Database connection: SUCCESS")
    else:
        print("   ❌ Database connection: FAILED")
        for issue in diagnosis['issues']:
            print(f"      - {issue}")
        return False
    
    if diagnosis['pgvector_available']:
        print("   ✅ pgvector extension: AVAILABLE")
    else:
        print("   ⚠️  pgvector extension: NOT AVAILABLE")
    
    if diagnosis['table_exists']:
        print(f"   ✅ Table exists: {config.TABLE_NAME}")
        print(f"   📊 Record count: {diagnosis['record_count']:,}")
        print(f"   📚 Indexes: {len(diagnosis['indexes_present'])} present")
    else:
        print(f"   ❌ Table missing: {config.TABLE_NAME}")
    
    if diagnosis['issues']:
        print("\n⚠️  ISSUES FOUND:")
        for issue in diagnosis['issues']:
            print(f"   - {issue}")
    
    if diagnosis['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in diagnosis['recommendations']:
            print(f"   - {rec}")
    
    print("="*60)
    return len(diagnosis['issues']) == 0
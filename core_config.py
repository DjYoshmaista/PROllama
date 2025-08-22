# core_config.py
import os
import json
import logging
import psutil
import requests
import psycopg2
import subprocess
import getpass
import socket
import time
from pathlib import Path
from typing import Dict, Any, Optional, ClassVar, Tuple
from dotenv import load_dotenv

# Set up dedicated logger for configuration
config_logger = logging.getLogger('config_setup')
config_logger.setLevel(logging.DEBUG)

# Create console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('🔧 CONFIG: %(message)s')
console_handler.setFormatter(console_formatter)
config_logger.addHandler(console_handler)

# Create debug file handler
debug_handler = logging.FileHandler('config_debug.log', mode='w', encoding='utf-8')
debug_handler.setLevel(logging.DEBUG)
debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(debug_formatter)
config_logger.addHandler(debug_handler)

logger = logging.getLogger(__name__)

def ensure_env_file_exists():
    """Ensure .env file exists with all required variables"""
    env_file = Path('.env')
    config_logger.debug(f"Checking for .env file at: {env_file.absolute()}")
    
    if not env_file.exists():
        config_logger.warning("📄 .env file not found, creating default .env file")
        try:
            env_file.touch()
            config_logger.info("✅ Created empty .env file")
        except Exception as e:
            config_logger.error(f"❌ Failed to create .env file: {e}")
            raise
    else:
        config_logger.debug("✅ .env file exists")

def get_env_var(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """Safely get environment variable with validation and auto-creation"""
    config_logger.debug(f"Requesting environment variable: {key}")
    
    # Ensure .env file exists first
    ensure_env_file_exists()
    
    value = os.environ.get(key, default)
    config_logger.debug(f"Environment variable {key} current value: {value}")
    
    if required and not value:
        config_logger.warning(f"⚠️ Required environment variable {key} not set or empty")
        config_logger.debug(f"Environment variable {key} is missing from environment")
        
        # Auto-create missing environment variable
        if _auto_create_env_var(key, default):
            # Reload .env after creating the variable
            load_dotenv(override=True)
            value = os.environ.get(key, default)
            config_logger.info(f"✅ Created and loaded environment variable {key} = {value}")
        else:
            error_msg = f"Required environment variable {key} not set and could not be created"
            config_logger.critical(f"❌ CRITICAL: {error_msg}")
            raise ValueError(error_msg)
    
    config_logger.debug(f"Final environment variable {key} = {value}")
    return value

def get_int_env_var(key: str, default: Optional[int] = None, required: bool = True) -> int:
    """Safely get integer environment variable with validation"""
    value_str = get_env_var(key, str(default) if default is not None else None, required)
    try:
        result = int(value_str) if value_str else default
        config_logger.debug(f"Integer environment variable {key} = {result}")
        return result
    except ValueError:
        error_msg = f"Environment variable {key} must be a valid integer, got: {value_str}"
        config_logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)

def _auto_create_env_var(key: str, default: Optional[str] = None) -> bool:
    """Auto-create missing environment variables with smart defaults"""
    config_logger.info(f"🔄 Auto-creating missing environment variable: {key}")
    
    # Define smart defaults for different variable types
    smart_defaults = {
        'EMB_DIM': '1024',
        'DB_NAME': 'rag_db',
        'DB_USER': 'postgres',
        'DB_PASSWORD': 'postgres',
        'DB_HOST': 'localhost',
        'DB_PORT': '5432',
        'TABLE_NAME': 'rag_db_code',
        'OLLAMA_API': 'http://localhost:11434/api',
        'EMBEDDING_MODEL': 'dengcao/Qwen3-Embedding-0.6B:Q8_0'
    }
    
    # Use smart default if available, otherwise use provided default
    default_value = smart_defaults.get(key, default or f"{key.lower()}_default")
    config_logger.debug(f"Using default value for {key}: {default_value}")
    
    try:
        # Check if .env file exists
        env_file = Path('.env')
        
        if not env_file.exists():
            config_logger.info("📄 Creating new .env file")
            env_file.touch()
        
        # Read existing .env content
        existing_content = env_file.read_text() if env_file.exists() else ""
        config_logger.debug(f"Current .env content length: {len(existing_content)} characters")
        
        # Check if key already exists in file
        if f"{key}=" in existing_content:
            config_logger.debug(f"Environment variable {key} exists in .env but may be empty")
            # Update the value if it's empty
            lines = existing_content.split('\n')
            updated_lines = []
            key_found = False
            
            for line in lines:
                if line.startswith(f"{key}="):
                    key_found = True
                    current_value = line.split('=', 1)[1] if '=' in line else ''
                    if not current_value.strip():
                        config_logger.info(f"Updating empty value for {key}")
                        updated_lines.append(f"{key}={default_value}")
                    else:
                        config_logger.debug(f"Keeping existing value for {key}")
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            
            if key_found:
                env_file.write_text('\n'.join(updated_lines))
                os.environ[key] = default_value
                config_logger.info(f"✅ Updated {key} in .env file")
                return True
        
        # Add new environment variable
        new_line = f"{key}={default_value}\n"
        
        with open(env_file, 'a') as f:
            f.write(new_line)
        
        # Set in current environment
        os.environ[key] = default_value
        
        config_logger.info(f"✅ Added {key}={default_value} to .env file")
        config_logger.debug(f"Environment variable {key} created with value: {default_value}")
        
        return True
        
    except Exception as e:
        config_logger.error(f"❌ Failed to create environment variable {key}: {e}")
        config_logger.debug(f"Auto-creation failed for {key}: {e}", exc_info=True)
        return False

def _check_postgresql_installation() -> bool:
    """Check if PostgreSQL is installed and accessible"""
    config_logger.debug("Checking PostgreSQL installation")
    
    try:
        # Try to find psql command
        result = subprocess.run(['psql', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            config_logger.info(f"✅ PostgreSQL found: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        config_logger.debug(f"PostgreSQL check failed: {e}")
    
    config_logger.warning("⚠️ PostgreSQL client (psql) not found in PATH")
    config_logger.info("💡 Install PostgreSQL:")
    config_logger.info("   Ubuntu/Debian: sudo apt install postgresql postgresql-contrib")
    config_logger.info("   Arch Linux: sudo pacman -S postgresql")
    config_logger.info("   Fedora: sudo dnf install postgresql-server postgresql-contrib")
    return False

def _check_postgresql_service() -> bool:
    """Check if PostgreSQL service is running"""
    config_logger.debug("Checking PostgreSQL service status")
    
    try:
        # Try to connect to default PostgreSQL port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 5432))
        sock.close()
        
        if result == 0:
            config_logger.info("✅ PostgreSQL service is running on localhost:5432")
            return True
        else:
            config_logger.warning("⚠️ PostgreSQL service not accessible on localhost:5432")
            config_logger.info("💡 Start PostgreSQL service:")
            config_logger.info("   Ubuntu/Debian: sudo systemctl start postgresql")
            config_logger.info("   Arch Linux: sudo systemctl start postgresql")
            config_logger.info("   Fedora: sudo systemctl start postgresql")
            return False
            
    except Exception as e:
        config_logger.error(f"❌ Error checking PostgreSQL service: {e}")
        return False

def _create_superuser_connection(host: str = 'localhost', port: str = '5432') -> Optional[psycopg2.extensions.connection]:
    """Try to establish superuser connection with various common credentials"""
    config_logger.debug("Attempting to create superuser connection")
    
    # Common superuser credentials to try
    superuser_configs = [
        {'user': 'postgres', 'password': ''},  # No password
        {'user': 'postgres', 'password': 'postgres'},  # Common default
        {'user': 'postgres', 'password': 'password'},  # Another common default
    ]
    
    for i, creds in enumerate(superuser_configs):
        config_logger.debug(f"Trying superuser connection attempt {i+1}: user={creds['user']}")
        try:
            conn_params = {
                'host': host,
                'port': port,
                'dbname': 'postgres',  # Connect to default postgres database
                'user': creds['user'],
                'password': creds['password'],
                'connect_timeout': 10
            }
            
            conn = psycopg2.connect(**conn_params)
            config_logger.info(f"✅ Superuser connection established with user: {creds['user']}")
            return conn
            
        except psycopg2.OperationalError as e:
            config_logger.debug(f"Superuser connection attempt {i+1} failed: {e}")
            continue
        except Exception as e:
            config_logger.debug(f"Unexpected error in superuser connection attempt {i+1}: {e}")
            continue
    
    config_logger.warning("⚠️ Could not establish superuser connection with common credentials")
    return None

def _create_user_if_not_exists(db_config: Dict[str, str]) -> bool:
    """Create database user if it doesn't exist"""
    config_logger.info(f"🔄 Checking if user '{db_config['user']}' exists")
    
    # Skip if trying to create postgres user (it should already exist)
    if db_config['user'] == 'postgres':
        config_logger.info("✅ Using postgres superuser (assumed to exist)")
        return True
    
    try:
        # Try to establish superuser connection
        admin_conn = _create_superuser_connection(db_config['host'], db_config['port'])
        
        if not admin_conn:
            config_logger.warning("⚠️ Could not establish superuser connection for user creation")
            return True  # Continue anyway, user might exist
        
        try:
            admin_conn.autocommit = True
            
            with admin_conn.cursor() as cur:
                # Check if user exists
                cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (db_config['user'],))
                user_exists = cur.fetchone() is not None
                
                if user_exists:
                    config_logger.info(f"✅ User '{db_config['user']}' already exists")
                else:
                    # Create user
                    create_user_query = f"""
                        CREATE USER "{db_config['user']}" 
                        WITH PASSWORD '{db_config['password']}' 
                        CREATEDB LOGIN
                    """
                    config_logger.debug(f"Creating user: {db_config['user']}")
                    cur.execute(create_user_query)
                    config_logger.info(f"✅ Created user '{db_config['user']}'")
            
            admin_conn.close()
            return True
            
        except psycopg2.Error as e:
            config_logger.warning(f"⚠️ Could not create user as postgres superuser: {e}")
            admin_conn.close()
            return True  # Continue anyway
            
    except Exception as e:
        config_logger.warning(f"⚠️ User creation attempt failed: {e}")
        config_logger.debug(f"User creation error details: {e}", exc_info=True)
        return True  # Continue anyway

def _create_database_if_not_exists(db_config: Dict[str, str]) -> bool:
    """Create database if it doesn't exist"""
    config_logger.info(f"🔄 Checking if database '{db_config['dbname']}' exists")
    
    try:
        # First, try to connect to the target database
        try:
            test_conn = psycopg2.connect(**db_config, connect_timeout=10)
            test_conn.close()
            config_logger.info(f"✅ Database '{db_config['dbname']}' already exists and is accessible")
            return True
        except psycopg2.OperationalError as e:
            if "does not exist" in str(e):
                config_logger.info(f"🔄 Database '{db_config['dbname']}' does not exist, creating...")
            else:
                config_logger.error(f"❌ Database connection error: {e}")
                return False
        
        # Try to establish superuser connection
        admin_conn = _create_superuser_connection(db_config['host'], db_config['port'])
        
        if not admin_conn:
            config_logger.error("❌ Could not establish superuser connection for database creation")
            return False
        
        try:
            admin_conn.autocommit = True
            
            with admin_conn.cursor() as cur:
                # Create database
                create_db_query = f'CREATE DATABASE "{db_config["dbname"]}"'
                config_logger.debug(f"Executing: {create_db_query}")
                cur.execute(create_db_query)
                
                config_logger.info(f"✅ Created database '{db_config['dbname']}'")
            
            admin_conn.close()
            return True
            
        except psycopg2.Error as e:
            if "already exists" in str(e):
                config_logger.info(f"✅ Database '{db_config['dbname']}' already exists")
                admin_conn.close()
                return True
            else:
                config_logger.error(f"❌ Failed to create database: {e}")
                admin_conn.close()
                return False
                
    except Exception as e:
        config_logger.error(f"❌ Database creation failed: {e}")
        config_logger.debug(f"Database creation error details: {e}", exc_info=True)
        return False

def _install_pgvector_extension(db_config: Dict[str, str]) -> bool:
    """Install pgvector extension if not already installed"""
    config_logger.info("🔄 Checking pgvector extension")
    
    try:
        conn = psycopg2.connect(**db_config, connect_timeout=10)
        
        with conn.cursor() as cur:
            # Check if extension exists
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            extension_exists = cur.fetchone() is not None
            
            if extension_exists:
                config_logger.info("✅ pgvector extension already installed")
                conn.close()
                return True
            
            # Try to install extension
            config_logger.info("📦 Installing pgvector extension...")
            try:
                cur.execute("CREATE EXTENSION vector")
                conn.commit()
                config_logger.info("✅ pgvector extension installed successfully")
                conn.close()
                return True
                
            except psycopg2.Error as e:
                if "does not exist" in str(e):
                    config_logger.error("❌ pgvector extension not available - please install pgvector")
                    config_logger.info("💡 Install instructions: https://github.com/pgvector/pgvector")
                    config_logger.info("   Ubuntu: sudo apt install postgresql-15-pgvector")
                    config_logger.info("   Arch: sudo pacman -S pgvector")
                else:
                    config_logger.error(f"❌ Failed to install pgvector: {e}")
                conn.close()
                return False
                
    except Exception as e:
        config_logger.error(f"❌ pgvector check failed: {e}")
        config_logger.debug(f"pgvector check error details: {e}", exc_info=True)
        return False

def _setup_database_components(db_config: Dict[str, str]) -> bool:
    """Setup all database components automatically"""
    config_logger.info("🚀 Setting up database components...")
    
    # Check PostgreSQL installation
    if not _check_postgresql_installation():
        config_logger.error("❌ PostgreSQL not found. Please install PostgreSQL first.")
        return False
    
    # Check if PostgreSQL service is running
    if not _check_postgresql_service():
        config_logger.error("❌ PostgreSQL service not running. Please start PostgreSQL service.")
        return False
    
    # Create user if needed
    if not _create_user_if_not_exists(db_config):
        config_logger.warning("⚠️ User creation failed, but continuing...")
    
    # Create database if needed
    if not _create_database_if_not_exists(db_config):
        config_logger.error("❌ Database creation failed")
        return False
    
    # Install pgvector extension
    if not _install_pgvector_extension(db_config):
        config_logger.warning("⚠️ pgvector extension not available, but continuing...")
    
    config_logger.info("✅ Database setup completed")
    return True

def _test_database_connection(db_config: Dict[str, str]) -> bool:
    """Test database connection with detailed error reporting"""
    config_logger.info(f"🔍 Testing database connection to {db_config['host']}:{db_config['port']}/{db_config['dbname']}")
    
    try:
        conn = psycopg2.connect(**db_config, connect_timeout=10)
        
        # Test basic query
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            config_logger.info(f"✅ Database connection successful: {version}")
        
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        config_logger.error(f"❌ Database connection failed: {e}")
        
        # Provide specific troubleshooting based on error
        error_str = str(e).lower()
        if "could not connect to server" in error_str:
            config_logger.error("💡 PostgreSQL server is not running or not accessible")
        elif "does not exist" in error_str and "database" in error_str:
            config_logger.error(f"💡 Database '{db_config['dbname']}' does not exist")
        elif "authentication failed" in error_str:
            config_logger.error(f"💡 Authentication failed for user '{db_config['user']}'")
        elif "role" in error_str and "does not exist" in error_str:
            config_logger.error(f"💡 User '{db_config['user']}' does not exist")
        
        return False
        
    except Exception as e:
        config_logger.error(f"❌ Unexpected database connection error: {e}")
        config_logger.debug(f"Database connection error details: {e}", exc_info=True)
        return False

# Load environment variables early
load_dotenv()

class UnifiedConfig:
    """Centralized configuration management with auto-setup and persistence"""
    
    _instance = None
    _config_file = "ragdb_config.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            config_logger.info("🔧 Initializing Enhanced RAG Database Configuration...")
            
            # Initialize configuration
            self.runtime_config = {}
            self.load_config()
            
            # Initialize core configuration with auto-setup
            self._setup_core_configuration()
            
            self._initialized = True
            config_logger.info("✅ Configuration initialization completed")
    
    def _setup_core_configuration(self):
        """Setup core configuration with auto-creation of missing components"""
        config_logger.info("🔄 Setting up core configuration...")
        
        try:
            # Core configuration with environment variable support and auto-creation
            self.EMBEDDING_MODEL = get_env_var('EMBEDDING_MODEL', "dengcao/Qwen3-Embedding-0.6B:Q8_0", False)
            self.EMB_DIM = get_int_env_var('EMB_DIM', 1024, False)
            
            # Database configuration with auto-setup
            self.DB_NAME = get_env_var('DB_NAME', 'rag_db', False)
            self.DB_USER = get_env_var('DB_USER', 'postgres', False)
            self.DB_PASSWORD = get_env_var('DB_PASSWORD', 'postgres', False)
            self.DB_HOST = get_env_var('DB_HOST', 'localhost', False)
            self.DB_PORT = get_int_env_var('DB_PORT', 5432, False)
            self.TABLE_NAME = get_env_var('TABLE_NAME', 'rag_db_code', False)
            
            # API endpoints
            self.OLLAMA_API = get_env_var('OLLAMA_API', "http://localhost:11434/api", False)
            
            config_logger.info("✅ Core configuration loaded successfully")
            
            # Auto-setup database components
            db_config = {
                'dbname': self.DB_NAME,
                'user': self.DB_USER,
                'password': self.DB_PASSWORD,
                'host': self.DB_HOST,
                'port': str(self.DB_PORT)
            }
            
            config_logger.info("🔧 Auto-setting up database components...")
            
            # Test connection first
            if _test_database_connection(db_config):
                config_logger.info("✅ Database connection successful, no setup needed")
            else:
                config_logger.info("🔧 Database connection failed, attempting auto-setup...")
                if _setup_database_components(db_config):
                    config_logger.info("✅ Database auto-setup completed successfully")
                    # Test again after setup
                    if _test_database_connection(db_config):
                        config_logger.info("✅ Database connection successful after setup")
                    else:
                        config_logger.warning("⚠️ Database setup completed but connection still fails")
                else:
                    config_logger.warning("⚠️ Database auto-setup had issues, but continuing...")
                
        except Exception as e:
            config_logger.error(f"❌ Core configuration setup failed: {e}")
            config_logger.debug(f"Configuration setup error details: {e}", exc_info=True)
            raise
    
    # Database configuration
    @property
    def DB_CONFIG(self) -> Dict[str, Any]:
        return {
            'dbname': self.get('DB_NAME', self.DB_NAME),
            'user': self.get('DB_USER', self.DB_USER),
            'password': self.get('DB_PASSWORD', self.DB_PASSWORD),
            'host': self.get('DB_HOST', self.DB_HOST),
            'port': self.get('DB_PORT', self.DB_PORT)
        }
    
    # Processing configuration with dynamic adjustment
    @property
    def CHUNK_SIZES(self) -> Dict[str, int]:
        return {
            'memory_safe': self.get('chunk_memory_safe', 100),
            'embedding_batch': self.get('chunk_embedding_batch', 100),
            'insert_batch': self.get('chunk_insert_batch', 500),
            'file_processing': self.get('chunk_file_processing', 100)
        }
    
    # Search configuration
    @property
    def SEARCH_DEFAULTS(self) -> Dict[str, Any]:
        return {
            'relevance_threshold': self.get('relevance_threshold', 0.3),
            'top_k': self.get('top_k', 25),
            'vector_search_limit': self.get('vector_search_limit', 999999999)
        }
    
    # Performance configuration with auto-detection
    @property
    def PERFORMANCE_CONFIG(self) -> Dict[str, Any]:
        cpu_count = os.cpu_count() or 1
        memory = psutil.virtual_memory()
        
        return {
            'max_concurrent_requests': self.get('max_concurrent_requests', 200),
            'max_cache_size': self.get('max_cache_size', 20000000),
            'cache_refresh_interval': self.get('cache_refresh_interval', 300),
            'memory_cleanup_threshold': self.get('memory_cleanup_threshold', 85),
            'max_concurrent_chunks': self.get('max_concurrent_chunks', 32),
            'worker_processes': self.get('worker_processes', min(8, cpu_count)),
            'embedding_queue_memory_gb': self.get('embedding_queue_memory_gb', 4),
            'cache_memory_gb': self.get('cache_memory_gb', 1),
            'max_file_size_mb': self.get('max_file_size_mb', 100)
        }
    
    # PostgreSQL optimization settings
    @property
    def PG_OPTIMIZATION_SETTINGS(self) -> Dict[str, str]:
        return {
            "statement_timeout": self.get('pg_statement_timeout', "300000"),
            "work_mem": self.get('pg_work_mem', "16MB"),
            "maintenance_work_mem": self.get('pg_maintenance_work_mem', "512MB")
        }
    
    # File processing configuration
    @property
    def FILE_CONFIG(self) -> Dict[str, Any]:
        return {
            'supported_extensions': self.get('supported_extensions', ["py", "txt", "csv", "json"]),
            'dangerous_extensions': self.get('dangerous_extensions', [
                'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'vbe', 'js', 'jse',
                'ws', 'wsf', 'wsc', 'wsh', 'ps1', 'ps1xml', 'ps2', 'ps2xml', 'psc1',
                'psc2', 'msh', 'msh1', 'msh2', 'mshxml', 'msh1xml', 'msh2xml'
            ]),
            'max_file_size': self.get('max_file_size_mb', 100) * 1024 * 1024,
            'max_path_length': self.get('max_path_length', 4096),
            'token_overlap_size': self.get('token_overlap_size', 32)
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback"""
        return self.runtime_config.get(key, default)
    
    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Set configuration value with optional persistence"""
        old_value = self.runtime_config.get(key)
        self.runtime_config[key] = value
        
        if persist:
            self.save_config()
        
        config_logger.info(f"Configuration updated: {key} = {value} (was: {old_value})")
    
    def update(self, config_dict: Dict[str, Any], persist: bool = True) -> None:
        """Update multiple configuration values"""
        for key, value in config_dict.items():
            self.runtime_config[key] = value
            config_logger.info(f"Configuration updated: {key} = {value}")
        
        if persist:
            self.save_config()
    
    def load_config(self) -> None:
        """Load configuration from file"""
        try:
            if os.path.exists(self._config_file):
                with open(self._config_file, 'r') as f:
                    self.runtime_config = json.load(f)
                config_logger.debug(f"Loaded configuration from {self._config_file}")
            else:
                self.runtime_config = {}
                config_logger.debug("No configuration file found, using defaults")
        except Exception as e:
            config_logger.warning(f"Could not load configuration: {e}")
            self.runtime_config = {}
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            # Create backup
            if os.path.exists(self._config_file):
                backup_file = f"{self._config_file}.backup"
                os.replace(self._config_file, backup_file)
            
            # Save configuration
            with open(self._config_file, 'w') as f:
                json.dump(self.runtime_config, f, indent=2)
            
            config_logger.debug(f"Configuration saved to {self._config_file}")
        except Exception as e:
            config_logger.error(f"Failed to save configuration: {e}")
    
    def validate_config(self) -> bool:
        """Validate all configuration values"""
        config_logger.info("🔍 Validating configuration...")
        
        try:
            # Validate dimensions
            if self.EMB_DIM <= 0:
                config_logger.error(f"Invalid embedding dimension: {self.EMB_DIM}")
                return False
            
            # Validate chunk sizes
            for key, value in self.CHUNK_SIZES.items():
                if not isinstance(value, int) or value <= 0:
                    config_logger.error(f"Invalid chunk size for {key}: {value}")
                    return False
            
            # Validate search defaults
            threshold = self.SEARCH_DEFAULTS['relevance_threshold']
            if not 0.0 <= threshold <= 1.0:
                config_logger.error(f"Invalid relevance threshold: {threshold}")
                return False
            
            if self.SEARCH_DEFAULTS['top_k'] <= 0:
                config_logger.error(f"Invalid top_k: {self.SEARCH_DEFAULTS['top_k']}")
                return False
           
           # Validate performance settings
            perf_config = self.PERFORMANCE_CONFIG
            if perf_config['worker_processes'] <= 0:
                config_logger.error(f"Invalid worker_processes: {perf_config['worker_processes']}")
                return False
            
            # Test database connection
            config_logger.debug("Testing database connection as part of validation...")
            db_config = self.DB_CONFIG
            if not _test_database_connection(db_config):
                config_logger.error("Database connection validation failed")
                return False
            
            config_logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            config_logger.error(f"Configuration validation failed: {e}")
            config_logger.debug(f"Validation error details: {e}", exc_info=True)
            return False
   
    def get_db_connection_string(self) -> str:
        """Get formatted database connection string"""
        db_config = self.DB_CONFIG
        return (f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}"
                f"/{db_config['dbname']}")
    
    def validate_embedding_model(self) -> bool:
        """Validate that the required embedding model is available"""
        config_logger.info("🔍 Validating embedding model...")
        
        try:
            response = requests.get(f"{self.OLLAMA_API}/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            if self.EMBEDDING_MODEL not in models:
                config_logger.error(f"❌ Embedding model {self.EMBEDDING_MODEL} not found in Ollama!")
                config_logger.info(f"Available models: {models}")
                config_logger.info("💡 Pull the model with: ollama pull dengcao/Qwen3-Embedding-0.6B:Q8_0")
                return False
                
            config_logger.info(f"✅ Embedding model {self.EMBEDDING_MODEL} validated successfully")
            return True
            
        except requests.RequestException as e:
            config_logger.error(f"❌ Failed to connect to Ollama API: {e}")
            config_logger.info("💡 Ensure Ollama is running: ollama serve")
            return False
        except Exception as e:
            config_logger.error(f"❌ Model validation failed: {e}")
            config_logger.debug(f"Model validation error: {e}", exc_info=True)
            return False
    
    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal settings based on system resources"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count() or 1
        
        # Calculate optimal settings based on available resources
        available_memory_gb = memory.available / (1024**3)
        
        optimal_settings = {
            'worker_processes': min(8, cpu_count),
            'chunk_embedding_batch': min(200, int(available_memory_gb * 50)),  # 50 items per GB
            'chunk_insert_batch': min(1000, int(available_memory_gb * 100)),   # 100 items per GB
            'embedding_queue_memory_gb': min(4, available_memory_gb * 0.3),    # 30% of available
            'cache_memory_gb': min(2, available_memory_gb * 0.2),              # 20% of available
            'memory_cleanup_threshold': 85 if available_memory_gb > 8 else 75,
        }
        
        config_logger.info(f"Calculated optimal settings for {available_memory_gb:.1f}GB available memory")
        return optimal_settings
    
    def apply_optimal_settings(self) -> None:
        """Apply optimal settings based on system resources"""
        optimal_settings = self.get_optimal_settings()
        self.update(optimal_settings, persist=True)
        config_logger.info("Applied optimal settings based on system resources")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        self.runtime_config.clear()
        self.save_config()
        config_logger.info("Configuration reset to defaults")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            'database': {
                'host': self.DB_CONFIG['host'],
                'database': self.DB_CONFIG['dbname'],
                'table': self.TABLE_NAME
            },
            'embedding': {
                'model': self.EMBEDDING_MODEL,
                'dimension': self.EMB_DIM,
                'api_endpoint': self.OLLAMA_API
            },
            'performance': self.PERFORMANCE_CONFIG,
            'search': self.SEARCH_DEFAULTS,
            'chunks': self.CHUNK_SIZES,
            'files': {
                'supported_extensions': self.FILE_CONFIG['supported_extensions'],
                'max_file_size_mb': self.FILE_CONFIG['max_file_size'] / (1024*1024)
            }
        }
    
    def diagnose_and_fix_issues(self) -> Dict[str, Any]:
        """Comprehensive diagnosis and automatic fixing of configuration issues"""
        config_logger.info("🔍 Running comprehensive configuration diagnosis...")
        
        diagnosis = {
            'env_file': False,
            'env_variables': {},
            'database_connection': False,
            'database_exists': False,
            'user_exists': False,
            'pgvector_available': False,
            'ollama_connection': False,
            'embedding_model': False,
            'fixes_applied': [],
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Check .env file
            config_logger.debug("Checking .env file...")
            env_file = Path('.env')
            diagnosis['env_file'] = env_file.exists()
            if not diagnosis['env_file']:
                config_logger.warning("⚠️ .env file missing")
                diagnosis['issues'].append(".env file does not exist")
                diagnosis['recommendations'].append("Create .env file with required variables")
            
            # 2. Check environment variables
            config_logger.debug("Checking environment variables...")
            required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'TABLE_NAME']
            for var in required_vars:
                value = os.environ.get(var)
                diagnosis['env_variables'][var] = {
                    'exists': value is not None,
                    'empty': not value if value is not None else True,
                    'value': value if value else 'NOT_SET'
                }
                
                if not value:
                    config_logger.warning(f"⚠️ Environment variable {var} missing or empty")
                    diagnosis['issues'].append(f"Environment variable {var} not set")
            
            # 3. Test database connection
            config_logger.debug("Testing database connection...")
            db_config = self.DB_CONFIG
            diagnosis['database_connection'] = _test_database_connection(db_config)
            
            if not diagnosis['database_connection']:
                config_logger.warning("⚠️ Database connection failed")
                diagnosis['issues'].append("Cannot connect to database")
                
                # Try to fix database issues
                config_logger.info("🔧 Attempting to fix database issues...")
                if _setup_database_components(db_config):
                    diagnosis['fixes_applied'].append("Database setup completed")
                    # Test again
                    diagnosis['database_connection'] = _test_database_connection(db_config)
                    if diagnosis['database_connection']:
                        diagnosis['fixes_applied'].append("Database connection restored")
            
            # 4. Check pgvector extension
            if diagnosis['database_connection']:
                config_logger.debug("Checking pgvector extension...")
                try:
                    conn = psycopg2.connect(**db_config, connect_timeout=10)
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                        diagnosis['pgvector_available'] = cur.fetchone() is not None
                    conn.close()
                    
                    if not diagnosis['pgvector_available']:
                        config_logger.warning("⚠️ pgvector extension not available")
                        diagnosis['issues'].append("pgvector extension not installed")
                        diagnosis['recommendations'].append("Install pgvector extension")
                except Exception as e:
                    config_logger.debug(f"pgvector check failed: {e}")
            
            # 5. Check Ollama connection
            config_logger.debug("Checking Ollama connection...")
            try:
                response = requests.get(f"{self.OLLAMA_API}/tags", timeout=5)
                diagnosis['ollama_connection'] = response.status_code == 200
                
                if diagnosis['ollama_connection']:
                    # Check embedding model
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    diagnosis['embedding_model'] = self.EMBEDDING_MODEL in models
                    
                    if not diagnosis['embedding_model']:
                        config_logger.warning(f"⚠️ Embedding model {self.EMBEDDING_MODEL} not available")
                        diagnosis['issues'].append(f"Embedding model {self.EMBEDDING_MODEL} not found")
                        diagnosis['recommendations'].append(f"Pull model: ollama pull {self.EMBEDDING_MODEL}")
                else:
                    config_logger.warning("⚠️ Cannot connect to Ollama API")
                    diagnosis['issues'].append("Cannot connect to Ollama API")
                    diagnosis['recommendations'].append("Start Ollama service: ollama serve")
            except Exception as e:
                config_logger.debug(f"Ollama check failed: {e}")
                diagnosis['ollama_connection'] = False
                diagnosis['issues'].append("Ollama API not accessible")
            
            # Summary
            total_checks = 6
            passed_checks = sum([
                diagnosis['env_file'],
                len([v for v in diagnosis['env_variables'].values() if v['exists'] and not v['empty']]) == len(required_vars),
                diagnosis['database_connection'],
                diagnosis['pgvector_available'],
                diagnosis['ollama_connection'],
                diagnosis['embedding_model']
            ])
            
            config_logger.info(f"📊 Diagnosis complete: {passed_checks}/{total_checks} checks passed")
            
            if diagnosis['fixes_applied']:
                config_logger.info(f"🔧 Applied {len(diagnosis['fixes_applied'])} fixes:")
                for fix in diagnosis['fixes_applied']:
                    config_logger.info(f"   ✅ {fix}")
            
            if diagnosis['issues']:
                config_logger.warning(f"⚠️ Found {len(diagnosis['issues'])} issues:")
                for issue in diagnosis['issues']:
                    config_logger.warning(f"   ❌ {issue}")
            
            if diagnosis['recommendations']:
                config_logger.info(f"💡 Recommendations:")
                for rec in diagnosis['recommendations']:
                    config_logger.info(f"   💡 {rec}")
            
            return diagnosis
            
        except Exception as e:
            config_logger.error(f"❌ Diagnosis failed: {e}")
            config_logger.debug(f"Diagnosis error details: {e}", exc_info=True)
            diagnosis['issues'].append(f"Diagnosis failed: {e}")
            return diagnosis

def auto_fix_missing_env_vars():
    """Auto-fix commonly missing environment variables"""
    config_logger.info("🔧 Auto-fixing missing environment variables...")
    
    missing_vars = []
    
    # Check for DB_PORT specifically since it's commonly missing
    if not os.environ.get('DB_PORT'):
        config_logger.warning("⚠️ DB_PORT missing from environment")
        missing_vars.append('DB_PORT')
        if _auto_create_env_var('DB_PORT', '5432'):
            config_logger.info("✅ Added DB_PORT=5432 to .env")
    
    # Check other critical variables
    critical_vars = {
        'DB_NAME': 'rag_db',
        'DB_USER': 'postgres', 
        'DB_PASSWORD': 'postgres',
        'DB_HOST': 'localhost',
        'TABLE_NAME': 'rag_db_code',
        'EMB_DIM': '1024',
        'EMBEDDING_MODEL': 'dengcao/Qwen3-Embedding-0.6B:Q8_0',
        'OLLAMA_API': 'http://localhost:11434/api'
    }
    
    for var, default_val in critical_vars.items():
        if not os.environ.get(var):
            missing_vars.append(var)
            if _auto_create_env_var(var, default_val):
                config_logger.info(f"✅ Added {var}={default_val} to .env")
    
    if missing_vars:
        config_logger.info(f"🔄 Reloading .env file after adding {len(missing_vars)} variables")
        load_dotenv(override=True)
        config_logger.info("✅ Environment variables reloaded")
    
    return len(missing_vars)

# Auto-fix missing variables on module load
auto_fix_missing_env_vars()

# Global configuration instance
config = UnifiedConfig()

# Run comprehensive diagnosis and fixing on module load
config_logger.info("🚀 Running startup diagnosis and auto-fix...")
startup_diagnosis = config.diagnose_and_fix_issues()

# Validate configuration on module load
if not config.validate_config():
    config_logger.warning("⚠️ Configuration validation failed, some features may not work correctly")
    config_logger.info("💡 Run the diagnostic tool for detailed information")

# Legacy compatibility exports
Config = config  # For backward compatibility
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMB_DIM = config.EMB_DIM
DB_NAME = config.DB_NAME
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD
DB_HOST = config.DB_HOST
TABLE_NAME = config.TABLE_NAME
OLLAMA_API = config.OLLAMA_API
# database_setup.py - Standalone Database Setup Utility
import os
import sys
import subprocess
import psycopg2
import getpass
import logging
from pathlib import Path
from typing import Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='🔧 SETUP: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('database_setup.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def check_postgresql_installation() -> bool:
    """Check if PostgreSQL is installed"""
    logger.info("Checking PostgreSQL installation...")
    
    try:
        result = subprocess.run(['psql', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info(f"✅ PostgreSQL found: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    logger.error("❌ PostgreSQL not found. Please install PostgreSQL first.")
    logger.info("💡 Installation instructions:")
    logger.info("   Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib")
    logger.info("   CentOS/RHEL: sudo yum install postgresql postgresql-server")
    logger.info("   macOS: brew install postgresql")
    logger.info("   Arch Linux: sudo pacman -S postgresql")
    return False

def check_postgresql_service() -> bool:
    """Check if PostgreSQL service is running"""
    logger.info("Checking PostgreSQL service...")
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 5432))
        sock.close()
        
        if result == 0:
            logger.info("✅ PostgreSQL service is running")
            return True
        else:
            logger.warning("⚠️  PostgreSQL service not running on localhost:5432")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking PostgreSQL service: {e}")
        return False

def start_postgresql_service() -> bool:
    """Attempt to start PostgreSQL service"""
    logger.info("Attempting to start PostgreSQL service...")
    
    # Try different service management commands
    commands = [
        ['sudo', 'systemctl', 'start', 'postgresql'],
        ['sudo', 'service', 'postgresql', 'start'],
        ['brew', 'services', 'start', 'postgresql'],
        ['pg_ctl', 'start', '-D', '/usr/local/var/postgres']
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"✅ PostgreSQL started using: {' '.join(cmd)}")
                return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    logger.warning("⚠️  Could not start PostgreSQL automatically")
    logger.info("💡 Please start PostgreSQL manually:")
    logger.info("   Ubuntu/Debian: sudo systemctl start postgresql")
    logger.info("   CentOS/RHEL: sudo systemctl start postgresql")
    logger.info("   macOS: brew services start postgresql")
    return False

def get_database_credentials() -> Dict[str, str]:
    """Get database credentials from user or environment"""
    logger.info("Setting up database credentials...")
    
    # Check if credentials are in environment
    db_config = {
        'host': os.environ.get('DB_HOST', 'localhost'),
        'port': os.environ.get('DB_PORT', '5432'),
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'dbname': os.environ.get('DB_NAME', 'ragdb')
    }
    
    # Ask user for missing credentials
    if not db_config['user']:
        db_config['user'] = input("Database username (default: postgres): ").strip() or 'postgres'
    
    if not db_config['password']:
        db_config['password'] = getpass.getpass("Database password: ")
    
    return db_config

def test_connection(db_config: Dict[str, str]) -> bool:
    """Test database connection"""
    logger.info(f"Testing connection to {db_config['host']}:{db_config['port']}...")
    
    try:
        # First try connecting to the target database
        conn = psycopg2.connect(**db_config, connect_timeout=10)
        conn.close()
        logger.info(f"✅ Successfully connected to database '{db_config['dbname']}'")
        return True
        
    except psycopg2.OperationalError as e:
        if "does not exist" in str(e):
            logger.info(f"ℹ️  Database '{db_config['dbname']}' does not exist (will create)")
            return True  # This is expected if database doesn't exist yet
        else:
            logger.error(f"❌ Connection failed: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

def create_user(db_config: Dict[str, str]) -> bool:
    """Create database user if it doesn't exist"""
    logger.info(f"Creating user '{db_config['user']}'...")
    
    try:
        # Connect as postgres superuser
        admin_config = db_config.copy()
        admin_config['user'] = 'postgres'
        admin_config['dbname'] = 'postgres'
        
        # Ask for postgres password if needed
        if admin_config['user'] != db_config['user']:
            postgres_password = getpass.getpass("PostgreSQL admin (postgres) password: ")
            admin_config['password'] = postgres_password
        
        conn = psycopg2.connect(**admin_config, connect_timeout=10)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Check if user exists
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (db_config['user'],))
            if cur.fetchone():
                logger.info(f"✅ User '{db_config['user']}' already exists")
            else:
                # Create user
                cur.execute(f"""
                    CREATE USER "{db_config['user']}" 
                    WITH PASSWORD '{db_config['password']}' 
                    CREATEDB LOGIN
                """)
                logger.info(f"✅ Created user '{db_config['user']}'")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.warning(f"⚠️  User creation failed: {e}")
        logger.info("💡 You may need to create the user manually or use an existing user")
        return True  # Continue anyway

def create_database(db_config: Dict[str, str]) -> bool:
    """Create database if it doesn't exist"""
    logger.info(f"Creating database '{db_config['dbname']}'...")
    
    try:
        # Connect to postgres database to create our database
        admin_config = db_config.copy()
        admin_config['dbname'] = 'postgres'
        
        conn = psycopg2.connect(**admin_config, connect_timeout=10)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_config['dbname'],))
            if cur.fetchone():
                logger.info(f"✅ Database '{db_config['dbname']}' already exists")
            else:
                # Create database
                cur.execute(f'CREATE DATABASE "{db_config["dbname"]}"')
                logger.info(f"✅ Created database '{db_config['dbname']}'")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database creation failed: {e}")
        return False

def install_pgvector(db_config: Dict[str, str]) -> bool:
    """Install pgvector extension"""
    logger.info("Installing pgvector extension...")
    
    try:
        conn = psycopg2.connect(**db_config, connect_timeout=10)
        
        with conn.cursor() as cur:
            # Check if extension exists
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if cur.fetchone():
                logger.info("✅ pgvector extension already installed")
                conn.close()
                return True
            
            # Install extension
            cur.execute("CREATE EXTENSION vector")
            conn.commit()
            logger.info("✅ pgvector extension installed successfully")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        if "does not exist" in str(e):
            logger.error("❌ pgvector extension not available")
            logger.info("💡 Please install pgvector first:")
            logger.info("   GitHub: https://github.com/pgvector/pgvector")
            logger.info("   Ubuntu: sudo apt install postgresql-15-pgvector")
            logger.info("   macOS: brew install pgvector")
        else:
            logger.error(f"❌ pgvector installation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ pgvector installation failed: {e}")
        return False

def create_env_file(db_config: Dict[str, str]) -> bool:
    """Create or update .env file with database configuration"""
    logger.info("Creating/updating .env file...")
    
    try:
        env_file = Path('.env')
        env_vars = {
            'DB_HOST': db_config['host'],
            'DB_PORT': db_config['port'],
            'DB_USER': db_config['user'],
            'DB_PASSWORD': db_config['password'],
            'DB_NAME': db_config['dbname'],
            'TABLE_NAME': 'rag_db_code',
            'EMB_DIM': '1024',
            'EMBEDDING_MODEL': 'dengcao/Qwen3-Embedding-0.6B:Q8_0',
            'OLLAMA_API': 'http://localhost:11434/api'
        }
        
        # Read existing .env if it exists
        existing_vars = {}
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        existing_vars[key] = value
        
        # Merge with new variables (don't overwrite existing)
        for key, value in env_vars.items():
            if key not in existing_vars:
                existing_vars[key] = value
        
        # Write updated .env file
        with open(env_file, 'w') as f:
            f.write("# RAG Database Configuration\n")
            f.write("# Auto-generated by database setup utility\n\n")
            
            f.write("# Database Configuration\n")
            f.write(f"DB_HOST={existing_vars['DB_HOST']}\n")
            f.write(f"DB_PORT={existing_vars['DB_PORT']}\n")
            f.write(f"DB_USER={existing_vars['DB_USER']}\n")
            f.write(f"DB_PASSWORD={existing_vars['DB_PASSWORD']}\n")
            f.write(f"DB_NAME={existing_vars['DB_NAME']}\n")
            f.write(f"TABLE_NAME={existing_vars['TABLE_NAME']}\n\n")
            
            f.write("# Embedding Configuration\n")
            f.write(f"EMB_DIM={existing_vars['EMB_DIM']}\n")
            f.write(f"EMBEDDING_MODEL={existing_vars['EMBEDDING_MODEL']}\n")
            f.write(f"OLLAMA_API={existing_vars['OLLAMA_API']}\n")
        
        logger.info("✅ .env file created/updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create .env file: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting RAG Database Setup...")
    logger.info("="*50)
    
    # Check PostgreSQL installation
    if not check_postgresql_installation():
        return False
    
    # Check if PostgreSQL service is running
    if not check_postgresql_service():
        if not start_postgresql_service():
            logger.error("❌ PostgreSQL service is not running. Please start it manually.")
            return False
        
        # Check again after attempting to start
        if not check_postgresql_service():
            logger.error("❌ PostgreSQL service still not accessible")
            return False
    
    # Get database credentials
    db_config = get_database_credentials()
    
    # Test initial connection
    if not test_connection(db_config):
        logger.error("❌ Cannot connect to PostgreSQL")
        return False
    
    # Setup database components
    success = True
    
    if not create_user(db_config):
        success = False
    
    if not create_database(db_config):
        success = False
    
    if not install_pgvector(db_config):
        logger.warning("⚠️  pgvector not installed, but continuing...")
    
    if not create_env_file(db_config):
        success = False
    
    if success:
        logger.info("="*50)
        logger.info("🎉 Database setup completed successfully!")
        logger.info("✅ You can now run the RAG Database application")
        logger.info("="*50)
    else:
        logger.error("="*50)
        logger.error("❌ Database setup completed with errors")
        logger.error("Please check the logs and fix any issues")
        logger.error("="*50)
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        sys.exit(1)
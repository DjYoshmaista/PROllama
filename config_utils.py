# config_utils.py - Configuration management utilities
import os
import requests
import logging
import psutil
import sys
from config import Config
from constants import OLLAMA_API
from utils import db_cursor

logger = logging.getLogger(__name__)

def configure_embedding_parameters():
    """Menu for configuring embedding parameters"""
    print("\nEmbedding Configuration:")
    print(f"1. Relevance Threshold (current: {Config.SEARCH_DEFAULTS['relevance_threshold']:.2f})")
    print(f"2. Top K Results (current: {Config.SEARCH_DEFAULTS['top_k']})")
    print(f"3. Vector Search Limit/Chunk Size (current: {Config.SEARCH_DEFAULTS['vector_search_limit']})")
    print("4. Show current configuration")
    print("5. Reset to defaults")
    print("6. Back to main menu")
    
    choice = input("Select option: ")
    
    if choice == "1":
        try:
            current = Config.SEARCH_DEFAULTS['relevance_threshold']
            print(f"Current relevance threshold: {current:.2f}")
            print("Relevance threshold determines minimum similarity for results (0.0-1.0)")
            print("Lower values = more results, higher values = more selective")
            
            new_threshold = float(input("Enter new relevance threshold (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                Config.SEARCH_DEFAULTS['relevance_threshold'] = new_threshold
                print(f"Threshold updated from {current:.2f} to {new_threshold:.2f}")
            else:
                print("Invalid value. Must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    elif choice == "2":
        try:
            current = Config.SEARCH_DEFAULTS['top_k']
            print(f"Current Top K: {current}")
            print("Top K determines maximum number of results to return")
            
            new_top_k = int(input("Enter new Top K value: "))
            if new_top_k > 0:
                Config.SEARCH_DEFAULTS['top_k'] = new_top_k
                print(f"Top K updated from {current} to {new_top_k}")
            else:
                print("Value must be positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
            
    elif choice == "3":
        try:
            current = Config.SEARCH_DEFAULTS['vector_search_limit']
            print(f"Current vector search limit: {current}")
            print("This limits how many vectors are searched at once (affects memory usage)")
            
            new_limit = int(input("Enter new vector search limit/chunk size: "))
            if new_limit > 0:
                Config.SEARCH_DEFAULTS['vector_search_limit'] = new_limit
                print(f"Vector search limit updated from {current} to {new_limit}")
            else:
                print("Value must be positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
            
    elif choice == "4":
        show_current_configuration()
        
    elif choice == "5":
        confirm = input("Reset all embedding parameters to defaults? (y/n): ").strip().lower()
        if confirm == 'y':
            # Store original values
            original = Config.SEARCH_DEFAULTS.copy()
            
            # Reset to defaults (you'll need to define these in config.py)
            Config.SEARCH_DEFAULTS = {
                'relevance_threshold': 0.7,
                'top_k': 10,
                'vector_search_limit': 100000
            }
            
            print("Parameters reset to defaults:")
            print(f"  Relevance threshold: {original['relevance_threshold']:.2f} → {Config.SEARCH_DEFAULTS['relevance_threshold']:.2f}")
            print(f"  Top K: {original['top_k']} → {Config.SEARCH_DEFAULTS['top_k']}")
            print(f"  Vector search limit: {original['vector_search_limit']} → {Config.SEARCH_DEFAULTS['vector_search_limit']}")
        
    elif choice == "6":
        return
    else:
        print("Invalid option.")

def show_current_configuration():
    """Display current configuration settings"""
    print("\n🔧 Current Configuration:")
    print("="*50)
    
    # Search defaults
    print("Search Parameters:")
    print(f"  Relevance Threshold: {Config.SEARCH_DEFAULTS['relevance_threshold']:.2f}")
    print(f"  Top K Results: {Config.SEARCH_DEFAULTS['top_k']}")
    print(f"  Vector Search Limit: {Config.SEARCH_DEFAULTS['vector_search_limit']:,}")
    
    # Chunk sizes if available
    if hasattr(Config, 'CHUNK_SIZES'):
        print("\nProcessing Chunk Sizes:")
        for key, value in Config.CHUNK_SIZES.items():
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
    
    # Worker processes
    if hasattr(Config, 'WORKER_PROCESSES'):
        print(f"\nWorker Processes: {Config.WORKER_PROCESSES}")
    
    # Memory settings
    if hasattr(Config, 'MEMORY_CLEANUP_THRESHOLD'):
        print(f"Memory Cleanup Threshold: {Config.MEMORY_CLEANUP_THRESHOLD}%")
    
    # Database settings
    print(f"\nDatabase Table: {Config.TABLE_NAME}")

def configure_postgres():
    """Apply PostgreSQL configuration optimizations"""
    try:
        print("\nPostgreSQL Configuration Options:")
        print("1. Apply optimizations from postgresql.conf")
        print("2. Show current PostgreSQL settings")
        print("3. Test database connection")
        print("4. Back to main menu")
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            apply_postgres_optimizations()
        elif choice == "2":
            show_postgres_settings()
        elif choice == "3":
            test_database_connection()
        elif choice == "4":
            return
        else:
            print("Invalid option.")
            
    except Exception as e:
        logger.error(f"PostgreSQL configuration menu failed: {e}")
        print(f"Error in PostgreSQL configuration: {e}")

def apply_postgres_optimizations():
    """Apply PostgreSQL configuration optimizations"""
    try:
        config_file = "postgresql.conf"
        if not os.path.exists(config_file):
            print(f"Configuration file {config_file} not found.")
            print("Please ensure postgresql.conf exists in the current directory.")
            return
        
        with db_cursor() as (conn, cur):
            applied_count = 0
            failed_count = 0
            
            print("Applying PostgreSQL optimizations...")
            
            with open(config_file, "r") as conf_file:
                for line_num, line in enumerate(conf_file, 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            if "=" in line:
                                setting = line.split("=")[0].strip()
                                value = line.split("=", 1)[1].strip().replace("'", "")
                                
                                cur.execute(f"ALTER SYSTEM SET {setting} = %s", (value,))
                                applied_count += 1
                                print(f"  ✓ Set {setting} = {value}")
                        except Exception as e:
                            failed_count += 1
                            logger.warning(f"Couldn't set {setting} (line {line_num}): {str(e)}")
                            print(f"  ✗ Failed to set {setting}: {str(e)}")
            
            conn.commit()
            
            print(f"\nConfiguration complete:")
            print(f"  ✓ Applied: {applied_count} settings")
            print(f"  ✗ Failed: {failed_count} settings")
            print("\n⚠️  Please restart PostgreSQL for changes to take effect.")
            
    except Exception as e:
        logger.error(f"Configuration failed: {str(e)}")
        print(f"Failed to configure PostgreSQL: {str(e)}")

def show_postgres_settings():
    """Show current PostgreSQL settings"""
    try:
        with db_cursor() as (conn, cur):
            # Get key performance settings
            settings_to_check = [
                'shared_buffers',
                'effective_cache_size',
                'maintenance_work_mem',
                'checkpoint_completion_target',
                'wal_buffers',
                'default_statistics_target',
                'random_page_cost',
                'effective_io_concurrency',
                'work_mem'
            ]
            
            print("\n🔧 Current PostgreSQL Settings:")
            print("="*50)
            
            for setting in settings_to_check:
                try:
                    cur.execute("SELECT current_setting(%s)", (setting,))
                    value = cur.fetchone()[0]
                    print(f"  {setting:<25}: {value}")
                except Exception as e:
                    print(f"  {setting:<25}: Unable to retrieve ({e})")
            
            # Show version
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            print(f"\nPostgreSQL Version: {version.split(',')[0]}")
            
    except Exception as e:
        logger.error(f"Failed to show PostgreSQL settings: {e}")
        print(f"Error retrieving PostgreSQL settings: {e}")

def test_database_connection():
    """Test database connection and basic functionality"""
    try:
        print("Testing database connection...")
        
        with db_cursor() as (conn, cur):
            # Test basic connection
            cur.execute("SELECT 1")
            result = cur.fetchone()
            if result[0] == 1:
                print("  ✓ Database connection successful")
            
            # Test table existence
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{Config.TABLE_NAME}'
                )
            """)
            table_exists = cur.fetchone()[0]
            if table_exists:
                print(f"  ✓ Table '{Config.TABLE_NAME}' exists")
                
                # Get table stats
                cur.execute(f"SELECT COUNT(*) FROM {Config.TABLE_NAME}")
                count = cur.fetchone()[0]
                print(f"  ℹ️  Records in table: {count:,}")
            else:
                print(f"  ⚠️  Table '{Config.TABLE_NAME}' does not exist")
            
            # Test pgvector extension
            try:
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                vector_ext = cur.fetchone()
                if vector_ext:
                    print("  ✓ pgvector extension available")
                else:
                    print("  ✗ pgvector extension not found")
            except Exception as e:
                print(f"  ✗ Error checking pgvector: {e}")
        
        print("Database test completed.")
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        print(f"✗ Database test failed: {e}")

def validate_embedding_model():
    """Validate that the required embedding model is available"""
    try:
        print("Validating embedding model...")
        
        response = requests.get(f"{OLLAMA_API}/tags", timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        
        # Get the list of models
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        
        # Define the expected model name
        expected_model = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
        
        print(f"Checking for model: {expected_model}")
        
        # Check if the model exists
        if expected_model not in models:
            logger.error(f"Embedding model {expected_model} not found in Ollama!")
            print(f"✗ Required embedding model not found: {expected_model}")
            print(f"Available models: {len(models)} total")
            for model in models[:5]:  # Show first 5 models
                print(f"  - {model}")
            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")
            return False
            
        logger.info(f"Embedding model {expected_model} validated successfully")
        print(f"✓ Embedding model validated: {expected_model}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Failed to connect to Ollama API: {str(e)}")
        print(f"✗ Failed to connect to Ollama API: {str(e)}")
        print("Please ensure Ollama is running and accessible.")
        return False
    except KeyError as e:
        logger.error(f"Unexpected API response structure: {str(e)}")
        print(f"✗ Unexpected API response structure: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        print(f"✗ Model validation failed: {str(e)}")
        return False

def show_system_info():
    """Display system information and status"""
    import psutil
    import sys
    
    print("\n💻 System Information:")
    print("="*50)
    
    # Python version
    print(f"Python Version: {sys.version.split()[0]}")
    
    # System resources
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB ({memory.percent:.1f}%)")
    
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent()
    print(f"CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")
    
    # Disk space
    disk = psutil.disk_usage('.')
    print(f"Disk: {disk.used/(1024**3):.1f}GB / {disk.total/(1024**3):.1f}GB ({(disk.used/disk.total)*100:.1f}%)")
    
    # Load average (Unix-like systems)
    try:
        load_avg = psutil.getloadavg()
        print(f"Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    except AttributeError:
        # Windows doesn't have load average
        pass
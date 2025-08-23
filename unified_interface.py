# unified_interface.py
import asyncio
import logging
import os
import sys
import time
import multiprocessing
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path

# CRITICAL: Enable all logging immediately
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ragdb_debug.log", encoding='utf-8', mode='w'),  # Fresh log file
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("🚀 unified_interface.py starting - ENHANCED DIAGNOSTIC MODE")

# Force environment diagnostic before any imports
logger.info("🔍 PRE-IMPORT ENVIRONMENT DIAGNOSTIC:")
env_vars = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'TABLE_NAME']
for var in env_vars:
    value = os.environ.get(var)
    logger.info(f"   {var}: {value if value else 'NOT_SET'}")

# Test .env file existence and content
logger.info("🔍 .ENV FILE DIAGNOSTIC:")
env_file_path = '.env'
if os.path.exists(env_file_path):
    logger.info(f"   .env file exists: {env_file_path}")
    try:
        with open(env_file_path, 'r') as f:
            content = f.read()
            logger.info(f"   .env file size: {len(content)} characters")
            logger.info(f"   .env file content:")
            for i, line in enumerate(content.split('\n'), 1):
                if line.strip() and not line.startswith('#'):
                    logger.info(f"      Line {i}: {line}")
    except Exception as env_read_error:
        logger.error(f"   Failed to read .env file: {env_read_error}")
else:
    logger.warning(f"   .env file does not exist: {env_file_path}")

# Import modules with detailed tracking
logger.info("📦 IMPORTING MODULES WITH TRACKING:")

try:
    logger.info("   Importing file_processor...")
    from document_processor import file_processor, file_tracker, document_parser
    logger.info(f"   ✅ file_processor imported, id: {id(file_processor)}\n  ✅ file_tracker improted, id: {id(file_tracker)}\n  ✅ document_parser imported, id: {id(document_parser)}")
except Exception as file_processor_import_error:
    logger.error(f"   ❌ file_processor import failed: {file_processor_import_error}")
    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("   Importing search_engine...")
    from search_engine import search_engine
    logger.info(f"   ✅ search_engine imported, id: {id(search_engine)}")
except Exception as search_engine_import_error:
    logger.error(f"   ❌ search_engine import failed: {search_engine_import_error}")
    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
    raise

# try:
#     logger.info("   Importing answer_generator...")
#     from answer_generator import answer_generator
#     logger.info(f"   ✅ answer_generator imported, id: {id(answer_generator)}")
# except Exception as answer_generator_import_error:
#     logger.error(f"   ❌ answer_generator import failed: {answer_generator_import_error}")
#     logger.error(f"   📋 Traceback: {traceback.format_exc()}")
#     raise

try:
    logger.info("   Importing embedding_service...")
    from embedding_manager import embedding_service
    logger.info(f"   ✅ embedding_service imported, id: {id(embedding_service)}")
except Exception as embedding_service_import_error:
    logger.error(f"   ❌ embedding_service import failed: {embedding_service_import_error}")
    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("   Improting embedding_queue...")
    from embedding_manager import embedding_queue
    logger.info(f"   ✅ embedding_queue imported, id: {id(embedding_queue)}")
except Exception as embedding_queue_import_error:
    logger.error(f"   ❌ embedding_queue import failed: {embedding_queue_import_error}")
    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("  Importing async_processor...")
    from async_processor import async_processor
    logger.info(f"   ✅ async_processor imported, id: {id(async_processor)}")
except Exception as async_processor_import_error:
    logger.error(f"   ❌ async_processor import failed: {async_processor_import_error}")
    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("   Importing core_config...")
    from core_config import config
    logger.info(f"   ✅ core_config imported, config id: {id(config)}")
except Exception as config_import_error:
    logger.error(f"   ❌ core_config import failed: {config_import_error}")
    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("   Importing database_manager...")
    from database_manager import db_manager, init_db, diagnose_db_issues
    logger.info(f"   ✅ database_manager imported, db_manager id: {id(db_manager)}")
    logger.info(f"   🔍 db_manager type: {type(db_manager)}")
    logger.info(f"   🔍 db_manager._initialized: {getattr(db_manager, '_initialized', 'NOT_FOUND')}")
except Exception as db_import_error:
    logger.error(f"   ❌ database_manager import failed: {db_import_error}")
    logger.error(f"   📋 Traceback: {traceback.format_exc()}")
    raise

# Test database manager configuration immediately after import
logger.info("🔍 IMMEDIATE DATABASE MANAGER DIAGNOSTIC:")
try:
    logger.info(f"   db_manager instance id: {id(db_manager)}")
    logger.info(f"   db_manager._db_config exists: {hasattr(db_manager, '_db_config')}")
    
    if hasattr(db_manager, '_db_config'):
        db_config = db_manager._db_config
        logger.info(f"   db_manager._db_config: {dict(db_config, password='***HIDDEN***') if db_config else 'NOT_SET'}")
    
    logger.info(f"   db_manager._connection_string exists: {hasattr(db_manager, '_connection_string')}")
    
    if hasattr(db_manager, '_connection_string'):
        conn_str = db_manager._connection_string
        logger.info(f"   db_manager._connection_string: {conn_str}")
    
    # Test connection_params property
    logger.info("   Testing db_manager.connection_params...")
    try:
        conn_params = db_manager.connection_params
        logger.info(f"   ✅ connection_params retrieved: {dict(conn_params, password='***HIDDEN***')}")
    except Exception as params_error:
        logger.error(f"   ❌ connection_params failed: {params_error}")
        logger.error(f"   📋 Params traceback: {traceback.format_exc()}")
    
except Exception as db_diag_error:
    logger.error(f"   ❌ Database manager diagnostic failed: {db_diag_error}")
    logger.error(f"   📋 DB diag traceback: {traceback.format_exc()}")

# Continue with other imports...
logger.info("   Importing remaining modules...")

class ProgressTracker:
    """Enhanced progress tracker for all operations"""
    
    def __init__(self, description: str):
        self.description = description
        self.start_time = time.time()
        self.detailed_stats = {
            'files_processed': 0,
            'items_queued': 0,
            'items_completed': 0,
            'current_file': None,
            'stage': 'initializing',
            'start_time': time.time()
        }
    
    def update(self, current: int, total: int, message: str = ""):
        """Update progress display"""
        elapsed = time.time() - self.start_time
        percentage = (current / total) * 100 if total > 0 else 0
        rate = current / elapsed if elapsed > 0 else 0
        
        status = f"\r{self.description}: {current:,}/{total:,} ({percentage:.1f}%) [{rate:.1f}/s] {message}"
        print(status, end="", flush=True)
    
    def update_file_progress(self, filename: str, stage: str, current: int, total: int, message: str):
        """Update file-specific progress"""
        self.detailed_stats.update({
            'current_file': filename,
            'stage': stage,
            'last_message': message
        })
        self.update(current, total, f"{stage}: {filename} - {message}")
    
    def update_overall_progress(self, processed: int, total: int, success: int, failed: int, items_queued: int):
        """Update overall processing progress"""
        self.detailed_stats.update({
            'files_processed': processed,
            'files_total': total,
            'files_success': success,
            'files_failed': failed,
            'items_queued': items_queued
        })
        message = f"Files: {processed}/{total} | Success: {success} | Failed: {failed} | Queued: {items_queued}"
        self.update(processed, total, message)
    
    def update_system_stats(self, cpu_percent: float, memory_percent: float, load_avg: float):
        """Update system resource statistics"""
        pass  # Simple tracker doesn't display system stats
    
    def update_queue_stats(self, queue_size: int, active_workers: int):
        """Update embedding queue statistics"""
        pass  # Simple tracker doesn't display queue stats
    
    def start_processing(self, total_files: int):
        """Initialize processing"""
        self.detailed_stats.update({
            'files_total': total_files,
            'stage': 'processing'
        })
    
    def finish_processing(self, success_count: int, failed_count: int, total_items: int):
        """Finalize processing"""
        elapsed = time.time() - self.detailed_stats['start_time']
        self.detailed_stats.update({
            'stage': 'completed',
            'elapsed_time': elapsed,
            'final_success': success_count,
            'final_failed': failed_count,
            'total_items_processed': total_items
        })
        self.complete(total_items)
    
    def report_error(self, error_message: str):
        """Report an error"""
        self.detailed_stats.update({
            'stage': 'error',
            'error_message': error_message
        })
        logger.error(f"Processing error: {error_message}")
    
    def complete(self, total_processed: int):
        """Mark operation as complete"""
        elapsed = time.time() - self.start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"\n✅ {self.description} complete: {total_processed:,} items in {elapsed:.2f}s ({rate:.1f}/s)")

class UnifiedInterface:
    """Main application interface with all functionality"""
    
    def __init__(self):
        self.running = True
    
    async def initialize(self) -> bool:
        """Initialize all systems with comprehensive diagnostics"""
        try:
            logger.info("🚀 UnifiedInterface.initialize() called")
            
            # Step 1: Pre-initialization diagnostics
            logger.info("📊 STEP 1: Pre-initialization comprehensive diagnostics")
            
            # Test config state
            logger.info("   🔍 Config state diagnostic:")
            try:
                logger.info(f"      Config object id: {id(config)}")
                logger.info(f"      Config DB_NAME: {getattr(config, 'DB_NAME', 'NOT_FOUND')}")
                logger.info(f"      Config DB_USER: {getattr(config, 'DB_USER', 'NOT_FOUND')}")
                logger.info(f"      Config DB_HOST: {getattr(config, 'DB_HOST', 'NOT_FOUND')}")
                logger.info(f"      Config DB_PORT: {getattr(config, 'DB_PORT', 'NOT_FOUND')}")
                
                # Test DB_CONFIG property
                db_config_test = config.DB_CONFIG
                logger.info(f"      Config DB_CONFIG: {dict(db_config_test, password='***HIDDEN***')}")
                
            except Exception as config_test_error:
                logger.error(f"      ❌ Config test failed: {config_test_error}")
                logger.error(f"      📋 Config test traceback: {traceback.format_exc()}")
            
            # Test database manager state
            logger.info("   🔍 Database manager state diagnostic:")
            try:
                logger.info(f"      DatabaseManager id: {id(db_manager)}")
                logger.info(f"      DatabaseManager type: {type(db_manager)}")
                logger.info(f"      DatabaseManager._initialized: {getattr(db_manager, '_initialized', 'NOT_FOUND')}")
                
                if hasattr(db_manager, '_db_config'):
                    db_config = db_manager._db_config
                    logger.info(f"      DatabaseManager._db_config: {dict(db_config, password='***HIDDEN***') if db_config else 'NOT_SET'}")
                else:
                    logger.error(f"      ❌ DatabaseManager._db_config not found")
                
                # Force test connection_params again
                logger.info("      Testing connection_params again...")
                conn_params = db_manager.connection_params
                logger.info(f"      ✅ Connection params: {dict(conn_params, password='***HIDDEN***')}")
                
            except Exception as db_mgr_error:
                logger.error(f"      ❌ Database manager test failed: {db_mgr_error}")
                logger.error(f"      📋 DB manager traceback: {traceback.format_exc()}")
            
            # Step 2: Manual database connection test
            logger.info("📊 STEP 2: Manual database connection test")
            try:
                import psycopg2
                
                # Get connection parameters manually
                manual_params = {
                    'dbname': config.DB_NAME,
                    'user': config.DB_USER,
                    'password': config.DB_PASSWORD,
                    'host': config.DB_HOST,
                    'port': int(config.DB_PORT),
                    'connect_timeout': 10
                }
                
                logger.info(f"   Manual connection params: {dict(manual_params, password='***HIDDEN***')}")
                
                # Test manual connection
                logger.info("   Attempting manual psycopg2 connection...")
                manual_conn = psycopg2.connect(**manual_params)
                
                with manual_conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    logger.info(f"   ✅ Manual connection successful: {result}")
                
                manual_conn.close()
                logger.info("   ✅ Manual connection test passed")
                
            except Exception as manual_error:
                logger.error(f"   ❌ Manual connection test failed: {manual_error}")
                logger.error(f"   📋 Manual connection traceback: {traceback.format_exc()}")
            
            # Step 3: DatabaseManager connection test with full tracing
            logger.info("📊 STEP 3: DatabaseManager connection test")
            try:
                logger.info("   Calling db_manager.test_connection()...")
                db_connection_result = db_manager.test_connection()
                logger.info(f"   DatabaseManager connection result: {db_connection_result}")
                
                if not db_connection_result:
                    logger.error("   ❌ DatabaseManager connection test failed")
                    
                    # Try to force re-initialization of DatabaseManager
                    logger.info("   🔄 Attempting to force re-initialize DatabaseManager...")
                    db_manager._initialized = False
                    
                    # Trigger re-initialization by accessing connection_params
                    try:
                        new_params = db_manager.connection_params
                        logger.info(f"   ✅ Re-initialization successful: {dict(new_params, password='***HIDDEN***')}")
                        
                        # Test again
                        db_connection_result = db_manager.test_connection()
                        logger.info(f"   Re-test result: {db_connection_result}")
                        
                    except Exception as reinit_error:
                        logger.error(f"   ❌ Re-initialization failed: {reinit_error}")
                        
            except Exception as db_test_error:
                logger.error(f"   ❌ DatabaseManager test failed: {db_test_error}")
                logger.error(f"   📋 DatabaseManager test traceback: {traceback.format_exc()}")
            
            # Continue with normal initialization only if we have a working connection
            logger.info("📊 STEP 4: Standard initialization process")
            
            # Validate configuration
            if not config.validate_config():
                print("❌ Configuration validation failed!")
                return False
            
            # Initialize database
            if not init_db():
                print("❌ Database initialization failed!")
                return False
            
            print("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(f"Full initialization traceback: {traceback.format_exc()}")
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def run(self):
        """Main application loop"""
        if not await self.initialize():
            return
        
        # Show startup information
        await self._show_startup_info()
        
        while self.running:
            await self._show_main_menu()
    
    async def _show_startup_info(self):
        """Display startup information and statistics"""
        print("\n" + "="*60)
        print("🚀 ENHANCED RAG DATABASE SYSTEM")
        print("="*60)
        
        # Show configuration summary
        config_summary = config.get_config_summary()
        print(f"📊 Configuration Summary:")
        print(f"   Database: {config_summary['database']['host']}/{config_summary['database']['database']}")
        print(f"   Table: {config_summary['database']['table']}")
        print(f"   Embedding Model: {config_summary['embedding']['model']}")
        print(f"   Embedding Dimension: {config_summary['embedding']['dimension']}")
        print(f"   Worker Processes: {config_summary['performance']['worker_processes']}")
        
        # Show database statistics
        db_stats = db_manager.get_stats()
        if db_stats.get('table_exists'):
            print(f"   Records in Database: {db_stats['record_count']:,}")
            print(f"   Database Size: {db_stats['table_size']}")
        
        # Show file processing statistics
        file_stats = file_processor.tracker.get_processed_files_stats()
        if file_stats['total_files'] > 0:
            print(f"   Files Processed: {file_stats['total_files']:,}")
            print(f"   Total Records Created: {file_stats['total_records']:,}")
        
        print("="*60)
    
    async def _show_main_menu(self):
        """Display main menu and handle user selection"""
        print("\n🔧 MAIN MENU:")
        print("1.  Ask questions (Search & Answer)")
        print("2.  Add single document manually")
        print("3.  Process single file")
        print("4.  Process folder (bulk processing)")
        print("5.  Search documents")
        print("6.  Configuration management")
        print("7.  System diagnostics")
        print("8.  Database management")
        print("9.  Processing status")
        print("10. Advanced operations")
        print("11. Exit")
        
        try:
            choice = input("\nSelect option (1-11): ").strip()
            
            if choice == "1":
                await self._ask_questions()
            elif choice == "2":
                await self._add_manual_document()
            elif choice == "3":
                await self._process_single_file()
            elif choice == "4":
                await self._process_folder()
            elif choice == "5":
                await self._search_documents()
            elif choice == "6":
                await self._configuration_menu()
            elif choice == "7":
                await self._system_diagnostics()
            elif choice == "8":
                await self._database_menu()
            elif choice == "9":
                await self._show_processing_status()
            elif choice == "10":
                await self._advanced_menu()
            elif choice == "11":
                await self._exit_application()
            else:
                print("Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
        except Exception as e:
            logger.error(f"Menu error: {e}")
            print(f"Error: {e}")
    
    async def _ask_questions(self):
        """Interactive question answering"""
        print("\n🤖 QUESTION ANSWERING")
        print("="*40)
        
        while True:
            try:
                question = input("\nEnter your question (or 'back' to return): ").strip()
                
                if question.lower() == 'back':
                    break
                
                if not question:
                    print("Please enter a question.")
                    continue
                
                print("\nSearching for relevant information...")
                tracker = ProgressTracker("Processing query")
                
                # Search for relevant documents
                tracker.update(1, 4, "Searching documents...")
                results = await search_engine.search(
                    query=question,
                    method="auto",
                    relevance_threshold=config.SEARCH_DEFAULTS['relevance_threshold'],
                    top_k=config.SEARCH_DEFAULTS['top_k']
                )
                
                if not results:
                    tracker.complete(0)
                    print("❌ No relevant documents found.")
                    continue
                
                # Generate answer
                tracker.update(3, 4, "Generating answer...")
                answer = answer_generator.generate_answer(question, results)
                tracker.complete(len(results))
                
                # Display results
                print(f"\n💡 Answer:")
                print(f"{answer}")
                
                print(f"\n📚 Sources ({len(results)} documents):")
                for i, doc in enumerate(results[:3], 1):
                    similarity = doc.get('cosine_similarity', 0)
                    file_path = doc.get('file_path', 'Unknown')
                    filename = os.path.basename(file_path) if file_path else 'Unknown'
                    print(f"   {i}. {filename} (similarity: {similarity:.3f})")
                    print(f"      {doc['content'][:100]}...")
                
                if len(results) > 3:
                    print(f"   ... and {len(results) - 3} more documents")
                
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                break
            except Exception as e:
                logger.error(f"Question answering error: {e}")
                print(f"Error: {e}")
    
    async def _add_manual_document(self):
        """Add a single document manually"""
        print("\n📝 ADD DOCUMENT MANUALLY")
        print("="*40)
        
        try:
            content = input("Enter document content: ").strip()
            if not content:
                print("Content cannot be empty!")
                return
            
            tags_input = input("Enter tags (comma-separated, optional): ").strip()
            tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()] if tags_input else []
            
            tracker = ProgressTracker("Adding document")
            tracker.update(0, 3, "Generating embedding...")
            
            # Generate embedding
            embedding = await embedding_service.generate_embedding(content)
            if not embedding:
                print("❌ Failed to generate embedding.")
                return
            
            tracker.update(1, 3, "Inserting into database...")
            
            # Insert into database
            records = [{
                'content': content,
                'tags': tags,
                'embedding': embedding,
                'file_path': 'manual_entry',
                'chunk_index': 0,
                'metadata': {'source': 'manual', 'timestamp': time.time()}
            }]
            
            inserted_count = await db_manager.insert_records_batch(records)
            
            tracker.complete(inserted_count)
            print(f"✅ Document added successfully! ({len(content)} characters)")
            
        except Exception as e:
            logger.error(f"Manual document addition failed: {e}")
            print(f"❌ Failed to add document: {e}")
    
    def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single file and return detailed results including record IDs
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Dictionary with processing results and record IDs for embedding
        """
        logger.info(f"🔧 process_file() called for: {file_path}")
        
        try:
            # Track record IDs for embedding queue
            created_record_ids = []
            
            # Check if file was already processed
            if self.is_file_processed(file_path):
                logger.debug(f"   ⏭️ File already processed: {file_path}")
                return {
                    'success': True,
                    'already_processed': True,
                    'records_created': 0,
                    'record_ids': [],
                    'message': 'File already processed'
                }
            
            # Parse the file
            parsed_result = self.parser.parse_file(file_path)
            
            if not parsed_result or not parsed_result.get('success'):
                logger.warning(f"   ⚠️ Failed to parse file: {file_path}")
                return {
                    'success': False,
                    'records_created': 0,
                    'record_ids': [],
                    'message': 'File parsing failed'
                }
            
            documents = parsed_result.get('documents', [])
            if not documents:
                logger.warning(f"   ⚠️ No documents extracted from: {file_path}")
                return {
                    'success': False,
                    'records_created': 0,
                    'record_ids': [],
                    'message': 'No documents extracted'
                }
            
            # Store documents in database
            records_created = 0
            
            with db_manager.get_sync_cursor() as (conn, cur):
                for i, doc in enumerate(documents):
                    try:
                        # Insert document record
                        insert_query = f"""
                            INSERT INTO {config.TABLE_NAME} 
                            (content, file_path, chunk_index, metadata, tags)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                        """
                        
                        # Prepare metadata
                        metadata = {
                            'file_size': file_path.stat().st_size,
                            'file_type': file_path.suffix.lower(),
                            'processing_timestamp': time.time(),
                            'chunk_count': len(documents),
                            **doc.get('metadata', {})
                        }
                        
                        # Execute insert
                        cur.execute(insert_query, (
                            doc['content'],
                            str(file_path),
                            i,
                            json.dumps(metadata),
                            doc.get('tags', [])
                        ))
                        
                        # Get the inserted record ID
                        result = cur.fetchone()
                        if result:
                            record_id = result['id']  # Handle RealDictCursor result
                            created_record_ids.append(record_id)
                            records_created += 1
                            
                            logger.debug(f"   📝 Created record {record_id} for chunk {i}")
                        
                    except Exception as insert_error:
                        logger.error(f"   ❌ Error inserting document {i}: {insert_error}")
                        continue
                
                # Commit all inserts
                conn.commit()
            
            # Mark file as processed
            self.mark_file_processed(file_path, records_created)
            
            logger.info(f"   ✅ Successfully processed {file_path}: {records_created} records")
            
            return {
                'success': True,
                'records_created': records_created,
                'record_ids': created_record_ids,
                'file_path': str(file_path),
                'message': f'Successfully created {records_created} records'
            }
            
        except Exception as e:
            logger.error(f"❌ process_file() failed for {file_path}: {e}")
            logger.debug(f"📋 Process file traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'records_created': 0,
                'record_ids': [],
                'file_path': str(file_path),
                'message': f'Processing failed: {str(e)}'
            }
    
    async def _process_folder(self, folder_path: str, file_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Process folder with concurrent file processing and embedding generation
        
        Args:
            folder_path: Path to folder to process
            file_patterns: Optional list of file patterns to match
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"🚀 process_folder() called - Enhanced concurrent processing")
        logger.info(f"   📁 Folder: {folder_path}")
        logger.info(f"   🔍 Patterns: {file_patterns or ['*']}")
        
        try:
            # Import concurrent processor
            from async_processor import concurrent_processor
            
            # Find files to process
            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")
            
            # Collect files based on patterns
            file_paths = []
            patterns = file_patterns or ['*']
            
            for pattern in patterns:
                if '*' in pattern or '?' in pattern:
                    # Glob pattern
                    file_paths.extend(folder.rglob(pattern))
                else:
                    # Extension pattern
                    file_paths.extend(folder.rglob(f"*.{pattern.lstrip('.')}"))
            
            # Remove duplicates and ensure they're files
            file_paths = list(set([p for p in file_paths if p.is_file()]))
            
            logger.info(f"   📊 Found {len(file_paths)} files to process")
            
            if not file_paths:
                logger.warning("   ⚠️ No files found matching patterns")
                return {
                    'success': False,
                    'message': 'No files found matching the specified patterns',
                    'files_processed': 0,
                    'files_failed': 0,
                    'records_created': 0,
                    'embeddings_generated': 0
                }
            
            # Setup progress tracking
            progress_data = {
                'total_files': len(file_paths),
                'processed_files': 0,
                'failed_files': 0,
                'current_file': None,
                'records_created': 0,
                'embeddings_generated': 0,
                'embeddings_pending': 0,
                'start_time': time.time()
            }
            
            def progress_callback(update_data: Dict[str, Any]):
                """Update progress tracking"""
                progress_data.update(update_data)
                
                # Calculate progress percentage
                progress_pct = (progress_data['processed_files'] / progress_data['total_files']) * 100
                elapsed = time.time() - progress_data['start_time']
                
                # Calculate rates
                file_rate = progress_data['processed_files'] / elapsed if elapsed > 0 else 0
                
                # Print progress update
                print(f"\r📊 Progress: {progress_pct:.1f}% | "
                    f"Files: {progress_data['processed_files']}/{progress_data['total_files']} "
                    f"({file_rate:.1f}/s) | "
                    f"Records: {progress_data['records_created']} | "
                    f"Embeddings: {progress_data['embeddings_generated']} | "
                    f"Pending: {progress_data['embeddings_pending']}", end='')
            
            # Start concurrent processing
            logger.info("   🚀 Starting concurrent file processing with embedding generation...")
            
            results = await concurrent_processor.process_files_concurrent(
                file_paths=file_paths,
                progress_callback=progress_callback
            )
            
            # Print final newline for clean output
            print()
            
            # Prepare final results
            total_time = results.get('total_time', 0)
            final_results = {
                'success': True,
                'message': f'Successfully processed {results["files_processed"]} files with concurrent embedding generation',
                'files_processed': results['files_processed'],
                'files_failed': results['files_failed'],
                'records_created': results['records_created'],
                'embeddings_generated': results['embeddings_generated'],
                'embeddings_pending': results['embeddings_pending'],
                'total_time': total_time,
                'file_processing_rate': results.get('file_rate', 0),
                'embedding_generation_rate': results.get('embedding_rate', 0),
                'success_rate': results.get('success_rate', 0)
            }
            
            logger.info(f"🎉 Concurrent folder processing completed:")
            logger.info(f"   📁 Files processed: {final_results['files_processed']}")
            logger.info(f"   📊 Records created: {final_results['records_created']}")
            logger.info(f"   🔧 Embeddings generated: {final_results['embeddings_generated']}")
            logger.info(f"   ⏱️ Total time: {total_time:.2f}s")
            logger.info(f"   📈 File rate: {final_results['file_processing_rate']:.2f}/s")
            logger.info(f"   📈 Embedding rate: {final_results['embedding_generation_rate']:.2f}/s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ process_folder() failed: {e}")
            logger.debug(f"📋 Process folder traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'message': f'Folder processing failed: {str(e)}',
                'files_processed': 0,
                'files_failed': 0,
                'records_created': 0,
                'embeddings_generated': 0
            }
    
    async def _search_documents(self):
        """Search documents with various methods"""
        print("\n🔍 SEARCH DOCUMENTS")
        print("="*40)
        
        while True:
            try:
                print("\nSearch Methods:")
                print("1. Semantic search (embeddings)")
                print("2. Text search (keywords)")
                print("3. Tag search")
                print("4. File path search")
                print("5. Hybrid search")
                print("6. Back to main menu")
                
                choice = input("\nSelect search method (1-6): ").strip()
                
                if choice == "6":
                    break
                
                if choice == "1":
                    await self._semantic_search()
                elif choice == "2":
                    await self._text_search()
                elif choice == "3":
                    await self._tag_search()
                elif choice == "4":
                    await self._file_path_search()
                elif choice == "5":
                    await self._hybrid_search()
                else:
                    print("Invalid option.")
                    
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                break
            except Exception as e:
                logger.error(f"Search error: {e}")
                print(f"Error: {e}")
    
    async def _semantic_search(self):
        """Perform semantic search using embeddings"""
        query = input("Enter search query: ").strip()
        if not query:
            return
        
        threshold = float(input(f"Similarity threshold (default {config.SEARCH_DEFAULTS['relevance_threshold']}): ").strip() or config.SEARCH_DEFAULTS['relevance_threshold'])
        top_k = int(input(f"Max results (default {config.SEARCH_DEFAULTS['top_k']}): ").strip() or config.SEARCH_DEFAULTS['top_k'])
        
        print("Searching...")
        results = await search_engine.search(
            query=query,
            method="auto",
            relevance_threshold=threshold,
            top_k=top_k
        )
        
        self._display_search_results(results, "Semantic Search")
    
    async def _text_search(self):
        """Perform full-text search"""
        query = input("Enter search keywords: ").strip()
        if not query:
            return
        
        top_k = int(input(f"Max results (default {config.SEARCH_DEFAULTS['top_k']}): ").strip() or config.SEARCH_DEFAULTS['top_k'])
        
        print("Searching...")
        results = await search_engine.search(
            query=query,
            method="text",
            top_k=top_k
        )
        
        self._display_search_results(results, "Text Search")
    
    async def _tag_search(self):
        """Search by tags"""
        tags_input = input("Enter tags (comma-separated): ").strip()
        if not tags_input:
            return
        
        tags = [tag.strip() for tag in tags_input.split(',')]
        top_k = int(input(f"Max results (default {config.SEARCH_DEFAULTS['top_k']}): ").strip() or config.SEARCH_DEFAULTS['top_k'])
        
        print("Searching...")
        results = await search_engine.search_by_tags(tags, top_k)
        
        self._display_search_results(results, "Tag Search")
    
    async def _file_path_search(self):
        """Search by file path"""
        file_path = input("Enter file path (partial matches supported): ").strip()
        if not file_path:
            return
        
        top_k = int(input(f"Max results (default {config.SEARCH_DEFAULTS['top_k']}): ").strip() or config.SEARCH_DEFAULTS['top_k'])
        
        print("Searching...")
        results = await search_engine.search_by_file_path(file_path, top_k)
        
        self._display_search_results(results, "File Path Search")
    
    async def _hybrid_search(self):
        """Perform hybrid search combining multiple methods"""
        query = input("Enter search query: ").strip()
        if not query:
            return
        
        threshold = float(input(f"Similarity threshold (default {config.SEARCH_DEFAULTS['relevance_threshold']}): ").strip() or config.SEARCH_DEFAULTS['relevance_threshold'])
        top_k = int(input(f"Max results (default {config.SEARCH_DEFAULTS['top_k']}): ").strip() or config.SEARCH_DEFAULTS['top_k'])
        
        print("Performing hybrid search...")
        results = await search_engine.search(
            query=query,
            method="hybrid",
            relevance_threshold=threshold,
            top_k=top_k
        )
        
        self._display_search_results(results, "Hybrid Search")
    
    def _display_search_results(self, results, search_type):
        """Display search results in a formatted way"""
        if not results:
            print(f"❌ No results found for {search_type}.")
            return
        
        print(f"\n📊 {search_type} Results ({len(results)} found):")
        print("="*60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document ID: {result['id']}")
            
            # Show similarity scores
            if result.get('cosine_similarity') is not None:
                print(f"   Similarity: {result['cosine_similarity']:.3f}")
            if result.get('text_rank') is not None:
                print(f"   Text Rank: {result['text_rank']:.3f}")
            if result.get('hybrid_score') is not None:
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
            
            # Show metadata
            if result.get('file_path'):
                print(f"   File: {os.path.basename(result['file_path'])}")
            if result.get('tags'):
                print(f"   Tags: {', '.join(result['tags'][:5])}")
            
            # Show content preview
            content = result.get('content', '')
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"   Content: {preview}")
    
    async def _configuration_menu(self):
        """Configuration management menu"""
        print("\n⚙️ CONFIGURATION MANAGEMENT")
        print("="*40)
        
        while True:
            try:
                print("\nConfiguration Options:")
                print("1. Show current configuration")
                print("2. Update search parameters")
                print("3. Update performance settings")
                print("4. Update chunk sizes")
                print("5. Apply optimal settings")
                print("6. Reset to defaults")
                print("7. Back to main menu")
                
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == "7":
                    break
                elif choice == "1":
                    self._show_configuration()
                elif choice == "2":
                    self._update_search_parameters()
                elif choice == "3":
                    self._update_performance_settings()
                elif choice == "4":
                    self._update_chunk_sizes()
                elif choice == "5":
                    config.apply_optimal_settings()
                    print("✅ Applied optimal settings based on system resources.")
                elif choice == "6":
                    confirm = input("Reset all configuration to defaults? (y/n): ").strip().lower()
                    if confirm == 'y':
                        config.reset_to_defaults()
                        print("✅ Configuration reset to defaults.")
                else:
                    print("Invalid option.")
                    
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                break
            except Exception as e:
                logger.error(f"Configuration error: {e}")
                print(f"Error: {e}")
    
    def _show_configuration(self):
        """Display current configuration"""
        summary = config.get_config_summary()
        
        print("\n📋 Current Configuration:")
        print("="*50)
        
        print("Database:")
        for key, value in summary['database'].items():
            print(f"  {key}: {value}")
        
        print("\nEmbedding:")
        for key, value in summary['embedding'].items():
            print(f"  {key}: {value}")
        
        print("\nPerformance:")
        for key, value in summary['performance'].items():
            print(f"  {key}: {value}")
        
        print("\nSearch:")
        for key, value in summary['search'].items():
            print(f"  {key}: {value}")
        
        print("\nChunk Sizes:")
        for key, value in summary['chunks'].items():
            print(f"  {key}: {value}")
        
        print("\nFiles:")
        for key, value in summary['files'].items():
            print(f"  {key}: {value}")
    
    def _update_search_parameters(self):
        """Update search-related parameters"""
        print("\nUpdating Search Parameters:")
        
        try:
            threshold = input(f"Relevance threshold (current: {config.SEARCH_DEFAULTS['relevance_threshold']}, 0.0-1.0): ").strip()
            if threshold:
                config.set('relevance_threshold', float(threshold))
            
            top_k = input(f"Top K results (current: {config.SEARCH_DEFAULTS['top_k']}): ").strip()
            if top_k:
                config.set('top_k', int(top_k))
            
            vector_limit = input(f"Vector search limit (current: {config.SEARCH_DEFAULTS['vector_search_limit']}): ").strip()
            if vector_limit:
                config.set('vector_search_limit', int(vector_limit))
            
            print("✅ Search parameters updated.")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def _update_performance_settings(self):
        """Update performance-related settings"""
        print("\nUpdating Performance Settings:")
        
        try:
            workers = input(f"Worker processes (current: {config.PERFORMANCE_CONFIG['worker_processes']}): ").strip()
            if workers:
                config.set('worker_processes', int(workers))
            
            memory_threshold = input(f"Memory cleanup threshold % (current: {config.PERFORMANCE_CONFIG['memory_cleanup_threshold']}): ").strip()
            if memory_threshold:
                config.set('memory_cleanup_threshold', int(memory_threshold))
            
            queue_memory = input(f"Queue memory GB (current: {config.PERFORMANCE_CONFIG['embedding_queue_memory_gb']}): ").strip()
            if queue_memory:
                config.set('embedding_queue_memory_gb', int(queue_memory))
            
            print("✅ Performance settings updated.")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    def _update_chunk_sizes(self):
        """Update chunk size settings"""
        print("\nUpdating Chunk Sizes:")
        
        try:
            for chunk_type, current_size in config.CHUNK_SIZES.items():
                new_size = input(f"{chunk_type} (current: {current_size}): ").strip()
                if new_size:
                    config.set(f'chunk_{chunk_type}', int(new_size))
            
            print("✅ Chunk sizes updated.")
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
    
    async def _system_diagnostics(self):
        """Run comprehensive system diagnostics"""
        print("\n🔧 SYSTEM DIAGNOSTICS")
        print("="*40)
        
        print("Running comprehensive diagnostics...")
        
        # Database diagnostics
        print("\n1. Database Diagnostics:")
        db_healthy = diagnose_db_issues()
        
        # Embedding service diagnostics
        print("\n2. Embedding Service:")
        embedding_healthy = await embedding_service.test_connection()
        if embedding_healthy:
            print("   ✅ Embedding service: HEALTHY")
        else:
            print("   ❌ Embedding service: UNHEALTHY")
        
        # System resources
        print("\n3. System Resources:")
        memory = psutil.virtual_memory()
        print(f"   Memory: {memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB ({memory.percent:.1f}%)")
        print(f"   CPU Count: {psutil.cpu_count()}")
        print(f"   CPU Usage: {psutil.cpu_percent():.1f}%")
        
        # Configuration validation
        print("\n4. Configuration:")
        config_valid = config.validate_config()
        if config_valid:
            print("   ✅ Configuration: VALID")
        else:
            print("   ❌ Configuration: INVALID")
        
        # Overall health
        overall_healthy = db_healthy and embedding_healthy and config_valid
        print(f"\n🏥 Overall System Health: {'✅ HEALTHY' if overall_healthy else '❌ ISSUES DETECTED'}")
    
    async def _database_menu(self):
        """Database management menu"""
        print("\n💾 DATABASE MANAGEMENT")
        print("="*40)
        
        while True:
            try:
                print("\nDatabase Options:")
                print("1. Show database statistics")
                print("2. Run database maintenance")
                print("3. Apply PostgreSQL optimizations")
                print("4. Query database directly")
                print("5. Export data")
                print("6. Back to main menu")
                
                choice = input("\nSelect option (1-6): ").strip()
                
                if choice == "6":
                    break
                elif choice == "1":
                    await self._show_database_stats()
                elif choice == "2":
                    await self._run_database_maintenance()
                elif choice == "3":
                    self._apply_database_optimizations()
                elif choice == "4":
                    await self._query_database()
                elif choice == "5":
                    await self._export_data()
                else:
                    print("Invalid option.")
                    
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                break
            except Exception as e:
                logger.error(f"Database menu error: {e}")
                print(f"Error: {e}")
    
    async def _show_database_stats(self):
        """Show comprehensive database statistics"""
        print("\n📊 Database Statistics:")
        
        stats = db_manager.get_stats()
        
        if stats.get('error'):
            print(f"❌ Error getting stats: {stats['error']}")
            return
        
        print(f"   Table: {stats.get('table_name', 'Unknown')}")
        print(f"   Records: {stats.get('record_count', 0):,}")
        print(f"   Size: {stats.get('table_size', 'Unknown')}")
        print(f"   Embeddings: {stats.get('embeddings_count', 0):,}")
        print(f"   Tagged Records: {stats.get('tagged_records', 0):,}")
        print(f"   Files with Path: {stats.get('files_with_path', 0):,}")
        print(f"   Recent 24h: {stats.get('recent_24h', 0):,}")
        
        if stats.get('avg_content_length'):
            print(f"   Avg Content Length: {stats['avg_content_length']:.0f} chars")
        
        # Show indexes
        if stats.get('indexes'):
            print(f"\n   Indexes ({len(stats['indexes'])}):")
            for idx in stats['indexes']:
                print(f"     - {idx['name']}: {idx['size']}")
    
    async def _run_database_maintenance(self):
        """Run database maintenance operations"""
        print("\nRunning database maintenance...")
        
        success = await db_manager.run_maintenance()
        
        if success:
            print("✅ Database maintenance completed successfully.")
        else:
            print("❌ Database maintenance failed.")
    
    def _apply_database_optimizations(self):
        """Apply PostgreSQL optimizations"""
        print("\nApplying PostgreSQL optimizations...")
        
        success = db_manager.apply_optimizations()
        
        if success:
            print("✅ PostgreSQL optimizations applied.")
            print("⚠️  Please restart PostgreSQL for changes to take effect.")
        else:
            print("❌ Failed to apply optimizations.")
    
    async def _query_database(self):
        """Direct database query interface"""
        print("\n💬 Direct Database Query")
        print("(Enter 'exit' to return)")
        
        while True:
            try:
                query = input("\nSQL> ").strip()
                
                if query.lower() == 'exit':
                    break
                
                if not query:
                    continue
                
                # Simple safety check
                if any(dangerous in query.upper() for dangerous in ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']):
                    confirm = input("⚠️  Potentially dangerous query. Continue? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                # Execute query (read-only for safety)
                with db_manager.get_sync_cursor() as (conn, cur):
                    cur.execute(query)
                    results = cur.fetchall()
                    
                    if results:
                        print(f"\nResults ({len(results)} rows):")
                        for i, row in enumerate(results[:10], 1):  # Limit to 10 rows
                            print(f"  {i}. {dict(row)}")
                        
                        if len(results) > 10:
                            print(f"  ... and {len(results) - 10} more rows")
                    else:
                        print("No results returned.")
                        
            except KeyboardInterrupt:
                print("\nReturning to database menu.")
                break
            except Exception as e:
                print(f"Query error: {e}")
    
    async def _export_data(self):
        """Export database data"""
        print("\n📤 Export Data")
        
        export_type = input("Export format (json/csv): ").strip().lower()
        if export_type not in ['json', 'csv']:
            print("Invalid format. Use 'json' or 'csv'.")
            return
        
        output_file = input("Output file name: ").strip()
        if not output_file:
            output_file = f"ragdb_export_{int(time.time())}.{export_type}"
        
        limit = input("Limit records (default: all): ").strip()
        limit = int(limit) if limit else None
        
        print(f"Exporting to {output_file}...")
        
        try:
            with db_manager.get_sync_cursor() as (conn, cur):
                query = f"SELECT * FROM {config.TABLE_NAME}"
                if limit:
                    query += f" LIMIT {limit}"
                
                cur.execute(query)
                results = cur.fetchall()
                
                if export_type == 'json':
                    data = [dict(row) for row in results]
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
                elif export_type == 'csv':
                    import csv
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        if results:
                            writer = csv.DictWriter(f, fieldnames=results[0].keys())
                            writer.writeheader()
                            for row in results:
                                writer.writerow(dict(row))
                
                print(f"✅ Exported {len(results)} records to {output_file}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            print(f"❌ Export failed: {e}")
    
    async def _show_processing_status(self):
        """Show current processing status"""
        print("\n📈 PROCESSING STATUS")
        print("="*40)
        
        status = async_processor.get_processing_status()
        
        # Processing state
        proc_state = status['processing_state']
        print(f"Progress: {proc_state.get('progress_percent', 0):.1f}%")
        print(f"Files remaining: {proc_state.get('files_remaining', 0)}")
        print(f"Failed files: {proc_state.get('failed_count', 0)}")
        print(f"Elapsed time: {proc_state.get('elapsed_time', 0):.1f}s")
        
        # Current stats
        current_stats = status['current_stats']
        print(f"\nCurrent Session:")
        print(f"  Total files: {current_stats['total_files']}")
        print(f"  Processed: {current_stats['processed_files']}")
        print(f"  Successful: {current_stats['successful_files']}")
        print(f"  Failed: {current_stats['failed_files']}")
        print(f"  Records created: {current_stats['total_records']}")
        
        # Queue stats
        queue_stats = status['queue_stats']
        print(f"\nEmbedding Queue:")
        print(f"  Queue size: {queue_stats.get('queue_size', 0)} items")
        print(f"  Memory usage: {queue_stats.get('current_memory_mb', 0):.1f}MB")
        print(f"  Processed items: {queue_stats.get('processed_items', 0)}")
        print(f"  Failed items: {queue_stats.get('failed_items', 0)}")
        
        # System stats
        system_stats = status.get('system_stats', {})
        if system_stats:
            print(f"\nSystem Resources:")
            print(f"  Memory: {system_stats.get('memory_percent', 0):.1f}%")
            print(f"  CPU: {system_stats.get('cpu_percent', 0):.1f}%")
            print(f"  Load: {system_stats.get('load_avg_1m', 0):.2f}")
    
    async def _advanced_menu(self):
        """Advanced operations menu"""
        print("\n🔬 ADVANCED OPERATIONS")
        print("="*40)
        
        while True:
            try:
                print("\nAdvanced Options:")
                print("1. Test embedding system")
                print("2. Cache management")
                print("3. File tracker management")
                print("4. Force reprocess files")
                print("5. System cleanup")
                print("6. Performance tuning")
                print("7. Back to main menu")
                
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == "7":
                    break
                elif choice == "1":
                    await self._test_embedding_system()
                elif choice == "2":
                    await self._cache_management()
                elif choice == "3":
                    await self._file_tracker_management()
                elif choice == "4":
                    await self._force_reprocess()
                elif choice == "5":
                    await self._system_cleanup()
                elif choice == "6":
                    await self._performance_tuning()
                else:
                    print("Invalid option.")
                    
            except KeyboardInterrupt:
                print("\nReturning to main menu.")
                break
            except Exception as e:
                logger.error(f"Advanced menu error: {e}")
                print(f"Error: {e}")
    
    async def _test_embedding_system(self):
        """Test the embedding system end-to-end"""
        print("\n🧪 Testing Embedding System...")
        
        test_results = {}
        
        try:
            # Test 1: API connectivity
            print("1. Testing API connectivity...")
            api_healthy = await embedding_service.test_connection()
            test_results['api'] = api_healthy
            print(f"   {'✅' if api_healthy else '❌'} API connectivity")
            
            # Test 2: Embedding generation
            print("2. Testing embedding generation...")
            test_text = "This is a test sentence for embedding generation."
            embedding = await embedding_service.generate_embedding(test_text)
            embedding_healthy = embedding is not None and len(embedding) == config.EMB_DIM
            test_results['embedding'] = embedding_healthy
            print(f"   {'✅' if embedding_healthy else '❌'} Embedding generation")
            
            # Test 3: Database insertion
            print("3. Testing database insertion...")
            if embedding_healthy:
                records = [{
                    'content': test_text,
                    'tags': ['test'],
                    'embedding': embedding,
                    'file_path': 'test_file.txt',
                    'chunk_index': 0,
                    'metadata': {'test': True}
                }]
                
                try:
                    inserted = await db_manager.insert_records_batch(records)
                    db_healthy = inserted > 0
                    test_results['database'] = db_healthy
                    print(f"   {'✅' if db_healthy else '❌'} Database insertion")
                except Exception as e:
                    test_results['database'] = False
                    print(f"   ❌ Database insertion failed: {e}")
            else:
                test_results['database'] = False
                print("   ⏭️  Database test skipped (embedding failed)")
            
            # Test 4: Search functionality
            print("4. Testing search functionality...")
            if embedding_healthy:
                try:
                    search_results = await search_engine.search("test sentence", top_k=5)
                    search_healthy = search_results is not None
                    test_results['search'] = search_healthy
                    print(f"   {'✅' if search_healthy else '❌'} Search functionality")
                except Exception as e:
                    test_results['search'] = False
                    print(f"   ❌ Search test failed: {e}")
            else:
                test_results['search'] = False
                print("   ⏭️  Search test skipped (embedding failed)")
            
            # Summary
            passed_tests = sum(test_results.values())
            total_tests = len(test_results)
            
            print(f"\n🎯 Test Summary: {passed_tests}/{total_tests} tests passed")
            
            if passed_tests == total_tests:
                print("🎉 All systems operational!")
            else:
                print("⚠️  Some systems need attention.")
                
        except Exception as e:
            logger.error(f"Embedding system test failed: {e}")
            print(f"❌ Test failed with error: {e}")
    
    async def _cache_management(self):
        """Manage embedding cache"""
        print("\n💾 Cache Management")
        
        cache_stats = search_engine.cache.stats
        print(f"Current cache status:")
        print(f"  Memory entries: {cache_stats['memory_entries']}")
        print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f}MB ({cache_stats['memory_usage_percent']:.1f}%)")
        print(f"  Unique embeddings: {cache_stats['unique_embeddings']}")
        print(f"  DB loaded: {cache_stats['db_loaded']}")
        
        print("\nOptions:")
        print("1. Load embeddings into cache")
        print("2. Clear cache")
        print("3. Force reload cache")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            print("Loading embeddings into cache...")
            success = await search_engine.cache.load()
            print(f"{'✅' if success else '❌'} Cache loading {'completed' if success else 'failed'}")
        elif choice == "2":
            confirm = input("Clear all cache data? (y/n): ").strip().lower()
            if confirm == 'y':
                search_engine.cache.memory_cache.clear()
                search_engine.cache.total_memory_bytes = 0
                search_engine.cache.embedding_signatures.clear()
                search_engine.cache.db_loaded = False
                print("✅ Cache cleared")
        elif choice == "3":
            print("Force reloading cache...")
            success = await search_engine.cache.load(force=True)
            print(f"{'✅' if success else '❌'} Cache reload {'completed' if success else 'failed'}")
    
    async def _file_tracker_management(self):
        """Manage file tracker"""
        print("\n📋 File Tracker Management")
        
        stats = file_tracker.get_processed_files_stats()
        print(f"File tracker status:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total records: {stats['total_records']}")
        print(f"  Total size: {stats['total_size_mb']:.2f}MB")
        print(f"  Avg records per file: {stats['avg_records_per_file']:.1f}")
        
        print("\nOptions:")
        print("1. Show recent files")
        print("2. Clean up missing files")
        print("3. Force save tracker")
        print("4. Reset tracker (clear all)")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            print("\nRecent files:")
            recent_files = sorted(
                file_tracker.processed_files.items(),
                key=lambda x: x[1].processed_at,
                reverse=True
            )[:10]
            
            for i, (filepath, record) in enumerate(recent_files, 1):
                filename = os.path.basename(filepath)
                print(f"  {i}. {filename} ({record.records_count} records)")
        
        elif choice == "2":
            print("Cleaning up missing files...")
            missing = file_tracker.cleanup_missing_files()
            print(f"✅ Cleaned up {len(missing)} missing file records")
        
        elif choice == "3":
            file_tracker.save_tracker(force=True)
            print("✅ File tracker saved")
        
        elif choice == "4":
            confirm = input("⚠️  Clear ALL file tracking data? (y/n): ").strip().lower()
            if confirm == 'y':
                file_tracker.processed_files.clear()
                file_tracker.save_tracker(force=True)
                print("✅ File tracker reset")
    
    async def _force_reprocess(self):
        """Force reprocessing of files"""
        print("\n🔄 Force Reprocess Files")
        
        print("Options:")
        print("1. Reprocess specific file")
        print("2. Reprocess directory")
        print("3. Reprocess by file pattern")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Enter file path: ").strip()
            if file_path and os.path.exists(file_path):
                abs_path = os.path.abspath(file_path)
                if abs_path in file_tracker.processed_files:
                    del file_tracker.processed_files[abs_path]
                    file_tracker.save_tracker()
                    print(f"✅ {file_path} will be reprocessed next time")
                else:
                    print("File not in tracker")
            else:
                print("File not found")
        
        elif choice == "2":
            dir_path = input("Enter directory path: ").strip()
            if dir_path and os.path.exists(dir_path):
                abs_dir = os.path.abspath(dir_path)
                removed_count = 0
                
                for filepath in list(file_tracker.processed_files.keys()):
                    if filepath.startswith(abs_dir):
                        del file_tracker.processed_files[filepath]
                        removed_count += 1
                
                file_tracker.save_tracker()
                print(f"✅ {removed_count} files will be reprocessed")
            else:
                print("Directory not found")
        
        elif choice == "3":
            pattern = input("Enter file pattern (e.g., '*.py'): ").strip()
            if pattern:
                import fnmatch
                removed_count = 0
                
                for filepath in list(file_tracker.processed_files.keys()):
                    if fnmatch.fnmatch(os.path.basename(filepath), pattern):
                        del file_tracker.processed_files[filepath]
                        removed_count += 1
                
                file_tracker.save_tracker()
                print(f"✅ {removed_count} files matching '{pattern}' will be reprocessed")
    
    async def _system_cleanup(self):
        """Perform comprehensive system cleanup"""
        print("\n🧹 System Cleanup")
        
        print("Running comprehensive cleanup...")
        
        # Memory cleanup
        print("1. Memory cleanup...")
        import gc
        collected = gc.collect()
        print(f"   Collected {collected} objects")
        
        # GPU cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   GPU cache cleared")
        except ImportError:
            pass
        
        # File tracker cleanup
        print("2. File tracker cleanup...")
        missing = file_tracker.cleanup_missing_files()
        print(f"   Removed {len(missing)} missing file records")
        
        # Database maintenance
        print("3. Database maintenance...")
        db_success = await db_manager.run_maintenance()
        print(f"   Database maintenance {'✅ completed' if db_success else '❌ failed'}")
        
        # Save all states
        print("4. Saving states...")
        file_tracker.save_tracker(force=True)
        async_processor.state_manager.save_state()
        config.save_config()
        print("   All states saved")
        
        print("✅ System cleanup completed")
    
    async def _performance_tuning(self):
        """Performance tuning and optimization"""
        print("\n⚡ Performance Tuning")
        
        print("Current performance settings:")
        perf_config = config.PERFORMANCE_CONFIG
        for key, value in perf_config.items():
            print(f"  {key}: {value}")
        
        print("\nOptions:")
        print("1. Apply optimal settings automatically")
        print("2. Manual performance tuning")
        print("3. Benchmark system")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            print("Applying optimal settings...")
            config.apply_optimal_settings()
            print("✅ Optimal settings applied")
        
        elif choice == "2":
            print("Manual performance tuning:")
            
            workers = input(f"Worker processes (current: {perf_config['worker_processes']}): ").strip()
            if workers:
                config.set('worker_processes', int(workers))
            
            memory_gb = input(f"Queue memory GB (current: {perf_config['embedding_queue_memory_gb']}): ").strip()
            if memory_gb:
                config.set('embedding_queue_memory_gb', int(memory_gb))
            
            cleanup_threshold = input(f"Memory cleanup threshold % (current: {perf_config['memory_cleanup_threshold']}): ").strip()
            if cleanup_threshold:
                config.set('memory_cleanup_threshold', int(cleanup_threshold))
            
            print("✅ Performance settings updated")
        
        elif choice == "3":
            await self._run_benchmark()
    
    async def _run_benchmark(self):
        """Run system benchmark"""
        print("\n🏃 Running System Benchmark...")
        
        # Test embedding generation speed
        print("1. Testing embedding generation speed...")
        test_texts = [
            "This is a test sentence for benchmarking.",
            "Another test sentence with different content.",
            "A third test sentence for speed measurement.",
            "Fourth benchmark text for performance testing.",
            "Final test sentence to complete the benchmark."
        ]
        
        start_time = time.time()
        embeddings = await embedding_service.generate_embeddings_batch(test_texts)
        embedding_time = time.time() - start_time
        
        successful_embeddings = sum(1 for e in embeddings if e is not None)
        embedding_rate = successful_embeddings / embedding_time if embedding_time > 0 else 0
        
        print(f"   Generated {successful_embeddings}/{len(test_texts)} embeddings in {embedding_time:.2f}s")
        print(f"   Rate: {embedding_rate:.1f} embeddings/second")
        
        # Test database operations
        print("2. Testing database operations...")
        
        # Insert test
        test_records = []
        for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
            if embedding:
                test_records.append({
                    'content': f"Benchmark test {i}: {text}",
                    'tags': ['benchmark', f'test_{i}'],
                    'embedding': embedding,
                    'file_path': 'benchmark_test.txt',
                    'chunk_index': i,
                    'metadata': {'benchmark': True, 'timestamp': time.time()}
                })
        
        if test_records:
            start_time = time.time()
            inserted = await db_manager.insert_records_batch(test_records)
            insert_time = time.time() - start_time
            insert_rate = inserted / insert_time if insert_time > 0 else 0
            
            print(f"   Inserted {inserted} records in {insert_time:.2f}s")
            print(f"   Rate: {insert_rate:.1f} records/second")
        
        # Search test
        print("3. Testing search performance...")
        start_time = time.time()
        search_results = await search_engine.search("benchmark test", top_k=10)
        search_time = time.time() - start_time
        
        result_count = len(search_results) if search_results else 0
        print(f"   Search completed in {search_time:.2f}s")
        print(f"   Found {result_count} results")
        
        # Summary
        print(f"\n📊 Benchmark Summary:")
        print(f"   Embedding rate: {embedding_rate:.1f}/s")
        if test_records:
            print(f"   Database insert rate: {insert_rate:.1f}/s")
        print(f"   Search time: {search_time:.2f}s")
        
        # Cleanup benchmark data
        try:
            with db_manager.get_sync_cursor() as (conn, cur):
                cur.execute(f"DELETE FROM {config.TABLE_NAME} WHERE metadata->>'benchmark' = 'true'")
                print("   Benchmark data cleaned up")
        except Exception as e:
            logger.warning(f"Benchmark cleanup failed: {e}")
    
    async def _exit_application(self):
        """Clean exit of the application"""
        print("\n👋 Exiting Enhanced RAG Database System...")
        
        try:
            # Stop any running processes
            if embedding_queue.started:
                print("Stopping embedding workers...")
                await embedding_queue.stop_workers()
            
            # Save all states
            print("Saving application state...")
            file_tracker.save_tracker(force=True)
            async_processor.state_manager.save_state()
            config.save_config()
            
            # Close database connections
            print("Closing database connections...")
            await db_manager.close_pools()
            
            # Close embedding service
            await embedding_service.close()
            
            print("✅ Clean shutdown completed.")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            print(f"⚠️  Shutdown error: {e}")
        
        finally:
            self.running = False

async def main():
    """Main application entry point"""
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Create and run the unified interface
    interface = UnifiedInterface()
    await interface.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        sys.exit(1)# unified_interface.py - Consolidated User Interface            
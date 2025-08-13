# cli/enhanced_interface.py - Enhanced CLI Interface with Queue-Based Processing
import asyncio
import logging
import os
from typing import Optional
from pathlib import Path

from core.config import config
from database.schema_init import initialize_database_schema
from database.batch_operations import batch_db_ops
from inference.async_embeddings import async_embedding_service
from file_management.discovery import file_selector, file_discovery
from file_management.instant_loader import instant_async_loader
from file_management.simple_async_loader import simple_async_loader
from file_management.lightning_loader import lightning_loader

# Optional imports with fallbacks
try:
    from core.memory import memory_manager
except ImportError:
    memory_manager = None

try:
    from inference.multi_query import enhanced_inference_engine
except ImportError:
    enhanced_inference_engine = None

try:
    from cli.validation import InputValidator
except ImportError:
    InputValidator = None

logger = logging.getLogger(__name__)

class EnhancedCLIInterface:
    """Enhanced CLI interface with queue-based processing"""
    
    def __init__(self):
        self.validator = InputValidator() if InputValidator else None
        self._running = False
        self.queue_enabled = True
        self.batch_processing_enabled = True
    
    async def initialize(self):
        """Initialize the enhanced system with queue-based services"""
        try:
            print("🚀 Initializing Enhanced Queue-Based RAG System...")
            
            # Initialize database schema
            print("📊 Setting up database schema...")
            if not initialize_database_schema():
                print("❌ Failed to initialize database schema!")
                return False
            
            # Initialize batch database operations
            print("🔧 Initializing batch database operations...")
            await batch_db_ops.initialize_connection_pool()
            
            # Validate and start embedding service
            print("🧠 Validating embedding model...")
            if not async_embedding_service.validate_model():
                print(f"❌ CRITICAL: Embedding model {config.embedding.model} not available!")
                print("Please ensure Ollama is running and the model is pulled:")
                print(f"   ollama pull {config.embedding.model}")
                return False
            
            print("🔄 Starting queue-based embedding service...")
            await async_embedding_service.start()
            
            # Display system configuration
            self._display_system_info()
            
            logger.info("Enhanced CLI system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"CLI initialization failed: {e}")
            print(f"❌ System initialization failed: {e}")
            return False
    
    def _display_system_info(self):
        """Display system configuration and capabilities"""
        print("\n" + "="*80)
        print("🎯 ENHANCED QUEUE-BASED RAG SYSTEM READY")
        print("="*80)
        print(f"🧠 Embedding Model:    {config.embedding.model}")
        print(f"📊 Database:           {config.database.name}@{config.database.host}")
        print(f"🔧 Queue Size:         {async_embedding_service.max_queue_size} embeddings")
        print(f"⚡ Workers:            {async_embedding_service.num_workers} embedding workers")
        print(f"📝 Chunking:           Enabled (512 tokens, 32 overlap)")
        print(f"📑 Summarization:      Enabled (hierarchical)")
        print(f"🔄 Parallel Loading:   Enabled (queue-based)")
        print(f"💾 Batch Processing:   Enabled (connection pooling)")
        print("="*80)
    
    async def run(self):
        """Main CLI loop with enhanced menu options"""
        if not await self.initialize():
            return
        
        self._running = True
        
        try:
            while self._running:
                await self._display_main_menu()
                choice = await self._get_user_choice()
                await self._handle_menu_choice(choice)
                
        except KeyboardInterrupt:
            print("\n👋 Gracefully shutting down...")
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print(f"❌ Unexpected error: {e}")
        finally:
            await self._cleanup()
    
    async def _display_main_menu(self):
        """Display the enhanced main menu"""
        # Get current statistics
        stats = await batch_db_ops.get_database_stats()
        embed_stats = async_embedding_service.get_stats()
        memory_info = memory_manager.get_memory_info() if memory_manager else {'percent': 0}
        
        print("\n" + "="*80)
        print("🚀 ENHANCED RAG SYSTEM - MAIN MENU")
        print("="*80)
        print("📊 Current Status:")
        print(f"   📝 Chunks: {stats.get('total_chunks', 0):,} | "
              f"📑 Summaries: {stats.get('total_summaries', 0):,} | "
              f"📁 Files: {stats.get('processed_files', 0):,}")
        print(f"   🧠 Embedding Queue: {embed_stats.get('queue_size', 0)}/{async_embedding_service.max_queue_size} | "
              f"⚡ Workers: {embed_stats.get('active_workers', 0)}/{async_embedding_service.num_workers} | "
              f"💾 Memory: {memory_info.get('percent', 0):.1f}%")
        print("="*80)
        print("1. 🔍 Ask AI (Enhanced Multi-Query RAG)")
        print("2. ➕ Add Single Document (with Chunking)")
        print("3. 📁 Load Single File (Queue-Based)")
        print("4. 📂 Bulk Load Folder (Instant Async)")
        print("5. ⚡ Lightning Load (Ultra-Fast Mode)")
        print("6. 🔎 Search Database Directly")
        print("7. 📋 List Database Contents")
        print("8. ⚙️  Configure System Parameters")
        print("9. 🗄️  Database Management")
        print("10. 📊 System Information & Performance")
        print("11. 🔧 Queue & Processing Status")
        print("12. 🧹 Optimize Database Performance")
        print("0. 🚪 Exit")
        print("="*80)
    
    async def _get_user_choice(self) -> str:
        """Get and validate user menu choice"""
        while True:
            try:
                choice = input("\n👉 Enter your choice (0-12): ").strip()
                if choice in [str(i) for i in range(13)]:
                    return choice
                print("❌ Invalid choice. Please enter a number between 0-12.")
            except EOFError:
                return "0"
            except Exception as e:
                print(f"❌ Input error: {e}")
    
    async def _handle_menu_choice(self, choice: str):
        """Handle user menu selection with enhanced options"""
        try:
            if choice == "1":
                await self._enhanced_ai_query()
            elif choice == "2":
                await self._add_single_document()
            elif choice == "3":
                await self._load_single_file()
            elif choice == "4":
                await self._bulk_load_folder()
            elif choice == "5":
                await self._lightning_load()
            elif choice == "6":
                await self._search_database()
            elif choice == "7":
                await self._list_database_contents()
            elif choice == "8":
                await self._configure_system()
            elif choice == "9":
                await self._database_management()
            elif choice == "10":
                await self._system_information()
            elif choice == "11":
                await self._queue_status()
            elif choice == "12":
                await self._optimize_database()
            elif choice == "0":
                self._running = False
            
        except Exception as e:
            logger.error(f"Menu handler error: {e}")
            print(f"❌ Error processing menu choice: {e}")
    
    async def _enhanced_ai_query(self):
        """Enhanced AI query with multi-query expansion"""
        print("\n🔍 Enhanced Multi-Query AI Assistant")
        print("="*50)
        
        query = input("❓ Enter your question: ").strip()
        if not query:
            print("❌ Empty query provided")
            return
        
        try:
            print("\n🔄 Processing your query with multi-query expansion...")
            
            # Use enhanced inference engine
            if enhanced_inference_engine:
                response = await enhanced_inference_engine.generate_response(query)
            else:
                print("❌ Enhanced inference engine not available")
                return
            
            if response:
                print("\n🤖 AI Response:")
                print("="*50)
                print(response.get('response', 'No response generated'))
                
                if response.get('sources'):
                    print(f"\n📚 Sources ({len(response['sources'])} found):")
                    for i, source in enumerate(response['sources'][:3], 1):
                        print(f"{i}. {source.get('source_file', 'Unknown')} "
                              f"(similarity: {source.get('similarity', 0):.3f})")
                
                if response.get('query_variations'):
                    print(f"\n🔍 Query Variations Used: {len(response['query_variations'])}")
                    for i, var in enumerate(response['query_variations'][:3], 1):
                        print(f"{i}. {var}")
            else:
                print("❌ Failed to generate response")
                
        except Exception as e:
            logger.error(f"Enhanced AI query failed: {e}")
            print(f"❌ Query processing failed: {e}")
    
    async def _bulk_load_folder(self):
        """Enhanced bulk folder loading with instant async processing"""
        print("\n📂 Instant Async Bulk Loading")
        print("="*50)
        
        try:
            # Get folder selection
            folder_path = file_selector.get_folder_path()
            if not folder_path:
                print("❌ No folder selected")
                return
            
            print(f"📁 Selected folder: {folder_path}")
            
            # Quick check for folder contents (don't count all files)
            print("🔍 Checking folder contents...")
            
            # Quick sample check instead of full discovery
            sample_files = 0
            for root, dirs, files in os.walk(folder_path):
                for file in files[:10]:  # Just check first 10 files
                    file_path = os.path.join(root, file)
                    if Path(file_path).suffix.lower() in {'.py', '.txt', '.csv', '.json'}:
                        sample_files += 1
                break  # Only check first directory
            
            if sample_files == 0:
                print("❌ No compatible files found in the selected folder")
                return
            
            print(f"📊 Found compatible files (will process up to 1,000 for efficiency)")
            
            # Confirm processing
            confirm = input(f"\n🚀 Start instant async processing? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Operation cancelled")
                return
            
            # Start instant async loading
            print(f"\n🔄 Starting instant async loading...")
            print("💡 Processing will begin immediately:")
            
            results = await instant_async_loader.load_folder_instant(
                folder_path=folder_path,
                total_files=1000,  # Estimated limit
                enable_chunking=True,
                enable_summarization=False
            )
            
            # Display final results
            if results.get('interrupted'):
                print(f"\n⚠️  Instant loading was interrupted!")
            else:
                print(f"\n✅ Instant async loading completed!")
            
            print(f"📁 Files processed: {results.get('processed_files', 0)}")
            print(f"📝 Chunks created: {results.get('total_chunks', 0)}")
            print(f"📑 Summaries created: {results.get('total_summaries', 0)}")
            print(f"⚡ Processing speed: {results.get('files_per_second', 0):.1f} files/sec")
            print(f"⏱️  Total time: {results.get('processing_time', 0):.1f} seconds")
            
            if results.get('failed_files', 0) > 0:
                print(f"⚠️  Failed files: {results['failed_files']}")
            
        except Exception as e:
            logger.error(f"Bulk loading failed: {e}")
            print(f"❌ Bulk loading failed: {e}")
    
    async def _lightning_load(self):
        """Ultra-fast lightning loading for massive datasets"""
        print("\n⚡ Lightning Load - Ultra-Fast Mode")
        print("="*50)
        
        try:
            # Get folder selection
            folder_path = file_selector.get_folder_path()
            if not folder_path:
                print("❌ No folder selected")
                return
            
            print(f"📁 Selected folder: {folder_path}")
            print("⚡ Lightning mode processes up to 2,000 files with minimal overhead")
            
            # Confirm processing
            confirm = input(f"\n🚀 Start lightning-fast processing? (y/N): ")
            if confirm.lower() != 'y':
                print("❌ Operation cancelled")
                return
            
            # Start lightning loading
            print(f"\n⚡ Starting lightning loading...")
            
            results = await lightning_loader.load_folder_lightning(
                folder_path=folder_path,
                enable_chunking=True
            )
            
            # Display final results
            if results.get('error'):
                print(f"\n❌ Lightning loading failed: {results['error']}")
            else:
                print(f"\n✅ Lightning loading completed!")
                print(f"📁 Files processed: {results.get('processed_files', 0)}")
                print(f"📝 Chunks created: {results.get('total_chunks', 0)}")
                print(f"⚡ Processing speed: {results.get('files_per_second', 0):.1f} files/sec")
                print(f"⏱️  Total time: {results.get('processing_time', 0):.1f} seconds")
                
                if results.get('failed_files', 0) > 0:
                    print(f"⚠️  Failed files: {results['failed_files']}")
            
        except Exception as e:
            logger.error(f"Lightning loading failed: {e}")
            print(f"❌ Lightning loading failed: {e}")
    
    async def _load_single_file(self):
        """Load single file with queue-based processing"""
        print("\n📄 Queue-Based Single File Loading")
        print("="*50)
        
        file_path = input("📁 Enter file path: ").strip()
        if not file_path:
            print("❌ No file path provided")
            return
        
        try:
            print(f"🔄 Processing file with simple async system...")
            
            # Use simple async loader for single file
            results = await simple_async_loader.load_folder_async(
                folder_path="",
                file_generator=iter([file_path]),
                total_files=1,
                enable_chunking=True,
                enable_summarization=False
            )
            
            if results.get('processed_files', 0) > 0:
                print(f"✅ File processed successfully!")
                print(f"📝 Chunks created: {results.get('total_chunks', 0)}")
                print(f"📑 Summaries created: {results.get('total_summaries', 0)}")
            else:
                print("❌ File processing failed")
                
        except Exception as e:
            logger.error(f"Single file loading failed: {e}")
            print(f"❌ File loading failed: {e}")
    
    async def _queue_status(self):
        """Display queue and processing status"""
        print("\n🔧 Queue & Processing Status")
        print("="*50)
        
        try:
            # Embedding service stats
            embed_stats = async_embedding_service.get_stats()
            print("🧠 Embedding Service:")
            print(f"   Queue Size: {embed_stats.get('queue_size', 0)}/{async_embedding_service.max_queue_size}")
            print(f"   Active Workers: {embed_stats.get('active_workers', 0)}/{async_embedding_service.num_workers}")
            print(f"   Total Requests: {embed_stats.get('total_requests', 0):,}")
            print(f"   Completed: {embed_stats.get('completed_requests', 0):,}")
            print(f"   Failed: {embed_stats.get('failed_requests', 0):,}")
            print(f"   Success Rate: {embed_stats.get('success_rate', 0):.1f}%")
            print(f"   Avg Processing Time: {embed_stats.get('avg_processing_time', 0):.3f}s")
            
            # Database connection pool status
            pool_size = len(batch_db_ops._connection_pool)
            print(f"\n📊 Database Connection Pool:")
            print(f"   Available Connections: {pool_size}/{batch_db_ops._pool_size}")
            
            # Memory usage
            if memory_manager:
                memory_info = memory_manager.get_memory_info()
                print(f"\n💾 Memory Usage:")
                print(f"   Used: {memory_info.get('percent', 0):.1f}% ({memory_info.get('used_mb', 0):.1f} MB)")
                print(f"   Available: {memory_info.get('available_mb', 0):.1f} MB")
            else:
                print(f"\n💾 Memory Usage: Monitoring not available")
            
        except Exception as e:
            logger.error(f"Queue status display failed: {e}")
            print(f"❌ Failed to get queue status: {e}")
    
    async def _optimize_database(self):
        """Optimize database performance"""
        print("\n🧹 Database Optimization")
        print("="*50)
        
        confirm = input("🔄 Run database optimization (VACUUM & ANALYZE)? This may take a while. (y/N): ")
        if confirm.lower() != 'y':
            print("❌ Optimization cancelled")
            return
        
        try:
            print("🔄 Running database optimization...")
            success = await batch_db_ops.optimize_database()
            
            if success:
                print("✅ Database optimization completed successfully")
                
                # Show updated stats
                stats = await batch_db_ops.get_database_stats()
                print(f"\n📊 Updated Database Statistics:")
                print(f"   Chunks table size: {stats.get('chunks_table_size', 'Unknown')}")
                print(f"   Summaries table size: {stats.get('summaries_table_size', 'Unknown')}")
                print(f"   Checksums table size: {stats.get('checksums_table_size', 'Unknown')}")
            else:
                print("❌ Database optimization failed")
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            print(f"❌ Optimization failed: {e}")
    
    async def _system_information(self):
        """Display enhanced system information"""
        print("\n📊 Enhanced System Information")
        print("="*50)
        
        try:
            # Database statistics
            stats = await batch_db_ops.get_database_stats()
            print("📊 Database Statistics:")
            print(f"   Total Chunks: {stats.get('total_chunks', 0):,}")
            print(f"   Total Summaries: {stats.get('total_summaries', 0):,}")
            print(f"   Processed Files: {stats.get('processed_files', 0):,}")
            print(f"   Avg Chunks/File: {stats.get('avg_chunks_per_file', 0):.1f}")
            print(f"   Avg Summaries/File: {stats.get('avg_summaries_per_file', 0):.1f}")
            
            # Queue service statistics
            embed_stats = async_embedding_service.get_stats()
            print(f"\n🧠 Embedding Service:")
            print(f"   Queue Utilization: {embed_stats.get('queue_size', 0)}/{async_embedding_service.max_queue_size}")
            print(f"   Worker Utilization: {embed_stats.get('active_workers', 0)}/{async_embedding_service.num_workers}")
            print(f"   Success Rate: {embed_stats.get('success_rate', 0):.1f}%")
            
            # Memory information
            if memory_manager:
                memory_info = memory_manager.get_memory_info()
                print(f"\n💾 System Resources:")
                print(f"   Memory Usage: {memory_info.get('percent', 0):.1f}%")
                print(f"   Memory Used: {memory_info.get('used_mb', 0):.1f} MB")
                print(f"   Memory Available: {memory_info.get('available_mb', 0):.1f} MB")
            else:
                print(f"\n💾 System Resources: Memory monitoring not available")
            
        except Exception as e:
            logger.error(f"System information display failed: {e}")
            print(f"❌ Failed to get system information: {e}")
    
    # Placeholder methods for backward compatibility
    async def _add_single_document(self):
        """Add single document (placeholder)"""
        print("📝 Add Single Document - Feature coming soon!")
    
    async def _search_database(self):
        """Search database directly (placeholder)"""
        print("🔎 Direct Database Search - Feature coming soon!")
    
    async def _list_database_contents(self):
        """List database contents (placeholder)"""
        print("📋 Database Contents Listing - Feature coming soon!")
    
    async def _configure_system(self):
        """Configure system parameters (placeholder)"""
        print("⚙️ System Configuration - Feature coming soon!")
    
    async def _database_management(self):
        """Database management options (placeholder)"""
        print("🗄️ Database Management - Feature coming soon!")
    
    async def _cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            print("\n🔄 Shutting down services...")
            
            # Stop embedding service
            await async_embedding_service.stop()
            
            # Close database connections
            await batch_db_ops.close_connection_pool()
            
            # Stop memory monitoring
            if memory_manager:
                memory_manager.stop_monitoring()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            print(f"⚠️ Cleanup warning: {e}")

# Global enhanced CLI interface instance
enhanced_cli = EnhancedCLIInterface()
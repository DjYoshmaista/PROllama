# cli/interface.py
import asyncio
import logging
from typing import Optional

from core.config import config
from core.memory import memory_manager
from database.operations import db_ops
from database.cache import embedding_cache
from inference.engine import inference_engine
from inference.embeddings import embedding_service
from file_management.discovery import file_selector, file_discovery
from file_management.loaders import bulk_loader
from cli.validation import InputValidator

logger = logging.getLogger(__name__)

class CLIInterface:
    """Main command-line interface for the RAG system"""
    
    def __init__(self):
        self.validator = InputValidator()
        self._running = False
    
    async def initialize(self):
        """Initialize the system"""
        try:
            # Initialize database schema
            db_ops.initialize_schema()
            
            # Validate embedding model
            if not embedding_service.validate_model():
                print(f"CRITICAL: Embedding model {config.embedding.model} not available!")
                return False
            
            # Start embedding service
            await embedding_service.start(concurrency=10)
            logger.info("CLI system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"CLI initialization failed: {e}")
            print(f"System initialization failed: {e}")
            return False
    
    async def run(self):
        """Main CLI loop"""
        if not await self.initialize():
            return
        
        self._running = True
        
        try:
            while self._running:
                await self._show_main_menu()
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print(f"Unexpected error: {e}")
        finally:
            await self._cleanup()
    
    async def _show_main_menu(self):
        """Display main menu and handle user input"""
        print("\n" + "="*60)
        print("RAG Database System")
        print("="*60)
        print("1. Ask the inference model for information")
        print("2. Add data to the database")
        print("3. Load data from a single file")
        print("4. Load documents from folder")
        print("5. Query the database directly")
        print("6. List database contents")
        print("7. Configure system parameters")
        print("8. Database management")
        print("9. System information")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ").strip()
        
        try:
            if choice == "1":
                await self._handle_inference()
            elif choice == "2":
                await self._handle_add_data()
            elif choice == "3":
                await self._handle_load_single_file()
            elif choice == "4":
                await self._handle_load_folder()
            elif choice == "5":
                await self._handle_query_database()
            elif choice == "6":
                await self._handle_list_contents()
            elif choice == "7":
                await self._handle_configuration()
            elif choice == "8":
                await self._handle_database_management()
            elif choice == "9":
                await self._handle_system_info()
            elif choice == "0":
                self._running = False
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            logger.error(f"Menu handler error: {e}")
            print(f"Error: {e}")
    
    async def _handle_inference(self):
        """Handle inference/question-answering"""
        question = input("Enter your question: ").strip()
        if not question:
            print("Question cannot be empty!")
            return
        
        # Check if user wants to configure search parameters
        print("\nSearch Configuration:")
        print("Press Enter to use defaults, or type 'config' to customize")
        search_config = input("Choice: ").strip().lower()
        
        kwargs = {}
        if search_config == 'config':
            kwargs = self._get_search_configuration()
        
        # Ask about cache usage
        cache_choice = input("Use embedding cache if available? (Y/n): ").strip().lower()
        if cache_choice == 'n':
            kwargs['use_cache'] = False
        
        print("\nProcessing your question...")
        
        try:
            with memory_manager:
                result = await inference_engine.ask_question(question, **kwargs)
            
            # Display results
            print("\n" + "="*60)
            print("ANSWER:")
            print("="*60)
            print(result['answer'])
            
            if result['sources']:
                print("\n" + "="*60)
                print("SOURCES:")
                print("="*60)
                for i, source in enumerate(result['sources'], 1):
                    print(f"\n[Source {i}] (Similarity: {source['similarity']:.4f})")
                    print(f"ID: {source['id']}")
                    print(f"Preview: {source['content_preview']}")
                    if source.get('tags'):
                        print(f"Tags: {', '.join(source['tags'][:5])}")
            
            # Show metadata
            metadata = result['metadata']
            print(f"\nMatches found: {metadata['matches_found']}")
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            print(f"Failed to process question: {e}")
    
    def _get_search_configuration(self) -> dict:
        """Get custom search configuration from user"""
        kwargs = {}
        
        try:
            top_k = input(f"Top K results (default: {config.inference.top_k}): ").strip()
            if top_k:
                kwargs['top_k'] = int(top_k)
        except ValueError:
            print("Invalid top_k value, using default")
        
        try:
            threshold = input(f"Relevance threshold (default: {config.inference.relevance_threshold}): ").strip()
            if threshold:
                kwargs['relevance_threshold'] = float(threshold)
        except ValueError:
            print("Invalid threshold value, using default")
        
        return kwargs
    
    async def _handle_add_data(self):
        """Handle manual data addition"""
        content = input("Enter the content to add: ").strip()
        if not content:
            print("Content cannot be empty!")
            return
        
        # Get optional tags
        tags_input = input("Enter tags (comma-separated, optional): ").strip()
        tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
        
        try:
            print("Generating embedding...")
            embedding = await embedding_service.generate_embedding(content)
            
            if embedding:
                success = db_ops.insert_document(content, tags, embedding)
                if success:
                    print("Data added successfully.")
                    # Invalidate cache
                    embedding_cache.invalidate()
                else:
                    print("Failed to add data to database.")
            else:
                print("Failed to generate embedding.")
                
        except Exception as e:
            logger.error(f"Error adding data: {e}")
            print(f"Failed to add data: {e}")
    
    async def _handle_load_single_file(self):
        """Handle single file loading"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="Select File to Load")
            root.destroy()
            
            if file_path:
                success = await bulk_loader.load_single_file_interactive(file_path)
                if success:
                    # Invalidate cache after loading
                    embedding_cache.invalidate()
            else:
                print("No file selected.")
                
        except Exception as e:
            logger.error(f"File loading error: {e}")
            print(f"Failed to load file: {e}")
    
    async def _handle_load_folder(self):
        """Handle folder loading"""
        folder_path = file_selector.browse_for_folder()
        if not folder_path:
            print("No folder selected.")
            return
        
        # Count files
        total_files = file_discovery.count_files(folder_path)
        if total_files == 0:
            print("No supported files found in the selected folder.")
            return
        
        print(f"Found {total_files} supported files.")
        confirm = input("Proceed with loading? (y/N): ").strip().lower()
        
        if confirm != 'y':
            print("Loading cancelled.")
            return
        
        # Create file generator
        file_generator = file_discovery.discover_files(folder_path)
        
        # Progress callback
        def progress_callback(processed, total, records):
            percent = (processed / total) * 100
            print(f"\rProgress: {processed}/{total} files ({percent:.1f}%), {records} records", end="")
        
        print("Starting bulk loading...")
        
        try:
            total_records = await bulk_loader.load_from_folder(
                folder_path, file_generator, total_files, progress_callback
            )
            
            print(f"\nCompleted! Loaded {total_records} records from {total_files} files.")
            
            # Invalidate cache after bulk loading
            embedding_cache.invalidate()
            
        except Exception as e:
            logger.error(f"Bulk loading error: {e}")
            print(f"\nBulk loading failed: {e}")
    
    async def _handle_query_database(self):
        """Handle direct database queries"""
        print("\nDatabase Query Options:")
        print("1. Count total documents")
        print("2. Show recent documents")
        print("3. Search by ID")
        print("4. Custom query")
        
        choice = input("Select option (1-4): ").strip()
        
        try:
            if choice == "1":
                count = db_ops.get_document_count()
                print(f"Total documents: {count}")
                
            elif choice == "2":
                limit = int(input("Number of documents to show (default 10): ") or "10")
                docs = db_ops.get_documents_page(limit, 0)
                
                for doc in docs:
                    print(f"\nID: {doc['id']}")
                    print(f"Content: {doc['content'][:200]}...")
                    print(f"Created: {doc['created_at']}")
                    
            elif choice == "3":
                doc_id = int(input("Enter document ID: "))
                docs = db_ops.get_documents_by_ids([doc_id])
                
                if docs:
                    doc = docs[0]
                    print(f"\nID: {doc['id']}")
                    print(f"Content: {doc['content']}")
                    print(f"Tags: {doc.get('tags', [])}")
                else:
                    print("Document not found.")
                    
            elif choice == "4":
                print("Custom queries not implemented yet.")
            else:
                print("Invalid option.")
                
        except ValueError:
            print("Invalid input.")
        except Exception as e:
            logger.error(f"Database query error: {e}")
            print(f"Query failed: {e}")
    
    async def _handle_list_contents(self):
        """Handle content listing"""
        preview_length = int(input("Preview length (default 200): ") or "200")
        
        try:
            docs = db_ops.list_all_documents(preview_length)
            
            if not docs:
                print("No documents found in database.")
                return
            
            print(f"\nFound {len(docs)} documents:")
            for doc in docs:
                print(f"ID: {doc['id']}, Preview: {doc['content_preview']}")
                
        except Exception as e:
            logger.error(f"Error listing contents: {e}")
            print(f"Failed to list contents: {e}")
    
    async def _handle_configuration(self):
        """Handle system configuration"""
        print("\nConfiguration Options:")
        print("1. Inference parameters")
        print("2. Database optimization")
        print("3. Cache management")
        print("4. View current configuration")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            await self._configure_inference()
        elif choice == "2":
            await self._configure_database()
        elif choice == "3":
            await self._configure_cache()
        elif choice == "4":
            await self._show_configuration()
        else:
            print("Invalid option.")
    
    async def _configure_inference(self):
        """Configure inference parameters"""
        try:
            print(f"\nCurrent inference configuration:")
            print(f"Relevance threshold: {config.inference.relevance_threshold}")
            print(f"Top K: {config.inference.top_k}")
            print(f"Vector search limit: {config.inference.vector_search_limit}")
            
            # Get new values
            threshold = input(f"New relevance threshold (0.0-1.0, current: {config.inference.relevance_threshold}): ").strip()
            if threshold:
                new_threshold = float(threshold)
                if 0.0 <= new_threshold <= 1.0:
                    config.inference.relevance_threshold = new_threshold
                    print("Threshold updated.")
                else:
                    print("Invalid threshold. Must be between 0.0 and 1.0")
            
            top_k = input(f"New Top K (current: {config.inference.top_k}): ").strip()
            if top_k:
                new_top_k = int(top_k)
                if new_top_k > 0:
                    config.inference.top_k = new_top_k
                    print("Top K updated.")
                else:
                    print("Invalid Top K. Must be positive.")
                    
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
    
    async def _configure_database(self):
        """Configure database settings"""
        print("\nDatabase Configuration:")
        print("1. Optimize PostgreSQL settings")
        print("2. Run database maintenance")
        print("3. View database statistics")
        
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            try:
                db_ops.optimize_configuration()
                print("PostgreSQL optimization applied. Please restart PostgreSQL service.")
            except Exception as e:
                print(f"Optimization failed: {e}")
                
        elif choice == "2":
            try:
                print("Running database maintenance...")
                await db_ops.run_maintenance()
                print("Maintenance completed.")
            except Exception as e:
                print(f"Maintenance failed: {e}")
                
        elif choice == "3":
            try:
                metrics = await db_ops.get_batch_metrics()
                print(f"\nDatabase Statistics:")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"Failed to get statistics: {e}")
    
    async def _configure_cache(self):
        """Configure cache settings"""
        print("\nCache Management:")
        print("1. Load cache")
        print("2. Invalidate cache")
        print("3. Cache information")
        print("4. Refresh cache")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            try:
                print("Loading embedding cache...")
                success = embedding_cache.load_cache(force_reload=True)
                if success:
                    print("Cache loaded successfully.")
                else:
                    print("Failed to load cache.")
            except Exception as e:
                print(f"Cache loading failed: {e}")
                
        elif choice == "2":
            embedding_cache.invalidate()
            print("Cache invalidated.")
            
        elif choice == "3":
            info = embedding_cache.cache_info
            print(f"\nCache Information:")
            for key, value in info.items():
                print(f"{key}: {value}")
                
        elif choice == "4":
            try:
                print("Refreshing cache...")
                success = embedding_cache.refresh_if_needed()
                if success:
                    print("Cache refreshed.")
                else:
                    print("Cache refresh failed.")
            except Exception as e:
                print(f"Cache refresh failed: {e}")
    
    async def _show_configuration(self):
        """Show current system configuration"""
        print(f"\nSystem Configuration:")
        print(f"Database: {config.database.name}@{config.database.host}")
        print(f"Embedding model: {config.embedding.model}")
        print(f"Embedding dimension: {config.embedding.dimension}")
        print(f"Relevance threshold: {config.inference.relevance_threshold}")
        print(f"Top K: {config.inference.top_k}")
        print(f"Cache max size: {config.inference.max_cache_size}")
        
        # System status
        doc_count = db_ops.get_document_count()
        cache_info = embedding_cache.cache_info
        
        print(f"\nSystem Status:")
        print(f"Total documents: {doc_count}")
        print(f"Cache loaded: {cache_info['loaded']}")
        if cache_info['loaded']:
            print(f"Cached documents: {cache_info['count']}")
            print(f"Cache memory: {cache_info.get('memory_usage_mb', 0):.1f} MB")
    
    async def _handle_database_management(self):
        """Handle database management tasks"""
        print("\nDatabase Management:")
        print("1. Initialize/Reset schema")
        print("2. Run maintenance")
        print("3. Backup (not implemented)")
        print("4. Statistics")
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            confirm = input("This will reinitialize the database schema. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                try:
                    db_ops.initialize_schema()
                    print("Schema initialized.")
                except Exception as e:
                    print(f"Schema initialization failed: {e}")
                    
        elif choice == "2":
            try:
                await db_ops.run_maintenance()
                print("Database maintenance completed.")
            except Exception as e:
                print(f"Maintenance failed: {e}")
                
        elif choice == "3":
            print("Database backup functionality not implemented yet.")
            
        elif choice == "4":
            try:
                metrics = await db_ops.get_batch_metrics()
                count = db_ops.get_document_count()
                
                print(f"\nDatabase Statistics:")
                print(f"Total documents: {count}")
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"Failed to get statistics: {e}")
    
    async def _handle_system_info(self):
        """Show system information"""
        memory_info = memory_manager.get_memory_info()
        
        print(f"\nSystem Information:")
        print(f"Memory usage: {memory_info['percent']:.1f}%")
        print(f"Memory used: {memory_info['used_mb']:.1f} MB")
        print(f"Memory available: {memory_info['available_mb']:.1f} MB")
        
        if 'gpu_allocated_mb' in memory_info:
            print(f"GPU memory: {memory_info['gpu_allocated_mb']:.1f} MB")
        
        # Embedding service status
        if embedding_service._started:
            print(f"Embedding queue size: {embedding_service.queue_size}")
            print(f"Active embedding workers: {embedding_service.active_workers}")
        else:
            print("Embedding service not started")
        
        # Database status
        doc_count = db_ops.get_document_count()
        print(f"Database documents: {doc_count}")
        
        cache_info = embedding_cache.cache_info
        print(f"Cache status: {'Loaded' if cache_info['loaded'] else 'Not loaded'}")
    
    async def _cleanup(self):
        """Cleanup system resources"""
        print("Cleaning up system resources...")
        
        try:
            # Stop embedding service
            await embedding_service.stop()
            
            # Stop memory monitoring
            memory_manager.stop_monitoring()
            
            # Invalidate cache
            embedding_cache.invalidate()
            
            # Final memory cleanup
            memory_manager.cleanup_memory()
            
            logger.info("CLI cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Global CLI interface
cli = CLIInterface()
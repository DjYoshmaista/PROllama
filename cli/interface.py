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
from cli.handlers.inference_handler import InferenceHandler
from cli.handlers.data_handler import DataHandler
from cli.handlers.file_handler import FileHandler
from cli.handlers.database_handler import DatabaseHandler
from cli.handlers.config_handler import ConfigHandler
from cli.handlers.system_handler import SystemHandler
from cli.menu import MenuDisplay
from cli.cleanup import CleanupManager

logger = logging.getLogger(__name__)

class CLIInterface:
    """Main command-line interface for the RAG system"""
    
    def __init__(self):
        self.validator = InputValidator()
        self._running = False
        
        # Initialize handlers
        self.inference_handler = InferenceHandler()
        self.data_handler = DataHandler()
        self.file_handler = FileHandler()
        self.database_handler = DatabaseHandler()
        self.config_handler = ConfigHandler()
        self.system_handler = SystemHandler()
        self.menu_display = MenuDisplay()
        self.cleanup_manager = CleanupManager()
    
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
            await self.cleanup_manager.cleanup()
    
    async def _show_main_menu(self):
        """Display main menu and handle user input"""
        self.menu_display.show_main_menu()
        choice = input("\nEnter your choice (0-9): ").strip()
        
        try:
            if choice == "1":
                await self.inference_handler.handle_inference()
            elif choice == "2":
                await self.data_handler.handle_add_data()
            elif choice == "3":
                await self.file_handler.handle_load_single_file()
            elif choice == "4":
                await self.file_handler.handle_load_folder()
            elif choice == "5":
                await self.database_handler.handle_query_database()
            elif choice == "6":
                await self.database_handler.handle_list_contents()
            elif choice == "7":
                await self.config_handler.handle_configuration()
            elif choice == "8":
                await self.database_handler.handle_database_management()
            elif choice == "9":
                await self.system_handler.handle_system_info()
            elif choice == "0":
                self._running = False
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            logger.error(f"Menu handler error: {e}")
            print(f"Error: {e}")

# Global CLI interface
cli = CLIInterface()
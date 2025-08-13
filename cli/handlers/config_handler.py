# cli/handlers/config_handler.py
import logging
from core.config import config
from database.operations import db_ops
from database.cache import embedding_cache
from cli.menu import MenuDisplay

logger = logging.getLogger(__name__)

class ConfigHandler:
    """Handles system configuration operations"""
    
    def __init__(self):
        self.menu_display = MenuDisplay()
    
    async def handle_configuration(self):
        """Handle system configuration"""
        self.menu_display.show_configuration_menu()
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
            self._update_relevance_threshold()
            self._update_top_k()
                    
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
    
    async def _configure_database(self):
        """Configure database settings"""
        self.menu_display.show_database_configuration_menu()
        choice = input("Select option (1-3): ").strip()
        
        if choice == "1":
            await self._optimize_database()
        elif choice == "2":
            await self._run_database_maintenance()
        elif choice == "3":
            await self._show_database_statistics()
    
    async def _configure_cache(self):
        """Configure cache settings"""
        self.menu_display.show_cache_management_menu()
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            await self._load_cache()
        elif choice == "2":
            await self._invalidate_cache()
        elif choice == "3":
            await self._show_cache_info()
        elif choice == "4":
            await self._refresh_cache()
    
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
    
    def _update_relevance_threshold(self):
        """Update relevance threshold"""
        threshold = input(f"New relevance threshold (0.0-1.0, current: {config.inference.relevance_threshold}): ").strip()
        if threshold:
            new_threshold = float(threshold)
            if 0.0 <= new_threshold <= 1.0:
                config.inference.relevance_threshold = new_threshold
                print("Threshold updated.")
            else:
                print("Invalid threshold. Must be between 0.0 and 1.0")
    
    def _update_top_k(self):
        """Update Top K value"""
        top_k = input(f"New Top K (current: {config.inference.top_k}): ").strip()
        if top_k:
            new_top_k = int(top_k)
            if new_top_k > 0:
                config.inference.top_k = new_top_k
                print("Top K updated.")
            else:
                print("Invalid Top K. Must be positive.")
    
    async def _optimize_database(self):
        """Optimize database configuration"""
        try:
            db_ops.optimize_configuration()
            print("PostgreSQL optimization applied. Please restart PostgreSQL service.")
        except Exception as e:
            print(f"Optimization failed: {e}")
    
    async def _run_database_maintenance(self):
        """Run database maintenance"""
        try:
            print("Running database maintenance...")
            await db_ops.run_maintenance()
            print("Maintenance completed.")
        except Exception as e:
            print(f"Maintenance failed: {e}")
    
    async def _show_database_statistics(self):
        """Show database statistics"""
        try:
            metrics = await db_ops.get_batch_metrics()
            print(f"\nDatabase Statistics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"Failed to get statistics: {e}")
    
    async def _load_cache(self):
        """Load embedding cache"""
        try:
            print("Loading embedding cache...")
            success = embedding_cache.load_cache(force_reload=True)
            if success:
                print("Cache loaded successfully.")
            else:
                print("Failed to load cache.")
        except Exception as e:
            print(f"Cache loading failed: {e}")
    
    async def _invalidate_cache(self):
        """Invalidate cache"""
        embedding_cache.invalidate()
        print("Cache invalidated.")
    
    async def _show_cache_info(self):
        """Show cache information"""
        info = embedding_cache.cache_info
        print(f"\nCache Information:")
        for key, value in info.items():
            print(f"{key}: {value}")
    
    async def _refresh_cache(self):
        """Refresh cache"""
        try:
            print("Refreshing cache...")
            success = embedding_cache.refresh_if_needed()
            if success:
                print("Cache refreshed.")
            else:
                print("Cache refresh failed.")
        except Exception as e:
            print(f"Cache refresh failed: {e}")
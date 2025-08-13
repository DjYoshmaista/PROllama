# cli/cleanup.py
import logging
from core.memory import memory_manager
from inference.embeddings import embedding_service
from database.cache import embedding_cache

logger = logging.getLogger(__name__)

class CleanupManager:
    """Manages system cleanup and resource deallocation"""
    
    async def cleanup(self):
        """Cleanup system resources"""
        print("Cleaning up system resources...")
        
        try:
            await self._stop_embedding_service()
            self._stop_memory_monitoring()
            self._invalidate_cache()
            self._cleanup_memory()
            
            logger.info("CLI cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            print(f"Cleanup error: {e}")
    
    async def _stop_embedding_service(self):
        """Stop the embedding service"""
        try:
            await embedding_service.stop()
            print("✓ Embedding service stopped")
        except Exception as e:
            logger.error(f"Error stopping embedding service: {e}")
            print(f"✗ Error stopping embedding service: {e}")
    
    def _stop_memory_monitoring(self):
        """Stop memory monitoring"""
        try:
            memory_manager.stop_monitoring()
            print("✓ Memory monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping memory monitoring: {e}")
            print(f"✗ Error stopping memory monitoring: {e}")
    
    def _invalidate_cache(self):
        """Invalidate the embedding cache"""
        try:
            embedding_cache.invalidate()
            print("✓ Cache invalidated")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            print(f"✗ Error invalidating cache: {e}")
    
    def _cleanup_memory(self):
        """Final memory cleanup"""
        try:
            memory_manager.cleanup_memory()
            print("✓ Memory cleanup completed")
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            print(f"✗ Error during memory cleanup: {e}")
    
    def force_cleanup(self):
        """Force cleanup without async operations"""
        try:
            self._stop_memory_monitoring()
            self._invalidate_cache()
            self._cleanup_memory()
            print("Force cleanup completed")
        except Exception as e:
            logger.error(f"Force cleanup error: {e}")
            print(f"Force cleanup error: {e}")
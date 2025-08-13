# cli/handlers/system_handler.py
import logging
from core.memory import memory_manager
from inference.embeddings import embedding_service
from database.operations import db_ops
from database.cache import embedding_cache

logger = logging.getLogger(__name__)

class SystemHandler:
    """Handles system information and monitoring operations"""
    
    async def handle_system_info(self):
        """Show comprehensive system information"""
        self._show_memory_info()
        self._show_embedding_service_status()
        self._show_database_status()
        self._show_cache_status()
    
    def _show_memory_info(self):
        """Display memory information"""
        memory_info = memory_manager.get_memory_info()
        
        print(f"\nSystem Information:")
        print(f"Memory usage: {memory_info['percent']:.1f}%")
        print(f"Memory used: {memory_info['used_mb']:.1f} MB")
        print(f"Memory available: {memory_info['available_mb']:.1f} MB")
        
        if 'gpu_allocated_mb' in memory_info:
            print(f"GPU memory: {memory_info['gpu_allocated_mb']:.1f} MB")
    
    def _show_embedding_service_status(self):
        """Display embedding service status"""
        if embedding_service._started:
            print(f"Embedding queue size: {embedding_service.queue_size}")
            print(f"Active embedding workers: {embedding_service.active_workers}")
        else:
            print("Embedding service not started")
    
    def _show_database_status(self):
        """Display database status"""
        doc_count = db_ops.get_document_count()
        print(f"Database documents: {doc_count}")
    
    def _show_cache_status(self):
        """Display cache status"""
        cache_info = embedding_cache.cache_info
        print(f"Cache status: {'Loaded' if cache_info['loaded'] else 'Not loaded'}")
        
        if cache_info['loaded']:
            print(f"Cached documents: {cache_info.get('count', 0)}")
            print(f"Cache memory usage: {cache_info.get('memory_usage_mb', 0):.1f} MB")
    
    def get_system_metrics(self) -> dict:
        """Get comprehensive system metrics"""
        memory_info = memory_manager.get_memory_info()
        cache_info = embedding_cache.cache_info
        doc_count = db_ops.get_document_count()
        
        return {
            'memory_usage_percent': memory_info['percent'],
            'memory_used_mb': memory_info['used_mb'],
            'memory_available_mb': memory_info['available_mb'],
            'gpu_memory_mb': memory_info.get('gpu_allocated_mb', 0),
            'embedding_service_started': embedding_service._started,
            'embedding_queue_size': getattr(embedding_service, 'queue_size', 0),
            'database_document_count': doc_count,
            'cache_loaded': cache_info['loaded'],
            'cache_document_count': cache_info.get('count', 0),
            'cache_memory_mb': cache_info.get('memory_usage_mb', 0)
        }
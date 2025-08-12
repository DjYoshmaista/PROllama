# database/cache.py - Optimized Embedding Cache
import time
import torch
import logging
from typing import Optional, Dict, Any, List, Tuple
from database.operations import db_ops
from core.config import config

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Optimized embedding cache that only loads when beneficial
    Uses smart loading strategies based on database size
    """
    
    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._last_refresh: float = 0
        self._refresh_interval = config.inference.cache_refresh_interval
        self._max_cache_size = config.inference.max_cache_size
        self._cache_threshold = 1000  # Only cache if less than 1000 documents
    
    @property
    def is_loaded(self) -> bool:
        """Check if cache is loaded and valid"""
        return (
            self._cache is not None and 
            (time.time() - self._last_refresh) < self._refresh_interval
        )
    
    @property
    def cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        if not self.is_loaded:
            return {"loaded": False, "count": 0, "age_seconds": 0}
        
        return {
            "loaded": True,
            "count": self._cache.get('count', 0),
            "age_seconds": time.time() - self._last_refresh,
            "memory_usage_mb": self._estimate_memory_usage(),
            "recommended": self._should_use_cache()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB"""
        if not self._cache or 'embeddings' not in self._cache:
            return 0.0
        
        embeddings_tensor = self._cache['embeddings']
        if isinstance(embeddings_tensor, torch.Tensor):
            return embeddings_tensor.element_size() * embeddings_tensor.nelement() / (1024 ** 2)
        return 0.0
    
    def _should_use_cache(self) -> bool:
        """Determine if caching is beneficial based on database size"""
        doc_count = db_ops.get_document_count()
        
        # Only cache small to medium databases where it provides benefit
        if doc_count == 0:
            return False
        elif doc_count <= self._cache_threshold:
            return True
        else:
            logger.info(f"Database size ({doc_count}) exceeds cache threshold ({self._cache_threshold})")
            return False
    
    def should_use_cache_for_search(self) -> bool:
        """
        Determine if cache should be used for similarity search
        Returns False if database-level search would be more efficient
        """
        doc_count = db_ops.get_document_count()
        
        # For small databases, cache provides benefit
        if doc_count <= 500:
            return True
        
        # For medium databases, only if cache is already loaded
        elif doc_count <= self._cache_threshold:
            return self.is_loaded
        
        # For large databases, always use database-level search
        else:
            return False
    
    def load_cache(self, force_reload: bool = False) -> bool:
        """
        Load embeddings into memory cache - only if beneficial
        """
        current_time = time.time()
        
        # Check if reload is needed
        if not force_reload and self.is_loaded:
            logger.debug("Cache is already loaded and valid")
            return True
        
        # Check if caching is beneficial
        if not self._should_use_cache():
            logger.info("Caching not recommended for current database size")
            return False
        
        try:
            logger.info("Loading embedding cache...")
            
            # Use optimized cache loading method
            cache_data = db_ops.get_embeddings_for_cache(self._max_cache_size)
            
            if cache_data is None:
                logger.warning("Database too large for caching")
                return False
            
            ids, embeddings = cache_data
            
            if not embeddings:
                logger.error("No valid embeddings found for caching")
                return False
            
            # Convert to tensor
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
            if torch.cuda.is_available():
                embedding_tensor = embedding_tensor.cuda()
                logger.info("Embeddings loaded to GPU")
            
            # Store cache
            self._cache = {
                'ids': ids,
                'embeddings': embedding_tensor,
                'count': len(ids),
                'timestamp': current_time
            }
            self._last_refresh = current_time
            
            memory_mb = self._estimate_memory_usage()
            logger.info(f"Embedding cache loaded: {len(ids)} vectors, {memory_mb:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding cache: {e}")
            self._cache = None
            return False
    
    def get_embeddings_tensor(self) -> Optional[torch.Tensor]:
        """Get the embeddings tensor from cache"""
        if not self.is_loaded:
            return None
        return self._cache.get('embeddings')
    
    def get_document_ids(self) -> Optional[List[int]]:
        """Get document IDs from cache"""
        if not self.is_loaded:
            return None
        return self._cache.get('ids')
    
    def invalidate(self):
        """Invalidate the cache"""
        if self._cache and 'embeddings' in self._cache:
            # Clean up GPU memory
            embeddings_tensor = self._cache['embeddings']
            if isinstance(embeddings_tensor, torch.Tensor) and embeddings_tensor.is_cuda:
                del embeddings_tensor
                torch.cuda.empty_cache()
        
        self._cache = None
        self._last_refresh = 0
        logger.info("Embedding cache invalidated")
    
    def refresh_if_needed(self) -> bool:
        """Refresh cache if it's stale"""
        if (time.time() - self._last_refresh) > self._refresh_interval:
            logger.info("Cache is stale, refreshing...")
            return self.load_cache(force_reload=True)
        return True
    
    def get_cache_recommendation(self) -> Dict[str, Any]:
        """Get recommendation on whether to use cache"""
        doc_count = db_ops.get_document_count()
        
        if doc_count == 0:
            return {
                "recommended": False,
                "reason": "No documents in database",
                "strategy": "none"
            }
        elif doc_count <= 100:
            return {
                "recommended": True,
                "reason": "Small database - cache provides significant speedup",
                "strategy": "always_cache"
            }
        elif doc_count <= self._cache_threshold:
            return {
                "recommended": True,
                "reason": "Medium database - cache beneficial for repeated searches",
                "strategy": "cache_if_available"
            }
        else:
            return {
                "recommended": False,
                "reason": f"Large database ({doc_count} docs) - database search more efficient",
                "strategy": "database_only"
            }
    
    def __del__(self):
        """Cleanup when cache is destroyed"""
        self.invalidate()

# Global embedding cache instance
embedding_cache = EmbeddingCache()
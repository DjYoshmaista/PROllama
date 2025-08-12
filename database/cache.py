# database/cache.py
import time
import torch
import logging
from typing import Optional, Dict, Any, List
from database.operations import db_ops
from core.config import config

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Manages in-memory embedding cache for fast similarity search"""
    
    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self._last_refresh: float = 0
        self._refresh_interval = config.inference.cache_refresh_interval
        self._max_cache_size = config.inference.max_cache_size
    
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
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB"""
        if not self._cache or 'embeddings' not in self._cache:
            return 0.0
        
        embeddings_tensor = self._cache['embeddings']
        if isinstance(embeddings_tensor, torch.Tensor):
            return embeddings_tensor.element_size() * embeddings_tensor.nelement() / (1024 ** 2)
        return 0.0
    
    def load_cache(self, force_reload: bool = False) -> bool:
        """Load all embeddings into memory for fast GPU processing"""
        current_time = time.time()
        
        # Check if reload is needed
        if not force_reload and self.is_loaded:
            logger.debug("Cache is already loaded and valid")
            return True
        
        # Check document count first
        total_docs = db_ops.get_document_count()
        if total_docs == 0:
            logger.warning("No documents in database to cache")
            return False
        
        if total_docs > self._max_cache_size:
            logger.warning(f"Database too large for cache ({total_docs} > {self._max_cache_size})")
            return False
        
        try:
            logger.info(f"Loading embedding cache for {total_docs} documents...")
            
            # Load all embeddings
            all_embeddings = []
            all_ids = []
            offset = 0
            chunk_size = 1000
            
            while offset < total_docs:
                embeddings_page = db_ops.get_embeddings_page(chunk_size, offset)
                if not embeddings_page:
                    break
                
                for doc_id, embedding in embeddings_page:
                    all_ids.append(doc_id)
                    all_embeddings.append(embedding)
                
                offset += chunk_size
            
            if not all_embeddings:
                logger.error("No valid embeddings found in database")
                return False
            
            # Convert to tensor
            embedding_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
            if torch.cuda.is_available():
                embedding_tensor = embedding_tensor.cuda()
                logger.info("Embeddings loaded to GPU")
            
            # Store cache
            self._cache = {
                'ids': all_ids,
                'embeddings': embedding_tensor,
                'count': len(all_ids),
                'timestamp': current_time
            }
            self._last_refresh = current_time
            
            memory_mb = self._estimate_memory_usage()
            logger.info(f"Embedding cache loaded: {len(all_ids)} vectors, {memory_mb:.1f}MB")
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
    
    def __del__(self):
        """Cleanup when cache is destroyed"""
        self.invalidate()

# Global embedding cache instance
embedding_cache = EmbeddingCache()
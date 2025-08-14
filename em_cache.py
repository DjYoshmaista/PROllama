# embedding_cache.py
import asyncpg
from constants import *
import time
import torch
import logging
import os
import heapq
import numpy as np
import tempfile
import shelve
import atexit
import shutil
from collections import OrderedDict
import json

logger = logging.getLogger(__name__)

class EmbeddingCache:
    _instance = None
    MAX_MEMORY_BYTES = 1 * 1024 * 1024 * 1024  # 1GB
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize only once
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize cache structures and setup cleanup"""
        # In-memory cache (OrderedDict: key -> (embedding, byte_size))
        self.memory_cache = OrderedDict()
        self.total_memory_bytes = 0
        
        # Disk cache setup
        self.disk_cache_dir = tempfile.mkdtemp(prefix="embedding_cache_")
        self.disk_cache_path = os.path.join(self.disk_cache_dir, "disk_cache.shelve")
        self.disk_cache = shelve.open(self.disk_cache_path, flag='c')
        
        # Unique embeddings tracking
        self.embedding_signatures = {}
        
        # Setup cleanup on exit
        atexit.register(self.cleanup)
    
    async def load(self, force=False):
        """Load embeddings from database into cache"""
        from tqdm import tqdm
        try:
            async with asyncpg.connect(DB_CONN_STRING) as conn:
                with tqdm(postfix={"Loading": "embeddings"}, unit="embeddings") as pbar:
                    records = await conn.fetch(
                        f"SELECT id, embedding FROM {TABLE_NAME}"
                    )
                
                    for record in records:
                        doc_id = record['id']
                        embedding = record['embedding']
                        
                        # Handle different embedding formats
                        if isinstance(embedding, str):
                            try:
                                embedding = json.loads(embedding)
                            except json.JSONDecodeError:
                                # Handle pgvector array format
                                if embedding.startswith('[') and embedding.endswith(']'):
                                    embedding = [
                                        float(x) 
                                        for x in embedding[1:-1].split(',')
                                    ]
                        
                        # Add to cache
                        await self.add(doc_id, embedding)
                        pbar.update(1)
                    
                logger.info(f"Loaded {len(records)} embeddings into cache")
                return True
        except Exception as e:
            logger.error(f"Cache load failed: {e}", exc_info=True)
            return False
    
    async def add(self, key, embedding):
        """Add embedding to cache with deduplication and memory management"""
        # Convert to tensor if needed
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        # Create signature for deduplication
        signature = self._create_signature(embedding)
        
        # Skip exact duplicates
        if signature in self.embedding_signatures:
            existing_key = self.embedding_signatures[signature]
            logger.debug(f"Duplicate embedding found. Key {key} duplicates {existing_key}")
            return
        
        # Calculate memory footprint
        byte_size = embedding.element_size() * embedding.nelement()
        
        # Remove existing entry if present
        await self._remove_key(key)
        
        # Add to memory cache
        self.memory_cache[key] = (embedding, byte_size)
        self.memory_cache.move_to_end(key)  # Mark as recently used
        self.total_memory_bytes += byte_size
        self.embedding_signatures[signature] = key
        
        # Evict to disk if over limit
        if self.total_memory_bytes > self.MAX_MEMORY_BYTES:
            await self._evict_to_disk()
    
    async def get(self, key):
        """Retrieve embedding from cache (memory or disk)"""
        # Check memory first
        if key in self.memory_cache:
            embedding, size = self.memory_cache[key]
            # Move to end to mark as recently used
            self.memory_cache.move_to_end(key)
            return embedding
        
        # Check disk cache
        if key in self.disk_cache:
            embedding = self.disk_cache[key]
            # Promote back to memory
            await self.add(key, embedding)
            return embedding
        
        return None
    
    async def _evict_to_disk(self):
        """Move oldest 25% of memory cache to disk"""
        num_entries = len(self.memory_cache)
        if num_entries == 0:
            return
        
        # Calculate 25% of entries (at least 1)
        num_to_evict = max(1, num_entries // 4)
        logger.info(f"Evicting {num_to_evict} oldest entries to disk")
        
        # Evict oldest entries
        for _ in range(num_to_evict):
            key, (embedding, byte_size) = self.memory_cache.popitem(last=False)
            
            # Move to disk
            self.disk_cache[key] = embedding.cpu().numpy()  # Store as numpy array
            
            # Update memory tracking
            self.total_memory_bytes -= byte_size
            del self.embedding_signatures[self._create_signature(embedding)]
    
    async def _remove_key(self, key):
        """Remove key from both memory and disk caches"""
        # Remove from memory cache
        if key in self.memory_cache:
            embedding, byte_size = self.memory_cache.pop(key)
            self.total_memory_bytes -= byte_size
            del self.embedding_signatures[self._create_signature(embedding)]
        
        # Remove from disk cache
        if key in self.disk_cache:
            del self.disk_cache[key]
    
    def _create_signature(self, embedding):
        """Create unique signature for embedding deduplication"""
        # Use a precision-reduced hash for efficiency
        if embedding.numel() > 1000:
            sample_points = min(1000, embedding.numel())
            stride = embedding.numel() // sample_points
            sample = embedding[::stride].cpu().numpy()
        else:
            sample = embedding.cpu().numpy()
        
        # Create hashable signature
        return hash(tuple(np.round(sample, decimals=3).tobytes()))
    
    def cleanup(self):
        """Clean up resources on exit"""
        # Close disk cache
        self.disk_cache.close()
        
        # Remove temporary directory
        try:
            shutil.rmtree(self.disk_cache_dir)
            logger.info("Cleaned up disk cache directory")
        except Exception as e:
            logger.error(f"Error cleaning cache directory: {e}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @property
    def memory_usage(self):
        """Get current memory usage in human-readable format"""
        mb = self.total_memory_bytes / (1024 * 1024)
        return f"{mb:.2f}MB"
    
    @property
    def stats(self):
        """Get cache statistics"""
        return {
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(self.disk_cache),
            "memory_usage": self.memory_usage,
            "unique_embeddings": len(self.embedding_signatures)
        }
import asyncpg
from constants import *
import time
import torch
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cache = None
            cls._instance.last_refresh = 0
        return cls._instance
    
    async def load(self, force=False):
        """Unified cache loading with async support"""
        if not force and self.cache and (time.time() - self.last_refresh) < CACHE_REFRESH_INTERVAL:
            return True
            
        try:
            async with asyncpg.connect(DB_CONN_STRING) as conn:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                if count > MAX_CACHE_SIZE:
                    return False
                    
                records = await conn.fetch(f"SELECT id, embedding FROM {TABLE_NAME}")
                ids = [r['id'] for r in records]
                embeddings = [r['embedding'] for r in records]
                
                self.cache = {
                    'ids': ids,
                    'embeddings': torch.tensor(embeddings).cuda() if torch.cuda.is_available() 
                                else torch.tensor(embeddings),
                    'timestamp': time.time()
                }
                return True
        except Exception as e:
            logger.error(f"Cache load failed: {e}")
            return False
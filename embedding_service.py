# embedding_service.py - Solution 1: Standalone embedding service
import asyncio
import aiohttp
import json
import logging
import random
from typing import List, Optional, Any
from config import Config
from constants import EMBEDDING_MODEL
from ollama import embed
logger = logging.getLogger(__name__)

def is_zero_vector(vector):
    """Check if vector contains only zeros"""
    return all(x == 0 for x in vector)

class EmbeddingService:
    """Standalone embedding service to break circular dependency"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.session = None
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, text: Optional[Any] = None, texts: Optional[List[str]] = None):
        self.text = text

    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if self.session is None:
            connector = aiohttp.TCPConnector(limit_per_host=10)
            self.session = aiohttp.ClientSession(connector=connector)
            self._initialized = True
    
    async def get_single_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text"""
        await self._ensure_session()
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with embed(model=EMBEDDING_MODEL, text=text) as response:
                    if response.status == 200:
                        data = await response.json()
                        embedding = data.get("embedding") or (data.get("embeddings")[0] if data.get("embeddings") else None)
                        if not embedding:
                            logger.error(f"Unexpected response format: {data}")
                            raise ValueError("Unexpected response format")
                        if len(embedding) != Config.EMBEDDING_DIM:
                            logger.error(f"Invalid dimension: {len(embedding)} (expected {Config.EMBEDDING_DIM})")
                            raise ValueError(f"Invalid dimension: {len(embedding)}")
                        if is_zero_vector(embedding):
                            logger.error("Zero vector generated")
                            raise ValueError("Zero vector generated")
                        return embedding
                    error = await response.text()
                    logger.error(f"API error {response.status}: {error}")
                    raise Exception(f"API error {response.status}: {error}")
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.warning(f"Embedding attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 1 + random.random() + attempt
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"Embedding failed after {max_retries} attempts")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode error: {str(e)}")
                raise
        return None
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for a batch of texts"""
        if not texts:
            return []
        
        # Process in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
        
        async def get_with_semaphore(text):
            async with semaphore:
                return await self.get_single_embedding(text)
        
        results = await asyncio.gather(*[get_with_semaphore(text) for text in texts], return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch embedding: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            self._initialized = False

# Global embedding service instance
embedding_service = EmbeddingService()

# Convenience functions that can be imported without circular dependency
async def get_single_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for a single text"""
    return await embedding_service.get_single_embedding(text)

async def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Get embeddings for a batch of texts"""
    return await embedding_service.get_embeddings_batch(texts)
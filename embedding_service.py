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
    
    async def fetch_embedding(self, text: str) -> Optional[List[float]]:
        """Fetch embedding for a single text - FIXED VERSION"""
        logger.debug(f"fetch_embedding() called for text length: {len(text)}")
        
        if not self.session:
            logger.error("HTTP session not initialized!")
            return None
            
        max_retries = 3  # Reduced from 5 for faster failure
        for attempt in range(max_retries):
            logger.debug(f"Embedding attempt {attempt + 1}/{max_retries}")
            try:
                # FIXED: Use correct Ollama embed function (synchronous, not async context manager)
                logger.debug(f"Making Ollama embed call")
                logger.debug(f"Model: {EMBEDDING_MODEL}")
                logger.debug(f"Text preview: {text[:100]}...")
                
                # FIXED: Use synchronous embed call, not async context manager
                response = embed(model=EMBEDDING_MODEL, input=text)
                logger.debug(f"Ollama embed response received")
                
                # FIXED: Handle Ollama response format directly
                if 'embedding' in response:
                    embedding = response['embedding']
                    logger.debug(f"Embedding received - dimension: {len(embedding)}")
                    
                    if len(embedding) != Config.EMBEDDING_DIM:
                        logger.error(f"Invalid dimension: {len(embedding)} (expected {Config.EMBEDDING_DIM})")
                        raise ValueError(f"Invalid dimension: {len(embedding)}")
                    
                    if is_zero_vector(embedding):
                        logger.error("Zero vector generated")
                        raise ValueError("Zero vector generated")
                    
                    logger.debug("Embedding successfully generated")
                    return embedding
                else:
                    logger.error(f"No embedding in response: {response}")
                    raise ValueError("No embedding in response")
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
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

    async def get_single_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text - FINAL FIXED VERSION"""
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # FIXED: Use synchronous embed call directly with Ollama library
                logger.debug(f"Making Ollama embed call for text length: {len(text)}")
                
                # Use synchronous Ollama embed function
                response = embed(model=EMBEDDING_MODEL, input=text)
                logger.debug(f"Ollama embed response received")
                logger.debug(f"Response type: {type(response)}")
                
                # FINAL FIX: Extract embedding using multiple methods
                embedding = None
                
                # Method 1: getattr approach (most reliable)
                try:
                    embeddings_list = getattr(response, 'embeddings', None)
                    if embeddings_list is not None and len(embeddings_list) > 0:
                        embedding = embeddings_list[0]
                        logger.debug(f"✓ Successfully extracted embedding via getattr - dimension: {len(embedding)}")
                except Exception as e:
                    logger.debug(f"getattr method failed: {e}")
                
                # Method 2: Direct access
                if embedding is None:
                    try:
                        if hasattr(response, 'embeddings') and response.embeddings:
                            embedding = response.embeddings[0]
                            logger.debug(f"✓ Successfully extracted embedding via direct access - dimension: {len(embedding)}")
                    except Exception as e:
                        logger.debug(f"Direct access failed: {e}")
                
                # Method 3: Try singular 'embedding'
                if embedding is None:
                    try:
                        if hasattr(response, 'embedding') and response.embedding:
                            embedding = response.embedding
                            logger.debug(f"✓ Successfully extracted via embedding attribute - dimension: {len(embedding)}")
                    except Exception as e:
                        logger.debug(f"Singular embedding access failed: {e}")
                
                # Final validation
                if embedding is None:
                    logger.error(f"Failed to extract embedding from response: {response}")
                    raise ValueError("Could not extract embedding from Ollama response")
                    
                if not isinstance(embedding, (list, tuple)):
                    raise ValueError(f"Invalid embedding type: {type(embedding)}")
                    
                if len(embedding) != Config.EMBEDDING_DIM:
                    logger.error(f"Invalid dimension: {len(embedding)} (expected {Config.EMBEDDING_DIM})")
                    raise ValueError(f"Invalid dimension: {len(embedding)}")
                    
                if is_zero_vector(embedding):
                    logger.error("Zero vector generated")
                    raise ValueError("Zero vector generated")
                    
                return embedding
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 1 + random.random() + attempt
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"Embedding failed after {max_retries} attempts")
                raise
                
        return None

# Global embedding service instance
embedding_service = EmbeddingService()

# Convenience functions that can be imported without circular dependency
async def get_single_embedding(text: str) -> Optional[List[float]]:
    """Get embedding for a single text"""
    return await embedding_service.get_single_embedding(text)

async def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Get embeddings for a batch of texts"""
    return await embedding_service.get_embeddings_batch(texts)
# inference/embeddings.py
import asyncio
import aiohttp
import json
import logging
import random
import time
from typing import List, Optional, Union
from core.config import config
from core.vector_ops import VectorOperations

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Unified embedding generation service with queue management"""
    
    def __init__(self):
        self.api_url = config.embedding.api_url
        self.model = config.embedding.model
        self.dimension = config.embedding.dimension
        self.max_concurrent = config.embedding.max_concurrent_requests
        
        self._queue = asyncio.Queue()
        self._workers = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._started = False
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def start(self, concurrency: int = 10):
        """Start the embedding service workers"""
        if self._started:
            return
        
        self._started = True
        
        # Create shared session with connection limits
        connector = aiohttp.TCPConnector(limit_per_host=2, limit=50)
        self._session = aiohttp.ClientSession(connector=connector)
        
        # Start worker tasks
        self._workers = [
            asyncio.create_task(self._worker()) 
            for _ in range(concurrency)
        ]
        
        logger.info(f"Started {concurrency} embedding workers")
    
    async def stop(self):
        """Stop the embedding service"""
        if not self._started:
            return
        
        self._started = False
        
        # Wait for queue to empty
        await self._queue.join()
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        
        # Close session
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("Embedding service stopped")
    
    async def _worker(self):
        """Worker task for processing embedding requests"""
        while self._started:
            try:
                text, future = await self._queue.get()
                try:
                    embedding = await self._generate_embedding(text)
                    future.set_result(embedding)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                async with self._semaphore:
                    async with self._session.post(
                        f"{self.api_url}/embed",
                        json={"model": self.model, "input": text},
                        timeout=30
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            # Handle different response formats
                            embedding = (
                                data.get("embedding") or 
                                (data.get("embeddings", [None])[0] if data.get("embeddings") else None)
                            )
                            
                            if not embedding:
                                raise ValueError(f"Unexpected response format: {data}")
                            
                            # Validate embedding
                            if len(embedding) != self.dimension:
                                raise ValueError(f"Invalid dimension: {len(embedding)} (expected {self.dimension})")
                            
                            if VectorOperations.is_zero_vector(embedding):
                                raise ValueError("Zero vector generated")
                            
                            return embedding
                        
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                        
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 1 + random.random() + attempt
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Embedding failed after {max_retries} attempts")
                raise
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                raise
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text (public interface)"""
        if not self._started:
            await self.start()
        
        future = asyncio.Future()
        await self._queue.put((text, future))
        
        try:
            return await future
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        if not self._started:
            await self.start()
        
        # Create futures for all texts
        futures = []
        for text in texts:
            future = asyncio.Future()
            await self._queue.put((text, future))
            futures.append(future)
        
        # Wait for all embeddings to complete
        try:
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Convert exceptions to None
            embeddings = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding error: {result}")
                    embeddings.append(None)
                else:
                    embeddings.append(result)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(texts)
    
    def generate_embedding_sync(self, text: str) -> Optional[List[float]]:
        """Synchronous wrapper for embedding generation (avoid in async contexts)"""
        logger.warning("Using synchronous embedding call - avoid in async contexts")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.generate_embedding(text))
        finally:
            loop.close()
    
    def validate_model(self) -> bool:
        """Validate that the embedding model is available"""
        try:
            import requests
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            if self.model not in models:
                logger.error(f"Embedding model {self.model} not found in Ollama!")
                logger.info(f"Available models: {models}")
                return False
            
            logger.info(f"Embedding model {self.model} validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    @property
    def queue_size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()
    
    @property
    def active_workers(self) -> int:
        """Get number of active workers"""
        return sum(1 for w in self._workers if not w.done())

# Global embedding service instance
embedding_service = EmbeddingService()
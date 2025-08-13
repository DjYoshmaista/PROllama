# inference/embeddings.py
import asyncio
import aiohttp
import logging
import json
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import time

from core.config import config

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRequest:
    """Container for embedding request"""
    text: str
    future: asyncio.Future
    retries: int = 0
    max_retries: int = 3

class EmbeddingService:
    """Centralized embedding service with queue-based processing"""
    
    def __init__(self):
        self.model = config.embedding.model
        self.dimension = config.embedding.dimension
        self.api_url = config.embedding.api_url
        self.max_concurrent = config.embedding.max_concurrent_requests
        
        self._queue: asyncio.Queue = None
        self._workers: List[asyncio.Task] = []
        self._session: aiohttp.ClientSession = None
        self._started = False
        self._stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'avg_time': 0
        }
    
    @property
    def queue_size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize() if self._queue else 0
    
    @property
    def active_workers(self) -> int:
        """Get number of active workers"""
        return sum(1 for w in self._workers if not w.done()) if self._workers else 0
    
    async def start(self, concurrency: Optional[int] = None):
        """Start the embedding service with worker pool"""
        if self._started:
            return
        
        concurrency = concurrency or min(self.max_concurrent, 50)
        
        # Create queue and session
        self._queue = asyncio.Queue(maxsize=1000)
        
        # Configure connector with connection pooling
        connector = aiohttp.TCPConnector(
            limit=concurrency * 2,
            limit_per_host=concurrency,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        # Start worker tasks
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(concurrency)
        ]
        
        self._started = True
        logger.info(f"Embedding service started with {concurrency} workers")
    
    async def stop(self):
        """Stop the embedding service"""
        if not self._started:
            return
        
        self._started = False
        
        # Wait for queue to empty
        if self._queue:
            while not self._queue.empty():
                await asyncio.sleep(0.1)
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        
        # Close session
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info(f"Embedding service stopped. Stats: {self._stats}")
    
    async def _worker(self, worker_id: int):
        """Worker task to process embedding requests"""
        logger.debug(f"Worker {worker_id} started")
        
        while self._started:
            try:
                # Get request from queue with timeout
                try:
                    request = await asyncio.wait_for(
                        self._queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                start_time = time.time()
                
                try:
                    embedding = await self._fetch_embedding(request.text)
                    
                    # Update stats
                    self._stats['successful'] += 1
                    elapsed = time.time() - start_time
                    self._stats['avg_time'] = (
                        self._stats['avg_time'] * 0.9 + elapsed * 0.1
                    )
                    
                    # Set result
                    if not request.future.done():
                        request.future.set_result(embedding)
                        
                except Exception as e:
                    # Retry logic
                    if request.retries < request.max_retries:
                        request.retries += 1
                        await self._queue.put(request)  # Re-queue for retry
                        logger.debug(f"Retrying request (attempt {request.retries})")
                    else:
                        self._stats['failed'] += 1
                        if not request.future.done():
                            request.future.set_result(None)
                        logger.error(f"Failed after {request.max_retries} retries: {e}")
                
                finally:
                    self._stats['total_requests'] += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _fetch_embedding(self, text: str) -> Optional[List[float]]:
        """Fetch embedding from API"""
        if not self._session:
            raise RuntimeError("Session not initialized")
        
        try:
            async with self._session.post(
                f"{self.api_url}/embed",
                json={"model": self.model, "input": text},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle different response formats
                    embedding = data.get("embedding")
                    if not embedding and "embeddings" in data:
                        embedding = data["embeddings"][0] if data["embeddings"] else None
                    
                    if not embedding:
                        raise ValueError(f"No embedding in response: {data}")
                    
                    # Validate embedding
                    if len(embedding) != self.dimension:
                        raise ValueError(
                            f"Invalid embedding dimension: {len(embedding)} != {self.dimension}"
                        )
                    
                    # Check for zero vector
                    if all(x == 0 for x in embedding):
                        raise ValueError("Zero vector generated")
                    
                    return embedding
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception("Request timeout")
        except aiohttp.ClientError as e:
            raise Exception(f"Client error: {e}")
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text"""
        if not self._started:
            await self.start()
        
        # Create future for result
        future = asyncio.Future()
        request = EmbeddingRequest(text=text, future=future)
        
        # Add to queue
        await self._queue.put(request)
        
        # Wait for result
        return await future
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str]
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for batch of texts"""
        if not texts:
            return []
        
        if not self._started:
            await self.start()
        
        # Create requests
        futures = []
        for text in texts:
            future = asyncio.Future()
            request = EmbeddingRequest(text=text, future=future)
            await self._queue.put(request)
            futures.append(future)
        
        # Wait for all results
        results = await asyncio.gather(*futures, return_exceptions=False)
        return results
    
    def validate_model(self) -> bool:
        """Validate that the embedding model is available"""
        # This would normally check if the model is available
        # For now, we'll assume it's available if config is set
        return bool(self.model and self.api_url)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self._stats,
            'queue_size': self.queue_size,
            'active_workers': self.active_workers,
            'model': self.model
        }

# Global embedding service instance
embedding_service = EmbeddingService()
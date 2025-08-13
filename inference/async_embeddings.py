# inference/async_embeddings.py - Pure Asyncio Embedding Service
import asyncio
import aiohttp
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
import uuid

from core.config import config
from core.vector_ops import VectorOperations

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRequest:
    """Represents an embedding request"""
    text: str
    request_id: str
    priority: int = 1
    timestamp: float = field(default_factory=time.time)

@dataclass
class EmbeddingStats:
    """Embedding service statistics"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    active_workers: int = 0
    avg_processing_time: float = 0.0
    queue_size: int = 0
    
    def __post_init__(self):
        self._processing_times = []
    
    def update_request_stats(self, success: bool, processing_time: float):
        self.completed_requests += 1
        if success:
            self._processing_times.append(processing_time)
            if len(self._processing_times) > 100:  # Keep last 100 times
                self._processing_times = self._processing_times[-100:]
            self.avg_processing_time = sum(self._processing_times) / len(self._processing_times)
        else:
            self.failed_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'completed_requests': self.completed_requests,
            'failed_requests': self.failed_requests,
            'queue_size': self.queue_size,
            'active_workers': self.active_workers,
            'success_rate': (self.completed_requests / max(self.total_requests, 1)) * 100,
            'avg_processing_time': self.avg_processing_time
        }

class AsyncEmbeddingService:
    """Pure asyncio-based embedding service with background processing"""
    
    def __init__(self, max_queue_size: int = 50, num_workers: int = 8):
        self.api_url = config.embedding.api_url
        self.model = config.embedding.model
        self.dimension = config.embedding.dimension
        self.max_concurrent = min(config.embedding.max_concurrent_requests, 20)
        
        # Queue configuration
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # Asyncio queues and state
        self.embedding_queue: Optional[asyncio.Queue] = None
        self._workers = []
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = EmbeddingStats()
        
        # Batch processing
        self.batch_size = 10
        self.batch_timeout = 0.5  # seconds
        
        logger.info(f"Initialized async embedding service: {max_queue_size} queue, {num_workers} workers")
    
    async def start(self):
        """Start the embedding service with background workers"""
        if self._running:
            return
        
        self._running = True
        
        # Create asyncio queue
        self.embedding_queue = asyncio.Queue(maxsize=self.max_queue_size)
        
        # Create HTTP session with optimized settings
        connector = aiohttp.TCPConnector(
            limit_per_host=self.max_concurrent,
            limit=self.max_concurrent * 2,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            json_serialize=json.dumps
        )
        
        # Start background workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker_async(i))
            self._workers.append(worker)
        
        logger.info(f"Started {self.num_workers} async embedding workers")
    
    async def stop(self):
        """Stop the embedding service gracefully"""
        if not self._running:
            return
        
        logger.info("Stopping async embedding service...")
        self._running = False
        
        # Signal workers to stop by adding sentinel values
        if self.embedding_queue:
            for _ in range(self.num_workers):
                try:
                    await self.embedding_queue.put(None)
                except asyncio.QueueFull:
                    pass
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
        
        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("Async embedding service stopped")
    
    async def _worker_async(self, worker_id: int):
        """Async worker for processing embedding requests"""
        logger.debug(f"Async worker {worker_id} started")
        
        try:
            while self._running:
                try:
                    # Get batch of requests
                    batch = await self._get_batch_requests_async()
                    if not batch:
                        continue
                    
                    # Process batch
                    await self._process_batch_async(batch, worker_id)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Async worker {worker_id} error: {e}")
                    await asyncio.sleep(0.1)
        
        finally:
            logger.debug(f"Async worker {worker_id} stopped")
    
    async def _get_batch_requests_async(self) -> List[EmbeddingRequest]:
        """Get a batch of requests from the async queue"""
        batch = []
        end_time = asyncio.get_event_loop().time() + self.batch_timeout
        
        while len(batch) < self.batch_size and asyncio.get_event_loop().time() < end_time and self._running:
            try:
                # Wait for request with timeout
                remaining_time = end_time - asyncio.get_event_loop().time()
                if remaining_time <= 0:
                    break
                
                request = await asyncio.wait_for(
                    self.embedding_queue.get(), 
                    timeout=min(0.1, remaining_time)
                )
                
                # Sentinel value to stop worker
                if request is None:
                    break
                
                batch.append(request)
                
            except asyncio.TimeoutError:
                # If we have some requests and timeout, process them
                if batch:
                    break
                continue
        
        return batch
    
    async def _process_batch_async(self, batch: List[EmbeddingRequest], worker_id: int):
        """Process a batch of embedding requests asynchronously"""
        if not batch:
            return
        
        self.stats.active_workers += 1
        
        try:
            # Extract texts for batch processing
            texts = [req.text for req in batch]
            
            # Generate embeddings
            start_time = time.time()
            embeddings = await self._generate_embeddings_batch_async(texts)
            processing_time = time.time() - start_time
            
            # Update statistics
            for request, embedding in zip(batch, embeddings):
                success = embedding is not None
                self.stats.update_request_stats(success, processing_time / len(batch))
        
        except Exception as e:
            logger.error(f"Batch processing error in worker {worker_id}: {e}")
            
            # Mark all requests as failed
            for request in batch:
                self.stats.update_request_stats(False, 0)
        
        finally:
            self.stats.active_workers = max(0, self.stats.active_workers - 1)
    
    async def _generate_embeddings_batch_async(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts asynchronously"""
        if not texts or not self._session:
            return [None] * len(texts)
        
        results = []
        
        # Process in smaller sub-batches to avoid overwhelming the API
        sub_batch_size = 5
        for i in range(0, len(texts), sub_batch_size):
            sub_batch = texts[i:i + sub_batch_size]
            sub_results = await self._generate_sub_batch_async(sub_batch)
            results.extend(sub_results)
        
        return results
    
    async def _generate_sub_batch_async(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a sub-batch of texts"""
        tasks = []
        semaphore = asyncio.Semaphore(min(len(texts), 3))  # Limit concurrent requests
        
        async def generate_single(text: str):
            async with semaphore:
                return await self._generate_single_embedding_async(text)
        
        for text in texts:
            task = asyncio.create_task(generate_single(text))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        embeddings = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Embedding generation error: {result}")
                embeddings.append(None)
            else:
                embeddings.append(result)
        
        return embeddings
    
    async def _generate_single_embedding_async(self, text: str) -> Optional[List[float]]:
        """Generate a single embedding with retries"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with self._session.post(
                    f"{self.api_url}/embed",
                    json={"model": self.model, "input": text}
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract embedding from response
                        embedding = (
                            data.get("embedding") or 
                            (data.get("embeddings", [None])[0] if data.get("embeddings") else None)
                        )
                        
                        if not embedding:
                            raise ValueError(f"No embedding in response: {data}")
                        
                        # Validate embedding
                        if len(embedding) != self.dimension:
                            raise ValueError(f"Wrong dimension: {len(embedding)} != {self.dimension}")
                        
                        if VectorOperations.is_zero_vector(embedding):
                            raise ValueError("Zero vector generated")
                        
                        return embedding
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Failed to generate embedding after {max_retries} attempts: {e}")
                return None
        
        return None
    
    async def generate_embedding(self, text: str, priority: int = 1) -> Optional[List[float]]:
        """Generate a single embedding asynchronously"""
        if not self._running:
            await self.start()
        
        # Create request
        request = EmbeddingRequest(
            text=text,
            request_id=str(uuid.uuid4())[:8],
            priority=priority
        )
        
        # Create a future to wait for the result
        result_future = asyncio.Future()
        
        # Store the future with the request (hack for now)
        request._future = result_future
        
        try:
            # Add to queue
            await self.embedding_queue.put(request)
            self.stats.total_requests += 1
            self.stats.queue_size = self.embedding_queue.qsize()
            
            # For single requests, generate immediately to avoid complexity
            return await self._generate_single_embedding_async(text)
            
        except Exception as e:
            logger.error(f"Failed to queue embedding request: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        if not self._running:
            await self.start()
        
        # For batch processing, use direct generation to avoid queue complexity
        return await self._generate_embeddings_batch_async(texts)
    
    def validate_model(self) -> bool:
        """Validate that the embedding model is available"""
        try:
            import requests
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            response.raise_for_status()
            
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            if self.model not in models:
                logger.error(f"Model {self.model} not found! Available: {models}")
                return False
            
            logger.info(f"Model {self.model} validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current service statistics"""
        stats = self.stats.get_stats()
        if self.embedding_queue:
            stats['queue_size'] = self.embedding_queue.qsize()
        return stats
    
    @property
    def queue_size(self) -> int:
        """Current queue size"""
        return self.embedding_queue.qsize() if self.embedding_queue else 0
    
    @property
    def is_queue_full(self) -> bool:
        """Check if queue is full"""
        return self.embedding_queue.full() if self.embedding_queue else False

# Global async embedding service
async_embedding_service = AsyncEmbeddingService(max_queue_size=50, num_workers=8)
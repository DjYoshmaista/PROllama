# inference/queue_embeddings.py - Queue-Based Embedding Service with Background Processing
import asyncio
import aiohttp
import json
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Callable
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import uuid

from core.config import config
from core.vector_ops import VectorOperations

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRequest:
    """Represents an embedding request in the queue"""
    text: str
    request_id: str
    priority: int = 1
    callback: Optional[Callable] = None
    future: Optional[asyncio.Future] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:8]

@dataclass
class EmbeddingResult:
    """Represents the result of an embedding request"""
    request_id: str
    embedding: Optional[List[float]]
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0

class EmbeddingStats:
    """Thread-safe embedding statistics"""
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.queue_size = 0
        self.active_workers = 0
        self.avg_processing_time = 0.0
        self._processing_times = []
    
    def update_request_stats(self, success: bool, processing_time: float):
        with self._lock:
            self.completed_requests += 1
            if success:
                self._processing_times.append(processing_time)
                if len(self._processing_times) > 100:  # Keep last 100 times
                    self._processing_times = self._processing_times[-100:]
                self.avg_processing_time = sum(self._processing_times) / len(self._processing_times)
            else:
                self.failed_requests += 1
    
    def update_queue_size(self, size: int):
        with self._lock:
            self.queue_size = size
    
    def update_active_workers(self, count: int):
        with self._lock:
            self.active_workers = count
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_requests': self.total_requests,
                'completed_requests': self.completed_requests,
                'failed_requests': self.failed_requests,
                'queue_size': self.queue_size,
                'active_workers': self.active_workers,
                'success_rate': (self.completed_requests / max(self.total_requests, 1)) * 100,
                'avg_processing_time': self.avg_processing_time
            }

class QueueBasedEmbeddingService:
    """Advanced queue-based embedding service with background processing"""
    
    def __init__(self, max_queue_size: int = 50, num_workers: int = 8):
        self.api_url = config.embedding.api_url
        self.model = config.embedding.model
        self.dimension = config.embedding.dimension
        self.max_concurrent = min(config.embedding.max_concurrent_requests, 20)
        
        # Queue configuration
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        
        # Thread-safe queues
        self.embedding_queue: Queue = Queue(maxsize=max_queue_size)
        self.result_queue: Queue = Queue()
        
        # Worker management
        self.workers = []
        self.worker_executor = ThreadPoolExecutor(max_workers=num_workers)
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Statistics and monitoring
        self.stats = EmbeddingStats()
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Batch processing
        self.batch_size = 10
        self.batch_timeout = 0.5  # seconds
        
        logger.info(f"Initialized queue-based embedding service: {max_queue_size} queue, {num_workers} workers")
    
    async def start(self):
        """Start the embedding service with background workers"""
        if self._running:
            return
        
        self._running = True
        
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
            worker = threading.Thread(
                target=self._worker_thread,
                args=(i,),
                daemon=True,
                name=f"EmbeddingWorker-{i}"
            )
            worker.start()
            self.workers.append(worker)
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="EmbeddingMonitor"
        )
        self._monitor_thread.start()
        
        logger.info(f"Started {self.num_workers} embedding workers with {self.max_queue_size} queue size")
    
    async def stop(self):
        """Stop the embedding service gracefully"""
        if not self._running:
            return
        
        logger.info("Stopping embedding service...")
        self._running = False
        
        # Signal workers to stop by adding sentinel values
        for _ in range(self.num_workers):
            try:
                self.embedding_queue.put(None, timeout=1)
            except:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        # Shutdown executor
        self.worker_executor.shutdown(wait=True, timeout=10)
        
        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None
        
        # Wait for monitor thread
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)
        
        self.workers.clear()
        logger.info("Embedding service stopped")
    
    def _worker_thread(self, worker_id: int):
        """Background worker thread for processing embedding requests"""
        logger.debug(f"Worker {worker_id} started")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self._running:
                try:
                    # Get batch of requests
                    batch = self._get_batch_requests()
                    if not batch:
                        continue
                    
                    # Process batch
                    loop.run_until_complete(self._process_batch(batch, worker_id))
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    time.sleep(0.1)
        finally:
            loop.close()
            logger.debug(f"Worker {worker_id} stopped")
    
    def _get_batch_requests(self) -> List[EmbeddingRequest]:
        """Get a batch of requests from the queue"""
        batch = []
        end_time = time.time() + self.batch_timeout
        
        while len(batch) < self.batch_size and time.time() < end_time and self._running:
            try:
                request = self.embedding_queue.get(timeout=0.1)
                
                # Sentinel value to stop worker
                if request is None:
                    break
                
                batch.append(request)
                
            except Empty:
                # If we have some requests and timeout, process them
                if batch:
                    break
                continue
        
        return batch
    
    async def _process_batch(self, batch: List[EmbeddingRequest], worker_id: int):
        """Process a batch of embedding requests"""
        if not batch:
            return
        
        self.stats.update_active_workers(self.stats.active_workers + 1)
        
        try:
            # Extract texts for batch processing
            texts = [req.text for req in batch]
            
            # Generate embeddings
            start_time = time.time()
            embeddings = await self._generate_embeddings_batch(texts)
            processing_time = time.time() - start_time
            
            # Process results
            for request, embedding in zip(batch, embeddings):
                success = embedding is not None
                
                result = EmbeddingResult(
                    request_id=request.request_id,
                    embedding=embedding,
                    success=success,
                    error=None if success else "Embedding generation failed",
                    processing_time=processing_time / len(batch)
                )
                
                # Update statistics
                self.stats.update_request_stats(success, result.processing_time)
                
                # Handle result
                if request.future and not request.future.done():
                    if success:
                        request.future.set_result(embedding)
                    else:
                        request.future.set_result(None)
                
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Put result in result queue for monitoring
                try:
                    self.result_queue.put(result, timeout=0.1)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Batch processing error in worker {worker_id}: {e}")
            
            # Mark all requests as failed
            for request in batch:
                if request.future and not request.future.done():
                    request.future.set_result(None)
                
                result = EmbeddingResult(
                    request_id=request.request_id,
                    embedding=None,
                    success=False,
                    error=str(e)
                )
                
                self.stats.update_request_stats(False, 0)
                
                if request.callback:
                    try:
                        request.callback(result)
                    except:
                        pass
        
        finally:
            self.stats.update_active_workers(max(0, self.stats.active_workers - 1))
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts"""
        if not texts or not self._session:
            return [None] * len(texts)
        
        results = []
        
        # Process in smaller sub-batches to avoid overwhelming the API
        sub_batch_size = 5
        for i in range(0, len(texts), sub_batch_size):
            sub_batch = texts[i:i + sub_batch_size]
            sub_results = await self._generate_sub_batch(sub_batch)
            results.extend(sub_results)
        
        return results
    
    async def _generate_sub_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a sub-batch of texts"""
        tasks = []
        semaphore = asyncio.Semaphore(min(len(texts), 3))  # Limit concurrent requests
        
        async def generate_single(text: str):
            async with semaphore:
                return await self._generate_single_embedding(text)
        
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
    
    async def _generate_single_embedding(self, text: str) -> Optional[List[float]]:
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
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                # Update queue size
                self.stats.update_queue_size(self.embedding_queue.qsize())
                
                # Process completed results
                try:
                    while True:
                        result = self.result_queue.get_nowait()
                        # Could log results, update metrics, etc.
                except Empty:
                    pass
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(1)
    
    async def generate_embedding(self, text: str, priority: int = 1) -> Optional[List[float]]:
        """Generate a single embedding asynchronously"""
        if not self._running:
            await self.start()
        
        # Create future for result
        future = asyncio.Future()
        
        # Create request
        request = EmbeddingRequest(
            text=text,
            request_id=str(uuid.uuid4())[:8],
            priority=priority,
            future=future
        )
        
        try:
            # Add to queue
            self.embedding_queue.put(request, timeout=1)
            self.stats.total_requests += 1
            
            # Wait for result
            return await future
            
        except Exception as e:
            logger.error(f"Failed to queue embedding request: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        if not self._running:
            await self.start()
        
        # Create futures for all texts
        futures = []
        
        for text in texts:
            future = asyncio.Future()
            request = EmbeddingRequest(
                text=text,
                request_id=str(uuid.uuid4())[:8],
                priority=1,
                future=future
            )
            
            try:
                self.embedding_queue.put(request, timeout=0.1)
                self.stats.total_requests += 1
                futures.append(future)
            except:
                # Queue is full, return None for this text
                logger.warning("Embedding queue full, skipping text")
                futures.append(None)
        
        # Wait for all results
        results = []
        for future in futures:
            if future is None:
                results.append(None)
            else:
                try:
                    result = await asyncio.wait_for(future, timeout=30)
                    results.append(result)
                except asyncio.TimeoutError:
                    logger.warning("Embedding request timed out")
                    results.append(None)
                except Exception as e:
                    logger.error(f"Embedding request failed: {e}")
                    results.append(None)
        
        return results
    
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
        return self.stats.get_stats()
    
    @property
    def queue_size(self) -> int:
        """Current queue size"""
        return self.embedding_queue.qsize()
    
    @property
    def is_queue_full(self) -> bool:
        """Check if queue is full"""
        return self.embedding_queue.full()

# Global queue-based embedding service
queue_embedding_service = QueueBasedEmbeddingService(max_queue_size=50, num_workers=8)
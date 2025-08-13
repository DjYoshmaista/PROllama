# inference/embeddings.py - Unified Embedding Service (REPLACES all embedding services)
import asyncio
import threading
from typing import List, Optional, Dict, Any
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import logging

from inference.base_embedding import (
    BaseEmbeddingService, 
    ProcessingStrategy, 
    EmbeddingRequest, 
    EmbeddingResult
)

logger = logging.getLogger(__name__)

class UnifiedEmbeddingService(BaseEmbeddingService):
    """
    Unified embedding service that supports multiple processing strategies:
    - ASYNC: Pure async/await processing
    - QUEUE_THREAD: Thread-based queue processing
    - QUEUE_ASYNC: Async queue processing
    - DIRECT: Direct processing without queuing
    """
    
    def __init__(self, 
                 strategy: ProcessingStrategy = ProcessingStrategy.ASYNC,
                 **kwargs):
        super().__init__(strategy=strategy, **kwargs)
        
        # Strategy-specific attributes
        if strategy == ProcessingStrategy.QUEUE_THREAD:
            self.embedding_queue: Queue = Queue(maxsize=self.max_queue_size)
            self.result_queue: Queue = Queue()
            self.workers = []
            self.worker_executor = ThreadPoolExecutor(max_workers=self.num_workers)
            self._monitor_thread: Optional[threading.Thread] = None
        
        elif strategy in [ProcessingStrategy.ASYNC, ProcessingStrategy.QUEUE_ASYNC]:
            self.embedding_queue: Optional[asyncio.Queue] = None
            self._workers = []
    
    async def _start_strategy(self):
        """Start strategy-specific components"""
        if self.strategy == ProcessingStrategy.QUEUE_THREAD:
            # Start thread-based workers
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
        
        elif self.strategy in [ProcessingStrategy.ASYNC, ProcessingStrategy.QUEUE_ASYNC]:
            # Create async queue
            self.embedding_queue = asyncio.Queue(maxsize=self.max_queue_size)
            
            # Start async workers
            for i in range(self.num_workers):
                worker = asyncio.create_task(self._worker_async(i))
                self._workers.append(worker)
    
    async def _stop_strategy(self):
        """Stop strategy-specific components"""
        if self.strategy == ProcessingStrategy.QUEUE_THREAD:
            # Signal thread workers to stop
            for _ in range(self.num_workers):
                try:
                    self.embedding_queue.put(None, timeout=1)
                except:
                    pass
            
            # Wait for workers
            for worker in self.workers:
                worker.join(timeout=5)
            
            # Shutdown executor
            self.worker_executor.shutdown(wait=True, timeout=10)
            
            # Stop monitor thread
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=2)
            
            self.workers.clear()
        
        elif self.strategy in [ProcessingStrategy.ASYNC, ProcessingStrategy.QUEUE_ASYNC]:
            # Signal async workers to stop
            if self.embedding_queue:
                for _ in range(self.num_workers):
                    try:
                        await self.embedding_queue.put(None)
                    except asyncio.QueueFull:
                        pass
            
            # Cancel workers
            for worker in self._workers:
                worker.cancel()
            
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers = []
    
    async def _process_single_request(self, request: EmbeddingRequest) -> Optional[List[float]]:
        """Process a single embedding request based on strategy"""
        if self.strategy == ProcessingStrategy.DIRECT:
            # Direct processing without queuing
            return await self._generate_single_embedding_with_retry(request.text)
        
        elif self.strategy == ProcessingStrategy.QUEUE_THREAD:
            # Thread-based queue processing
            future = asyncio.Future()
            request.future = future
            
            try:
                self.embedding_queue.put(request, timeout=1)
                self.stats.total_requests += 1
                
                # For thread strategy, we need to bridge sync/async
                # Create a simple polling mechanism
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._wait_for_thread_result, request)
                
            except Exception as e:
                logger.error(f"Failed to queue request: {e}")
                return None
        
        else:  # ASYNC or QUEUE_ASYNC
            # Async queue processing
            if self.strategy == ProcessingStrategy.QUEUE_ASYNC:
                # Use queue for processing
                await self.embedding_queue.put(request)
                self.stats.total_requests += 1
                
                # Create a future to wait for result
                future = asyncio.Future()
                request.future = future
                
                # Process immediately in a worker
                return await self._generate_single_embedding_with_retry(request.text)
            else:
                # Direct async processing
                return await self._generate_single_embedding_with_retry(request.text)
    
    async def _process_batch_requests(self, requests: List[EmbeddingRequest]) -> List[Optional[List[float]]]:
        """Process batch embedding requests based on strategy"""
        if self.strategy == ProcessingStrategy.DIRECT:
            # Direct batch processing
            texts = [req.text for req in requests]
            return await self._generate_embeddings_batch_internal(texts)
        
        elif self.strategy == ProcessingStrategy.QUEUE_THREAD:
            # Queue all requests and wait for results
            futures = []
            
            for request in requests:
                future = asyncio.Future()
                request.future = future
                
                try:
                    self.embedding_queue.put(request, timeout=0.1)
                    self.stats.total_requests += 1
                    futures.append(future)
                except:
                    futures.append(None)
            
            # Wait for all results
            results = []
            for future in futures:
                if future is None:
                    results.append(None)
                else:
                    try:
                        # Bridge sync/async
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, self._wait_for_thread_future, future)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to get result: {e}")
                        results.append(None)
            
            return results
        
        else:  # ASYNC or QUEUE_ASYNC
            # Process batch directly with async
            texts = [req.text for req in requests]
            return await self._generate_embeddings_batch_internal(texts)
    
    def _wait_for_thread_result(self, request: EmbeddingRequest) -> Optional[List[float]]:
        """Wait for thread-based result (blocking)"""
        # This is a simplified version - in production, use proper synchronization
        import time
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request.future and request.future.done():
                try:
                    return request.future.result()
                except:
                    return None
            time.sleep(0.1)
        
        return None
    
    def _wait_for_thread_future(self, future: asyncio.Future) -> Optional[List[float]]:
        """Wait for a future from thread context"""
        import time
        timeout = 30
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if future.done():
                try:
                    return future.result()
                except:
                    return None
            time.sleep(0.1)
        
        return None
    
    def _worker_thread(self, worker_id: int):
        """Thread worker for QUEUE_THREAD strategy"""
        logger.debug(f"Thread worker {worker_id} started")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self._running:
                try:
                    # Get batch of requests
                    batch = self._get_batch_requests_sync()
                    if not batch:
                        continue
                    
                    # Process batch
                    loop.run_until_complete(self._process_batch_thread(batch, worker_id))
                    
                except Exception as e:
                    logger.error(f"Thread worker {worker_id} error: {e}")
                    import time
                    time.sleep(0.1)
        finally:
            loop.close()
            logger.debug(f"Thread worker {worker_id} stopped")
    
    def _get_batch_requests_sync(self) -> List[EmbeddingRequest]:
        """Get batch of requests from sync queue"""
        batch = []
        import time
        end_time = time.time() + self.batch_timeout
        
        while len(batch) < self.batch_size and time.time() < end_time and self._running:
            try:
                request = self.embedding_queue.get(timeout=0.1)
                
                if request is None:  # Sentinel
                    break
                
                batch.append(request)
                
            except Empty:
                if batch:
                    break
                continue
        
        return batch
    
    async def _process_batch_thread(self, batch: List[EmbeddingRequest], worker_id: int):
        """Process batch in thread worker"""
        if not batch:
            return
        
        self.stats.active_workers += 1
        
        try:
            texts = [req.text for req in batch]
            
            # Generate embeddings
            import time
            start_time = time.time()
            embeddings = await self._generate_embeddings_batch_internal(texts)
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
                await self.stats.update_request_stats(success, result.processing_time)
                
                # Set future result
                if request.future and not request.future.done():
                    if success:
                        request.future.set_result(embedding)
                    else:
                        request.future.set_result(None)
                
                # Call callback if provided
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Put result in result queue
                try:
                    self.result_queue.put(result, timeout=0.1)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Batch processing error in worker {worker_id}: {e}")
            
            # Mark all as failed
            for request in batch:
                if request.future and not request.future.done():
                    request.future.set_result(None)
                
                await self.stats.update_request_stats(False, 0)
                
                if request.callback:
                    try:
                        request.callback(EmbeddingResult(
                            request_id=request.request_id,
                            embedding=None,
                            success=False,
                            error=str(e)
                        ))
                    except:
                        pass
        
        finally:
            self.stats.active_workers = max(0, self.stats.active_workers - 1)
    
    async def _worker_async(self, worker_id: int):
        """Async worker for ASYNC/QUEUE_ASYNC strategies"""
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
        """Get batch of requests from async queue"""
        batch = []
        end_time = asyncio.get_event_loop().time() + self.batch_timeout
        
        while len(batch) < self.batch_size and asyncio.get_event_loop().time() < end_time and self._running:
            try:
                remaining_time = end_time - asyncio.get_event_loop().time()
                if remaining_time <= 0:
                    break
                
                request = await asyncio.wait_for(
                    self.embedding_queue.get(), 
                    timeout=min(0.1, remaining_time)
                )
                
                if request is None:  # Sentinel
                    break
                
                batch.append(request)
                
            except asyncio.TimeoutError:
                if batch:
                    break
                continue
        
        return batch
    
    async def _process_batch_async(self, batch: List[EmbeddingRequest], worker_id: int):
        """Process batch in async worker"""
        if not batch:
            return
        
        self.stats.active_workers += 1
        
        try:
            texts = [req.text for req in batch]
            
            # Generate embeddings
            import time
            start_time = time.time()
            embeddings = await self._generate_embeddings_batch_internal(texts)
            processing_time = time.time() - start_time
            
            # Process results
            for request, embedding in zip(batch, embeddings):
                success = embedding is not None
                await self.stats.update_request_stats(success, processing_time / len(batch))
        
        except Exception as e:
            logger.error(f"Batch processing error in worker {worker_id}: {e}")
            
            for request in batch:
                await self.stats.update_request_stats(False, 0)
        
        finally:
            self.stats.active_workers = max(0, self.stats.active_workers - 1)
    
    def _monitor_loop(self):
        """Background monitoring for thread strategy"""
        import time
        while self._running:
            try:
                self.stats.queue_size = self.embedding_queue.qsize()
                
                # Process completed results
                try:
                    while True:
                        result = self.result_queue.get_nowait()
                        # Could log or process results here
                except Empty:
                    pass
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(1)
    
    @property
    def queue_size(self) -> int:
        """Get current queue size"""
        if self.strategy == ProcessingStrategy.QUEUE_THREAD:
            return self.embedding_queue.qsize() if self.embedding_queue else 0
        elif self.strategy in [ProcessingStrategy.ASYNC, ProcessingStrategy.QUEUE_ASYNC]:
            return self.embedding_queue.qsize() if self.embedding_queue else 0
        return 0
    
    @property
    def is_queue_full(self) -> bool:
        """Check if queue is full"""
        if self.strategy == ProcessingStrategy.QUEUE_THREAD:
            return self.embedding_queue.full() if self.embedding_queue else False
        elif self.strategy in [ProcessingStrategy.ASYNC, ProcessingStrategy.QUEUE_ASYNC]:
            return self.embedding_queue.full() if self.embedding_queue else False
        return False

# Global embedding service instances with different strategies
# Replace all previous global instances
embedding_service = UnifiedEmbeddingService(strategy=ProcessingStrategy.ASYNC)
queue_embedding_service = UnifiedEmbeddingService(strategy=ProcessingStrategy.QUEUE_THREAD, max_queue_size=50, num_workers=8)
async_embedding_service = UnifiedEmbeddingService(strategy=ProcessingStrategy.QUEUE_ASYNC, max_queue_size=50, num_workers=8)
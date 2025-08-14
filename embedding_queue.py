# embedding_queue.py
import asyncio
import logging
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
from collections import deque
import aiohttp
from config import Config
from utils import get_embeddings_batch

logger = logging.getLogger(__name__)

@dataclass
class QueueItem:
    """Single item in the embedding queue"""
    id: str
    content: str
    tags: List[str]
    file_path: str
    chunk_index: int
    timestamp: float
    size_bytes: int
    
    def to_dict(self):
        return asdict(self)

class EmbeddingQueue:
    """
    Embedding queue that maintains up to 4GB of data in memory
    Processes items sequentially while allowing parallelization within batches
    """
    
    MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024  # 4GB
    BATCH_SIZE = 50  # Items per batch
    PERSIST_INTERVAL = 30  # Seconds between queue state saves
    
    def __init__(self):
        self.queue = deque()
        self.current_memory_usage = 0
        self.total_processed = 0
        self.workers = []
        self.started = False
        self.processing_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._persist_task = None
        self._stats = {
            'queued_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'current_memory_mb': 0,
            'peak_memory_mb': 0,
            'last_update': time.time()
        }
        
        # Persistence
        self.state_file = "embedding_queue_state.json"
        self.load_persistent_state()
        
        # Setup cleanup
        import atexit
        atexit.register(self.cleanup)
    
    def calculate_item_size(self, item: QueueItem) -> int:
        """Calculate approximate memory footprint of queue item"""
        # Base object overhead + string content + metadata
        base_size = sys.getsizeof(item)
        content_size = len(item.content.encode('utf-8'))
        tags_size = sum(len(tag.encode('utf-8')) for tag in item.tags)
        path_size = len(item.file_path.encode('utf-8'))
        
        return base_size + content_size + tags_size + path_size
    
    async def enqueue(self, content: str, tags: List[str], file_path: str, chunk_index: int) -> bool:
        """
        Add item to queue if memory allows
        Returns True if queued, False if rejected due to memory limits
        """
        item = QueueItem(
            id=f"{file_path}:{chunk_index}:{int(time.time() * 1000)}",
            content=content,
            tags=tags,
            file_path=file_path,
            chunk_index=chunk_index,
            timestamp=time.time(),
            size_bytes=0  # Will be calculated below
        )
        
        item.size_bytes = self.calculate_item_size(item)
        
        # Check memory limits
        if self.current_memory_usage + item.size_bytes > self.MAX_MEMORY_BYTES:
            logger.warning(f"Queue memory limit reached. Current: {self.current_memory_usage / (1024**2):.1f}MB")
            return False
        
        # Add to queue
        self.queue.append(item)
        self.current_memory_usage += item.size_bytes
        self._stats['queued_items'] += 1
        self._stats['current_memory_mb'] = self.current_memory_usage / (1024**2)
        self._stats['peak_memory_mb'] = max(self._stats['peak_memory_mb'], self._stats['current_memory_mb'])
        
        logger.debug(f"Queued item from {file_path}:{chunk_index}, queue size: {len(self.queue)}")
        return True
    
    async def dequeue_batch(self) -> List[QueueItem]:
        """Remove and return a batch of items from the queue"""
        batch = []
        batch_memory = 0
        
        while len(batch) < self.BATCH_SIZE and self.queue:
            item = self.queue.popleft()
            batch.append(item)
            batch_memory += item.size_bytes
            self.current_memory_usage -= item.size_bytes
        
        if batch:
            self._stats['current_memory_mb'] = self.current_memory_usage / (1024**2)
            logger.debug(f"Dequeued batch of {len(batch)} items, freed {batch_memory / (1024**2):.1f}MB")
        
        return batch
    
    async def process_batch(self, batch: List[QueueItem], session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """Process a batch of items to generate embeddings"""
        if not batch:
            return []
        
        try:
            # Extract content for batch embedding
            contents = [item.content for item in batch]
            
            # Generate embeddings for the batch
            embeddings = await get_embeddings_batch(session, contents)
            
            # Combine with metadata
            processed_records = []
            for i, (item, embedding) in enumerate(zip(batch, embeddings)):
                if embedding is not None and len(embedding) == Config.EMBEDDING_DIM:
                    processed_records.append({
                        'content': item.content,
                        'tags': item.tags,
                        'embedding': embedding,
                        'file_path': item.file_path,
                        'chunk_index': item.chunk_index
                    })
                    self._stats['processed_items'] += 1
                else:
                    logger.warning(f"Failed to generate embedding for item {item.id}")
                    self._stats['failed_items'] += 1
            
            return processed_records
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self._stats['failed_items'] += len(batch)
            return []
    
    async def worker(self, worker_id: int, session: aiohttp.ClientSession, insert_callback: Callable):
        """Worker process for handling queue items"""
        logger.info(f"Embedding worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for items or shutdown
                if not self.queue:
                    await asyncio.sleep(0.1)
                    continue
                
                async with self.processing_lock:
                    batch = await self.dequeue_batch()
                
                if not batch:
                    continue
                
                # Process the batch
                processed_records = await self.process_batch(batch, session)
                
                if processed_records:
                    # Insert into database via callback
                    await insert_callback(processed_records)
                    logger.debug(f"Worker {worker_id} processed batch of {len(processed_records)} records")
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"Embedding worker {worker_id} stopped")
    
    async def start_workers(self, concurrency: int = 4, insert_callback: Optional[Callable] = None):
        """Start worker processes"""
        if self.started:
            logger.warning("Workers already started")
            return
        
        if not insert_callback:
            raise ValueError("insert_callback is required")
        
        self.started = True
        self._shutdown_event.clear()
        
        # Create HTTP session for workers
        connector = aiohttp.TCPConnector(limit=100)
        session = aiohttp.ClientSession(connector=connector)
        
        # Start workers
        for i in range(concurrency):
            worker_task = asyncio.create_task(
                self.worker(i, session, insert_callback)
            )
            self.workers.append(worker_task)
        
        # Start persistence task
        self._persist_task = asyncio.create_task(self._persist_state_periodically())
        
        logger.info(f"Started {concurrency} embedding workers")
    
    async def stop_workers(self):
        """Stop all workers gracefully"""
        if not self.started:
            return
        
        logger.info("Stopping embedding workers...")
        self._shutdown_event.set()
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Cancel persistence task
        if self._persist_task:
            self._persist_task.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        self.started = False
        
        # Save final state
        self.save_persistent_state()
        
        logger.info("All embedding workers stopped")
    
    async def _persist_state_periodically(self):
        """Periodically save queue state to disk"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.PERSIST_INTERVAL)
                if not self._shutdown_event.is_set():
                    self.save_persistent_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"State persistence error: {e}")
    
    def save_persistent_state(self):
        """Save current queue state to disk"""
        try:
            state = {
                'queue_items': [item.to_dict() for item in list(self.queue)],
                'stats': self._stats.copy(),
                'timestamp': time.time()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"Saved queue state with {len(self.queue)} items")
            
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")
    
    def load_persistent_state(self):
        """Load queue state from disk"""
        try:
            import os
            if not os.path.exists(self.state_file):
                logger.info("No previous queue state found")
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore queue items
            for item_data in state.get('queue_items', []):
                item = QueueItem(**item_data)
                self.queue.append(item)
                self.current_memory_usage += item.size_bytes
            
            # Restore stats
            if 'stats' in state:
                self._stats.update(state['stats'])
            
            logger.info(f"Restored queue state with {len(self.queue)} items, "
                       f"memory usage: {self.current_memory_usage / (1024**2):.1f}MB")
            
        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")
            # Clear corrupted state
            self.queue.clear()
            self.current_memory_usage = 0
    
    def cleanup(self):
        """Cleanup resources"""
        if self.started:
            # Can't run async cleanup from atexit, just save state
            try:
                self.save_persistent_state()
            except:
                pass
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        self._stats['last_update'] = time.time()
        self._stats['queue_size'] = len(self.queue)
        self._stats['current_memory_mb'] = self.current_memory_usage / (1024**2)
        return self._stats.copy()
    
    def is_full(self) -> bool:
        """Check if queue is approaching memory limits"""
        return self.current_memory_usage > (self.MAX_MEMORY_BYTES * 0.9)
    
    def wait_for_space(self) -> bool:
        """Check if we should wait for queue space"""
        return self.current_memory_usage > (self.MAX_MEMORY_BYTES * 0.8)

# Global embedding queue instance
embedding_queue = EmbeddingQueue()
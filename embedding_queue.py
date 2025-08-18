# embedding_queue.py - FINAL FIXED VERSION
import asyncio
import logging
import json
import sys
import psutil
import time
import threading
import aiohttp
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
from collections import deque
from config import Config
from constants import OLLAMA_API, EMBEDDING_MODEL
import os
os.environ['OLLAMA_NUM_PARALLEL'] = '4'  # Set parallelism for Ollama
from ollama import embed

# Set up logger for this module
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

def is_zero_vector(vector):
    """Check if vector contains only zeros"""
    return all(x == 0 for x in vector)

class EmbeddingQueue:
    """
    Embedding queue that maintains up to 4GB of data in memory
    Processes items sequentially while allowing parallelization within batches
    """
    
    MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024  # 4GB
    BATCH_SIZE = 50  # Items per batch
    PERSIST_INTERVAL = 30  # Seconds between queue state saves
    
    def __init__(self):
        logger.debug("Initializing EmbeddingQueue")
        self.queue = deque()
        self.current_memory_usage = 0
        self.total_processed = 0
        self.workers = []
        self.started = False
        self.processing_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._persist_task = None
        self.session = None  # HTTP session for API calls
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
        logger.debug("EmbeddingQueue initialized")
    
    def calculate_item_size(self, item: QueueItem) -> int:
        """Calculate approximate memory footprint of queue item"""
        # Base object overhead + string content + metadata
        base_size = sys.getsizeof(item)
        content_size = len(item.content.encode('utf-8'))
        tags_size = sum(len(tag.encode('utf-8')) for tag in item.tags)
        path_size = len(item.file_path.encode('utf-8'))
        
        total_size = base_size + content_size + tags_size + path_size
        logger.debug(f"Item size calculated: {total_size} bytes for content length {len(item.content)}")
        return total_size
    
    async def enqueue(self, content: str, tags: List[str] = None, file_path: str = "", chunk_index: int = 0) -> bool:
        """
        Add item to queue if memory allows (for backward compatibility)
        Returns True if queued, False if rejected due to memory limits
        """
        logger.debug(f"enqueue() called with content length: {len(content)}, tags: {tags}, file_path: {file_path}")
        if tags is None:
            tags = []
            
        return await self.enqueue_for_embedding(content, tags, file_path, chunk_index)
    
    async def enqueue_for_embedding(self, content: str, tags: List[str], file_path: str, chunk_index: int) -> bool:
        """
        Add item to queue for embedding processing
        Returns True if queued, False if rejected due to memory limits
        """
        logger.debug(f"enqueue_for_embedding() called - content length: {len(content)}, file: {file_path}, chunk: {chunk_index}")
        
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
        
        logger.debug(f"Item queued successfully. Queue size: {len(self.queue)}, Memory: {self.current_memory_usage / (1024**2):.1f}MB")
        return True
    
    async def dequeue_batch(self) -> List[QueueItem]:
        """Remove and return a batch of items from the queue"""
        logger.debug(f"dequeue_batch() called - queue size: {len(self.queue)}")
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
        
    async def fetch_embedding(self, text: str) -> Optional[List[float]]:
        """Fetch embedding for a single text - FINAL FIXED VERSION"""
        logger.debug(f"fetch_embedding() called for text length: {len(text)}")
        
        max_retries = 3
        for attempt in range(max_retries):
            logger.debug(f"Embedding attempt {attempt + 1}/{max_retries}")
            try:
                # FIXED: Use synchronous Ollama embed call directly
                logger.debug(f"Making Ollama embed call")
                logger.debug(f"Model: {EMBEDDING_MODEL}")
                logger.debug(f"Text preview: {text[:100]}...")
                
                # Call Ollama embed function directly (synchronous)
                response = embed(model=EMBEDDING_MODEL, input=text)
                logger.debug(f"Ollama embed response received")
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response str representation: {str(response)}")
                
                # FINAL FIX: The Ollama response object has embeddings as an attribute
                # From the logs we can see the response contains embeddings=[[-0.035149314, ...]]
                # We need to access this directly as response.embeddings[0]
                
                embedding = None
                
                try:
                    # Method 1: Direct attribute access (most likely to work)
                    embeddings_list = getattr(response, 'embeddings', None)
                    logger.debug(f"embeddings attribute type: {type(embeddings_list)}")
                    logger.debug(f"embeddings attribute value: {embeddings_list is not None}")
                    
                    if embeddings_list is not None and len(embeddings_list) > 0:
                        embedding = embeddings_list[0]
                        logger.debug(f"✓ Successfully extracted embedding via getattr - dimension: {len(embedding)}")
                    else:
                        logger.debug("embeddings attribute is None or empty")
                        
                except Exception as attr_error:
                    logger.debug(f"getattr method failed: {attr_error}")
                
                # Method 2: Try direct dot notation access
                if embedding is None:
                    try:
                        if hasattr(response, 'embeddings'):
                            embeddings_list = response.embeddings
                            logger.debug(f"Direct access embeddings type: {type(embeddings_list)}")
                            if embeddings_list and len(embeddings_list) > 0:
                                embedding = embeddings_list[0]
                                logger.debug(f"✓ Successfully extracted embedding via direct access - dimension: {len(embedding)}")
                    except Exception as direct_error:
                        logger.debug(f"Direct access method failed: {direct_error}")
                
                # Method 3: Try alternative attribute names
                if embedding is None:
                    try:
                        # Try singular 'embedding' attribute
                        if hasattr(response, 'embedding'):
                            embedding = response.embedding
                            logger.debug(f"✓ Successfully extracted via embedding attribute - dimension: {len(embedding)}")
                    except Exception as singular_error:
                        logger.debug(f"Singular embedding access failed: {singular_error}")
                
                # Method 4: Inspect all attributes and find embeddings
                if embedding is None:
                    try:
                        all_attrs = dir(response)
                        embedding_attrs = [attr for attr in all_attrs if 'embed' in attr.lower()]
                        logger.debug(f"All embedding-related attributes: {embedding_attrs}")
                        
                        for attr in embedding_attrs:
                            try:
                                attr_value = getattr(response, attr)
                                logger.debug(f"Attribute {attr}: type={type(attr_value)}, is_list={isinstance(attr_value, list)}")
                                if isinstance(attr_value, list) and len(attr_value) > 0:
                                    potential_embedding = attr_value[0] if isinstance(attr_value[0], list) else attr_value
                                    if isinstance(potential_embedding, list) and len(potential_embedding) == Config.EMBEDDING_DIM:
                                        embedding = potential_embedding
                                        logger.debug(f"✓ Found embedding via attribute {attr} - dimension: {len(embedding)}")
                                        break
                            except Exception as inspect_error:
                                logger.debug(f"Failed to inspect attribute {attr}: {inspect_error}")
                                
                    except Exception as inspect_all_error:
                        logger.debug(f"Attribute inspection failed: {inspect_all_error}")
                
                # If we still don't have an embedding, this is a critical error
                if embedding is None:
                    logger.error(f"CRITICAL: Could not extract embedding from response")
                    logger.error(f"Response object details:")
                    logger.error(f"  Type: {type(response)}")
                    logger.error(f"  String representation: {str(response)}")
                    logger.error(f"  Dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                    
                    # Try to get more information about the embeddings attribute specifically
                    try:
                        embeddings_attr = getattr(response, 'embeddings', 'NOT_FOUND')
                        logger.error(f"  embeddings attribute: {embeddings_attr}")
                        logger.error(f"  embeddings type: {type(embeddings_attr)}")
                        if hasattr(embeddings_attr, '__len__'):
                            logger.error(f"  embeddings length: {len(embeddings_attr)}")
                    except Exception as debug_error:
                        logger.error(f"  Error inspecting embeddings: {debug_error}")
                    
                    raise ValueError("Could not extract embedding from Ollama response")
                
                # Validate the extracted embedding
                if not isinstance(embedding, (list, tuple)):
                    logger.error(f"Embedding is not a list/tuple, got: {type(embedding)}")
                    raise ValueError(f"Invalid embedding type: {type(embedding)}")
                    
                if len(embedding) != Config.EMBEDDING_DIM:
                    logger.error(f"Invalid dimension: {len(embedding)} (expected {Config.EMBEDDING_DIM})")
                    raise ValueError(f"Invalid dimension: {len(embedding)}")
                
                if is_zero_vector(embedding):
                    logger.error("Zero vector generated")
                    raise ValueError("Zero vector generated")
                
                logger.debug("Embedding successfully generated and validated")
                return embedding
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise
                
        return None
    
    async def process_batch(self, batch: List[QueueItem]) -> List[Dict[str, Any]]:
        """Process a batch of items to generate embeddings"""
        logger.debug(f"process_batch() called with {len(batch)} items")
        if not batch:
            logger.debug("Empty batch, returning")
            return []
        
        try:
            # Generate embeddings for the batch
            logger.debug("Starting embedding generation for batch")
            processed_records = []
            
            for i, item in enumerate(batch):
                logger.debug(f"Processing item {i+1}/{len(batch)}: {item.id}")
                try:
                    embedding = await self.fetch_embedding(item.content)
                    
                    if embedding is not None and len(embedding) == Config.EMBEDDING_DIM:
                        record = {
                            'content': item.content,
                            'tags': item.tags,
                            'embedding': embedding,
                            'file_path': item.file_path,
                            'chunk_index': item.chunk_index
                        }
                        processed_records.append(record)
                        self._stats['processed_items'] += 1
                        logger.debug(f"Successfully processed item {i+1}: {item.id}")
                    else:
                        logger.warning(f"Failed to generate embedding for item {item.id}")
                        self._stats['failed_items'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing item {item.id}: {e}")
                    self._stats['failed_items'] += 1
            
            logger.info(f"Batch processing complete: {len(processed_records)}/{len(batch)} successful")
            return processed_records
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            self._stats['failed_items'] += len(batch)
            return []
    
    async def worker(self, worker_id: int, insert_callback: Callable):
        """Worker process for handling queue items"""
        logger.info(f"Embedding worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for items or shutdown
                if not self.queue:
                    await asyncio.sleep(0.1)
                    continue
                
                logger.debug(f"Worker {worker_id}: Queue has {len(self.queue)} items, acquiring lock")
                async with self.processing_lock:
                    batch = await self.dequeue_batch()
                
                if not batch:
                    continue
                
                logger.info(f"Worker {worker_id}: Processing batch of {len(batch)} items")
                
                # Process the batch
                processed_records = await self.process_batch(batch)
                
                if processed_records:
                    logger.debug(f"Worker {worker_id}: Calling insert callback with {len(processed_records)} records")
                    try:
                        await insert_callback(processed_records)
                        logger.info(f"Worker {worker_id}: Successfully inserted {len(processed_records)} records")
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: Insert callback failed: {e}")
                else:
                    logger.warning(f"Worker {worker_id}: No records to insert")
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"Embedding worker {worker_id} stopped")
    
    async def start_workers(self, concurrency: int = 4, insert_callback: Optional[Callable] = None):
        """Start worker processes"""
        logger.info(f"start_workers() called with concurrency={concurrency}, callback={'PROVIDED' if insert_callback else 'NONE'}")
        
        if self.started:
            logger.warning("Workers already started")
            return
        
        if not insert_callback:
            logger.error("insert_callback is required but not provided!")
            raise ValueError("insert_callback is required")
        
        # Reset state
        self.started = True
        self._shutdown_event.clear()
        
        # FIXED: Create session with proper configuration
        logger.debug("Creating HTTP session")
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=10,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        logger.debug("HTTP session created")
        
        # Test the session with a simple request
        try:
            logger.debug("Testing API connectivity...")
            async with self.session.get(f"{OLLAMA_API}/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                logger.debug(f"API test response: {response.status}")
                if response.status == 200:
                    logger.info("API connectivity test successful")
                else:
                    logger.warning(f"API test returned status {response.status}")
        except Exception as e:
            logger.error(f"API connectivity test failed: {e}")
            # Don't fail startup on connectivity test failure
        
        # Start workers
        logger.debug(f"Starting {concurrency} workers")
        for i in range(concurrency):
            worker_task = asyncio.create_task(
                self.worker(i, insert_callback)
            )
            self.workers.append(worker_task)
            logger.debug(f"Worker {i} task created")
        
        # Start persistence task
        self._persist_task = asyncio.create_task(self._persist_state_periodically())
        logger.debug("Persistence task started")
        
        logger.info(f"Successfully started {concurrency} embedding workers")
    
    async def stop_workers(self):
        """Stop all workers gracefully - FIXED to wait for queue completion"""
        if not self.started:
            logger.debug("Workers not started, nothing to stop")
            return
        
        logger.info("Stopping embedding workers...")
        
        # FIXED: Wait for queue to be processed before stopping
        if len(self.queue) > 0:
            logger.info(f"Waiting for {len(self.queue)} remaining items to be processed...")
            wait_count = 0
            while len(self.queue) > 0 and wait_count < 120:  # 2 minute timeout
                await asyncio.sleep(1)
                wait_count += 1
                if wait_count % 10 == 0:
                    logger.info(f"Still waiting... {len(self.queue)} items remaining")
            
            if len(self.queue) > 0:
                logger.warning(f"Timeout reached, {len(self.queue)} items still in queue")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel workers
        for i, worker in enumerate(self.workers):
            logger.debug(f"Cancelling worker {i}")
            worker.cancel()
        
        # Cancel persistence task
        if self._persist_task:
            logger.debug("Cancelling persistence task")
            self._persist_task.cancel()
        
        # Wait for workers to finish
        if self.workers:
            logger.debug("Waiting for workers to finish")
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Close session
        if self.session:
            logger.debug("Closing HTTP session")
            await self.session.close()
        
        self.workers.clear()
        self.started = False
        
        # Save final state
        self.save_persistent_state()
        
        logger.info("All embedding workers stopped")
    
    async def _persist_state_periodically(self):
        """Periodically save queue state to disk"""
        logger.debug("Persistence task started")
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.PERSIST_INTERVAL)
                if not self._shutdown_event.is_set():
                    logger.debug("Saving periodic state")
                    self.save_persistent_state()
            except asyncio.CancelledError:
                logger.debug("Persistence task cancelled")
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
        logger.debug("Cleanup called")
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
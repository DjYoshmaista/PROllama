# embedding_manager.py - Unified Embedding Generation
import asyncio
import aiohttp
import logging
import json
import sys
import psutil
import time
import threading
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Union
from collections import deque
from core_config import config
import os

# Set Ollama parallelism
os.environ['OLLAMA_NUM_PARALLEL'] = '4'

try:
    from ollama import embed
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    logging.error("Ollama library not available")

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingItem:
    """Single item in the embedding queue with comprehensive metadata"""
    id: str
    content: str
    tags: List[str]
    file_path: str
    chunk_index: int
    timestamp: float
    size_bytes: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self):
        return asdict(self)

def is_zero_vector(vector):
    """Check if vector contains only zeros"""
    return all(x == 0 for x in vector)

def validate_embedding(embedding: Any, expected_dim: int = None) -> bool:
    """Validate embedding format and dimensions"""
    expected_dim = expected_dim or config.EMB_DIM
    
    if not isinstance(embedding, (list, tuple)):
        return False
    
    if len(embedding) != expected_dim:
        return False
    
    if is_zero_vector(embedding):
        return False
    
    # Check for NaN or infinite values
    try:
        for val in embedding:
            if not isinstance(val, (int, float)) or not (-1e6 < val < 1e6):
                return False
    except (TypeError, ValueError):
        return False
    
    return True

class EmbeddingService:
    """Core embedding generation service with connection management"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.session = None
            self.api_url = config.OLLAMA_API
            self.model = config.EMBEDDING_MODEL
            self.dimension = config.EMB_DIM
            self._stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_processing_time': 0.0,
                'average_processing_time': 0.0
            }
            self._initialized = True
    
    async def _ensure_session(self):
        """Ensure HTTP session is available"""
        if self.session is None:
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
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama API"""
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.api_url}/tags", timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    if self.model in models:
                        logger.info(f"Embedding model {self.model} validated successfully")
                        return True
                    else:
                        logger.error(f"Model {self.model} not found. Available: {models}")
                        return False
                else:
                    logger.error(f"API test failed with status {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def generate_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Generate embedding for text with comprehensive error handling"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        if not HAS_OLLAMA:
            logger.error("Ollama library not available")
            return None
        
        start_time = time.time()
        self._stats['total_requests'] += 1
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Generating embedding (attempt {attempt + 1}/{max_retries})")
                logger.debug(f"Text preview: {text[:100]}...")
                
                # Use synchronous Ollama embed function
                response = embed(model=self.model, input=text)
                logger.debug(f"Ollama response received: {type(response)}")
                
                # Extract embedding using multiple methods
                embedding = None
                
                # Method 1: Direct embeddings attribute access
                try:
                    if hasattr(response, 'embeddings') and response.embeddings:
                        embeddings_list = response.embeddings
                        if isinstance(embeddings_list, list) and len(embeddings_list) > 0:
                            embedding = embeddings_list[0]
                            logger.debug(f"✓ Extracted via embeddings attribute - dim: {len(embedding)}")
                except Exception as e:
                    logger.debug(f"Embeddings attribute access failed: {e}")
                
                # Method 2: Try singular 'embedding' attribute
                if embedding is None:
                    try:
                        if hasattr(response, 'embedding') and response.embedding:
                            embedding = response.embedding
                            logger.debug(f"✓ Extracted via embedding attribute - dim: {len(embedding)}")
                    except Exception as e:
                        logger.debug(f"Embedding attribute access failed: {e}")
                
                # Method 3: Dictionary-style access
                if embedding is None and isinstance(response, dict):
                    try:
                        if 'embedding' in response:
                            embedding = response['embedding']
                            logger.debug(f"✓ Extracted via dict access - dim: {len(embedding)}")
                        elif 'embeddings' in response and response['embeddings']:
                            embedding = response['embeddings'][0]
                            logger.debug(f"✓ Extracted via dict embeddings - dim: {len(embedding)}")
                    except Exception as e:
                        logger.debug(f"Dictionary access failed: {e}")
                
                # Method 4: Inspect all attributes
                if embedding is None:
                    try:
                        attrs = [attr for attr in dir(response) if 'embed' in attr.lower()]
                        logger.debug(f"Available embedding-related attributes: {attrs}")
                        
                        for attr in attrs:
                            try:
                                attr_value = getattr(response, attr)
                                if isinstance(attr_value, list):
                                    if len(attr_value) > 0 and isinstance(attr_value[0], list):
                                        potential = attr_value[0]
                                    else:
                                        potential = attr_value
                                    
                                    if isinstance(potential, list) and len(potential) == self.dimension:
                                        embedding = potential
                                        logger.debug(f"✓ Found via attribute {attr} - dim: {len(embedding)}")
                                        break
                            except Exception:
                                continue
                    except Exception as e:
                        logger.debug(f"Attribute inspection failed: {e}")
                
                # Validate extracted embedding
                if embedding is None:
                    logger.error("CRITICAL: Could not extract embedding from response")
                    logger.error(f"Response type: {type(response)}")
                    logger.error(f"Response str: {str(response)}")
                    raise ValueError("Failed to extract embedding from Ollama response")
                
                if not validate_embedding(embedding, self.dimension):
                    logger.error(f"Invalid embedding: dim={len(embedding)}, expected={self.dimension}")
                    raise ValueError(f"Invalid embedding dimensions or content")
                
                # Success
                processing_time = time.time() - start_time
                self._stats['successful_requests'] += 1
                self._stats['total_processing_time'] += processing_time
                self._stats['average_processing_time'] = (
                    self._stats['total_processing_time'] / self._stats['successful_requests']
                )
                
                logger.debug(f"Embedding generated successfully in {processing_time:.3f}s")
                return embedding
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 1 + random.random() + attempt
                    await asyncio.sleep(wait_time)
                    continue
                
                # Final failure
                self._stats['failed_requests'] += 1
                logger.error(f"Embedding generation failed after {max_retries} attempts")
                return None
        
        return None
    
    async def generate_embeddings_batch(self, texts: List[str], 
                                       batch_size: int = None,
                                       max_concurrent: int = None) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts with controlled concurrency"""
        if not texts:
            return []
        
        batch_size = batch_size or config.CHUNK_SIZES['embedding_batch']
        max_concurrent = max_concurrent or min(5, config.PERFORMANCE_CONFIG['worker_processes'])
        
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(text: str) -> Optional[List[float]]:
            async with semaphore:
                return await self.generate_embedding(text)
        
        # Process in batches to manage memory
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} texts")
            
            # Generate embeddings for batch
            tasks = [generate_with_semaphore(text) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding error: {result}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
            
            # Brief pause between batches to prevent overwhelming the service
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return all_results
    
    async def close(self):
        """Close HTTP session and cleanup"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return self._stats.copy()

class EmbeddingQueue:
    """Advanced embedding queue with memory management and worker coordination"""
    
    def __init__(self):
        self.queue = deque()
        self.max_memory_bytes = config.PERFORMANCE_CONFIG['embedding_queue_memory_gb'] * 1024**3
        self.current_memory_usage = 0
        self.workers = []
        self.started = False
        self.processing_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._persist_task = None
        self.embedding_service = EmbeddingService()
        
        # Statistics
        self._stats = {
            'queued_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'current_memory_mb': 0,
            'peak_memory_mb': 0,
            'queue_size': 0,
            'worker_count': 0,
            'last_update': time.time()
        }
        
        # Persistence
        self.state_file = "embedding_queue_state.json"
        self.persist_interval = 30
        self.load_persistent_state()
        
        # Cleanup registration
        import atexit
        atexit.register(self.cleanup)
    
    def calculate_item_size(self, item: EmbeddingItem) -> int:
        """Calculate memory footprint of queue item"""
        base_size = sys.getsizeof(item)
        content_size = len(item.content.encode('utf-8'))
        tags_size = sum(len(tag.encode('utf-8')) for tag in item.tags)
        path_size = len(item.file_path.encode('utf-8'))
        metadata_size = len(json.dumps(item.metadata).encode('utf-8')) if item.metadata else 0
        
        total_size = base_size + content_size + tags_size + path_size + metadata_size
        return total_size
    
    async def enqueue_for_embedding(self, content: str, tags: List[str], 
                                   file_path: str, chunk_index: int,
                                   metadata: Dict[str, Any] = None) -> bool:
        """Add item to queue for embedding processing"""
        if not content or not content.strip():
            logger.warning("Empty content provided for embedding queue")
            return False
        
        item = EmbeddingItem(
            id=f"{file_path}:{chunk_index}:{int(time.time() * 1000)}",
            content=content,
            tags=tags or [],
            file_path=file_path,
            chunk_index=chunk_index,
            timestamp=time.time(),
            size_bytes=0,
            metadata=metadata or {}
        )
        
        item.size_bytes = self.calculate_item_size(item)
        
        # Check memory limits
        if self.current_memory_usage + item.size_bytes > self.max_memory_bytes:
            logger.warning(f"Queue memory limit reached: {self.current_memory_usage / (1024**2):.1f}MB")
            return False
        
        # Add to queue
        async with self.processing_lock:
            self.queue.append(item)
            self.current_memory_usage += item.size_bytes
            self._stats['queued_items'] += 1
            self._stats['current_memory_mb'] = self.current_memory_usage / (1024**2)
            self._stats['peak_memory_mb'] = max(self._stats['peak_memory_mb'], 
                                               self._stats['current_memory_mb'])
            self._stats['queue_size'] = len(self.queue)
        
        logger.debug(f"Item queued: {item.id} (queue size: {len(self.queue)})")
        return True
    
    async def dequeue_batch(self, batch_size: int = None) -> List[EmbeddingItem]:
        """Remove and return a batch of items from queue"""
        batch_size = batch_size or config.CHUNK_SIZES['embedding_batch']
        batch = []
        batch_memory = 0
        
        async with self.processing_lock:
            while len(batch) < batch_size and self.queue:
                item = self.queue.popleft()
                batch.append(item)
                batch_memory += item.size_bytes
                self.current_memory_usage -= item.size_bytes
            
            if batch:
                self._stats['current_memory_mb'] = self.current_memory_usage / (1024**2)
                self._stats['queue_size'] = len(self.queue)
        
        if batch:
            logger.debug(f"Dequeued batch: {len(batch)} items, freed {batch_memory / (1024**2):.1f}MB")
        
        return batch
    
    async def process_batch(self, batch: List[EmbeddingItem]) -> List[Dict[str, Any]]:
        """Process batch of items to generate embeddings and create records"""
        if not batch:
            return []
        
        try:
            logger.debug(f"Processing batch of {len(batch)} items")
            
            # Extract texts for batch processing
            texts = [item.content for item in batch]
            
            # Generate embeddings
            embeddings = await self.embedding_service.generate_embeddings_batch(texts)
            
            # Create records
            processed_records = []
            for i, (item, embedding) in enumerate(zip(batch, embeddings)):
                if embedding and validate_embedding(embedding):
                    record = {
                        'content': item.content,
                        'tags': item.tags,
                        'embedding': embedding,
                        'file_path': item.file_path,
                        'chunk_index': item.chunk_index,
                        'metadata': item.metadata
                    }
                    processed_records.append(record)
                    self._stats['processed_items'] += 1
                    logger.debug(f"Successfully processed item {i+1}: {item.id}")
                else:
                    logger.warning(f"Failed to generate embedding for item: {item.id}")
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
                # Wait for items
                if not self.queue:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get batch
                batch = await self.dequeue_batch()
                if not batch:
                    continue
                
                logger.debug(f"Worker {worker_id}: Processing batch of {len(batch)} items")
                
                # Process batch
                processed_records = await self.process_batch(batch)
                
                # Insert into database
                if processed_records:
                    try:
                        await insert_callback(processed_records)
                        logger.info(f"Worker {worker_id}: Inserted {len(processed_records)} records")
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: Insert failed: {e}")
                        # Re-queue failed items
                        for record in processed_records:
                            await self.enqueue_for_embedding(
                                record['content'],
                                record['tags'],
                                record['file_path'],
                                record['chunk_index'],
                                record.get('metadata', {})
                            )
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await asyncio.sleep(1)
        
        logger.info(f"Embedding worker {worker_id} stopped")
    
    async def start_workers(self, concurrency: int = None, insert_callback: Callable = None):
        """Start embedding workers"""
        if self.started:
            logger.warning("Workers already started")
            return
        
        if not insert_callback:
            raise ValueError("insert_callback is required")
        
        concurrency = concurrency or config.PERFORMANCE_CONFIG['worker_processes']
        
        # Test embedding service connection
        if not await self.embedding_service.test_connection():
            raise RuntimeError("Cannot connect to embedding service")
        
        # Reset state
        self.started = True
        self._shutdown_event.clear()
        self._stats['worker_count'] = concurrency
        
        # Start workers
        for i in range(concurrency):
            worker_task = asyncio.create_task(self.worker(i, insert_callback))
            self.workers.append(worker_task)
        
        # Start persistence task
        self._persist_task = asyncio.create_task(self._persist_state_periodically())
        
        logger.info(f"Started {concurrency} embedding workers")
    
    async def stop_workers(self):
        """Stop all workers gracefully"""
        if not self.started:
            return
        
        logger.info("Stopping embedding workers...")
        
        # Wait for queue to empty
        if len(self.queue) > 0:
            logger.info(f"Waiting for {len(self.queue)} items to complete...")
            wait_count = 0
            while len(self.queue) > 0 and wait_count < 120:  # 2 minute timeout
                await asyncio.sleep(1)
                wait_count += 1
                if wait_count % 10 == 0:
                    logger.info(f"Still waiting... {len(self.queue)} items remaining")
        
        # Signal shutdown
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
        
        # Close embedding service
        await self.embedding_service.close()
        
        self.workers.clear()
        self.started = False
        self._stats['worker_count'] = 0
        
        # Save final state
        self.save_persistent_state()
        
        logger.info("All embedding workers stopped")
    
    async def _persist_state_periodically(self):
        """Periodically save queue state"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.persist_interval)
                if not self._shutdown_event.is_set():
                    self.save_persistent_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"State persistence error: {e}")
    
    def save_persistent_state(self):
        """Save queue state to disk"""
        try:
            state = {
                'queue_items': [item.to_dict() for item in list(self.queue)],
                'stats': self._stats.copy(),
                'timestamp': time.time()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"Saved queue state: {len(self.queue)} items")
            
        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")
    
    def load_persistent_state(self):
        """Load queue state from disk"""
        try:
            if not os.path.exists(self.state_file):
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore queue items
            for item_data in state.get('queue_items', []):
                item = EmbeddingItem(**item_data)
                self.queue.append(item)
                self.current_memory_usage += item.size_bytes
            
            # Restore stats
            if 'stats' in state:
                self._stats.update(state['stats'])
                self._stats['queue_size'] = len(self.queue)
                self._stats['current_memory_mb'] = self.current_memory_usage / (1024**2)
            
            logger.info(f"Restored queue state: {len(self.queue)} items, "
                       f"{self.current_memory_usage / (1024**2):.1f}MB")
            
        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")
            self.queue.clear()
            self.current_memory_usage = 0
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.save_persistent_state()
        except:
            pass
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get current queue statistics"""
        self._stats.update({
            'last_update': time.time(),
            'queue_size': len(self.queue),
            'current_memory_mb': self.current_memory_usage / (1024**2)
        })
        return self._stats.copy()
    
    def is_full(self) -> bool:
        """Check if queue is approaching memory limits"""
        return self.current_memory_usage > (self.max_memory_bytes * 0.9)
    
    def wait_for_space(self) -> bool:
        """Check if should wait for queue space"""
        return self.current_memory_usage > (self.max_memory_bytes * 0.8)

# Global instances
embedding_service = EmbeddingService()
embedding_queue = EmbeddingQueue()

# Legacy compatibility functions
async def get_single_embedding(text: str) -> Optional[List[float]]:
    """Legacy compatibility function"""
    return await embedding_service.generate_embedding(text)

async def get_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """Legacy compatibility function"""
    return await embedding_service.generate_embeddings_batch(texts)
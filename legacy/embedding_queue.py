# embedding_queue.py
import asyncio
import aiohttp
import json
import logging
import random
import time
from constants import *

OLLAMA_API = "http://localhost:11434/api"
EMBEDDING_DIMENSION = EMB_DIM
logger = logging.getLogger()

def is_zero_vector(vector):
    """Check if vector contains only zeros"""
    return all(x == 0 for x in vector)

class EmbeddingQueue:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.queue = asyncio.Queue()
            cls._instance.workers = []
            cls._instance.started = False
            cls._instance.session = None
        return cls._instance
    
    async def start_workers(self, concurrency=10):
        if self.started:
            return
        self.started = True
        # Create shared session with connection limits
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit_per_host=2))
        self.workers = [asyncio.create_task(self.worker()) for _ in range(concurrency)]
        logger.info(f"Started {concurrency} embedding workers")
    
    async def stop_workers(self):
        self.started = False
        await self.queue.join()
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers = []
        if self.session:
            await self.session.close()
        logger.info("Embedding workers stopped")
    
    async def worker(self):
        while self.started:
            try:
                (text, future) = await self.queue.get()
                try:
                    embedding = await self.fetch_embedding(text)
                    future.set_result(embedding)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.queue.task_done()
            except asyncio.CancelledError:
                return
    
    async def fetch_embedding(self, text):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    f"{OLLAMA_API}/embed",
                    json={"model": EMBEDDING_MODEL, "input": text},
                    timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embedding = data.get("embedding") or (data.get("embeddings")[0] if data.get("embeddings") else None)
                        if not embedding:
                            logger.error(f"Unexpected response format: {data}")
                            raise ValueError("Unexpected response format")
                        if len(embedding) != EMBEDDING_DIMENSION:
                            logger.error(f"Invalid dimension: {len(embedding)} (expected {EMBEDDING_DIMENSION})")
                            raise ValueError(f"Invalid dimension: {len(embedding)}")
                        if is_zero_vector(embedding):
                            logger.error("Zero vector generated")
                            raise ValueError("Zero vector generated")
                        return embedding
                    error = await response.text()
                    logger.error(f"API error {response.status}: {error}")
                    raise Exception(f"API error {response.status}: {error}")
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 1 + random.random() + attempt
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"Embedding failed after {max_retries} attempts")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode error: {str(e)}")
                raise
    
    async def enqueue(self, text):
        future = asyncio.Future()
        await self.queue.put((text, future))
        return await future

# Global queue instance
embedding_queue = EmbeddingQueue()

# utils.py
import json
import requests
import aiohttp
import asyncio
import logging
import inspect
import traceback
import time
import random
import math
import torch
from embedding_queue import embedding_queue

MAX_CONCURRENT_REQUESTS = 200
EMBEDDING_DIMENSION = 1024
OLLAMA_API = "http://localhost:11434/api"
logger = logging.getLogger()
l_pre = f"{inspect.currentframe().f_code.co_name}-{inspect.currentframe().f_lineno}::"
embedding_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Memory cleanup
def cleanup_memory():
    # Force garbage collection and clear GPU cache if necessary
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Memory cleanup performed")
    return

# Vector math utilities
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def is_zero_vector(vector):
    """Check if vector contains only zeros"""
    return all(x == 0 for x in vector)

def batched_gpu_cosine_similarity(query_embedding, db_embeddings_tensor):
    """
    Compute cosine similarity between a query vector and a batch of vectors on GPU.
    Args:
        query_embedding (list or np.array): The query embedding [dim].
        db_embeddings_tensor (torch.Tensor): Database embeddings [batch_size, dim] on GPU.
    Returns:
        np.array: Cosine similarities [batch_size].
    """
    # Convert query to tensor and move to same device as db_embeddings_tensor
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    if db_embeddings_tensor.is_cuda:
        query_tensor = query_tensor.cuda()
    
    # Ensure query is 2D [1, dim] for matrix multiplication
    if query_tensor.dim() == 1:
        query_tensor = query_tensor.unsqueeze(0)  # Shape: [1, dim]
    
    # Normalize vectors
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)  # Shape: [1, dim]
    db_norm = torch.nn.functional.normalize(db_embeddings_tensor, p=2, dim=1)  # Shape: [batch_size, dim]
    
    # Compute cosine similarity via dot product
    # Result shape: [1, batch_size]
    cos_sim = torch.mm(query_norm, db_norm.t())
    
    # Flatten and convert to numpy - always move to CPU first
    return cos_sim.squeeze().cpu().numpy()

async def get_embeddings_batch(session, texts):
    """Batch process embeddings using centralized queue"""
    if not texts:
        return []
    
    # Start workers if not already started
    if not embedding_queue.started:
        await embedding_queue.start_workers(concurrency=10)
    
    # Return exceptions
    return await asyncio.gather(*[embedding_queue.enqueue(text) for text in texts], return_exceptions=False)

async def get_embedding_async(text):
    # Async version for embedding generation
    return await embedding_queue.enqueue(text)

def get_embedding(text):
    """Synchronous wrapper for async embedding generation"""
    logging.warning("Using synchronous embedding call - avoid in async contexts")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(embedding_queue.enqueue(text))
    finally:
        loop.close()

# utils.py
import json
import requests
import aiohttp
import asyncio
import logging
import inspect
import traceback
import time  # For retry mechanism
import random
import math  # For vector calculations

MAX_CONCURRENT_REQUESTS = 200
EMBEDDING_DIMENSION = 1024
OLLAMA_API = "http://localhost:11434/api"
logger = logging.getLogger()
l_pre = f"{inspect.currentframe().f_code.co_name}-{inspect.currentframe().f_lineno}::"
embedding_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

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

# Synchronous embedding with enhanced error handling
def get_embedding(text):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use the embeddings endpoint directly
            response = requests.post(
                f"{OLLAMA_API}/embed",
                json={"model": "dengcao/Qwen3-Embedding-0.6B:Q8_0", "input": text},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                if "embedding" in data:
                    embedding = data["embedding"]
                elif "embeddings" in data and data["embeddings"]:
                    embedding = data["embeddings"][0]
                else:
                    raise ValueError("Unexpected response format: no embedding found")
                
                # Validate embedding
                if len(embedding) != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Invalid embedding dimension: {len(embedding)} "
                        f"(expected {EMBEDDING_DIMENSION})"
                    )
                
                if is_zero_vector(embedding):
                    raise ValueError("Zero vector embedding generated")
                
                return embedding
            else:
                raise Exception(f"API error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = 1 + random.random()  # Add jitter
                time.sleep(sleep_time)
                continue
            raise Exception(f"Failed to get embedding after {max_retries} attempts: {str(e)}")

# Async batch embeddings with comprehensive error handling
async def get_embeddings_batch(session, texts):
    """Batch process embeddings with comprehensive error handling"""
    if not texts:
        return []

    async def fetch_one(text):
        """Fetch a single embedding with proper timeout handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with embedding_semaphore:
                    async with session.post(
                        f"{OLLAMA_API}/embed",
                        json={"model": "dengcao/Qwen3-Embedding-0.6B:Q8_0", "input": text},
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Handle both response formats
                            if "embedding" in data:
                                emb = data["embedding"]
                            elif "embeddings" in data and data["embeddings"]:
                                emb = data["embeddings"][0]
                            else:
                                raise ValueError("Unexpected response format")
                            
                            # Validate embedding
                            if not emb:
                                raise ValueError("Empty embedding received")
                                
                            if len(emb) != EMBEDDING_DIMENSION:
                                raise ValueError(
                                    f"Invalid dimension: {len(emb)} "
                                    f"(expected {EMBEDDING_DIMENSION})"
                                )
                                
                            if is_zero_vector(emb):
                                raise ValueError("Zero vector generated")
                                
                            return emb
                        else:
                            error = await response.text()
                            raise Exception(f"API error {response.status}: {error}")
            except asyncio.TimeoutError:
                logger.warning(f"Embedding timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 + random.random())  # Add jitter
                    continue
                return None
            except Exception as e:
                logger.error(f"Embedding exception: {str(e)}\n{traceback.format_exc()}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 + random.random())  # Add jitter
                    continue
                return None

    # Create and process all tasks
    tasks = [fetch_one(text) for text in texts]
    
    # Process in chunks to avoid too many simultaneous tasks
    embeddings = []
    for i in range(0, len(tasks), 100):  # Process 100 at a time
        chunk = tasks[i:i+100]
        results = await asyncio.gather(*chunk, return_exceptions=True)
        
        # Process results
        for j, result in enumerate(results):
            if isinstance(result, Exception) or result is None:
                embeddings.append(None)
            else:
                embeddings.append(result)
                
    return embeddings
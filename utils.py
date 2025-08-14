# utils.py
import asyncio
import logging
import psycopg2
import heapq
import json
from json import JSONDecodeError
import inspect
import numpy as np
import torch
from embedding_queue import embedding_queue
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor
from constants import *

MAX_CONCURRENT_REQUESTS = 200
EMBEDDING_DIMENSION = 1024
OLLAMA_API = "http://localhost:11434/api"
logger = logging.getLogger()
l_pre = f"{inspect.currentframe().f_code.co_name}-{inspect.currentframe().f_lineno}::"
embedding_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
table_name = TABLE_NAME

# Helper Functions
def get_user_question():
    """Prompt and validate user question"""
    question = input("Enter your question: ").strip()
    if not question:
        print("Question cannot be empty!")
    return question

async def get_question_embedding(question):
    """Generate embedding for question"""
    embedding = await get_embedding_async(question)
    if not embedding:
        print("Failed to generate embedding for question.")
    return embedding

def handle_cache_decision():
    """Determine if embedding cache should be used"""
    from em_cache import EmbeddingCache
    cache = EmbeddingCache()
    
    # Check if cache has content
    has_content = cache.stats['memory_entries'] > 0 or cache.stats['disk_entries'] > 0
    
    if not has_content:
        choice = input("Cache embeddings for faster processing? (y/N): ").lower()
        if choice == 'y':
            # Load embeddings into cache
            cache.load()
            if not cache.stats['memory_entries'] and not cache.stats['disk_entries']:
                try:
                    with db_cursor() as (conn, cur):
                        for record in cur.fetchall():
                            doc_id = record['id']
                            embedding = record['embedding']
                            if embedding and len(embedding) == EMBEDDING_DIMENSION:
                                cache.set(doc_id, embedding)
                        cur.close()
                        conn.close()
                    logger.info("Embeddings loaded into cache")
                    cached = True
                except Exception as e:
                    logger.error(f"Failed to load embeddings into cache: {e}")
                    cached = False
            return cached
        return False
    
    choice = input("Use available embedding cache? (Y/n): ").lower()
    if choice == 'n':
        logger.info("User opted out of cache")
        return False
    return True

async def perform_gpu_cache_search(question_embedding, threshold, top_k):
    """Search using pre-loaded GPU embeddings"""
    from em_cache import EmbeddingCache
    cache = EmbeddingCache()
    
    logger.info("Computing GPU similarities...")
    
    # Get all in-memory embeddings with their IDs
    all_embeddings = []
    all_ids = []
    for key in cache.memory_cache.keys():
        embedding = await cache.get(key)
        if embedding is not None:
            all_embeddings.append(embedding)
            all_ids.append(key)
    
    if not all_embeddings:
        logger.warning("No embeddings found in cache memory")
        return None
    
    # Convert to tensor and move to GPU
    embeddings_tensor = torch.stack(all_embeddings).cuda()
    
    # Calculate similarities
    similarities = gpu_cosine_similarity(question_embedding.cuda(), embeddings_tensor)
    
    # Process results
    valid_indices = torch.where(similarities >= threshold)[0]
    if len(valid_indices) == 0:
        print("No documents passed threshold.")
        return None

    # Get top-k indices
    sorted_indices = torch.argsort(similarities[valid_indices], descending=True)
    top_indices = valid_indices[sorted_indices][:top_k]

    # Prepare results
    top_docs = []
    for idx in top_indices:
        doc_id = all_ids[idx]
        sim_score = similarities[idx]
        top_docs.append({
            'id': doc_id,
            'cosine_similarity': float(sim_score)
        })

    logger.info(f"Found {len(top_docs)} GPU matches")
    
    # Cleanup
    del embeddings_tensor, similarities
    cleanup_gpu()
    return top_docs

def perform_chunked_database_search(question_embedding, threshold, top_k, chunk_size):
    """Search database using chunked processing"""
    logger.info("Starting chunked search...")
    total_docs = get_document_count()
    if not total_docs:
        return None

    top_k_heap = []
    docs_processed = 0
    
    with db_cursor() as (conn, cur):
        offset = 0
        while offset < total_docs:
            chunk_ids, chunk_embeddings = fetch_embedding_chunk(
                cur, offset, chunk_size
            )
            if not chunk_ids:
                break
                
            docs_processed += len(chunk_ids)
            log_progress(docs_processed, total_docs, chunk_size)
            
            similarities = calculate_chunk_similarities(
                question_embedding, 
                chunk_embeddings
            )
            update_top_k_heap(
                top_k_heap, 
                similarities, 
                chunk_ids, 
                threshold, 
                top_k
            )
            
            if docs_processed % (chunk_size * 5) == 0:
                cleanup_memory()
                
            offset += chunk_size

    return process_top_k_heap(top_k_heap)

def retrieve_document_contents(top_docs):
    """Fetch document contents for top matches"""
    doc_ids = [doc['id'] for doc in top_docs]
    with db_cursor() as (conn, cur):
        cur.execute(
            f"SELECT id, content FROM {table_name} WHERE id = ANY(%s)",
            (doc_ids,)
        )
        id_to_content = {r['id']: r['content'] for r in cur.fetchall()}
    
    final_docs = []
    for doc in top_docs:
        if content := id_to_content.get(doc['id']):
            doc['content'] = content
            final_docs.append(doc)
        else:
            logger.warning(f"Missing content for ID: {doc['id']}")
    
    return final_docs or None

def generate_and_display_answer(question, final_docs):
    """Generate answer from context and display results"""
    context = "\n".join(
        f"[Doc {i+1}] (CosSim: {doc['cosine_similarity']:.4f})\n{doc['content']}"
        for i, doc in enumerate(final_docs)
    )
    
    prompt = f"""Answer using only context below. If unsure, say 'I don't know'.
Question: {question}
Context (ranked by relevance):
{context}
"""
    answer = generate_answer(prompt)
    print("\nAnswer:", answer)
    print("\n--- Top Matching Documents ---")
    for doc in final_docs:
        print(f"ID {doc['id']} | Cosine Similarity: {doc['cosine_similarity']:.4f}")

def gpu_cosine_similarity(query_embedding, embeddings_tensor):
    """Compute cosine similarity using GPU acceleration"""
    # Convert query to tensor
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    if torch.cuda.is_available() and embeddings_tensor.is_cuda:
        query_tensor = query_tensor.cuda()
    
    # Ensure query is the right shape [1, dim] for matrix multiplication
    if query_tensor.dim() == 1:
        query_tensor = query_tensor.unsqueeze(0)  # Shape: [1, dim]
    
    # Normalize vectors
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)  # Shape: [1, dim]
    embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)  # Shape: [N, dim]
    
    # Compute cosine similarity via matrix multiplication
    # [1, dim] @ [N, dim].T = [1, N]
    cos_sim = torch.mm(query_norm, embeddings_norm.t())
    
    # Squeeze to remove the extra dimension and convert to numpy
    # Always move to CPU before converting to numpy
    return cos_sim.squeeze().cpu().numpy()

# Database Utilities
@contextmanager
def db_cursor():
    """Context manager for database connections"""
    conn = cur = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            connect_timeout=30
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        yield (conn, cur)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def get_document_count():
    """Get total document count from database"""
    with db_cursor() as (conn, cur):
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()['count'] or 0

def fetch_embedding_chunk(cursor, offset, chunk_size):
    """Retrieve chunk of embeddings from database"""
    cursor.execute(
        f"SELECT id, embedding FROM {table_name} ORDER BY id LIMIT %s OFFSET %s",
        (chunk_size, offset)
    )
    ids = []
    embeddings = []
    for row in cursor.fetchall():
        if not (embedding := parse_embedding(row['embedding'])):
            continue
        ids.append(row['id'])
        embeddings.append(embedding)
    return ids, embeddings

# Similarity Calculation
def calculate_chunk_similarities(question_embedding, chunk_embeddings):
    """Calculate similarities for chunk with GPU fallback"""
    if not chunk_embeddings:
        return None
        
    if torch.cuda.is_available() and len(chunk_embeddings) > 100:
        try:
            db_tensor = torch.tensor(chunk_embeddings, dtype=torch.float32).cuda()
            similarities = gpu_cosine_similarity(question_embedding, db_tensor)
            cleanup_gpu([db_tensor])
            return similarities
        except Exception as e:
            logger.warning(f"GPU failed: {e}")
    
    return np.array([
        VectorMath.cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ])

# Result Processing
def update_top_k_heap(heap, similarities, doc_ids, threshold, top_k):
    """Update top-k heap with new similarities"""
    if not similarities:
        return
        
    for i, doc_id in enumerate(doc_ids):
        if (score := similarities[i]) < threshold:
            continue
        if len(heap) < top_k:
            heapq.heappush(heap, (score, doc_id))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, doc_id))

def process_top_k_heap(heap):
    """Convert heap to final document list"""
    if not heap:
        print("No documents passed threshold.")
        return None
        
    return [{
        'id': doc_id,
        'cosine_similarity': float(score)
    } for score, doc_id in sorted(heap, reverse=True)]

def parse_embedding(embedding):
    """Parse embedding from various formats"""
    if embedding is None:
        return None
    if isinstance(embedding, (list, tuple)):
        return list(embedding)
    if isinstance(embedding, str):
        try:
            if embedding.startswith('['):
                return [float(x) for x in embedding[1:-1].split(',')]
            return json.loads(embedding)
        except (JSONDecodeError, ValueError) as e:
            logger.error(f"Embedding parse error: {e}")
    return None

def log_progress(processed, total, chunk_size=100):
    """Log progress at regular intervals"""
    if processed % (chunk_size * 10) == 0 or processed >= total:
        logger.info(f"Processed {processed}/{total} documents")


def cleanup_gpu(objects):
    """Cleanup GPU objects and memory"""
    for obj in objects:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
class VectorMath:
    @staticmethod
    def cosine_similarity(a, b):
        """Compute cosine similarity between vectors"""
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.cpu().numpy()
            
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0

    @staticmethod
    def euclidean_distance(a, b):
        """Compute Euclidean distance between vectors"""
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.cpu().numpy()
        return np.linalg.norm(np.array(a) - np.array(b))

    @staticmethod
    def batched_similarity(query, db_vectors):
        """Optimized batch similarity calculation"""
        if not isinstance(query, torch.Tensor):
            query = torch.tensor(query, dtype=torch.float32)
        if not isinstance(db_vectors, torch.Tensor):
            db_vectors = torch.tensor(db_vectors, dtype=torch.float32)
            
        if torch.cuda.is_available():
            query = query.cuda()
            db_vectors = db_vectors.cuda()
            
        query_norm = torch.nn.functional.normalize(query.unsqueeze(0), p=2, dim=1)
        db_norm = torch.nn.functional.normalize(db_vectors, p=2, dim=1)
        return torch.mm(query_norm, db_norm.t()).squeeze().cpu().numpy()

async def get_embeddings_batch(session, texts):
    """Batch process embeddings using centralized queue"""
    if not texts:
        return []
    if not embedding_queue.started:
        await embedding_queue.start_workers(concurrency=10)
    return await asyncio.gather(*[embedding_queue.enqueue(text) for text in texts])

async def get_embedding_async(text):
    return await embedding_queue.enqueue(text)

def get_embedding(text):
    """Synchronous wrapper for async embedding"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(embedding_queue.enqueue(text))
    finally:
        loop.close()
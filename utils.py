# utils.py
import asyncio
import logging
import heapq
import json
from json import JSONDecodeError
import numpy as np
import torch
from embedding_queue import embedding_queue
from contextlib import contextmanager
from config import Config
from db import db_manager
from vector_math import batched_cosine_similarity
from gpu_utils import cleanup_memory
import inspect

logger = logging.getLogger(__name__)
table_name = Config.TABLE_NAME

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
    """Simplified cache decision logic"""
    from em_cache import EmbeddingCache
    cache = EmbeddingCache()
    
    if cache.stats['memory_entries'] > 0 or cache.stats['disk_entries'] > 0:
        choice = input("Use available embedding cache? (Y/n): ").lower()
        return choice != 'n'
    
    choice = input("Cache embeddings for faster processing? (y/N): ").lower()
    if choice == 'y':
        # Load embeddings into cache
        asyncio.create_task(cache.load())
        return True
    return False

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
    
    # Use unified similarity calculation
    similarities = batched_cosine_similarity(
        question_embedding, 
        all_embeddings, 
        use_gpu=True
    )
    
    # Process results
    valid_indices = np.where(similarities >= threshold)[0]
    if len(valid_indices) == 0:
        print("No documents passed threshold.")
        return None

    # Get top-k indices
    sorted_indices = np.argsort(similarities[valid_indices])[::-1]
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

# Database Utilities
@contextmanager
def db_cursor():
    """Context manager for database connections"""
    with db_manager.get_sync_cursor() as (conn, cur):
        yield (conn, cur)

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
    """Calculate similarities for chunk using unified function"""
    if not chunk_embeddings:
        return None
    
    return batched_cosine_similarity(
        question_embedding, 
        chunk_embeddings,
        use_gpu=len(chunk_embeddings) > 100
    )

# Result Processing
def update_top_k_heap(heap, similarities, doc_ids, threshold, top_k):
    """Update top-k heap with new similarities"""
    if similarities is None:
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

def generate_answer(prompt):
    """Generate answer using Ollama (moved from rag_db.py for consistency)"""
    import requests
    from constants import OLLAMA_API
    
    try:
        response = requests.post(
            f"{OLLAMA_API}/generate",
            json={"model": "qwen3:8b", "prompt": prompt},
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    full_response += data.get("response", "")
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    logger.warning("Error decoding line from Ollama")
        return full_response
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        return "Error generating answer."
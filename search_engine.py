# search_engine.py - Unified Search and Retrieval
import asyncio
import logging
import heapq
import json
import numpy as np
import torch
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager
from core_config import config
from database_manager import db_manager
from embedding_manager import embedding_service

logger = logging.getLogger(__name__)

def batched_cosine_similarity(
    query_embedding: Union[List[float], np.ndarray, torch.Tensor], 
    embeddings: Union[List[List[float]], np.ndarray, torch.Tensor], 
    batch_size: int = 5000, 
    use_gpu: bool = None
) -> np.ndarray:
    """
    Optimized cosine similarity calculation with automatic GPU/CPU selection
    """
    # Convert inputs to numpy arrays
    if isinstance(query_embedding, (list, tuple)):
        query_array = np.array(query_embedding, dtype=np.float32)
    elif isinstance(query_embedding, torch.Tensor):
        query_array = query_embedding.cpu().numpy().astype(np.float32)
    else:
        query_array = query_embedding.astype(np.float32)
    
    if isinstance(embeddings, (list, tuple)):
        embeddings_array = np.array(embeddings, dtype=np.float32)
    elif isinstance(embeddings, torch.Tensor):
        embeddings_array = embeddings.cpu().numpy().astype(np.float32)
    else:
        embeddings_array = embeddings.astype(np.float32)
    
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = torch.cuda.is_available() and len(embeddings_array) > 1000
    
    if use_gpu and torch.cuda.is_available():
        return _gpu_batch_similarity(query_array, embeddings_array, batch_size)
    else:
        return _cpu_batch_similarity(query_array, embeddings_array)

def _gpu_batch_similarity(query_array: np.ndarray, embeddings_array: np.ndarray, batch_size: int) -> np.ndarray:
    """GPU-optimized similarity calculation"""
    device = torch.cuda.current_device()
    
    # Convert to tensors
    query_tensor = torch.from_numpy(query_array).float().to(device)
    
    all_similarities = []
    
    for i in range(0, len(embeddings_array), batch_size):
        batch_embeddings = embeddings_array[i:i + batch_size]
        batch_tensor = torch.from_numpy(batch_embeddings).float().to(device)
        
        # Normalize vectors
        query_norm = torch.nn.functional.normalize(query_tensor.unsqueeze(0), p=2, dim=1)
        batch_norm = torch.nn.functional.normalize(batch_tensor, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.mm(query_norm, batch_norm.t()).squeeze()
        
        # Move to CPU and store
        all_similarities.append(similarities.cpu().numpy())
        
        # Clear GPU memory
        del batch_tensor, similarities
    
    torch.cuda.empty_cache()
    return np.concatenate(all_similarities)

def _cpu_batch_similarity(query_array: np.ndarray, embeddings_array: np.ndarray) -> np.ndarray:
    """CPU-optimized similarity calculation"""
    # Normalize vectors
    query_norm = query_array / np.linalg.norm(query_array)
    embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    
    # Compute dot product (cosine similarity for normalized vectors)
    return np.dot(embeddings_norm, query_norm)

class EmbeddingCache:
    """Hybrid embedding cache with memory and disk storage"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.max_memory_bytes = config.PERFORMANCE_CONFIG['cache_memory_gb'] * 1024**3
            self.memory_cache = {}  # id -> (embedding, size)
            self.total_memory_bytes = 0
            self.db_loaded = False
            self.embedding_signatures = {}  # signature -> id (for deduplication)
            self._initialized = True
    
    async def load(self, force: bool = False):
        """Load embeddings from database into cache"""
        if self.db_loaded and not force:
            logger.info("Embeddings already loaded, skipping")
            return True
        
        try:
            loaded_count = 0
            async with db_manager.get_async_connection() as conn:
                # Load in chunks to manage memory
                offset = 0
                chunk_size = 1000
                
                while True:
                    query = f"""
                        SELECT id, embedding 
                        FROM {config.TABLE_NAME} 
                        WHERE embedding IS NOT NULL
                        ORDER BY id 
                        LIMIT $1 OFFSET $2
                    """
                    
                    rows = await conn.fetch(query, chunk_size, offset)
                    if not rows:
                        break
                    
                    for row in rows:
                        doc_id = row['id']
                        embedding = list(row['embedding'])
                        
                        await self.add(doc_id, embedding)
                        loaded_count += 1
                    
                    offset += chunk_size
                    
                    # Break if memory is getting full
                    if self.total_memory_bytes > self.max_memory_bytes * 0.8:
                        logger.warning("Cache memory limit approaching, stopping load")
                        break
            
            self.db_loaded = True
            logger.info(f"Loaded {loaded_count} embeddings into cache")
            return True
            
        except Exception as e:
            logger.error(f"Cache load failed: {e}", exc_info=True)
            return False
    
    async def add(self, key: str, embedding: List[float]):
        """Add embedding to cache with deduplication"""
        if not isinstance(embedding, (list, tuple)):
            embedding = list(embedding)
        
        # Create signature for deduplication
        signature = self._create_signature(embedding)
        
        # Skip duplicates
        if signature in self.embedding_signatures:
            existing_key = self.embedding_signatures[signature]
            logger.debug(f"Duplicate embedding: {key} duplicates {existing_key}")
            return
        
        # Calculate size
        embedding_array = np.array(embedding, dtype=np.float32)
        byte_size = embedding_array.nbytes
        
        # Remove existing if present
        if key in self.memory_cache:
            old_embedding, old_size = self.memory_cache[key]
            self.total_memory_bytes -= old_size
            old_sig = self._create_signature(old_embedding)
            if old_sig in self.embedding_signatures:
                del self.embedding_signatures[old_sig]
        
        # Check memory limit
        if self.total_memory_bytes + byte_size > self.max_memory_bytes:
            self._evict_oldest()
        
        # Add to cache
        self.memory_cache[key] = (embedding_array, byte_size)
        self.total_memory_bytes += byte_size
        self.embedding_signatures[signature] = key
        
        logger.debug(f"Added embedding to cache: {key}")
    
    async def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache"""
        if key in self.memory_cache:
            embedding, size = self.memory_cache[key]
            return embedding
        return None
    
    def _create_signature(self, embedding: Union[List[float], np.ndarray]) -> str:
        """Create unique signature for deduplication"""
        if isinstance(embedding, np.ndarray):
            sample = embedding
        else:
            sample = np.array(embedding)
        
        # Use reduced precision for signature
        if len(sample) > 100:
            stride = len(sample) // 100
            sample = sample[::stride]
        
        # Create hash
        rounded = np.round(sample, decimals=3)
        return hash(rounded.tobytes())
    
    def _evict_oldest(self):
        """Evict oldest 25% of cache entries"""
        if not self.memory_cache:
            return
        
        num_to_evict = max(1, len(self.memory_cache) // 4)
        logger.info(f"Evicting {num_to_evict} cache entries")
        
        # Convert to list and sort by access time (simplified - just remove first items)
        items_to_remove = list(self.memory_cache.keys())[:num_to_evict]
        
        for key in items_to_remove:
            embedding, size = self.memory_cache.pop(key)
            self.total_memory_bytes -= size
            
            # Remove from signatures
            sig = self._create_signature(embedding)
            if sig in self.embedding_signatures:
                del self.embedding_signatures[sig]
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "memory_entries": len(self.memory_cache),
            "memory_usage_mb": self.total_memory_bytes / (1024 * 1024),
            "memory_usage_percent": (self.total_memory_bytes / self.max_memory_bytes) * 100,
            "unique_embeddings": len(self.embedding_signatures),
            "db_loaded": self.db_loaded
        }

class SearchEngine:
    """Unified search engine with multiple search strategies"""
    
    def __init__(self):
        self.cache = EmbeddingCache()
        self.table_name = config.TABLE_NAME
    
    async def search(self, 
                    query: str, 
                    method: str = "auto",
                    relevance_threshold: float = None,
                    top_k: int = None,
                    use_cache: bool = None) -> Optional[List[Dict[str, Any]]]:
        """
        Unified search interface with multiple methods
        
        Args:
            query: Search query text
            method: "auto", "cache", "database", "text", "hybrid"
            relevance_threshold: Minimum similarity threshold
            top_k: Maximum number of results
            use_cache: Force cache usage
        
        Returns:
            List of search results with content and metadata
        """
        # Set defaults
        relevance_threshold = relevance_threshold or config.SEARCH_DEFAULTS['relevance_threshold']
        top_k = top_k or config.SEARCH_DEFAULTS['top_k']
        
        # Generate query embedding
        query_embedding = await embedding_service.generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return None
        
        # Determine search method
        if method == "auto":
            method = await self._determine_optimal_method(use_cache)
        
        logger.info(f"Using search method: {method}")
        
        # Execute search
        if method == "cache":
            return await self._search_cache(query_embedding, relevance_threshold, top_k)
        elif method == "database":
            return await self._search_database(query_embedding, relevance_threshold, top_k)
        elif method == "text":
            return await self._search_text(query, top_k)
        elif method == "hybrid":
            return await self._search_hybrid(query, query_embedding, relevance_threshold, top_k)
        else:
            logger.error(f"Unknown search method: {method}")
            return None
    
    async def _determine_optimal_method(self, use_cache: bool = None) -> str:
        """Determine optimal search method based on system state"""
        if use_cache is True:
            return "cache"
        
        cache_stats = self.cache.stats
        
        # Use cache if it has significant data
        if cache_stats['memory_entries'] > 100:
            return "cache"
        
        # Check database size
        db_stats = db_manager.get_stats()
        record_count = db_stats.get('record_count', 0)
        
        if record_count > 10000:
            return "database"
        elif record_count > 0:
            return "database"
        else:
            return "text"
    
    async def _search_cache(self, query_embedding: List[float], 
                           threshold: float, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Search using pre-loaded cache"""
        logger.info("Performing cache-based search")
        
        # Ensure cache is loaded
        if not self.cache.db_loaded:
            await self.cache.load()
        
        if len(self.cache.memory_cache) == 0:
            logger.warning("Cache is empty")
            return None
        
        # Get all embeddings and IDs
        all_embeddings = []
        all_ids = []
        
        for doc_id, (embedding, _) in self.cache.memory_cache.items():
            all_embeddings.append(embedding)
            all_ids.append(doc_id)
        
        if not all_embeddings:
            logger.warning("No embeddings found in cache")
            return None
        
        # Calculate similarities
        similarities = batched_cosine_similarity(query_embedding, all_embeddings, use_gpu=True)
        
        # Filter by threshold
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            logger.info("No documents passed similarity threshold")
            return None
        
        # Get top-k
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]
        
        # Get document IDs and similarities
        result_docs = []
        for idx in top_indices:
            doc_id = all_ids[idx]
            similarity = float(similarities[idx])
            result_docs.append({
                'id': doc_id,
                'cosine_similarity': similarity
            })
        
        # Fetch full document content
        return await self._enrich_results(result_docs)
    
    async def _search_database(self, query_embedding: List[float], 
                              threshold: float, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Search using chunked database processing"""
        logger.info("Performing database search")
        
        # Get total document count
        db_stats = db_manager.get_stats()
        total_docs = db_stats.get('record_count', 0)
        
        if total_docs == 0:
            logger.warning("Database is empty")
            return None
        
        # Use chunked processing for large datasets
        chunk_size = min(config.SEARCH_DEFAULTS['vector_search_limit'], 10000)
        top_k_heap = []
        
        offset = 0
        processed = 0
        
        while offset < total_docs:
            # Get chunk of embeddings
            chunk_ids, chunk_embeddings = await db_manager.get_embeddings_chunk(offset, chunk_size)
            
            if not chunk_ids:
                break
            
            processed += len(chunk_ids)
            logger.debug(f"Processing chunk: {processed}/{total_docs}")
            
            # Calculate similarities
            similarities = batched_cosine_similarity(
                query_embedding, 
                chunk_embeddings,
                use_gpu=len(chunk_embeddings) > 100
            )
            
            # Update top-k heap
            for i, doc_id in enumerate(chunk_ids):
                similarity = similarities[i]
                if similarity >= threshold:
                    if len(top_k_heap) < top_k:
                        heapq.heappush(top_k_heap, (similarity, doc_id))
                    elif similarity > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (similarity, doc_id))
            
            offset += chunk_size
            
            # Memory cleanup for large datasets
            if processed % (chunk_size * 5) == 0:
                self._cleanup_memory()
        
        if not top_k_heap:
            logger.info("No documents passed similarity threshold")
            return None
        
        # Convert heap to results
        result_docs = [{
            'id': doc_id,
            'cosine_similarity': float(similarity)
        } for similarity, doc_id in sorted(top_k_heap, reverse=True)]
        
        # Fetch full content
        return await self._enrich_results(result_docs)
    
    async def _search_text(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Search using full-text search"""
        logger.info("Performing text-based search")
        
        try:
            results = db_manager.search_by_text(query, top_k)
            if not results:
                return None
            
            # Convert to standard format
            enriched_results = []
            for result in results:
                enriched_results.append({
                    'id': result['id'],
                    'content': result['content'],
                    'tags': result.get('tags', []),
                    'file_path': result.get('file_path'),
                    'text_rank': result.get('rank', 0.0),
                    'cosine_similarity': None  # Not applicable for text search
                })
            
            return enriched_results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return None
    
    async def _search_hybrid(self, query: str, query_embedding: List[float],
                            threshold: float, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Hybrid search combining vector and text search"""
        logger.info("Performing hybrid search")
        
        # Perform both searches
        vector_results = await self._search_database(query_embedding, threshold, top_k * 2)
        text_results = await self._search_text(query, top_k)
        
        if not vector_results and not text_results:
            return None
        
        # Combine and rank results
        combined_results = {}
        
        # Add vector results
        if vector_results:
            for result in vector_results:
                doc_id = result['id']
                combined_results[doc_id] = result
                combined_results[doc_id]['vector_score'] = result['cosine_similarity']
        
        # Add text results
        if text_results:
            for result in text_results:
                doc_id = result['id']
                if doc_id in combined_results:
                    combined_results[doc_id]['text_score'] = result.get('text_rank', 0.0)
                else:
                    combined_results[doc_id] = result
                    combined_results[doc_id]['text_score'] = result.get('text_rank', 0.0)
                    combined_results[doc_id]['vector_score'] = 0.0
        
        # Calculate hybrid scores
        for doc_id, result in combined_results.items():
            vector_score = result.get('vector_score', 0.0)
            text_score = result.get('text_score', 0.0)
            
            # Weighted combination (adjust weights as needed)
            hybrid_score = 0.7 * vector_score + 0.3 * text_score
            result['hybrid_score'] = hybrid_score
        
        # Sort by hybrid score and limit
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:top_k]
        
        return sorted_results
    
    async def _enrich_results(self, result_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich search results with full document content"""
        if not result_docs:
            return []
        
        doc_ids = [doc['id'] for doc in result_docs]
        documents = await db_manager.get_documents_by_ids(doc_ids)
        
        # Create ID to document mapping
        id_to_doc = {doc['id']: doc for doc in documents}
        
        # Enrich results
        enriched_results = []
        for result in result_docs:
            doc_id = result['id']
            if doc_id in id_to_doc:
                doc = id_to_doc[doc_id]
                enriched_result = {
                    'id': doc_id,
                    'content': doc['content'],
                    'tags': doc.get('tags', []),
                    'file_path': doc.get('file_path'),
                    'chunk_index': doc.get('chunk_index', 0),
                    'metadata': doc.get('metadata', {}),
                    'cosine_similarity': result.get('cosine_similarity'),
                    'text_rank': result.get('text_rank'),
                    'hybrid_score': result.get('hybrid_score')
                }
                enriched_results.append(enriched_result)
            else:
                logger.warning(f"Missing document content for ID: {doc_id}")
        
        return enriched_results
    
    def _cleanup_memory(self):
        """Clean up memory during intensive operations"""
        try:
            import gc
            gc.collect()
            
            # GPU cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.debug(f"Memory cleanup failed: {e}")
    
    async def search_by_tags(self, tags: List[str], limit: int = None) -> List[Dict[str, Any]]:
        """Search documents by tags"""
        limit = limit or config.SEARCH_DEFAULTS['top_k']
        return db_manager.search_by_tags(tags, limit)
    
    async def search_by_file_path(self, file_path: str, limit: int = None) -> List[Dict[str, Any]]:
        """Search documents by file path"""
        limit = limit or config.SEARCH_DEFAULTS['top_k']
        
        try:
            async with db_manager.get_async_connection() as conn:
                query = f"""
                    SELECT id, content, tags, file_path, chunk_index, metadata
                    FROM {self.table_name}
                    WHERE file_path = $1
                    ORDER BY chunk_index
                    LIMIT $2
                """
                
                rows = await conn.fetch(query, file_path, limit)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"File path search failed: {e}")
            return []

class AnswerGenerator:
    """Generate answers using retrieved context"""
    
    def __init__(self):
        self.api_url = config.OLLAMA_API
        self.model = "qwen3:8b"  # Answer generation model
    
    def generate_answer(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer from context documents"""
        if not context_docs:
            return "I don't have enough information to answer this question."
        
        # Prepare context
        context_parts = []
        for i, doc in enumerate(context_docs[:5]):  # Limit to top 5 docs
            similarity = doc.get('cosine_similarity', 0.0)
            content = doc['content'][:1000]  # Limit content length
            context_parts.append(f"[Doc {i+1}] (Similarity: {similarity:.3f})\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Answer the question using only the context provided below. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Question: {question}

Context:
{context}

Answer:"""
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "Error generating answer.")
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Error generating answer. Please try again."
    
    def generate_answer_streaming(self, question: str, context_docs: List[Dict[str, Any]]):
        """Generate answer with streaming response"""
        if not context_docs:
            yield "I don't have enough information to answer this question."
            return
        
        # Prepare context (same as above)
        context_parts = []
        for i, doc in enumerate(context_docs[:5]):
            similarity = doc.get('cosine_similarity', 0.0)
            content = doc['content'][:1000]
            context_parts.append(f"[Doc {i+1}] (Similarity: {similarity:.3f})\n{content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Answer the question using only the context provided below. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Question: {question}

Context:
{context}

Answer:"""
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming answer generation failed: {e}")
            yield "Error generating answer. Please try again."

# Global instances
search_engine = SearchEngine()
embedding_cache = search_engine.cache
answer_generator = AnswerGenerator()

# Legacy compatibility functions
async def get_question_embedding(question: str) -> Optional[List[float]]:
    """Legacy compatibility function"""
    return await embedding_service.generate_embedding(question)

async def perform_gpu_cache_search(question_embedding: List[float], 
                                  threshold: float, top_k: int) -> Optional[List[Dict[str, Any]]]:
    """Legacy compatibility function"""
    return await search_engine._search_cache(question_embedding, threshold, top_k)

def perform_chunked_database_search(question_embedding: List[float], 
                                   threshold: float, top_k: int, chunk_size: int) -> Optional[List[Dict[str, Any]]]:
    """Legacy compatibility function - synchronous wrapper"""
    async def _async_search():
        return await search_engine._search_database(question_embedding, threshold, top_k)
    
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If we're already in an async context, create a new task
        task = asyncio.create_task(_async_search())
        return asyncio.run_coroutine_threadsafe(task, loop).result()
    else:
        return asyncio.run(_async_search())

def generate_answer(prompt: str) -> str:
    """Legacy compatibility function"""
    # Extract question from prompt (simple heuristic)
    lines = prompt.split('\n')
    question = ""
    for line in lines:
        if line.startswith("Question:"):
            question = line[9:].strip()
            break
    
    if not question:
        question = prompt
    
    return answer_generator.generate_answer(question, [])

def retrieve_document_contents(top_docs: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Legacy compatibility function - synchronous wrapper"""
    if not top_docs:
        return None
    
    async def _async_retrieve():
        return await search_engine._enrich_results(top_docs)
    
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = asyncio.create_task(_async_retrieve())
        return asyncio.run_coroutine_threadsafe(task, loop).result()
    else:
        return asyncio.run(_async_retrieve())
# inference/engine.py - Optimized Inference Engine
import json
import logging
from typing import List, Dict, Any, Optional
import requests

from core.config import config
from core.vector_ops import VectorOperations
from core.memory import memory_manager
from database.operations import db_ops
from database.cache import embedding_cache
from inference.embeddings import embedding_service

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Optimized inference engine that intelligently chooses between:
    1. Database-level vector similarity (most efficient for large DBs)
    2. GPU-cached similarity (efficient for small/medium DBs with repeated searches)
    3. Chunked processing (fallback for edge cases)
    """
    
    def __init__(self):
        self.config = config.inference
        self.vector_ops = VectorOperations()
    
    async def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimized semantic search that automatically chooses the best strategy
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            relevance_threshold: Minimum similarity threshold
            use_cache: Whether to use embedding cache (None = auto-decide)
            
        Returns:
            List of matching documents with metadata
        """
        # Use config defaults if not specified
        top_k = top_k or self.config.top_k
        relevance_threshold = relevance_threshold or self.config.relevance_threshold
        
        # Generate query embedding
        query_embedding = await embedding_service.generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Determine optimal search strategy
        strategy = self._choose_search_strategy(use_cache)
        logger.info(f"Using search strategy: {strategy}")
        
        # Execute search based on strategy
        if strategy == "database_direct":
            return await self._search_database_direct(query_embedding, top_k, relevance_threshold)
        elif strategy == "cache_gpu":
            return await self._search_with_cache(query_embedding, top_k, relevance_threshold)
        else:  # fallback
            return await self._search_chunked(query_embedding, top_k, relevance_threshold)
    
    def _choose_search_strategy(self, use_cache: Optional[bool]) -> str:
        """
        Intelligently choose the optimal search strategy based on:
        - Database size
        - Cache availability
        - User preference
        
        Returns:
            Strategy name: "database_direct", "cache_gpu", or "chunked"
        """
        doc_count = db_ops.get_document_count()
        
        # User explicitly requested cache usage
        if use_cache is True:
            if embedding_cache.should_use_cache_for_search():
                return "cache_gpu"
            else:
                logger.warning("Cache requested but not optimal for current database size")
                return "database_direct"
        
        # User explicitly disabled cache
        elif use_cache is False:
            return "database_direct"
        
        # Auto-decide based on database size and cache status
        else:
            # For very small databases, try cache
            if doc_count <= 500:
                if embedding_cache.is_loaded or embedding_cache.should_use_cache_for_search():
                    return "cache_gpu"
                else:
                    return "database_direct"
            
            # For medium databases, use cache if already loaded
            elif doc_count <= 2000:
                if embedding_cache.is_loaded:
                    return "cache_gpu"
                else:
                    return "database_direct"
            
            # For large databases, always use database direct
            else:
                return "database_direct"
    
    async def _search_database_direct(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Use PostgreSQL's vector similarity directly - most efficient for large databases
        Falls back to sync operations if async has issues
        """
        logger.info("Using database-direct vector similarity search")
        
        try:
            # Try async first
            results = await db_ops.similarity_search_database_async(
                query_embedding, top_k, threshold
            )
            
            if results:
                # Convert database results to expected format
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'id': result['id'],
                        'content': result['content'],
                        'tags': result.get('tags', []),
                        'cosine_similarity': float(result['cosine_similarity']),
                        'euclidean_distance': 0  # Not calculated for performance
                    })
                
                logger.info(f"Found {len(formatted_results)} matches via async database search")
                return formatted_results
            else:
                logger.info("No documents passed relevance threshold")
                return []
                
        except Exception as async_e:
            logger.warning(f"Async database search failed ({async_e}), trying sync fallback")
            
            # Fallback to sync database operations which we know work
            try:
                results = db_ops.similarity_search_database(
                    query_embedding, top_k, threshold
                )
                
                if results:
                    # Convert database results to expected format
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'id': result['id'],
                            'content': result['content'],
                            'tags': result.get('tags', []),
                            'cosine_similarity': float(result['cosine_similarity']),
                            'euclidean_distance': 0  # Not calculated for performance
                        })
                    
                    logger.info(f"Found {len(formatted_results)} matches via sync database search")
                    return formatted_results
                else:
                    logger.info("No documents passed relevance threshold")
                    return []
                    
            except Exception as sync_e:
                logger.error(f"Both async and sync database search failed: async={async_e}, sync={sync_e}")
                # Final fallback to Python-based similarity
                return await self._search_python_fallback(query_embedding, top_k, threshold)
    
    async def _search_python_fallback(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Python-based similarity search as fallback when database operations fail
        This loads embeddings in chunks and computes similarity in Python
        """
        import heapq
        import numpy as np
        
        logger.info("Using Python fallback similarity search")
        
        try:
            # Check if we can use cache
            if embedding_cache.should_use_cache_for_search():
                cache_data = db_ops.get_embeddings_for_cache(10000)  # Limit for safety
                if cache_data:
                    ids, embeddings = cache_data
                    
                    # Compute similarities in Python
                    similarities = []
                    for emb in embeddings:
                        sim = self.vector_ops.cosine_similarity(query_embedding, emb)
                        similarities.append(sim)
                    
                    # Find top-k above threshold
                    results = []
                    for i, sim in enumerate(similarities):
                        if sim >= threshold:
                            results.append((ids[i], sim))
                    
                    # Sort and take top-k
                    results.sort(key=lambda x: x[1], reverse=True)
                    results = results[:top_k]
                    
                    if results:
                        # Get document content
                        doc_ids = [doc_id for doc_id, _ in results]
                        documents = db_ops.get_documents_by_ids(doc_ids)
                        id_to_content = {doc['id']: doc for doc in documents}
                        
                        formatted_results = []
                        for doc_id, similarity in results:
                            doc_content = id_to_content.get(doc_id)
                            if doc_content:
                                formatted_results.append({
                                    'id': doc_id,
                                    'content': doc_content['content'],
                                    'tags': doc_content.get('tags', []),
                                    'cosine_similarity': similarity,
                                    'euclidean_distance': 0
                                })
                        
                        logger.info(f"Found {len(formatted_results)} matches via Python fallback")
                        return formatted_results
            
            # If cache approach doesn't work, return empty results
            logger.warning("Python fallback could not find matches")
            return []
            
        except Exception as e:
            logger.error(f"Python fallback search also failed: {e}")
            return []
    
    async def _search_with_cache(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Use cached embeddings with GPU acceleration - efficient for small/medium DBs"""
        logger.info("Using GPU-cached similarity search")
        
        # Ensure cache is loaded
        if not embedding_cache.is_loaded:
            cache_loaded = embedding_cache.load_cache()
            if not cache_loaded:
                logger.warning("Failed to load cache, falling back to database search")
                return await self._search_database_direct(query_embedding, top_k, threshold)
        
        embeddings_tensor = embedding_cache.get_embeddings_tensor()
        doc_ids = embedding_cache.get_document_ids()
        
        if not embeddings_tensor or not doc_ids:
            logger.error("Cache data is invalid")
            return await self._search_database_direct(query_embedding, top_k, threshold)
        
        try:
            # Compute similarities using GPU
            similarities = self.vector_ops.gpu_cosine_similarity_batch(
                query_embedding, embeddings_tensor
            )
            
            # Find indices that meet threshold and get top-k
            results = self.vector_ops.similarity_search(
                query_embedding,
                embeddings_tensor,
                top_k=top_k,
                threshold=threshold,
                use_gpu=True
            )
            
            if not results:
                logger.info("No documents passed relevance threshold in cache")
                return []
            
            # Get document IDs and fetch content
            result_ids = [doc_ids[idx] for idx, _ in results]
            documents = db_ops.get_documents_by_ids(result_ids)
            
            # Map content to similarity scores
            id_to_content = {doc['id']: doc for doc in documents}
            
            formatted_results = []
            for idx, similarity in results:
                doc_id = doc_ids[idx]
                doc_content = id_to_content.get(doc_id)
                
                if doc_content:
                    formatted_results.append({
                        'id': doc_id,
                        'content': doc_content['content'],
                        'tags': doc_content.get('tags', []),
                        'cosine_similarity': similarity,
                        'euclidean_distance': 0  # Not calculated for performance
                    })
            
            logger.info(f"Found {len(formatted_results)} matches via GPU cache")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Cache search failed: {e}")
            return await self._search_database_direct(query_embedding, top_k, threshold)
        finally:
            # Cleanup GPU memory
            if embeddings_tensor and embeddings_tensor.is_cuda:
                import torch
                torch.cuda.empty_cache()
    
    async def _search_chunked(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Fallback chunked search - used when database direct fails
        This loads embeddings in small chunks and computes similarity
        """
        logger.info("Using chunked fallback search")
        
        try:
            # Delegate to Python fallback which is more robust
            return await self._search_python_fallback(query_embedding, top_k, threshold)
        except Exception as e:
            logger.error(f"All search methods failed: {e}")
            return []
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using the configured LLM"""
        try:
            response = requests.post(
                f"{config.embedding.api_url}/generate",
                json={"model": self.config.generation_model, "prompt": prompt},
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
            logger.error(f"Answer generation failed: {e}")
            return "Error generating answer."
    
    async def ask_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Complete question-answering pipeline with optimized search
        
        Args:
            question: The question to answer
            top_k: Number of top results to use for context
            relevance_threshold: Minimum similarity threshold
            use_cache: Whether to use embedding cache
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        # Perform optimized semantic search
        matching_docs = await self.semantic_search(
            question, top_k, relevance_threshold, use_cache
        )
        
        if not matching_docs:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "metadata": {
                    "matches_found": 0,
                    "search_strategy": self._choose_search_strategy(use_cache)
                }
            }
        
        # Build context for answer generation
        context_parts = []
        for idx, doc in enumerate(matching_docs):
            # Truncate content for context
            truncated_content = doc['content'][:800]
            context_parts.append(
                f"[Doc {idx+1}] (Similarity: {doc['cosine_similarity']:.4f})\n{truncated_content}"
            )
        
        context = "\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Answer the question using only the context below. If unsure, say 'I don't know'.

Question: {question}

Context (ranked by relevance):
{context}
"""
        
        answer = self.generate_answer(prompt)
        
        # Prepare sources for response
        sources = []
        for doc in matching_docs:
            sources.append({
                "id": doc['id'],
                "similarity": doc['cosine_similarity'],
                "content_preview": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                "tags": doc.get('tags', [])
            })
        
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "matches_found": len(matching_docs),
                "context_length": len(context),
                "question_length": len(question),
                "search_strategy": self._choose_search_strategy(use_cache)
            }
        }
    
    def get_search_strategy_info(self) -> Dict[str, Any]:
        """Get information about available search strategies and recommendations"""
        doc_count = db_ops.get_document_count()
        cache_info = embedding_cache.cache_info
        cache_recommendation = embedding_cache.get_cache_recommendation()
        
        return {
            "database_size": doc_count,
            "cache_status": cache_info,
            "cache_recommendation": cache_recommendation,
            "available_strategies": {
                "database_direct": {
                    "description": "Use PostgreSQL vector similarity directly",
                    "best_for": "Large databases (>2000 docs)",
                    "pros": ["Fastest for large DBs", "Uses HNSW index", "Low memory usage"],
                    "cons": ["Requires PostgreSQL with pgvector"]
                },
                "cache_gpu": {
                    "description": "Load embeddings into GPU memory for similarity search",
                    "best_for": "Small to medium databases (<2000 docs) with repeated searches",
                    "pros": ["Very fast repeated searches", "GPU acceleration"],
                    "cons": ["High memory usage", "Load time", "Not suitable for large DBs"]
                },
                "chunked": {
                    "description": "Fallback method - loads embeddings in chunks",
                    "best_for": "Error recovery only",
                    "pros": ["Works with any database size"],
                    "cons": ["Slowest method", "High memory usage"]
                }
            },
            "recommended_strategy": self._choose_search_strategy(None)
        }
    
    async def benchmark_search_methods(
        self, 
        test_query: str = "test query",
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark different search methods for performance comparison
        
        Args:
            test_query: Query to use for benchmarking
            iterations: Number of iterations to average
            
        Returns:
            Performance comparison data
        """
        import time
        
        query_embedding = await embedding_service.generate_embedding(test_query)
        if not query_embedding:
            return {"error": "Failed to generate test embedding"}
        
        results = {}
        
        # Benchmark database direct
        try:
            times = []
            for i in range(iterations):
                start = time.time()
                await self._search_database_direct(query_embedding, 10, 0.1)
                times.append(time.time() - start)
            
            results["database_direct"] = {
                "avg_time_ms": sum(times) / len(times) * 1000,
                "min_time_ms": min(times) * 1000,
                "max_time_ms": max(times) * 1000
            }
        except Exception as e:
            results["database_direct"] = {"error": str(e)}
        
        # Benchmark cache if appropriate
        if embedding_cache.should_use_cache_for_search():
            try:
                times = []
                for i in range(iterations):
                    start = time.time()
                    await self._search_with_cache(query_embedding, 10, 0.1)
                    times.append(time.time() - start)
                
                results["cache_gpu"] = {
                    "avg_time_ms": sum(times) / len(times) * 1000,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000
                }
            except Exception as e:
                results["cache_gpu"] = {"error": str(e)}
        else:
            results["cache_gpu"] = {"skipped": "Not recommended for current database size"}
        
        # Add recommendations
        results["recommendation"] = {
            "best_method": min(
                [(k, v.get("avg_time_ms", float('inf'))) for k, v in results.items() 
                 if isinstance(v, dict) and "avg_time_ms" in v],
                key=lambda x: x[1],
                default=("database_direct", 0)
            )[0],
            "database_size": db_ops.get_document_count(),
            "test_iterations": iterations
        }
        
        return results

# Global inference engine instance
inference_engine = InferenceEngine()
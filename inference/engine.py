# inference/engine.py
import json
import heapq
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests

from core.config import config
from core.vector_ops import VectorOperations
from core.memory import memory_manager
from database.operations import db_ops
from database.cache import embedding_cache
from inference.embeddings import embedding_service

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Main inference engine for semantic search and answer generation"""
    
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
        Perform semantic search for the given query
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            relevance_threshold: Minimum similarity threshold
            use_cache: Whether to use embedding cache
            
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
        
        # Decide on search strategy
        if use_cache is None:
            # Auto-decide based on cache availability
            use_cache = embedding_cache.is_loaded or self._should_use_cache()
        
        if use_cache and not embedding_cache.is_loaded:
            cache_loaded = embedding_cache.load_cache()
            if not cache_loaded:
                logger.warning("Failed to load cache, using chunked search")
                use_cache = False
        
        # Perform search
        if use_cache and embedding_cache.is_loaded:
            return await self._search_with_cache(query_embedding, top_k, relevance_threshold)
        else:
            return await self._search_chunked(query_embedding, top_k, relevance_threshold)
    
    def _should_use_cache(self) -> bool:
        """Determine if cache should be used based on database size"""
        doc_count = db_ops.get_document_count()
        return doc_count <= self.config.max_cache_size and doc_count > 0
    
    async def _search_with_cache(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Search using cached embeddings with GPU acceleration"""
        logger.info("Computing similarities using GPU cache...")
        
        embeddings_tensor = embedding_cache.get_embeddings_tensor()
        doc_ids = embedding_cache.get_document_ids()
        
        if not embeddings_tensor or not doc_ids:
            logger.error("Cache data is invalid")
            return []
        
        try:
            # Compute similarities using GPU
            similarities = self.vector_ops.gpu_cosine_similarity_batch(
                query_embedding, embeddings_tensor
            )
            
            # Find valid indices above threshold
            valid_indices = np.where(similarities >= threshold)[0]
            if len(valid_indices) == 0:
                logger.info("No documents passed relevance threshold")
                return []
            
            # Get top-k results
            valid_similarities = similarities[valid_indices]
            sorted_indices = valid_indices[np.argsort(valid_similarities)[::-1]]
            top_indices = sorted_indices[:top_k]
            
            # Prepare results with metadata
            top_docs = []
            for idx in top_indices:
                top_docs.append({
                    'id': doc_ids[idx],
                    'cosine_similarity': float(similarities[idx]),
                    'euclidean_distance': 0  # Not calculated for performance
                })
            
            logger.info(f"Found {len(top_docs)} matches via GPU cache")
            return top_docs
            
        except Exception as e:
            logger.error(f"Cache search failed: {e}")
            return []
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
        """Search using chunked processing for large databases"""
        logger.info("Starting chunked similarity search...")
        
        chunk_size = min(self.config.vector_search_limit, 500000)
        top_k_heap = []  # Min-heap for top-k results
        docs_processed = 0
        total_docs = db_ops.get_document_count()
        
        if total_docs == 0:
            logger.warning("No documents in database to search")
            return []
        
        try:
            offset = 0
            while offset < total_docs:
                # Get chunk of embeddings
                embeddings_page = db_ops.get_embeddings_page(chunk_size, offset)
                if not embeddings_page:
                    break
                
                chunk_ids = []
                chunk_embeddings = []
                
                for doc_id, embedding in embeddings_page:
                    chunk_ids.append(doc_id)
                    chunk_embeddings.append(embedding)
                
                docs_processed += len(chunk_ids)
                if docs_processed % (chunk_size * 10) == 0:
                    logger.info(f"Processed {docs_processed}/{total_docs} documents...")
                
                if not chunk_embeddings:
                    offset += chunk_size
                    continue
                
                # Compute similarities for chunk
                similarities = self._compute_chunk_similarities(query_embedding, chunk_embeddings)
                
                # Update top-k heap
                for i, doc_id in enumerate(chunk_ids):
                    sim_score = similarities[i]
                    
                    if sim_score >= threshold:
                        if len(top_k_heap) < top_k:
                            heapq.heappush(top_k_heap, (sim_score, doc_id))
                        elif sim_score > top_k_heap[0][0]:
                            heapq.heapreplace(top_k_heap, (sim_score, doc_id))
                
                # Memory cleanup
                if docs_processed % (chunk_size * 5) == 0:
                    memory_manager.force_cleanup_if_needed()
                
                offset += chunk_size
            
            # Prepare final results
            if not top_k_heap:
                logger.info("No documents passed relevance threshold")
                return []
            
            # Sort by similarity (descending)
            sorted_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
            
            top_docs = []
            for sim_score, doc_id in sorted_results:
                top_docs.append({
                    'id': doc_id,
                    'cosine_similarity': float(sim_score),
                    'euclidean_distance': 0  # Not calculated for performance
                })
            
            logger.info(f"Found {len(top_docs)} matches via chunked search")
            return top_docs
            
        except Exception as e:
            logger.error(f"Chunked search failed: {e}")
            return []
    
    def _compute_chunk_similarities(
        self,
        query_embedding: List[float],
        chunk_embeddings: List[List[float]]
    ) -> np.ndarray:
        """Compute similarities for a chunk, with GPU fallback"""
        import torch
        
        # Try GPU acceleration for large chunks
        if torch.cuda.is_available() and len(chunk_embeddings) > 100:
            try:
                db_tensor = torch.tensor(chunk_embeddings, dtype=torch.float32).cuda()
                similarities = self.vector_ops.gpu_cosine_similarity_batch(query_embedding, db_tensor)
                del db_tensor
                torch.cuda.empty_cache()
                return similarities
            except Exception as e:
                logger.warning(f"GPU similarity calculation failed, using CPU: {e}")
        
        # CPU fallback
        return np.array([
            self.vector_ops.cosine_similarity(query_embedding, emb) 
            for emb in chunk_embeddings
        ])
    
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
        Complete question-answering pipeline
        
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        # Perform semantic search
        matching_docs = await self.semantic_search(
            question, top_k, relevance_threshold, use_cache
        )
        
        if not matching_docs:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
                "metadata": {"matches_found": 0}
            }
        
        # Get document content
        doc_ids = [doc['id'] for doc in matching_docs]
        documents = db_ops.get_documents_by_ids(doc_ids)
        
        # Map content to similarity scores
        id_to_content = {doc['id']: doc for doc in documents}
        final_docs = []
        
        for match in matching_docs:
            doc_content = id_to_content.get(match['id'])
            if doc_content:
                final_docs.append({
                    **match,
                    'content': doc_content['content'],
                    'tags': doc_content.get('tags', [])
                })
        
        if not final_docs:
            return {
                "answer": "Found matching documents but couldn't retrieve content.",
                "sources": [],
                "metadata": {"matches_found": len(matching_docs)}
            }
        
        # Build context for answer generation
        context_parts = []
        for idx, doc in enumerate(final_docs):
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
        for doc in final_docs:
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
                "question_length": len(question)
            }
        }

# Global inference engine instance
inference_engine = InferenceEngine()
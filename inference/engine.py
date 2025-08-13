# inference/engine.py - Unified Inference Engine
import json
import logging
import asyncio
import requests
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from core.config import config
from core.vector_ops import VectorOperations
from core.memory import memory_manager
from database.operations import db_ops
from database.cache import embedding_cache
from inference.embeddings import embedding_service

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """Search strategy options"""
    DATABASE_DIRECT = "database_direct"
    CACHE_GPU = "cache_gpu"
    CHUNKED = "chunked"
    MULTI_QUERY = "multi_query"
    HIERARCHICAL = "hierarchical"

@dataclass
class QueryVariation:
    """Represents a variation of the original query"""
    original_query: str
    variation: str
    variation_type: str
    embedding: Optional[List[float]] = None

@dataclass
class ChunkSummary:
    """Represents a summary of a text chunk"""
    chunk_id: str
    source_file: str
    summary: str
    key_topics: List[str]
    chunk_indices: List[int]
    related_chunk_ids: List[str]
    original_length: int
    summary_length: int
    importance_score: float = 0.0

@dataclass
class SearchResult:
    """Unified search result structure"""
    original_query: str
    query_variations: List[QueryVariation]
    summary_matches: List[Dict[str, Any]]
    chunk_matches: List[Dict[str, Any]]
    combined_context: str
    confidence_score: float
    answer: str
    sources: List[Dict[str, Any]]
    search_strategy: str
    metadata: Dict[str, Any]

class UnifiedInferenceEngine:
    """
    Unified inference engine that combines all search strategies:
    - Database-level vector similarity
    - GPU-cached similarity
    - Chunked processing
    - Multi-query expansion
    - Hierarchical search with summaries
    """
    
    def __init__(self):
        self.config = config.inference
        self.vector_ops = VectorOperations()
        self._query_expansion_cache = {}
    
    async def search(
        self,
        query: str,
        strategy: Optional[SearchStrategy] = None,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None,
        use_multi_query: Optional[bool] = None,
        search_summaries_first: Optional[bool] = None,
        top_k_summaries: Optional[int] = None
    ) -> SearchResult:
        """
        Unified search interface with automatic strategy selection
        
        Args:
            query: Search query text
            strategy: Explicit strategy to use (None = auto-select)
            top_k: Number of top results to return
            relevance_threshold: Minimum similarity threshold
            use_cache: Whether to use embedding cache
            use_multi_query: Whether to use query expansion
            search_summaries_first: Whether to search summaries before chunks
            top_k_summaries: Number of summaries to retrieve
            
        Returns:
            SearchResult with answer, sources, and metadata
        """
        # Set defaults
        top_k = top_k or self.config.top_k
        relevance_threshold = relevance_threshold or self.config.relevance_threshold
        top_k_summaries = top_k_summaries or 10
        
        # Auto-select strategy if not specified
        if strategy is None:
            strategy = self._choose_optimal_strategy(
                query, use_cache, use_multi_query, search_summaries_first
            )
        
        logger.info(f"Using search strategy: {strategy.value}")
        
        # Execute search based on strategy
        if strategy == SearchStrategy.DATABASE_DIRECT:
            return await self._search_database_strategy(
                query, top_k, relevance_threshold
            )
        elif strategy == SearchStrategy.CACHE_GPU:
            return await self._search_cache_strategy(
                query, top_k, relevance_threshold
            )
        elif strategy == SearchStrategy.CHUNKED:
            return await self._search_chunked_strategy(
                query, top_k, relevance_threshold
            )
        elif strategy == SearchStrategy.MULTI_QUERY:
            return await self._search_multi_query_strategy(
                query, top_k, relevance_threshold, False, top_k_summaries
            )
        elif strategy == SearchStrategy.HIERARCHICAL:
            return await self._search_multi_query_strategy(
                query, top_k, relevance_threshold, True, top_k_summaries
            )
        else:
            # Fallback to database direct
            return await self._search_database_strategy(
                query, top_k, relevance_threshold
            )
    
    def _choose_optimal_strategy(
        self,
        query: str,
        use_cache: Optional[bool],
        use_multi_query: Optional[bool],
        search_summaries_first: Optional[bool]
    ) -> SearchStrategy:
        """
        Intelligently choose the optimal search strategy
        
        Returns:
            Optimal SearchStrategy based on conditions
        """
        doc_count = db_ops.get_document_count()
        
        # Check for explicit multi-query or hierarchical search
        if use_multi_query is True and search_summaries_first is True:
            return SearchStrategy.HIERARCHICAL
        elif use_multi_query is True:
            return SearchStrategy.MULTI_QUERY
        
        # Check cache preference
        if use_cache is True:
            if embedding_cache.should_use_cache_for_search():
                return SearchStrategy.CACHE_GPU
            else:
                logger.warning("Cache requested but not optimal")
                return SearchStrategy.DATABASE_DIRECT
        elif use_cache is False:
            return SearchStrategy.DATABASE_DIRECT
        
        # Auto-decide based on database size
        if doc_count <= 500:
            if embedding_cache.is_loaded or embedding_cache.should_use_cache_for_search():
                return SearchStrategy.CACHE_GPU
            else:
                return SearchStrategy.DATABASE_DIRECT
        elif doc_count <= 2000:
            if embedding_cache.is_loaded:
                return SearchStrategy.CACHE_GPU
            else:
                return SearchStrategy.DATABASE_DIRECT
        else:
            return SearchStrategy.DATABASE_DIRECT
    
    async def _search_database_strategy(
        self,
        query: str,
        top_k: int,
        threshold: float
    ) -> SearchResult:
        """Database-direct vector similarity search"""
        # Generate query embedding
        query_embedding = await embedding_service.generate_embedding(query)
        if not query_embedding:
            return self._create_empty_result(query, SearchStrategy.DATABASE_DIRECT)
        
        # Search database
        chunk_matches = await self._search_database_direct(
            query_embedding, top_k, threshold
        )
        
        # Build context and generate answer
        context = self._build_context_from_chunks(chunk_matches)
        answer = self._generate_answer(query, context)
        
        return SearchResult(
            original_query=query,
            query_variations=[QueryVariation(query, query, "original", query_embedding)],
            summary_matches=[],
            chunk_matches=chunk_matches,
            combined_context=context,
            confidence_score=self._calculate_confidence([], chunk_matches),
            answer=answer,
            sources=self._prepare_sources([], chunk_matches),
            search_strategy=SearchStrategy.DATABASE_DIRECT.value,
            metadata={
                "matches_found": len(chunk_matches),
                "context_length": len(context)
            }
        )
    
    async def _search_cache_strategy(
        self,
        query: str,
        top_k: int,
        threshold: float
    ) -> SearchResult:
        """GPU-cached similarity search"""
        # Generate query embedding
        query_embedding = await embedding_service.generate_embedding(query)
        if not query_embedding:
            return self._create_empty_result(query, SearchStrategy.CACHE_GPU)
        
        # Search using cache
        chunk_matches = await self._search_with_cache(
            query_embedding, top_k, threshold
        )
        
        # Build context and generate answer
        context = self._build_context_from_chunks(chunk_matches)
        answer = self._generate_answer(query, context)
        
        return SearchResult(
            original_query=query,
            query_variations=[QueryVariation(query, query, "original", query_embedding)],
            summary_matches=[],
            chunk_matches=chunk_matches,
            combined_context=context,
            confidence_score=self._calculate_confidence([], chunk_matches),
            answer=answer,
            sources=self._prepare_sources([], chunk_matches),
            search_strategy=SearchStrategy.CACHE_GPU.value,
            metadata={
                "matches_found": len(chunk_matches),
                "context_length": len(context),
                "cache_used": True
            }
        )
    
    async def _search_chunked_strategy(
        self,
        query: str,
        top_k: int,
        threshold: float
    ) -> SearchResult:
        """Chunked fallback search"""
        # Generate query embedding
        query_embedding = await embedding_service.generate_embedding(query)
        if not query_embedding:
            return self._create_empty_result(query, SearchStrategy.CHUNKED)
        
        # Search using chunked approach
        chunk_matches = await self._search_chunked(
            query_embedding, top_k, threshold
        )
        
        # Build context and generate answer
        context = self._build_context_from_chunks(chunk_matches)
        answer = self._generate_answer(query, context)
        
        return SearchResult(
            original_query=query,
            query_variations=[QueryVariation(query, query, "original", query_embedding)],
            summary_matches=[],
            chunk_matches=chunk_matches,
            combined_context=context,
            confidence_score=self._calculate_confidence([], chunk_matches),
            answer=answer,
            sources=self._prepare_sources([], chunk_matches),
            search_strategy=SearchStrategy.CHUNKED.value,
            metadata={
                "matches_found": len(chunk_matches),
                "context_length": len(context)
            }
        )
    
    async def _search_multi_query_strategy(
        self,
        query: str,
        top_k: int,
        threshold: float,
        search_summaries_first: bool,
        top_k_summaries: int
    ) -> SearchResult:
        """Multi-query search with optional hierarchical retrieval"""
        logger.info(f"Starting multi-query search (hierarchical={search_summaries_first})")
        
        # Generate query variations
        query_variations = await self._generate_query_variations(query)
        
        # Generate embeddings for all variations
        await self._generate_embeddings_for_variations(query_variations)
        
        # Search summaries if hierarchical
        summary_matches = []
        if search_summaries_first:
            summary_matches = await self._search_summaries_multi_query(
                query_variations, top_k_summaries, threshold
            )
        
        # Search chunks
        chunk_matches = await self._search_chunks_multi_query(
            query_variations, summary_matches, top_k, threshold
        )
        
        # Build enhanced context
        context = await self._build_enhanced_context(
            summary_matches, chunk_matches
        )
        
        # Calculate confidence
        confidence = self._calculate_multi_query_confidence(
            summary_matches, chunk_matches, query_variations
        )
        
        # Generate answer
        answer = self._generate_enhanced_answer(
            query, context, summary_matches, chunk_matches
        )
        
        # Prepare sources
        sources = self._prepare_enhanced_sources(
            summary_matches, chunk_matches
        )
        
        strategy_name = (SearchStrategy.HIERARCHICAL if search_summaries_first 
                        else SearchStrategy.MULTI_QUERY)
        
        return SearchResult(
            original_query=query,
            query_variations=query_variations,
            summary_matches=summary_matches,
            chunk_matches=chunk_matches,
            combined_context=context,
            confidence_score=confidence,
            answer=answer,
            sources=sources,
            search_strategy=strategy_name.value,
            metadata={
                "matches_found": len(chunk_matches),
                "summaries_found": len(summary_matches),
                "query_variations": len(query_variations),
                "context_length": len(context)
            }
        )
    
    # Query expansion methods
    async def _generate_query_variations(self, query: str) -> List[QueryVariation]:
        """Generate query variations for multi-query search"""
        # Check cache first
        if query in self._query_expansion_cache:
            return self._query_expansion_cache[query]
        
        variation_prompts = [
            self._create_rephrase_prompt(query),
            self._create_technical_prompt(query),
            self._create_context_prompt(query),
            self._create_specific_prompt(query),
            self._create_broader_prompt(query)
        ]
        
        variations = []
        
        for i, prompt in enumerate(variation_prompts):
            try:
                variation_text = self.generate_answer(prompt)
                if variation_text and variation_text.strip() != query.strip():
                    variation_type = [
                        "rephrase", "technical", "contextual", "specific", "broader"
                    ][i]
                    
                    variations.append(QueryVariation(
                        original_query=query,
                        variation=variation_text.strip(),
                        variation_type=variation_type
                    ))
            except Exception as e:
                logger.error(f"Failed to generate query variation {i}: {e}")
        
        # Always include original
        variations.insert(0, QueryVariation(
            original_query=query,
            variation=query,
            variation_type="original"
        ))
        
        # Cache the result
        self._query_expansion_cache[query] = variations
        return variations
    
    def _create_rephrase_prompt(self, query: str) -> str:
        return f"""Rephrase the following question using different words while keeping the same meaning:

Original question: {query}

Rephrased question:"""
    
    def _create_technical_prompt(self, query: str) -> str:
        return f"""Rewrite the following question using more technical or specific terminology:

Original question: {query}

Technical version:"""
    
    def _create_context_prompt(self, query: str) -> str:
        return f"""Expand the following question to include more context and background:

Original question: {query}

Expanded question:"""
    
    def _create_specific_prompt(self, query: str) -> str:
        return f"""Make the following question more specific and detailed:

Original question: {query}

More specific question:"""
    
    def _create_broader_prompt(self, query: str) -> str:
        return f"""Rewrite the following question to be more general and broader in scope:

Original question: {query}

Broader question:"""
    
    async def _generate_embeddings_for_variations(
        self, 
        query_variations: List[QueryVariation]
    ):
        """Generate embeddings for all query variations"""
        texts = [var.variation for var in query_variations]
        embeddings = await embedding_service.generate_embeddings_batch(texts)
        
        for variation, embedding in zip(query_variations, embeddings):
            variation.embedding = embedding
    
    # Search methods
    async def _search_database_direct(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Database-direct vector similarity search"""
        logger.info("Using database-direct vector similarity search")
        
        try:
            # Try async first
            results = await db_ops.similarity_search_database_async(
                query_embedding, top_k, threshold
            )
            
            if results:
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        'id': result['id'],
                        'chunk_id': result.get('chunk_id', result['id']),
                        'content': result['content'],
                        'source_file': result.get('source_file', ''),
                        'chunk_index': result.get('chunk_index', 0),
                        'tags': result.get('tags', []),
                        'cosine_similarity': float(result['cosine_similarity']),
                        'euclidean_distance': 0
                    })
                
                logger.info(f"Found {len(formatted_results)} matches via async database search")
                return formatted_results
            else:
                return []
                
        except Exception as async_e:
            logger.warning(f"Async database search failed ({async_e}), trying sync fallback")
            
            try:
                results = db_ops.similarity_search_database(
                    query_embedding, top_k, threshold
                )
                
                if results:
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            'id': result['id'],
                            'chunk_id': result.get('chunk_id', result['id']),
                            'content': result['content'],
                            'source_file': result.get('source_file', ''),
                            'chunk_index': result.get('chunk_index', 0),
                            'tags': result.get('tags', []),
                            'cosine_similarity': float(result['cosine_similarity']),
                            'euclidean_distance': 0
                        })
                    
                    logger.info(f"Found {len(formatted_results)} matches via sync database search")
                    return formatted_results
                else:
                    return []
                    
            except Exception as sync_e:
                logger.error(f"Both database searches failed: {sync_e}")
                return await self._search_python_fallback(query_embedding, top_k, threshold)
    
    async def _search_with_cache(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """GPU-cached similarity search"""
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
            results = self.vector_ops.similarity_search(
                query_embedding,
                embeddings_tensor,
                top_k=top_k,
                threshold=threshold,
                use_gpu=True
            )
            
            if not results:
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
                        'chunk_id': doc_content.get('chunk_id', doc_id),
                        'content': doc_content['content'],
                        'source_file': doc_content.get('source_file', ''),
                        'chunk_index': doc_content.get('chunk_index', 0),
                        'tags': doc_content.get('tags', []),
                        'cosine_similarity': similarity,
                        'euclidean_distance': 0
                    })
            
            logger.info(f"Found {len(formatted_results)} matches via GPU cache")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Cache search failed: {e}")
            return await self._search_database_direct(query_embedding, top_k, threshold)
        finally:
            # Cleanup GPU memory
            if embeddings_tensor and hasattr(embeddings_tensor, 'is_cuda') and embeddings_tensor.is_cuda:
                import torch
                torch.cuda.empty_cache()
    
    async def _search_chunked(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Chunked fallback search"""
        logger.info("Using chunked fallback search")
        return await self._search_python_fallback(query_embedding, top_k, threshold)
    
    async def _search_python_fallback(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Python-based similarity search as fallback"""
        logger.info("Using Python fallback similarity search")
        
        try:
            if embedding_cache.should_use_cache_for_search():
                cache_data = db_ops.get_embeddings_for_cache(10000)
                if cache_data:
                    ids, embeddings = cache_data
                    
                    # Compute similarities
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
                        doc_ids = [doc_id for doc_id, _ in results]
                        documents = db_ops.get_documents_by_ids(doc_ids)
                        id_to_content = {doc['id']: doc for doc in documents}
                        
                        formatted_results = []
                        for doc_id, similarity in results:
                            doc_content = id_to_content.get(doc_id)
                            if doc_content:
                                formatted_results.append({
                                    'id': doc_id,
                                    'chunk_id': doc_content.get('chunk_id', doc_id),
                                    'content': doc_content['content'],
                                    'source_file': doc_content.get('source_file', ''),
                                    'chunk_index': doc_content.get('chunk_index', 0),
                                    'tags': doc_content.get('tags', []),
                                    'cosine_similarity': similarity,
                                    'euclidean_distance': 0
                                })
                        
                        logger.info(f"Found {len(formatted_results)} matches via Python fallback")
                        return formatted_results
            
            return []
            
        except Exception as e:
            logger.error(f"Python fallback search failed: {e}")
            return []
    
    async def _search_summaries_multi_query(
        self,
        query_variations: List[QueryVariation],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Search summaries using multiple query variations"""
        all_matches = []
        seen_chunk_ids = set()
        
        for variation in query_variations:
            if not variation.embedding:
                continue
            
            try:
                matches = db_ops.similarity_search_summaries(
                    variation.embedding, 
                    top_k=top_k, 
                    threshold=threshold,
                    min_importance=0.5
                )
                
                for match in matches:
                    chunk_id = match['chunk_id']
                    if chunk_id not in seen_chunk_ids:
                        match['matched_query'] = variation.variation
                        match['query_type'] = variation.variation_type
                        all_matches.append(match)
                        seen_chunk_ids.add(chunk_id)
                        
            except Exception as e:
                logger.error(f"Summary search failed for variation '{variation.variation}': {e}")
        
        # Sort by importance and similarity
        all_matches.sort(
            key=lambda x: (x.get('importance_score', 0) * 0.7 + x.get('cosine_similarity', 0) * 0.3),
            reverse=True
        )
        
        return all_matches[:top_k]
    
    async def _search_chunks_multi_query(
        self,
        query_variations: List[QueryVariation],
        summary_matches: List[Dict[str, Any]],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Search chunks using multiple queries"""
        # Determine search strategy
        relevant_files = None
        if summary_matches:
            relevant_files = list(set(match['source_file'] for match in summary_matches))
            logger.info(f"Focusing chunk search on {len(relevant_files)} relevant files")
        
        all_matches = []
        seen_chunk_ids = set()
        
        for variation in query_variations:
            if not variation.embedding:
                continue
            
            try:
                matches = db_ops.similarity_search_chunks(
                    variation.embedding,
                    top_k=top_k,
                    threshold=threshold,
                    source_files=relevant_files
                )
                
                for match in matches:
                    chunk_id = match.get('chunk_id', match.get('id'))
                    if chunk_id not in seen_chunk_ids:
                        match['matched_query'] = variation.variation
                        match['query_type'] = variation.variation_type
                        all_matches.append(match)
                        seen_chunk_ids.add(chunk_id)
                        
            except Exception as e:
                logger.error(f"Chunk search failed for variation '{variation.variation}': {e}")
        
        # Sort by similarity
        all_matches.sort(key=lambda x: x.get('cosine_similarity', 0), reverse=True)
        
        return all_matches[:top_k]
    
    # Context building methods
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Build basic context from chunks"""
        context_parts = []
        
        for idx, chunk in enumerate(chunks[:15]):  # Limit to prevent overflow
            truncated_content = chunk['content'][:800]
            context_parts.append(
                f"[Doc {idx+1}] (Similarity: {chunk.get('cosine_similarity', 0):.4f})\n"
                f"{truncated_content}"
            )
        
        return "\n".join(context_parts)
    
    async def _build_enhanced_context(
        self,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced context with hierarchical information"""
        context_parts = []
        
        # Add summary context first
        if summary_matches:
            context_parts.append("=== HIGH-LEVEL OVERVIEW ===")
            for i, summary in enumerate(summary_matches[:3]):
                context_parts.append(
                    f"[Summary {i+1}] ({summary.get('importance_score', 0):.2f} importance)\n"
                    f"File: {summary['source_file']}\n"
                    f"{summary['summary']}"
                )
        
        # Add detailed chunk context
        if chunk_matches:
            context_parts.append("\n=== DETAILED INFORMATION ===")
            
            # Get related chunks for better context
            expanded_chunks = db_ops.reconstruct_context_from_chunks(
                chunk_matches[:10], expand_context=True
            )
            
            for i, chunk in enumerate(expanded_chunks[:15]):
                context_parts.append(
                    f"[Detail {i+1}] (Similarity: {chunk.get('cosine_similarity', 0):.3f})\n"
                    f"File: {chunk.get('source_file', 'Unknown')} [Chunk {chunk.get('chunk_index', 0)}]\n"
                    f"{chunk['content'][:600]}..."
                )
        
        return "\n\n".join(context_parts)
    
    # Confidence calculation methods
    def _calculate_confidence(
        self,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]]
    ) -> float:
        """Calculate basic confidence score"""
        score = 0.0
        
        # Base score from matches
        score += min(len(chunk_matches) * 0.1, 1.0)
        
        # Quality score from similarity
        if chunk_matches:
            avg_sim = sum(m.get('cosine_similarity', 0) for m in chunk_matches[:10]) / min(10, len(chunk_matches))
            score += avg_sim * 3.0
        
        # Normalize to 0-10
        return min(score, 10.0)
    
    def _calculate_multi_query_confidence(
        self,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]],
        query_variations: List[QueryVariation]
    ) -> float:
        """Calculate confidence score for multi-query search"""
        score = 0.0
        
        # Base score from number of matches
        score += min(len(summary_matches) * 0.3, 1.0)
        score += min(len(chunk_matches) * 0.1, 1.0)
        
        # Quality score from similarity values
        if summary_matches:
            avg_summary_sim = sum(m.get('cosine_similarity', 0) for m in summary_matches[:5]) / min(5, len(summary_matches))
            score += avg_summary_sim * 2.0
        
        if chunk_matches:
            avg_chunk_sim = sum(m.get('cosine_similarity', 0) for m in chunk_matches[:10]) / min(10, len(chunk_matches))
            score += avg_chunk_sim * 1.5
        
        # Bonus for high-importance summaries
        high_importance = sum(1 for m in summary_matches if m.get('importance_score', 0) > 7.0)
        score += high_importance * 0.5
        
        # Bonus for multi-query consensus
        query_types = set(m.get('query_type', 'original') for m in chunk_matches + summary_matches)
        if len(query_types) > 2:
            score += 0.5
        
        # Normalize to 0-10 scale
        return min(score, 10.0)
    
    # Answer generation methods
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using basic context"""
        prompt = f"""Answer the question using only the context below. If unsure, say 'I don't know'.

Question: {query}

Context (ranked by relevance):
{context}
"""
        return self.generate_answer(prompt)
    
    def _generate_enhanced_answer(
        self,
        original_query: str,
        context: str,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using enhanced context"""
        prompt = f"""You are an expert assistant with access to both high-level summaries and detailed information. 
Answer the question using the provided context, leveraging both overview and specific details.

Question: {original_query}

Context Information:
{context}

Instructions:
1. Use the high-level overview to understand the broader context
2. Use detailed information to provide specific, accurate answers
3. Synthesize information from multiple sources when relevant
4. If information is contradictory, explain the different perspectives
5. Be specific about which sources support your conclusions

Answer:"""
        
        return self.generate_answer(prompt)
    
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
    
    # Source preparation methods
    def _prepare_sources(
        self,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare basic source information"""
        sources = []
        
        for chunk in chunk_matches[:10]:
            sources.append({
                "type": "chunk",
                "id": chunk.get('id'),
                "chunk_id": chunk.get('chunk_id', chunk.get('id')),
                "source_file": chunk.get('source_file', ''),
                "similarity": chunk.get('cosine_similarity', 0),
                "content_preview": chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            })
        
        return sources
    
    def _prepare_enhanced_sources(
        self,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare enhanced source information"""
        sources = []
        
        # Add summary sources
        for summary in summary_matches[:5]:
            sources.append({
                "type": "summary",
                "chunk_id": summary['chunk_id'],
                "source_file": summary['source_file'],
                "similarity": summary.get('cosine_similarity', 0),
                "importance": summary.get('importance_score', 0),
                "content_preview": summary['summary'][:200] + "...",
                "matched_query": summary.get('matched_query', ''),
                "query_type": summary.get('query_type', 'original')
            })
        
        # Add chunk sources
        for chunk in chunk_matches[:10]:
            sources.append({
                "type": "chunk",
                "chunk_id": chunk.get('chunk_id', chunk.get('id')),
                "source_file": chunk.get('source_file', ''),
                "similarity": chunk.get('cosine_similarity', 0),
                "chunk_index": chunk.get('chunk_index', 0),
                "content_preview": chunk['content'][:200] + "...",
                "matched_query": chunk.get('matched_query', ''),
                "query_type": chunk.get('query_type', 'original')
            })
        
        return sources
    
    # Utility methods
    def _create_empty_result(self, query: str, strategy: SearchStrategy) -> SearchResult:
        """Create an empty result when no matches found"""
        return SearchResult(
            original_query=query,
            query_variations=[QueryVariation(query, query, "original")],
            summary_matches=[],
            chunk_matches=[],
            combined_context="",
            confidence_score=0.0,
            answer="I don't have enough information to answer this question.",
            sources=[],
            search_strategy=strategy.value,
            metadata={"matches_found": 0}
        )
    
    # Backward compatibility methods
    async def ask_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Backward compatibility wrapper for original interface"""
        result = await self.search(
            question,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            use_cache=use_cache
        )
        
        return {
            "answer": result.answer,
            "sources": result.sources,
            "metadata": result.metadata
        }
    
    async def semantic_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Backward compatibility wrapper for semantic search"""
        result = await self.search(
            query,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            use_cache=use_cache
        )
        
        return result.chunk_matches
    
    async def enhanced_search(
        self,
        query: str,
        use_multi_query: bool = True,
        search_summaries_first: bool = True,
        top_k_summaries: int = 10,
        top_k_chunks: int = 25,
        relevance_threshold: float = 0.3
    ) -> Any:
        """Backward compatibility wrapper for enhanced search"""
        result = await self.search(
            query,
            strategy=SearchStrategy.HIERARCHICAL if search_summaries_first else SearchStrategy.MULTI_QUERY,
            top_k=top_k_chunks,
            relevance_threshold=relevance_threshold,
            use_multi_query=use_multi_query,
            search_summaries_first=search_summaries_first,
            top_k_summaries=top_k_summaries
        )
        
        # Return in the old MultiQueryResult format for compatibility
        return result
    
    # Information methods
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
                },
                "multi_query": {
                    "description": "Expand query into multiple variations for better recall",
                    "best_for": "Complex queries requiring different perspectives",
                    "pros": ["Better recall", "Handles ambiguous queries"],
                    "cons": ["Slower due to multiple searches", "More API calls"]
                },
                "hierarchical": {
                    "description": "Search summaries first, then relevant chunks",
                    "best_for": "Large documents with good summarization",
                    "pros": ["Efficient for large documents", "Better context understanding"],
                    "cons": ["Requires summaries", "Two-stage process"]
                }
            },
            "recommended_strategy": self._choose_optimal_strategy(None, None, None, None).value
        }
    
    async def benchmark_search_methods(
        self, 
        test_query: str = "test query",
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Benchmark different search methods for performance comparison"""
        import time
        
        query_embedding = await embedding_service.generate_embedding(test_query)
        if not query_embedding:
            return {"error": "Failed to generate test embedding"}
        
        results = {}
        strategies = [
            SearchStrategy.DATABASE_DIRECT,
            SearchStrategy.CACHE_GPU if embedding_cache.should_use_cache_for_search() else None,
            SearchStrategy.MULTI_QUERY,
            SearchStrategy.HIERARCHICAL
        ]
        
        for strategy in strategies:
            if strategy is None:
                continue
                
            try:
                times = []
                for i in range(iterations):
                    start = time.time()
                    await self.search(
                        test_query,
                        strategy=strategy,
                        top_k=10,
                        relevance_threshold=0.1
                    )
                    times.append(time.time() - start)
                
                results[strategy.value] = {
                    "avg_time_ms": sum(times) / len(times) * 1000,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000
                }
            except Exception as e:
                results[strategy.value] = {"error": str(e)}
        
        # Add recommendations
        best_method = min(
            [(k, v.get("avg_time_ms", float('inf'))) for k, v in results.items() 
             if isinstance(v, dict) and "avg_time_ms" in v],
            key=lambda x: x[1],
            default=("database_direct", 0)
        )[0]
        
        results["recommendation"] = {
            "best_method": best_method,
            "database_size": db_ops.get_document_count(),
            "test_iterations": iterations
        }
        
        return results

# Global instances for backward compatibility
inference_engine = UnifiedInferenceEngine()
enhanced_inference_engine = UnifiedInferenceEngine()  # Same instance, different name for compatibility
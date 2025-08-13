# inference/multi_query.py - Multi-Query Inference Engine
import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

from core.config import config
from inference.embeddings import embedding_service
from inference.engine import InferenceEngine
from database.operations import db_ops

logger = logging.getLogger(__name__)

@dataclass
class QueryVariation:
    """Represents a variation of the original query"""
    original_query: str
    variation: str
    variation_type: str
    embedding: Optional[List[float]] = None

@dataclass
class MultiQueryResult:
    """Results from multi-query search"""
    original_query: str
    query_variations: List[QueryVariation]
    summary_matches: List[Dict[str, Any]]
    chunk_matches: List[Dict[str, Any]]
    combined_context: str
    confidence_score: float
    answer: str
    sources: List[Dict[str, Any]]

class QueryExpansionService:
    """Service for generating query variations"""
    
    def __init__(self):
        self.inference_engine = InferenceEngine()
    
    async def generate_query_variations(self, original_query: str) -> List[QueryVariation]:
        """Generate 5 different variations of the original query"""
        
        variation_prompts = [
            self._create_rephrase_prompt(original_query),
            self._create_technical_prompt(original_query),
            self._create_context_prompt(original_query),
            self._create_specific_prompt(original_query),
            self._create_broader_prompt(original_query)
        ]
        
        variations = []
        
        for i, prompt in enumerate(variation_prompts):
            try:
                variation_text = self.inference_engine.generate_answer(prompt)
                if variation_text and variation_text.strip() != original_query.strip():
                    variation_type = [
                        "rephrase", "technical", "contextual", "specific", "broader"
                    ][i]
                    
                    variations.append(QueryVariation(
                        original_query=original_query,
                        variation=variation_text.strip(),
                        variation_type=variation_type
                    ))
            except Exception as e:
                logger.error(f"Failed to generate query variation {i}: {e}")
        
        # Always include the original query
        variations.insert(0, QueryVariation(
            original_query=original_query,
            variation=original_query,
            variation_type="original"
        ))
        
        return variations
    
    def _create_rephrase_prompt(self, query: str) -> str:
        """Create prompt for rephrasing the query"""
        return f"""Rephrase the following question using different words while keeping the same meaning:

Original question: {query}

Rephrased question:"""
    
    def _create_technical_prompt(self, query: str) -> str:
        """Create prompt for a more technical version"""
        return f"""Rewrite the following question using more technical or specific terminology:

Original question: {query}

Technical version:"""
    
    def _create_context_prompt(self, query: str) -> str:
        """Create prompt for adding context"""
        return f"""Expand the following question to include more context and background:

Original question: {query}

Expanded question:"""
    
    def _create_specific_prompt(self, query: str) -> str:
        """Create prompt for a more specific version"""
        return f"""Make the following question more specific and detailed:

Original question: {query}

More specific question:"""
    
    def _create_broader_prompt(self, query: str) -> str:
        """Create prompt for a broader version"""
        return f"""Rewrite the following question to be more general and broader in scope:

Original question: {query}

Broader question:"""

class MultiQueryInferenceEngine:
    """Enhanced inference engine with multi-query and hierarchical search"""
    
    def __init__(self):
        self.query_expansion = QueryExpansionService()
        self.base_engine = InferenceEngine()
    
    async def enhanced_search(
        self,
        query: str,
        use_multi_query: bool = True,
        search_summaries_first: bool = True,
        top_k_summaries: int = 10,
        top_k_chunks: int = 25,
        relevance_threshold: float = 0.3
    ) -> MultiQueryResult:
        """
        Enhanced search with multi-query expansion and hierarchical retrieval
        """
        logger.info(f"Starting enhanced search for: {query}")
        
        # Step 1: Generate query variations
        query_variations = []
        if use_multi_query:
            query_variations = await self.query_expansion.generate_query_variations(query)
        else:
            # Use only original query
            query_variations = [QueryVariation(
                original_query=query,
                variation=query,
                variation_type="original"
            )]
        
        # Step 2: Generate embeddings for all query variations
        await self._generate_embeddings_for_variations(query_variations)
        
        # Step 3: Search summaries first for high-level matches
        summary_matches = []
        if search_summaries_first:
            summary_matches = await self._search_summaries_multi_query(
                query_variations, top_k_summaries, relevance_threshold
            )
        
        # Step 4: Search chunks (either from summary context or full search)
        chunk_matches = await self._search_chunks_multi_query(
            query_variations, summary_matches, top_k_chunks, relevance_threshold
        )
        
        # Step 5: Reconstruct context with related chunks
        combined_context = await self._build_enhanced_context(
            summary_matches, chunk_matches
        )
        
        # Step 6: Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            summary_matches, chunk_matches, query_variations
        )
        
        # Step 7: Generate final answer
        answer = self._generate_enhanced_answer(
            query, combined_context, summary_matches, chunk_matches
        )
        
        # Step 8: Prepare sources
        sources = self._prepare_enhanced_sources(summary_matches, chunk_matches)
        
        result = MultiQueryResult(
            original_query=query,
            query_variations=query_variations,
            summary_matches=summary_matches,
            chunk_matches=chunk_matches,
            combined_context=combined_context,
            confidence_score=confidence_score,
            answer=answer,
            sources=sources
        )
        
        logger.info(f"Enhanced search completed with confidence: {confidence_score:.2f}")
        return result
    
    async def _generate_embeddings_for_variations(
        self, 
        query_variations: List[QueryVariation]
    ):
        """Generate embeddings for all query variations"""
        texts = [var.variation for var in query_variations]
        embeddings = await embedding_service.generate_embeddings_batch(texts)
        
        for variation, embedding in zip(query_variations, embeddings):
            variation.embedding = embedding
    
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
                    min_importance=0.5  # Only get important summaries
                )
                
                # Add variation info and deduplicate
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
        """Search chunks using multiple queries, optionally filtered by summary results"""
        
        # Determine search strategy
        if summary_matches:
            # Focus search on files that had summary matches
            relevant_files = list(set(match['source_file'] for match in summary_matches))
            logger.info(f"Focusing chunk search on {len(relevant_files)} relevant files")
        else:
            relevant_files = None
        
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
                
                # Add variation info and deduplicate
                for match in matches:
                    chunk_id = match['chunk_id']
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
    
    async def _build_enhanced_context(
        self,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced context with hierarchical information"""
        context_parts = []
        
        # Add summary context first (high-level overview)
        if summary_matches:
            context_parts.append("=== HIGH-LEVEL OVERVIEW ===")
            for i, summary in enumerate(summary_matches[:3]):  # Top 3 summaries
                context_parts.append(
                    f"[Summary {i+1}] ({summary.get('importance_score', 0):.2f} importance)\n"
                    f"File: {summary['source_file']}\n"
                    f"{summary['summary']}"
                )
        
        # Add detailed chunk context
        if chunk_matches:
            context_parts.append("\n=== DETAILED INFORMATION ===")
            
            # Get related chunks for better context
            chunk_ids = [chunk['chunk_id'] for chunk in chunk_matches[:10]]
            expanded_chunks = db_ops.reconstruct_context_from_chunks(chunk_matches[:10], expand_context=True)
            
            for i, chunk in enumerate(expanded_chunks[:15]):  # Limit to prevent context overflow
                context_parts.append(
                    f"[Detail {i+1}] (Similarity: {chunk.get('cosine_similarity', 0):.3f})\n"
                    f"File: {chunk['source_file']} [Chunk {chunk.get('chunk_index', 0)}]\n"
                    f"{chunk['content'][:600]}..."  # Truncate for context
                )
        
        return "\n\n".join(context_parts)
    
    def _calculate_confidence_score(
        self,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]],
        query_variations: List[QueryVariation]
    ) -> float:
        """Calculate confidence score based on match quality"""
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
            score += 0.5  # Multiple query types found matches
        
        # Normalize to 0-10 scale
        return min(score, 10.0)
    
    def _generate_enhanced_answer(
        self,
        original_query: str,
        context: str,
        summary_matches: List[Dict[str, Any]],
        chunk_matches: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using enhanced context"""
        # Create enhanced prompt with hierarchical context
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
        
        return self.base_engine.generate_answer(prompt)
    
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
                "chunk_id": chunk['chunk_id'],
                "source_file": chunk['source_file'],
                "similarity": chunk.get('cosine_similarity', 0),
                "chunk_index": chunk.get('chunk_index', 0),
                "content_preview": chunk['content'][:200] + "...",
                "matched_query": chunk.get('matched_query', ''),
                "query_type": chunk.get('query_type', 'original')
            })
        
        return sources
    
    # Backward compatibility method
    async def ask_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Backward compatibility wrapper"""
        # Use enhanced search with multi-query disabled for compatibility
        result = await self.enhanced_search(
            question,
            use_multi_query=False,
            top_k_chunks=top_k or 25,
            relevance_threshold=relevance_threshold or 0.3
        )
        
        return {
            "answer": result.answer,
            "sources": result.sources,
            "metadata": {
                "matches_found": len(result.chunk_matches),
                "confidence_score": result.confidence_score,
                "search_strategy": "enhanced_single_query"
            }
        }

# Global enhanced inference engine
enhanced_inference_engine = MultiQueryInferenceEngine()
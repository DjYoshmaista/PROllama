# inference/summarization.py - Text Summarization Service
import json
import logging
import requests
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.config import config
from file_management.chunking import TextChunk

logger = logging.getLogger(__name__)

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

class SummarizationService:
    """Service for creating intelligent summaries of text chunks"""
    
    def __init__(self):
        self.api_url = config.embedding.api_url
        self.model = config.inference.generation_model
        self.max_chunk_length = 2000  # Max characters to send for summarization
    
    async def summarize_chunks(self, chunks: List[TextChunk]) -> List[ChunkSummary]:
        """Summarize a list of text chunks"""
        if not chunks:
            return []
        
        summaries = []
        
        # Group chunks by source file for better context
        chunks_by_file = {}
        for chunk in chunks:
            if chunk.source_file not in chunks_by_file:
                chunks_by_file[chunk.source_file] = []
            chunks_by_file[chunk.source_file].append(chunk)
        
        # Process each file's chunks
        for source_file, file_chunks in chunks_by_file.items():
            file_summaries = await self._summarize_file_chunks(source_file, file_chunks)
            summaries.extend(file_summaries)
        
        return summaries
    
    async def _summarize_file_chunks(
        self, 
        source_file: str, 
        chunks: List[TextChunk]
    ) -> List[ChunkSummary]:
        """Summarize chunks from a single file with context awareness"""
        summaries = []
        
        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: x.chunk_index)
        
        for chunk in sorted_chunks:
            try:
                summary = await self._create_chunk_summary(chunk, sorted_chunks)
                if summary:
                    summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to summarize chunk {chunk.chunk_id}: {e}")
                # Create fallback summary
                fallback_summary = self._create_fallback_summary(chunk)
                summaries.append(fallback_summary)
        
        return summaries
    
    async def _create_chunk_summary(
        self, 
        chunk: TextChunk, 
        all_chunks: List[TextChunk]
    ) -> Optional[ChunkSummary]:
        """Create a detailed summary for a single chunk"""
        
        # Prepare context from surrounding chunks
        context = self._build_context_for_chunk(chunk, all_chunks)
        
        # Truncate content if too long
        content_to_summarize = chunk.content
        if len(content_to_summarize) > self.max_chunk_length:
            content_to_summarize = content_to_summarize[:self.max_chunk_length] + "..."
        
        # Create summarization prompt
        prompt = self._create_summarization_prompt(content_to_summarize, context, chunk)
        
        # Generate summary
        summary_text = await self._generate_summary(prompt)
        if not summary_text:
            return None
        
        # Extract key topics
        key_topics = await self._extract_key_topics(content_to_summarize)
        
        # Calculate importance score
        importance_score = self._calculate_importance_score(chunk, summary_text, key_topics)
        
        # Find related chunks
        related_chunk_ids = self._find_related_chunks(chunk, all_chunks)
        
        return ChunkSummary(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            summary=summary_text,
            key_topics=key_topics,
            chunk_indices=[chunk.chunk_index],
            related_chunk_ids=related_chunk_ids,
            original_length=len(chunk.content),
            summary_length=len(summary_text),
            importance_score=importance_score
        )
    
    def _build_context_for_chunk(
        self, 
        target_chunk: TextChunk, 
        all_chunks: List[TextChunk]
    ) -> str:
        """Build context information for better summarization"""
        context_parts = []
        
        # Add file context
        context_parts.append(f"Source: {target_chunk.source_file}")
        context_parts.append(f"Chunk {target_chunk.chunk_index + 1} of {target_chunk.total_chunks}")
        
        # Add surrounding context
        prev_chunk = None
        next_chunk = None
        
        for chunk in all_chunks:
            if chunk.chunk_index == target_chunk.chunk_index - 1:
                prev_chunk = chunk
            elif chunk.chunk_index == target_chunk.chunk_index + 1:
                next_chunk = chunk
        
        if prev_chunk:
            context_parts.append(f"Previous context: ...{prev_chunk.content[-100:]}")
        
        if next_chunk:
            context_parts.append(f"Following context: {next_chunk.content[:100]}...")
        
        return "\n".join(context_parts)
    
    def _create_summarization_prompt(
        self, 
        content: str, 
        context: str, 
        chunk: TextChunk
    ) -> str:
        """Create an effective summarization prompt"""
        return f"""You are an expert at creating concise, informative summaries. 

Context Information:
{context}

Text to Summarize:
{content}

Create a comprehensive summary that:
1. Captures the main ideas and key points
2. Maintains important details and context
3. Is roughly 1/4 the length of the original
4. Uses clear, professional language
5. Preserves technical terms and proper nouns
6. Indicates relationships to surrounding content

Summary:"""
    
    async def _generate_summary(self, prompt: str) -> Optional[str]:
        """Generate summary using the LLM"""
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more focused summaries
                        "max_tokens": 512,
                        "top_p": 0.9
                    }
                },
                stream=True,
                timeout=60
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
                        continue
            
            return full_response.strip() if full_response else None
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return None
    
    async def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        topic_prompt = f"""Extract 3-5 key topics or themes from the following text. 
Return only the topics as a comma-separated list, no explanations.

Text:
{content[:1000]}

Key topics:"""
        
        try:
            topics_text = await self._generate_summary(topic_prompt)
            if topics_text:
                # Parse comma-separated topics
                topics = [topic.strip() for topic in topics_text.split(",")]
                # Filter and clean topics
                topics = [topic for topic in topics if topic and len(topic) > 2 and len(topic) < 50]
                return topics[:5]  # Limit to 5 topics
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
        
        return []
    
    def _calculate_importance_score(
        self, 
        chunk: TextChunk, 
        summary: str, 
        key_topics: List[str]
    ) -> float:
        """Calculate importance score for the chunk"""
        score = 0.0
        
        # Base score from content length
        score += min(len(chunk.content) / 1000, 2.0)
        
        # Boost for key topics
        score += len(key_topics) * 0.5
        
        # Boost for technical content
        technical_indicators = [
            'class ', 'def ', 'function', 'import', 'from', 'return',
            'error', 'exception', 'algorithm', 'method', 'parameter',
            'configuration', 'setting', 'option', 'property'
        ]
        
        content_lower = chunk.content.lower()
        technical_count = sum(1 for indicator in technical_indicators if indicator in content_lower)
        score += technical_count * 0.2
        
        # Boost for structural elements (headers, lists, etc.)
        structural_indicators = ['#', '##', '###', '- ', '* ', '1.', '2.']
        structural_count = sum(1 for indicator in structural_indicators if indicator in chunk.content)
        score += structural_count * 0.1
        
        # Normalize score to 0-10 range
        return min(score, 10.0)
    
    def _find_related_chunks(
        self, 
        target_chunk: TextChunk, 
        all_chunks: List[TextChunk]
    ) -> List[str]:
        """Find chunks that are related to the target chunk"""
        related = []
        
        # Add parent and child chunks
        related.extend(target_chunk.parent_chunks)
        related.extend(target_chunk.child_chunks)
        
        # Add chunks with overlapping content (simple heuristic)
        target_words = set(target_chunk.content.lower().split())
        
        for chunk in all_chunks:
            if chunk.chunk_id == target_chunk.chunk_id:
                continue
            
            # Skip immediate neighbors (already added as parent/child)
            if abs(chunk.chunk_index - target_chunk.chunk_index) <= 1:
                continue
            
            chunk_words = set(chunk.content.lower().split())
            overlap = len(target_words & chunk_words)
            
            # If significant overlap, consider related
            if overlap > 20 and overlap / len(target_words) > 0.3:
                related.append(chunk.chunk_id)
        
        return related[:5]  # Limit to 5 related chunks
    
    def _create_fallback_summary(self, chunk: TextChunk) -> ChunkSummary:
        """Create a basic fallback summary when AI summarization fails"""
        # Simple extractive summary - first and last sentences
        sentences = chunk.content.split('.')
        if len(sentences) > 2:
            summary = sentences[0] + '. ' + sentences[-2] + '.'
        else:
            summary = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        
        return ChunkSummary(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            summary=summary,
            key_topics=["general"],
            chunk_indices=[chunk.chunk_index],
            related_chunk_ids=[],
            original_length=len(chunk.content),
            summary_length=len(summary),
            importance_score=1.0
        )
    
    async def create_hierarchical_summaries(
        self, 
        summaries: List[ChunkSummary]
    ) -> List[ChunkSummary]:
        """Create higher-level summaries by grouping related summaries"""
        if len(summaries) <= 1:
            return summaries
        
        # Group summaries by file and importance
        grouped_summaries = []
        
        summaries_by_file = {}
        for summary in summaries:
            if summary.source_file not in summaries_by_file:
                summaries_by_file[summary.source_file] = []
            summaries_by_file[summary.source_file].append(summary)
        
        # Create file-level summaries for files with many chunks
        for source_file, file_summaries in summaries_by_file.items():
            if len(file_summaries) > 10:  # Only create meta-summaries for large files
                meta_summary = await self._create_meta_summary(source_file, file_summaries)
                if meta_summary:
                    grouped_summaries.append(meta_summary)
        
        return summaries + grouped_summaries
    
    async def _create_meta_summary(
        self, 
        source_file: str, 
        summaries: List[ChunkSummary]
    ) -> Optional[ChunkSummary]:
        """Create a meta-summary from multiple chunk summaries"""
        # Combine all summaries
        combined_text = "\n\n".join(summary.summary for summary in summaries)
        
        if len(combined_text) > self.max_chunk_length:
            combined_text = combined_text[:self.max_chunk_length] + "..."
        
        # Create meta-summary prompt
        prompt = f"""Create a high-level summary of the following document summaries from {source_file}:

{combined_text}

Create a comprehensive overview that:
1. Captures the main themes and structure of the entire document
2. Highlights the most important concepts
3. Shows relationships between different sections
4. Maintains key technical details
5. Is roughly 300-500 words

Document Overview:"""
        
        try:
            meta_summary_text = await self._generate_summary(prompt)
            if not meta_summary_text:
                return None
            
            # Collect all topics and related chunks
            all_topics = []
            all_chunk_ids = []
            all_indices = []
            
            for summary in summaries:
                all_topics.extend(summary.key_topics)
                all_chunk_ids.append(summary.chunk_id)
                all_indices.extend(summary.chunk_indices)
            
            # Remove duplicates and limit
            unique_topics = list(dict.fromkeys(all_topics))[:10]
            
            return ChunkSummary(
                chunk_id=f"{source_file}_meta_summary",
                source_file=source_file,
                summary=meta_summary_text,
                key_topics=unique_topics,
                chunk_indices=sorted(set(all_indices)),
                related_chunk_ids=all_chunk_ids,
                original_length=sum(s.original_length for s in summaries),
                summary_length=len(meta_summary_text),
                importance_score=8.0  # High importance for meta-summaries
            )
            
        except Exception as e:
            logger.error(f"Meta-summary creation failed: {e}")
            return None

# Global summarization service
summarization_service = SummarizationService()
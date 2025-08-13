# file_management/chunking.py - Intelligent Text Chunking
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    chunk_id: str
    source_file: str
    chunk_index: int
    total_chunks: int
    start_token: int
    end_token: int
    overlap_before: str = ""
    overlap_after: str = ""
    parent_chunks: List[str] = None
    child_chunks: List[str] = None
    
    def __post_init__(self):
        if self.parent_chunks is None:
            self.parent_chunks = []
        if self.child_chunks is None:
            self.child_chunks = []

class IntelligentTextSplitter:
    """
    Intelligent text splitting with semantic awareness and token overlaps
    """
    
    def __init__(
        self, 
        chunk_size: int = 512, 
        overlap_tokens: int = 32,
        model_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(model_name)
        
        # Hierarchical splitting patterns (in order of preference)
        self.split_patterns = [
            # Document sections
            (r'\n\n#{1,6}\s+', 'section_header'),
            (r'\n\n\*\*[^*]+\*\*\n', 'markdown_header'),
            
            # Paragraphs and logical breaks
            (r'\n\n\n+', 'triple_newline'),
            (r'\n\n', 'double_newline'),
            
            # Sentence boundaries
            (r'(?<=[.!?])\s+(?=[A-Z])', 'sentence_end'),
            
            # Code blocks and special sections
            (r'\n```[\s\S]*?\n```\n', 'code_block'),
            (r'\n---+\n', 'horizontal_rule'),
            
            # List items
            (r'\n(?=\s*[-*+]\s)', 'list_item'),
            (r'\n(?=\s*\d+\.\s)', 'numbered_list'),
            
            # Fallback splits
            (r'\n(?=\s)', 'any_newline'),
            (r'\s+', 'whitespace'),
        ]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to approximate word count * 1.3
            return int(len(text.split()) * 1.3)
    
    def get_token_substring(self, text: str, start_token: int, end_token: int) -> str:
        """Get substring based on token positions"""
        try:
            tokens = self.tokenizer.encode(text)
            if start_token >= len(tokens):
                return ""
            
            end_token = min(end_token, len(tokens))
            subset_tokens = tokens[start_token:end_token]
            return self.tokenizer.decode(subset_tokens)
        except Exception as e:
            logger.warning(f"Token substring failed: {e}")
            # Fallback to character-based approximation
            char_start = int(start_token * len(text) / self.count_tokens(text))
            char_end = int(end_token * len(text) / self.count_tokens(text))
            return text[char_start:char_end]
    
    def split_text_hierarchical(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Split text using hierarchical patterns with semantic awareness
        """
        total_tokens = self.count_tokens(text)
        
        # If text is small enough, return as single chunk
        if total_tokens <= self.chunk_size:
            return [TextChunk(
                content=text,
                chunk_id=f"{source_file}_chunk_0",
                source_file=source_file,
                chunk_index=0,
                total_chunks=1,
                start_token=0,
                end_token=total_tokens
            )]
        
        # Find optimal split points
        split_points = self._find_split_points(text)
        
        # Create chunks with overlaps
        chunks = self._create_overlapped_chunks(text, split_points, source_file)
        
        # Add cross-references
        self._add_cross_references(chunks)
        
        return chunks
    
    def _find_split_points(self, text: str) -> List[Tuple[int, str, str]]:
        """Find optimal split points using hierarchical patterns"""
        split_points = []
        
        for pattern, split_type in self.split_patterns:
            matches = list(re.finditer(pattern, text))
            for match in matches:
                position = match.start()
                split_points.append((position, split_type, match.group()))
        
        # Sort by position and remove duplicates
        split_points = sorted(list(set(split_points)))
        
        return split_points
    
    def _create_overlapped_chunks(
        self, 
        text: str, 
        split_points: List[Tuple[int, str, str]], 
        source_file: str
    ) -> List[TextChunk]:
        """Create chunks with token-based overlaps"""
        chunks = []
        total_tokens = self.count_tokens(text)
        current_token = 0
        chunk_index = 0
        
        while current_token < total_tokens:
            # Determine chunk boundaries
            chunk_start = max(0, current_token - self.overlap_tokens if chunk_index > 0 else 0)
            chunk_end = min(total_tokens, current_token + self.chunk_size)
            
            # Find the best split point near chunk_end
            best_split = self._find_best_split_near_position(
                text, split_points, chunk_end, total_tokens
            )
            
            if best_split and best_split < total_tokens - 50:  # Don't split too close to end
                chunk_end = best_split
            
            # Extract chunk content
            chunk_content = self.get_token_substring(text, chunk_start, chunk_end)
            
            # Extract overlaps
            overlap_before = ""
            overlap_after = ""
            
            if chunk_start > 0:
                overlap_before = self.get_token_substring(
                    text, chunk_start, chunk_start + self.overlap_tokens
                )
            
            if chunk_end < total_tokens:
                overlap_after = self.get_token_substring(
                    text, chunk_end - self.overlap_tokens, chunk_end
                )
            
            # Create chunk
            chunk = TextChunk(
                content=chunk_content,
                chunk_id=f"{source_file}_chunk_{chunk_index}",
                source_file=source_file,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                start_token=chunk_start,
                end_token=chunk_end,
                overlap_before=overlap_before,
                overlap_after=overlap_after
            )
            
            chunks.append(chunk)
            
            # Move to next chunk
            current_token = chunk_end - self.overlap_tokens
            chunk_index += 1
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _find_best_split_near_position(
        self, 
        text: str, 
        split_points: List[Tuple[int, str, str]], 
        target_position: int,
        total_tokens: int
    ) -> Optional[int]:
        """Find the best split point near target position"""
        # Convert token position to character position (approximate)
        target_char = int(target_position * len(text) / total_tokens)
        
        # Look for split points within reasonable range
        search_range = int(len(text) * 0.1)  # 10% of text length
        
        best_split = None
        best_distance = float('inf')
        best_priority = float('inf')
        
        # Priority order for split types
        priority_map = {
            'section_header': 1,
            'markdown_header': 2,
            'triple_newline': 3,
            'double_newline': 4,
            'sentence_end': 5,
            'code_block': 6,
            'horizontal_rule': 7,
            'list_item': 8,
            'numbered_list': 9,
            'any_newline': 10,
            'whitespace': 11
        }
        
        for char_pos, split_type, match_text in split_points:
            if abs(char_pos - target_char) <= search_range:
                distance = abs(char_pos - target_char)
                priority = priority_map.get(split_type, 12)
                
                # Prefer closer splits, but heavily weight by type priority
                score = priority * 1000 + distance
                
                if score < (best_priority * 1000 + best_distance):
                    best_split = char_pos
                    best_distance = distance
                    best_priority = priority
        
        if best_split is not None:
            # Convert back to token position (approximate)
            return int(best_split * total_tokens / len(text))
        
        return None
    
    def _add_cross_references(self, chunks: List[TextChunk]):
        """Add parent/child relationships between chunks"""
        for i, chunk in enumerate(chunks):
            # Previous chunk is parent
            if i > 0:
                chunk.parent_chunks.append(chunks[i-1].chunk_id)
                chunks[i-1].child_chunks.append(chunk.chunk_id)
            
            # Next chunk is child
            if i < len(chunks) - 1:
                chunk.child_chunks.append(chunks[i+1].chunk_id)
    
    def reconstruct_text_from_chunks(
        self, 
        chunks: List[TextChunk], 
        remove_overlaps: bool = True
    ) -> str:
        """Reconstruct original text from chunks"""
        if not chunks:
            return ""
        
        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: x.chunk_index)
        
        if not remove_overlaps:
            return "".join(chunk.content for chunk in sorted_chunks)
        
        # Remove overlaps intelligently
        reconstructed = sorted_chunks[0].content
        
        for i in range(1, len(sorted_chunks)):
            current_chunk = sorted_chunks[i]
            
            # Find overlap with previous chunk
            overlap_text = current_chunk.overlap_before
            
            if overlap_text and overlap_text in reconstructed[-len(overlap_text)*2:]:
                # Find where overlap starts in current chunk
                overlap_start = current_chunk.content.find(overlap_text)
                if overlap_start >= 0:
                    # Add content after overlap
                    reconstructed += current_chunk.content[overlap_start + len(overlap_text):]
                else:
                    reconstructed += current_chunk.content
            else:
                reconstructed += current_chunk.content
        
        return reconstructed

# Global text splitter instance
text_splitter = IntelligentTextSplitter()
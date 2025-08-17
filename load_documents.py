# load_documents.py - Fixed version with corrected async iteration
import parse_documents
from embedding_service import get_embeddings_batch  # FIXED: Import from embedding_service instead of utils
from config import Config
from progress_manager import SimpleProgressTracker
import os
import logging

logger = logging.getLogger(__name__)

async def load_file(file_path, file_type, session):
    """Parses a file and returns data for insertion with progress tracking"""
    if os.path.getsize(file_path) == 0:
        logger.info(f"Skipping empty file: {file_path}")
        return []
    
    # Use simple progress tracker for single file loading
    filename = os.path.basename(file_path)
    tracker = SimpleProgressTracker(f"Processing {filename}")
    
    records = []
    chunk_count = 0
    
    try:
        # FIXED: Use regular for loop since load_file_chunked returns a sync generator
        for chunk in load_file_chunked(file_path, file_type, session):
            records.extend(chunk)
            chunk_count += 1
            tracker.update(chunk_count, chunk_count + 1, f"Processed {len(records)} records")
        
        tracker.complete(len(records))
        return records
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []

async def process_chunk(chunk, file_path, session):
    """Process chunk with embeddings"""
    valid_data = [item for item in chunk if item["content"] and item["content"].strip()]
    if not valid_data:
        return []
        
    contents = [item["content"] for item in valid_data]
    tags_list = [item.get("tags", []) for item in valid_data]
    
    # FIXED: Use embedding_service instead of utils
    embeddings = await get_embeddings_batch(contents)
    
    return [
        {"content": contents[j], "tags": tags_list[j], "embedding": embedding}
        for j, embedding in enumerate(embeddings)
        if embedding is not None and len(embedding) == Config.EMBEDDING_DIM
    ]

def load_file_chunked(file_path, file_type, session, chunk_size=None):
    """
    Streaming file loader - FIXED: Now returns a sync generator
    NOTE: This function is NOT async because parse_documents.stream_parse_file 
    returns a sync generator, not an async generator
    """
    chunk_size = chunk_size or Config.CHUNK_SIZES['memory_safe']
    
    if os.path.getsize(file_path) == 0:
        return
        
    for records in parse_documents.stream_parse_file(file_path, file_type, chunk_size):
        # Process each chunk synchronously for now
        yield records

async def load_file_chunked_async(file_path, file_type, session, chunk_size=None):
    """
    Async wrapper for streaming file loader that processes embeddings
    This version actually generates embeddings for each chunk
    """
    chunk_size = chunk_size or Config.CHUNK_SIZES['memory_safe']
    
    if os.path.getsize(file_path) == 0:
        return
        
    # Use the sync generator from parse_documents
    for records in parse_documents.stream_parse_file(file_path, file_type, chunk_size):
        # Process the chunk to add embeddings
        processed = await process_chunk(records, file_path, session)
        if processed:
            yield processed
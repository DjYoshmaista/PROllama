# load_documents.py
import parse_documents
from utils import get_embeddings_batch
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
        async for chunk in load_file_chunked(file_path, file_type, session):
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
    
    embeddings = await get_embeddings_batch(session, contents)
    
    return [
        {"content": contents[j], "tags": tags_list[j], "embedding": embedding}
        for j, embedding in enumerate(embeddings)
        if embedding is not None and len(embedding) == Config.EMBEDDING_DIM
    ]

async def load_file_chunked(file_path, file_type, session, chunk_size=None):
    """Streaming file loader"""
    chunk_size = chunk_size or Config.CHUNK_SIZES['memory_safe']
    
    if os.path.getsize(file_path) == 0:
        return
        
    for records in parse_documents.stream_parse_file(file_path, file_type, chunk_size):
        processed = await process_chunk(records, file_path, session)
        if processed:
            yield processed
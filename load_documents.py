# load_documents.py
import gc
import parse_documents
from utils import get_embeddings_batch, EMBEDDING_DIMENSION
import os
import json
import logging
import inspect
import asyncio

logger = logging.getLogger()
MAX_CONCURRENT_CHUNKS_PER_FILE = 8
MEMORY_CHUNK_SIZE = 100

async def load_file(file_path, file_type, session):
    """Parses a file and returns data for insertion"""
    if os.path.getsize(file_path) == 0:
        logger.info(f"Skipping empty file: {file_path}")
        return []
    
    records = []
    async for chunk in load_file_chunked(file_path, file_type, session):
        records.extend(chunk)
    return records

async def process_chunk(chunk, file_path, session):
    """Modern chunk processor"""
    valid_data = [item for item in chunk if item["content"] and item["content"].strip()]
    if not valid_data:
        return []
        
    contents = [item["content"] for item in valid_data]
    tags_list = [item.get("tags", []) for item in valid_data]
    
    embeddings = await get_embeddings_batch(session, contents)
    
    return [
        (contents[j], tags_list[j], embedding)
        for j, embedding in enumerate(embeddings)
        if embedding is not None and len(embedding) == EMBEDDING_DIMENSION
    ]

async def load_file_chunked(file_path, file_type, session, chunk_size=MEMORY_CHUNK_SIZE):
    """Streaming file loader"""
    if os.path.getsize(file_path) == 0:
        return
        
    async for records in parse_documents.stream_parse_file(file_path, file_type, chunk_size):
        processed = await process_chunk(records, file_path, session)
        if processed:
            yield processed

async def streaming_parse_file(file_path, file_type, session, chunk_size):
    """Streaming parser for large files - properly handle sync generators"""
    
    if file_type == "csv":
        # Use regular for loop with synchronous generator
        for chunk in parse_documents.stream_parse_csv(file_path, chunk_size):
            records = await process_chunk(chunk, file_path, 0, 0, session)
            if records:
                yield records
                
    elif file_type == "json":
        # Use regular for loop with synchronous generator
        for chunk in parse_documents.stream_parse_json(file_path, chunk_size):
            records = await process_chunk(chunk, file_path, 0, 0, session)
            if records:
                yield records
                
    elif file_type == "py":
        # Use regular for loop with synchronous generator
        for chunk in parse_documents.stream_parse_py(file_path, chunk_size):
            records = await process_chunk(chunk, file_path, 0, 0, session)
            if records:
                yield records
                
    else:  # txt files or fallback
        # For text files, read in chunks directly
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                while True:
                    content = f.read(8192)
                    if not content:
                        break
                    chunk = [{"content": content, "tags": [file_type, file_path]}]
                    records = await process_chunk(chunk, file_path, 0, 0, session)
                    if records:
                        yield records
        except Exception as e:
            logger.error(f"Error streaming {file_path}: {e}")
            # Fallback: try to read entire file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(1000000)  # Limit to 1MB
                    chunk = [{"content": content, "tags": ["fallback", file_path]}]
                    records = await process_chunk(chunk, file_path, 0, 0, session)
                    if records:
                        yield records
            except Exception as fallback_e:
                logger.error(f"Fallback also failed for {file_path}: {fallback_e}")

def load_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_type = os.path.splitext(file_path)[1][1:].lower()
            if file_type in ["txt", "csv", "json", "py"]:
                print(f"{inspect.currentframe().f_lineno}:Processing file: {file_path}")
                if os.path.getsize(file_path) == 0:
                    print(f"{inspect.currentframe().f_lineno}:Skipping empty file: {file_path}")
                    continue

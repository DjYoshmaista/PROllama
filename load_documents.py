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

async def process_chunk(chunk, file_path, chunk_index, total_chunks, session):
    """Process a single chunk of data"""
    valid_data = [
        item for item in chunk 
        if item["content"] and item["content"].strip()
    ]
    
    if not valid_data:
        return []
        
    contents = [item["content"] for item in valid_data]
    tags_list = [item.get("tags", []) for item in valid_data]  # Keep as lists
    
    embeddings = await get_embeddings_batch(session, contents)
    
    records = []
    failed_count = 0
    for j, embedding in enumerate(embeddings):
        if embedding is None:
            failed_count += 1
            continue
            
        if len(embedding) != EMBEDDING_DIMENSION:
            failed_count += 1
            continue
        
        # Return tags as list, not JSON string
        records.append((
            contents[j],      # content
            tags_list[j],     # tags as list (JSONB will handle it)
            embedding         # embedding
        ))
    
    if failed_count > 0:
        logger.warning(f"Failed to generate {failed_count}/{len(contents)} "
                      f"embeddings in chunk {chunk_index} of {file_path}")
    
    return records

async def load_file_chunked(file_path, file_type, session, chunk_size=MEMORY_CHUNK_SIZE):
    """Generator that processes files in chunks for memory efficiency"""
    if os.path.getsize(file_path) == 0:
        return

    async for records in streaming_parse_file(file_path, file_type, session, chunk_size):
        yield records

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

# async def process_chunk(chunk, file_path, chunk_index, total_chunks, session):
#     """Process a single chunk of data including embedding generation"""
#     # Filter out empty content
#     valid_data = [
#         item for item in chunk 
#         if item["content"] and item["content"].strip()
#     ]
    
#     if not valid_data:
#         logger.info(f"{inspect.currentframe().f_lineno}:No valid data in chunk {chunk_index}/{total_chunks} for {file_path}")
#         return []
        
#     # Prepare batch for embedding
#     contents = [item["content"] for item in valid_data]
#     tags = [json.dumps(item.get("tags", [])) for item in valid_data]
    
#     # Get embeddings for batch
#     logger.info(f"{inspect.currentframe().f_lineno}:Generating embeddings for {len(contents)} records in chunk {chunk_index}/{total_chunks} of {file_path}")
#     embeddings = await get_embeddings_batch(session, contents)
#     if embeddings:
#         # Validate against previous batch
#         if "last_embedding" in globals() and last_embedding:
#             for i, emb in enumerate(embeddings):
#                 if emb:
#                     sim = cosine_similarity(emb, last_embedding)
#                     dist = euclidean_distance(emb, last_embedding)
#                     logger.info(f"Embedding {i} similarity: cos={sim:.4f}, eucl={dist:.4f}")
        
#         # Update reference
#         last_embedding = embeddings[0]
    
#     # Prepare records with validation
#     records = []
#     success_count = 0
#     for j, embedding in enumerate(embeddings):
#         if embedding is None:
#             logger.warning(f"{inspect.currentframe().f_lineno}:Skipping record due to failed embedding in {file_path}")
#             continue
            
#         # Validate embedding dimensions
#         if len(embedding) != EMBEDDING_DIMENSION:
#             logger.warning(
#                 f"{inspect.currentframe().f_lineno}:Invalid embedding dimensions ({len(embedding)}) in {file_path}. "
#                 f"Content: {contents[j][:500]}..."
#             )
#             continue
        
#         records.append((
#             contents[j],       # content
#             tags[j],           # tags
#             embedding          # embedding vector
#         ))
#         success_count += 1
    
#     # Log batch results
#     if success_count:
#         logger.info(f"{inspect.currentframe().f_lineno}:Generated {success_count}/{len(contents)} embeddings for chunk {chunk_index}/{total_chunks} of {file_path}")
#     else:
#         logger.warning(f"{inspect.currentframe().f_lineno}:No valid embeddings generated for chunk {chunk_index}/{total_chunks} of {file_path}")
    
#     return records

# async def load_file_chunked(file_path, file_type, session, chunk_size=50):
#     """
#     Generator that processes files in chunks for memory efficiency
#     Yields batches of records ready for database insertion
#     """
#     if os.path.getsize(file_path) == 0:
#         logger.info(f"{inspect.currentframe().f_lineno}:Skipping empty file: {file_path}")
#         return

#     # Parse entire file
#     parsed_data = parse_documents.parse_file(file_path, file_type)
#     if not parsed_data:
#         logger.info(f"{inspect.currentframe().f_lineno}:No data to load from: {file_path}")
#         return

#     # Calculate chunk metrics
#     total_items = len(parsed_data)
#     total_chunks = (total_items + chunk_size - 1) // chunk_size
    
#     # Process small files sequentially
#     if total_chunks <= 20:
#         for i in range(0, total_items, chunk_size):
#             chunk = parsed_data[i:i+chunk_size]
#             logger.info(f"{inspect.currentframe().f_lineno}:Processing chunk {i//chunk_size+1}/{total_chunks} for {file_path}")
#             records = await process_chunk(
#                 chunk, file_path, i//chunk_size+1, total_chunks, session
#             )
#             if records:
#                 yield records
#         return

#     # Process large files with parallel chunk processing
#     logger.info(f"{inspect.currentframe().f_lineno}:Processing large file {file_path} with {total_chunks} chunks in parallel")
    
#     # Create all chunk tasks
#     tasks = []
#     for i in range(0, total_items, chunk_size):
#         chunk = parsed_data[i:i+chunk_size]
#         chunk_index = i//chunk_size + 1
#         tasks.append(
#             process_chunk(chunk, file_path, chunk_index, total_chunks, session)
#         )
    
#     # Process chunks with limited concurrency
#     sem = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS_PER_FILE)
    
#     async def process_with_semaphore(task):
#         async with sem:
#             return await task
    
#     # Create limited tasks
#     limited_tasks = [process_with_semaphore(task) for task in tasks]
    
#     # Process tasks as they complete
#     for future in asyncio.as_completed(limited_tasks):
#         records = await future
#         if records:
#             yield records

# def load_folder(folder_path):
#     """Legacy function for synchronous folder loading"""
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             file_type = os.path.splitext(file_path)[1][1:].lower()
#             if file_type in ["txt", "csv", "json"]:
#                 print(f"{inspect.currentframe().f_lineno}:Processing file: {file_path}")
#                 if os.path.getsize(file_path) == 0:
#                     print(f"{inspect.currentframe().f_lineno}:Skipping empty file: {file_path}")
#                     continue
#                 # Note: This will need async handling in real usage
#                 # load_file(file_path, file_type)

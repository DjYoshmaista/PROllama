import psycopg2
import requests
import inspect
import json
import aiohttp
import numpy as np
import heapq
import torch
import os
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import tkinter as tk
import asyncio
import multiprocessing
import time
from tkinter import filedialog
from progress_tracker import track_progress
from constants import *
from utils import OLLAMA_API, get_embedding, get_embedding_async, cosine_similarity, euclidean_distance, batched_gpu_cosine_similarity, cleanup_memory
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_db.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

table_name = TABLE_NAME

# Configuration for inference
DEFAULT_RELEVANCE_THRESHOLD = 0.3
DEFAULT_TOP_K = 25
DEFAULT_VECTOR_SEARCH_LIMIT = 999999999
MAX_CACHE_SIZE = 20000000
CACHE_REFRESH_INTERVAL = 300

# Global embedding cache
global embedding_cache, last_cache_refresh
embedding_cache = None
last_cache_refresh = 0

def init_db():
    """Initialize database with schema migration support"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            connect_timeout=5
        )
        cur = conn.cursor()
        
        # Create documents table with JSONB tags
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                tags JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding VECTOR(1024)
            )
        """
        cur.execute(query)
        
        # Check if tags column exists and migrate if needed
        cur.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}' 
            AND column_name = 'tags'
        """)
        if not cur.fetchone():
            logger.info("Migrating tags column to JSONB")
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN tags JSONB DEFAULT '[]'::jsonb")
            conn.commit()

        # Create index
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
            ON {table_name}
            USING hnsw (embedding vector_cosine_ops)
        """)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

# Generate answer using Ollama
def generate_answer(prompt):
    try:
        response = requests.post(
            f"{OLLAMA_API}/generate",
            json={"model": "qwen3:8b", "prompt": prompt},
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
        logger.error(f"Answer generation failed: {str(e)}")
        return "Error generating answer."

# Add data to the database
def add_data():
    global embedding_cache
    content = input("Enter the content to add: ")
    if not content.strip():
        print("Content cannot be empty!")
        return
    try:
        embedding = get_embedding(content)
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            connect_timeout=5
        )
        cur = conn.cursor()
        query = f"INSERT INTO {table_name} (content, embedding) VALUES (%s, %s)",
        cur.execute(query, (content, embedding))
        conn.commit()
        cur.close()
        conn.close()
        print("Data added successfully.")
        # Invalidate cache
        embedding_cache = None
    except Exception as e:
        logger.error(f"Error adding data: {str(e)}")
        print("Failed to add data.")

# Query database directly
def query_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            connect_timeout=5
        )
        cur = conn.cursor()
        query = f"SELECT id, content FROM {table_name}"
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            print(f"ID: {row[0]}, Content: {row[1][:1000]}...")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        print("Failed to query database.")

# List full contents
def list_contents():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            connect_timeout=5
        )
        cur = conn.cursor()
        query = f"SELECT id, content FROM {table_name}"
        cur.execute(query)
        rows = cur.fetchall()
        for row in rows:
            print(f"ID: {row[0]}, Content: {row[1][:200]}...")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to list contents: {str(e)}")
        print("Failed to list database contents.")

def load_embedding_cache():
    """Load all embeddings into memory for fast GPU processing"""
    global embedding_cache, last_cache_refresh
    current_time = time.time()
    
    if embedding_cache is None or (current_time - last_cache_refresh) > CACHE_REFRESH_INTERVAL:
        try:
            logger.info("Loading embedding cache...")
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                connect_timeout=30
            )
            cur = conn.cursor()
            
            # Get total count
            query = f"SELECT COUNT(*) FROM {table_name}"
            cur.execute(query)
            total_docs = cur.fetchone()[0]
            
            if total_docs > MAX_CACHE_SIZE:
                logger.warning(f"Database too large for cache ({total_docs} > {MAX_CACHE_SIZE})")
                embedding_cache = None
                return False
            
            # Load IDs and embeddings
            query = f"SELECT id, embedding FROM {table_name}"
            cur.execute(query)
            rows = cur.fetchall()
            
            ids = []
            embeddings = []
            for row in rows:
                doc_id, doc_embedding = row
                ids.append(doc_id)
                
                # Handle different embedding formats from pgvector
                if doc_embedding is None:
                    logger.warning(f"Null embedding for document ID {doc_id}, skipping")
                    continue
                    
                # pgvector returns embeddings as lists, not strings
                if isinstance(doc_embedding, str):
                    # Only parse if it's actually a string (shouldn't happen with pgvector)
                    try:
                        # Remove brackets and split by comma if it's a pgvector string format
                        if doc_embedding.startswith('[') and doc_embedding.endswith(']'):
                            doc_embedding = [float(x) for x in doc_embedding[1:-1].split(',')]
                        else:
                            doc_embedding = json.loads(doc_embedding)
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse embedding for ID {doc_id}: {e}")
                        continue
                elif isinstance(doc_embedding, (list, tuple)):
                    # Already in correct format from pgvector
                    doc_embedding = list(doc_embedding)
                else:
                    # Handle numpy arrays or other array-like objects
                    try:
                        doc_embedding = list(doc_embedding)
                    except Exception as e:
                        logger.error(f"Cannot convert embedding to list for ID {doc_id}: {e}")
                        continue
                
                embeddings.append(doc_embedding)
            
            if not embeddings:
                logger.error("No valid embeddings found in database")
                cur.close()
                conn.close()
                return False
            
            # Convert to tensor
            embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
            if torch.cuda.is_available():
                embedding_tensor = embedding_tensor.cuda()
            
            embedding_cache = {
                'ids': ids,
                'embeddings': embedding_tensor,
                'count': len(ids),
                'timestamp': current_time
            }
            last_cache_refresh = current_time
            logger.info(f"Embedding cache loaded with {len(ids)} vectors")
            
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding cache: {str(e)}", exc_info=True)
            embedding_cache = None
            return False
    return True

def gpu_cosine_similarity(query_embedding, embeddings_tensor):
    """Compute cosine similarity using GPU acceleration"""
    # Convert query to tensor
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    if torch.cuda.is_available() and embeddings_tensor.is_cuda:
        query_tensor = query_tensor.cuda()
    
    # Ensure query is the right shape [1, dim] for matrix multiplication
    if query_tensor.dim() == 1:
        query_tensor = query_tensor.unsqueeze(0)  # Shape: [1, dim]
    
    # Normalize vectors
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)  # Shape: [1, dim]
    embeddings_norm = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)  # Shape: [N, dim]
    
    # Compute cosine similarity via matrix multiplication
    # [1, dim] @ [N, dim].T = [1, N]
    cos_sim = torch.mm(query_norm, embeddings_norm.t())
    
    # Squeeze to remove the extra dimension and convert to numpy
    # Always move to CPU before converting to numpy
    return cos_sim.squeeze().cpu().numpy()

async def ask_inference_async(relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
                              top_k=DEFAULT_TOP_K,
                              vector_search_limit=DEFAULT_VECTOR_SEARCH_LIMIT):
    """Perform semantic search and generate answer with comprehensive error handling"""
    # Use vector_search_limit as CHUNK_SIZE for memory management
    CHUNK_SIZE = min(vector_search_limit, 500000)  # Use a reasonable default chunk size
    
    conn = None
    cur = None
    
    try:
        question = input("Enter your question: ")
        if not question.strip():
            print("Question cannot be empty!")
            return

        # Get embedding of the question
        question_embedding = await get_embedding_async(question)
        if question_embedding is None:
            print("Failed to generate embedding for question.")
            return

        # --- Prompt for Caching ---
        use_cache = False
        if embedding_cache is None:
            cache_choice = input("Would you like to cache database embeddings for faster processing? (y/N): ").strip().lower()
            if cache_choice == 'y':
                use_cache = load_embedding_cache()
                if not use_cache:
                    print("Failed to load cache, proceeding with chunked search.")
        else:
            # Cache is already loaded, check if user wants to use it
            cache_choice = input("Embedding cache is available. Use it for faster search? (Y/n): ").strip().lower()
            if cache_choice != 'n':  # Default to using cache if available
                use_cache = True
            else:
                logger.info("User chose not to use the existing cache.")

        # --- Use Cache Path ---
        if use_cache and embedding_cache:
            logger.info("Computing similarities using GPU cache...")
            similarities = gpu_cosine_similarity(question_embedding, embedding_cache['embeddings'])

            # --- Top-K Selection with Threshold ---
            # Find indices where similarity meets threshold
            valid_indices_mask = similarities >= relevance_threshold
            valid_indices = np.where(valid_indices_mask)[0]

            if len(valid_indices) == 0:
                print("No documents passed relevance threshold.")
                return

            # Get top_k indices among valid ones
            sorted_valid_indices = valid_indices[np.argsort(similarities[valid_indices])]  # Ascending sort
            top_indices = sorted_valid_indices[-top_k:]  # Take top_k from the end (highest similarity)
            top_indices = top_indices[::-1]  # Reverse to get descending order

            # Prepare top documents info
            top_docs = []
            for idx in top_indices:
                sim_score = similarities[idx]
                top_docs.append({
                    'id': embedding_cache['ids'][idx],
                    'cosine_similarity': float(sim_score),
                    'euclidean_distance': 0  # Not calculated in this path for performance
                })
            logger.info(f"Found {len(top_docs)} matches via GPU cache")
            
            # Cleanup GPU memory
            del similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Chunked Search Path ---
        else:
            logger.info("Starting chunked similarity search...")
            # Use a min-heap to keep track of top-k results efficiently
            top_k_heap = []
            docs_processed = 0

            try:
                conn = psycopg2.connect(
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    host=DB_HOST,
                    connect_timeout=30
                )
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Get total count with proper error handling
                query = f"SELECT COUNT(*) FROM {table_name}"
                cur.execute(query)
                result = cur.fetchone()
                
                # Safely access the count
                if result and 'count' in result:
                    total_docs = result['count']
                else:
                    # Fallback for non-dict cursor or empty result
                    cur.close()
                    cur = conn.cursor()  # Regular cursor
                    cur.execute(query)
                    count_result = cur.fetchone()
                    total_docs = count_result[0] if count_result else 0
                
                if total_docs == 0:
                    logger.warning("No documents found in database")
                    print("No documents in database to search.")
                    return
                    
                logger.info(f"Total documents to process: {total_docs}")

                # --- Chunked Processing ---
                offset = 0
                while offset < total_docs:
                    # Fetch a chunk of (id, embedding) from the database
                    query = f"""
                        SELECT id, embedding
                        FROM {table_name}
                        ORDER BY id
                        LIMIT %s OFFSET %s
                    """
                    cur.execute(query, (CHUNK_SIZE, offset))
                    rows = cur.fetchall()
                    
                    if not rows:
                        break  # No more data

                    chunk_ids = []
                    chunk_embeddings = []

                    # Prepare the chunk data
                    for row in rows:
                        doc_id = row['id']
                        doc_embedding = row['embedding']

                        # Handle potential string embeddings from pgvector
                        if doc_embedding is None:
                            logger.warning(f"Null embedding for document ID {doc_id}, skipping")
                            continue
                            
                        if isinstance(doc_embedding, str):
                            try:
                                # Handle pgvector string format
                                if doc_embedding.startswith('[') and doc_embedding.endswith(']'):
                                    doc_embedding = [float(x) for x in doc_embedding[1:-1].split(',')]
                                else:
                                    doc_embedding = json.loads(doc_embedding)
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.error(f"Failed to parse embedding for ID {doc_id}: {e}")
                                continue
                        elif isinstance(doc_embedding, (list, tuple)):
                            doc_embedding = list(doc_embedding)
                        else:
                            try:
                                doc_embedding = list(doc_embedding)
                            except Exception as e:
                                logger.error(f"Cannot convert embedding to list for ID {doc_id}: {e}")
                                continue

                        chunk_ids.append(doc_id)
                        chunk_embeddings.append(doc_embedding)

                    docs_processed += len(chunk_ids)
                    if docs_processed % (CHUNK_SIZE * 10) == 0 or docs_processed >= total_docs:
                        logger.info(f"Processed {docs_processed}/{total_docs} documents...")

                    if not chunk_embeddings:
                        offset += CHUNK_SIZE
                        continue

                    # --- Similarity Calculation for Chunk ---
                    similarities = None
                    # Try GPU acceleration if available and chunk is large enough
                    if torch.cuda.is_available() and len(chunk_embeddings) > 100:
                        try:
                            # Convert chunk embeddings to tensor and move to GPU
                            db_tensor = torch.tensor(chunk_embeddings, dtype=torch.float32).cuda()
                            
                            # Calculate similarities using the helper function
                            similarities = batched_gpu_cosine_similarity(question_embedding, db_tensor)
                            
                            # Clean up GPU tensor immediately
                            del db_tensor
                            torch.cuda.empty_cache()
                            
                        except Exception as gpu_e:
                            logger.warning(f"GPU similarity calculation failed, falling back to CPU: {gpu_e}")
                            similarities = None

                    # CPU fallback or primary path if GPU not used
                    if similarities is None:
                        similarities = np.array([
                            cosine_similarity(question_embedding, emb) for emb in chunk_embeddings
                        ])

                    # --- Update Top-K Results with Correct Heap Logic ---
                    for i, doc_id in enumerate(chunk_ids):
                        sim_score = similarities[i]
                        
                        if sim_score >= relevance_threshold:
                            if len(top_k_heap) < top_k:
                                # Heap not full, add the item
                                heapq.heappush(top_k_heap, (sim_score, doc_id))
                            elif sim_score > top_k_heap[0][0]:
                                # New score is better than the worst in heap
                                heapq.heapreplace(top_k_heap, (sim_score, doc_id))

                    # Memory cleanup every 5 chunks
                    if docs_processed % (CHUNK_SIZE * 5) == 0:
                        del chunk_ids
                        del chunk_embeddings
                        del similarities
                        cleanup_memory()
                        logger.debug(f"Memory cleanup at {docs_processed} documents")
                    
                    offset += CHUNK_SIZE

            finally:
                # Always close database connections
                if cur:
                    cur.close()
                if conn:
                    conn.close()

            # --- Prepare Final Results ---
            if not top_k_heap:
                print("No documents passed relevance threshold.")
                return

            # Sort by similarity descending (highest first)
            sorted_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)

            # Extract final document info
            top_docs = []
            for sim_score, doc_id in sorted_results:
                top_docs.append({
                    'id': doc_id,
                    'cosine_similarity': float(sim_score),
                    'euclidean_distance': 0  # Not calculated for performance
                })

            logger.info(f"Found {len(top_docs)} matches passing threshold.")

        # --- Retrieve Content for Top Matches ---
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                connect_timeout=30
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)

            top_ids = [doc['id'] for doc in top_docs]
            if not top_ids:
                print("No valid document IDs to retrieve.")
                return

            # Use parameterized query for safety
            placeholders = ','.join(['%s'] * len(top_ids))
            query = f"SELECT id, content FROM {table_name} WHERE id IN ({placeholders})"
            cur.execute(query, top_ids)
            content_rows = cur.fetchall()
            
            # Map content to final docs
            id_to_content = {row['id']: row['content'] for row in content_rows}
            final_docs_with_content = []
            
            for doc_info in top_docs:
                content = id_to_content.get(doc_info['id'])
                if content:
                    doc_info['content'] = content
                    final_docs_with_content.append(doc_info)
                else:
                    logger.warning(f"Content not found for retrieved document ID: {doc_info['id']}")

            if not final_docs_with_content:
                print("No document content found for the top matches.")
                return

            # --- Build Context and Generate Answer ---
            context_parts = []
            for idx, doc in enumerate(final_docs_with_content):
                # Truncate content for context prompt
                truncated_content = doc['content'][:800]
                context_parts.append(
                    f"[Doc {idx+1}] (CosSim: {doc['cosine_similarity']:.4f})\n{truncated_content}"
                )
            context = "\n".join(context_parts)

            # Enhanced prompt with metadata
            prompt = f"""Answer the question using only the context below. If unsure, say 'I don't know'.
Question: {question}
Context (ranked by relevance):
{context}
"""
            answer = generate_answer(prompt)
            print("\nAnswer:", answer)

            # Show sources (top matching documents)
            print("\n--- Top Matching Documents ---")
            for doc in final_docs_with_content:
                print(f"ID {doc['id']} | "
                      f"Cosine Similarity: {doc['cosine_similarity']:.4f}")

        finally:
            # Always close database connections
            if cur:
                cur.close()
            if conn:
                conn.close()

    except psycopg2.Error as db_e:
        logger.error(f"Database error during inference: {db_e}", exc_info=True)
        print("Failed to query database. Please check your database connection.")
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        print("Failed to generate answer due to an internal error.")
    finally:
        # Final cleanup
        if 'cur' in locals() and cur and not cur.closed:
            cur.close()
        if 'conn' in locals() and conn and not conn.closed:
            conn.close()
        
        # Memory cleanup
        if embedding_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        cleanup_memory()
        
        logger.info("Inference query completed")

def configure_embedding_parameters():
    """Menu for configuring embedding parameters"""
    global DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K, DEFAULT_VECTOR_SEARCH_LIMIT
    print("\nEmbedding Configuration:")
    print(f"1. Relevance Threshold (current: {DEFAULT_RELEVANCE_THRESHOLD:.2f})")
    print(f"2. Top K Results (current: {DEFAULT_TOP_K})")
    print(f"3. Vector Search Limit/Chunk Size (current: {DEFAULT_VECTOR_SEARCH_LIMIT})")
    print("4. Back to main menu")
    choice = input("Select option: ")
    if choice == "1":
        try:
            new_threshold = float(input("Enter new relevance threshold (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                DEFAULT_RELEVANCE_THRESHOLD = new_threshold
                print("Threshold updated.")
            else:
                print("Invalid value. Must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a number.")
    elif choice == "2":
        try:
            new_top_k = int(input("Enter new Top K value: "))
            if new_top_k > 0:
                DEFAULT_TOP_K = new_top_k
                print("Top K updated.")
            else:
                print("Value must be positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    elif choice == "3":
        try:
            new_limit = int(input("Enter new vector search limit/chunk size: "))
            if new_limit > 0:
                DEFAULT_VECTOR_SEARCH_LIMIT = new_limit
                print("Vector search limit/chunk size updated.")
            else:
                print("Value must be positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    elif choice == "4":
        return
    else:
        print("Invalid option.")

def browse_files():
    """Browse for files with fallback to manual input if GUI unavailable"""
    try:
        # Try to use GUI file browser
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="Select Folder to Load Files")
        root.destroy()  # Clean up the tkinter root window
        
        if folder_path:
            logger.info(f"Folder selected via GUI: {folder_path}")
            return folder_path
        else:
            logger.info("No folder selected via GUI dialog")
            return None
            
    except Exception as e:
        # GUI not available (SSH, no display, etc.)
        logger.warning(f"GUI file browser not available: {e}")
        print("\n" + "="*60)
        print("GUI file browser is not available (no display detected)")
        print("Please enter the folder path manually")
        print("="*60)
        
        return get_manual_folder_path()

def get_manual_folder_path():
    """Get folder path via manual input with validation"""
    while True:
        print("\nOptions:")
        print("1. Enter absolute path (e.g., /home/user/documents)")
        print("2. Enter relative path from current directory")
        print("3. Use current directory")
        print("4. Cancel")
        
        # Show current working directory for reference
        current_dir = os.getcwd()
        print(f"\nCurrent directory: {current_dir}")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Absolute path
            folder_path = input("Enter absolute folder path: ").strip()
            
            # Expand user home directory if ~ is used
            folder_path = os.path.expanduser(folder_path)
            
        elif choice == "2":
            # Relative path
            relative_path = input("Enter relative path from current directory: ").strip()
            folder_path = os.path.join(current_dir, relative_path)
            folder_path = os.path.abspath(folder_path)  # Convert to absolute
            
        elif choice == "3":
            # Use current directory
            folder_path = current_dir
            print(f"Using current directory: {folder_path}")
            
        elif choice == "4":
            # Cancel
            logger.info("Folder selection cancelled by user")
            return None
            
        else:
            print("Invalid option. Please try again.")
            continue
        
        # Validate the folder path
        if validate_folder_path(folder_path):
            return folder_path
        else:
            print(f"\nError: Invalid folder path: {folder_path}")
            retry = input("Would you like to try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None

def validate_folder_path(folder_path):
    """Validate that the folder path exists and is accessible"""
    try:
        # Check if path exists
        if not os.path.exists(folder_path):
            logger.error(f"Path does not exist: {folder_path}")
            print(f"Error: Path does not exist: {folder_path}")
            return False
        
        # Check if it's a directory
        if not os.path.isdir(folder_path):
            logger.error(f"Path is not a directory: {folder_path}")
            print(f"Error: Path is not a directory: {folder_path}")
            return False
        
        # Check if we have read permissions
        if not os.access(folder_path, os.R_OK):
            logger.error(f"No read permission for directory: {folder_path}")
            print(f"Error: No read permission for directory: {folder_path}")
            return False
        
        # Count supported files in the directory
        supported_extensions = ["py", "txt", "csv", "json"]
        file_count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1][1:].lower()
                if ext in supported_extensions:
                    file_count += 1
        
        if file_count == 0:
            logger.warning(f"No supported files found in: {folder_path}")
            print(f"\nWarning: No supported files found in: {folder_path}")
            print(f"Supported file types: {', '.join(supported_extensions)}")
            
            # Ask if user wants to continue anyway
            continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
            if continue_anyway != 'y':
                return False
        else:
            print(f"\nFound {file_count} supported file(s) in the directory")
            logger.info(f"Found {file_count} supported files in: {folder_path}")
        
        # Show a preview of files that will be processed
        print("\nPreview of files to be processed (first 10):")
        preview_count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                if preview_count >= 10:
                    break
                ext = os.path.splitext(file)[1][1:].lower()
                if ext in supported_extensions:
                    rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                    print(f"  - {rel_path}")
                    preview_count += 1
            if preview_count >= 10:
                if file_count > 10:
                    print(f"  ... and {file_count - 10} more file(s)")
                break
        
        # Final confirmation
        confirm = input(f"\nProcess {file_count} file(s) from this directory? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("User cancelled after folder validation")
            return False
        
        logger.info(f"Folder path validated successfully: {folder_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating folder path: {e}")
        print(f"Error validating folder path: {e}")
        return False

def list_directory_tree(folder_path, max_depth=2):
    """List directory structure for user reference"""
    print(f"\nDirectory structure (depth={max_depth}):")
    
    def print_tree(path, prefix="", depth=0):
        if depth >= max_depth:
            return
            
        try:
            items = sorted(os.listdir(path))
            dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
            files = [item for item in items if os.path.isfile(os.path.join(path, item))]
            
            # Print directories first
            for i, dir_name in enumerate(dirs[:5]):  # Limit to first 5 dirs
                is_last = (i == len(dirs) - 1) and len(files) == 0
                print(f"{prefix}{'└── ' if is_last else '├── '}{dir_name}/")
                
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(os.path.join(path, dir_name), new_prefix, depth + 1)
            
            if len(dirs) > 5:
                print(f"{prefix}├── ... ({len(dirs) - 5} more directories)")
            
            # Print files
            supported_files = []
            for file_name in files[:10]:  # Limit to first 10 files
                ext = os.path.splitext(file_name)[1][1:].lower()
                if ext in ["py", "txt", "csv", "json"]:
                    supported_files.append(file_name)
            
            for i, file_name in enumerate(supported_files[:5]):
                is_last = i == len(supported_files) - 1
                print(f"{prefix}{'└── ' if is_last else '├── '}{file_name}")
            
            if len(supported_files) > 5:
                print(f"{prefix}└── ... ({len(supported_files) - 5} more supported files)")
                
        except PermissionError:
            print(f"{prefix}[Permission Denied]")
    
    print_tree(folder_path)

def generate_file_paths(folder_path):
    """Generator yielding file paths one by one with progress tracking"""
    file_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1][1:].lower()
            if ext in ["py", "txt", "csv", "json"]:
                file_count += 1
                yield file_path
    return file_count  # Return total count for progress tracking

def count_files(folder_path):
    """Count supported files without storing paths"""
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1][1:].lower()
            if ext in ["py", "txt", "csv", "json"]:
                count += 1
    return count

def configure_postgres():
    """Apply PostgreSQL configuration optimizations"""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            connect_timeout=5
        )
        cur = conn.cursor()
        # Apply settings from postgresql.conf
        with open("postgresql.conf", "r") as conf_file:
            for line in conf_file:
                if line.strip() and not line.startswith("#"):
                    setting = line.split("=")[0].strip()
                    value = line.split("=")[1].strip().replace("'", "")
                    try:
                        cur.execute(f"ALTER SYSTEM SET {setting} = '{value}'")
                    except Exception as e:
                        logger.warning(f"Couldn't set {setting}: {str(e)}")
        conn.commit()
        cur.close()
        conn.close()
        print("PostgreSQL configuration applied. Please restart PostgreSQL.")
    except Exception as e:
        logger.error(f"Configuration failed: {str(e)}")
        print(f"Failed to configure PostgreSQL: {str(e)}")

def validate_embedding_model():
    try:
        response = requests.get(f"{OLLAMA_API}/tags", timeout=5)
        response.raise_for_status()  # Check for HTTP errors
        
        # Get the list of models
        data = response.json()
        models = [model['name'] for model in data.get('models', [])]
        
        # Define the expected model name
        expected_model = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
        
        # Check if the model exists
        if expected_model not in models:
            logger.error(f"Embedding model {expected_model} not found in Ollama!")
            logger.info(f"Available models: {models}")  # Log available models for debugging
            return False
            
        logger.info(f"Embedding model {expected_model} validated successfully")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Failed to connect to Ollama API: {str(e)}")
        return False
    except KeyError as e:
        logger.error(f"Unexpected API response structure: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

# Main menu
async def main():
    global embedding_cache, last_cache_refresh
    init_db()

    if not validate_embedding_model():
        emb_mdl = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
        print(f"CRITICAL: Embedding model {emb_mdl} not available!")
        return

    from embedding_queue import embedding_queue
    if not embedding_queue.started:
        await embedding_queue.start_workers(concurrency=10)
        logger.info("Starting embedding queue...")

    while True:
        print("\nRAG Database Menu:")
        print("1. Ask the inference model for information")
        print("2. Add data to the database")
        print("3. Load data from a file")
        print("4. Load documents from folder")
        print("5. Query the database directly")
        print("6. List the full contents of the database")
        print("7. Configure embedding parameters")
        print("8. Optimize PostgreSQL Configuration")
        print("9. Exit")
        choice = input("Enter your choice (1-9): ") # Updated choice number
        if choice == "1":
            await ask_inference_async(
                relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD,
                top_k=DEFAULT_TOP_K,
                vector_search_limit=DEFAULT_VECTOR_SEARCH_LIMIT
            )
        elif choice == "2":
            add_data()
        elif choice == "3":
            file_path = filedialog.askopenfilename(title="Select File to Load")
            if file_path:
                file_type = os.path.splitext(file_path)[1][1:].lower()
                try:
                    from load_documents import load_file
                    print(f"Loading file: {file_path} as {file_type}")
                    async with aiohttp.ClientSession() as session:
                        records = await load_file(file_path, file_type, session)
                    if records:
                        conn = psycopg2.connect(
                            dbname=DB_NAME,
                            user=DB_USER,
                            password=DB_PASSWORD,
                            host=DB_HOST,
                            connect_timeout=5
                        )
                        cur = conn.cursor()
                        for record in records:
                            content = record["content"]
                            tags = record["tags"]  # This is a list
                            embedding = record["embedding"]

                            # Serialize tags list to JSON string
                            tags_json = json.dumps(tags)

                            query = f"INSERT INTO {table_name} (content, tags, embedding) VALUES (%s, %s, %s)",
                            cur.execute(query, (content, tags_json, embedding))
                            
                        conn.commit()
                        cur.close()
                        conn.close()
                        print(f"Inserted {len(records)} records from file.")
                        # Invalidate cache
                        embedding_cache = None
                    else:
                        print("No records to insert.")
                except Exception as e:
                    logger.error(f"Error loading file: {str(e)}")
                    print(f"Failed to load file: {str(e)}")
        elif choice == "4":
            folder_path = browse_files()
            if folder_path:
                # Get total count while generating paths
                total_files = count_files(folder_path)
                if total_files == 0:
                    print("No supported files found in the selected folder.")
                    continue
                # Create a generator for the file paths
                file_generator = generate_file_paths(folder_path)
                print("Starting processing...")
                total_processed = await track_progress(file_generator, total_files)
                # Invalidate cache after bulk load
                embedding_cache = None
                print(f"\nFinished! Total documents loaded: {total_processed}/{total_files}")
        elif choice == "5":
            query_db()
        elif choice == "6":
            list_contents()
        elif choice == "7":
            configure_embedding_parameters()
        elif choice == "8":
            configure_postgres()
        elif choice == "9": # Updated exit choice
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())

# rag_db.py
import psycopg2
import requests
import inspect
import json
import aiohttp
import numpy as np
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
import os
import tkinter as tk
import asyncio
import multiprocessing
import torch
import time
from tkinter import filedialog
from progress_tracker import track_progress
from utils import OLLAMA_API, get_embedding, cosine_similarity, euclidean_distance
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

# Configuration for inference - now configurable via menu
DEFAULT_RELEVANCE_THRESHOLD = 0.3
DEFAULT_TOP_K = 10
DEFAULT_VECTOR_SEARCH_LIMIT = 1000
MAX_CACHE_SIZE = 2000000  # Max embeddings to cache in memory

# Global embedding cache
global embedding_cache, last_cache_refresh
embedding_cache = None
last_cache_refresh = 0
CACHE_REFRESH_INTERVAL = 300  # Refresh cache every 5 minutes

# Initialize database
def init_db():
    try:
        conn = psycopg2.connect(
            dbname="rag_db",
            user="postgres",
            password="postgres",
            host="localhost",
            connect_timeout=5
        )
        cur = conn.cursor()
        # Create pgvector extension if not exists
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create documents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding VECTOR(1024)
            )
        """)
        # Add index for faster similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents
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
            dbname="rag_db",
            user="postgres",
            password="postgres",
            host="localhost",
            connect_timeout=5
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)", 
            (content, embedding)
        )
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
            dbname="rag_db",
            user="postgres",
            password="postgres",
            host="localhost",
            connect_timeout=5
        )
        cur = conn.cursor()
        cur.execute("SELECT id, content FROM documents")
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
            dbname="rag_db",
            user="postgres",
            password="postgres",
            host="localhost",
            connect_timeout=5
        )
        cur = conn.cursor()
        cur.execute("SELECT id, content FROM documents")
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
    
    # Refresh cache if expired or not loaded
    if embedding_cache is None or (current_time - last_cache_refresh) > CACHE_REFRESH_INTERVAL:
        try:
            logger.info("Loading embedding cache...")
            conn = psycopg2.connect(
                dbname="rag_db",
                user="postgres",
                password="postgres",
                host="localhost",
                connect_timeout=30
            )
            cur = conn.cursor()
            
            # Get total count
            cur.execute("SELECT COUNT(*) FROM documents")
            total_docs = cur.fetchone()[0]
            
            if total_docs > MAX_CACHE_SIZE:
                logger.warning(f"Database too large for cache ({total_docs} > {MAX_CACHE_SIZE})")
                embedding_cache = None
                return False
                
            # Load only IDs and embeddings, NOT content
            cur.execute("SELECT id, embedding FROM documents")
            rows = cur.fetchall()

            # Explicitly process rows to handle potential string embeddings
            ids = []
            embeddings = []
            for row in rows:
                doc_id, doc_embedding = row
                ids.append(doc_id)

                # Check if the embedding is a string (common issue with pgvector)
                if isinstance(doc_embedding, str):
                    try:
                        # Attempt to parse the string as JSON
                        doc_embedding = json.loads(doc_embedding)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse embedding string for ID {doc_id}: {e}")
                        # Decide how to handle: skip, raise error, or use a zero vector?
                        # For now, let's raise an error to highlight the data issue
                        cur.close()
                        conn.close()
                        raise ValueError(f"Invalid embedding string format for document ID {doc_id}: {doc_embedding}") from e

                embeddings.append(doc_embedding)
            
            # Convert to tensor and move to GPU if available
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
            logger.error(f"Failed to load embedding cache: {str(e)}")
            embedding_cache = None
            return False
    return True

def gpu_cosine_similarity(query_embedding, embeddings_tensor):
    """Compute cosine similarity using GPU acceleration"""
    # Convert to tensor
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    if torch.cuda.is_available():
        query_tensor = query_tensor.cuda()
    print(f"Query Tensor: {query_tensor}")
    # Normalize vectors
    query_norm = query_tensor / torch.norm(query_tensor)
    print(f"Query Norm: {query_norm}\nEmbeddings Tensor: {embeddings_tensor}")
    embeddings_norm = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)
    print(f"Embeddings Norm: {embeddings_norm}")

    # Compute cosine similarity
    cos_sim = torch.mm(embeddings_norm, query_norm.view(-1, 1)).flatten()
    print(f"Cosine Similarity: {cos_sim}")

    return cos_sim.cpu().numpy() if torch.cuda.is_available() else cos_sim.numpy()

# Ask inference model with optimized memory usage
def ask_inference(relevance_threshold=DEFAULT_RELEVANCE_THRESHOLD, 
                  top_k=DEFAULT_TOP_K,
                  vector_search_limit=DEFAULT_VECTOR_SEARCH_LIMIT):
    global embedding_cache
    question = input("Enter your question: ")
    if not question.strip():
        print("Question cannot be empty!")
        return
        
    try:
        # Get embedding of the question
        question_embedding = get_embedding(question)
        if question_embedding is None:
            print("Failed to generate embedding for question.")
            return

        # Try to use cache for faster processing
        use_cache = load_embedding_cache()
        
        if use_cache and embedding_cache:
            # GPU-accelerated similarity calculation
            logger.info("Computing similarities using GPU cache...")
            similarities = gpu_cosine_similarity(question_embedding, embedding_cache['embeddings'])
            
            # Get top matches
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]
            
            # Filter by threshold
            top_docs = []
            for idx in top_indices:
                sim_score = similarities[idx]
                if sim_score >= relevance_threshold:
                    top_docs.append({
                        'id': embedding_cache['ids'][idx],
                        'cosine_similarity': float(sim_score),
                        'euclidean_distance': 0  # Not calculated
                    })
            
            logger.info(f"Found {len(top_docs)} matches via GPU cache")
        else:
            # Fallback to database method
            logger.info("Using database vector search")
            conn = psycopg2.connect(
                dbname="rag_db",
                user="postgres",
                password="postgres",
                host="localhost",
                connect_timeout=30
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get total count for progress
            cur.execute("SELECT COUNT(*) FROM documents")
            total_docs = cur.fetchone()['count']
            logger.info(f"Searching {total_docs} documents...")
            
            # Retrieve all embeddings
            cur.execute("SELECT id, embedding FROM documents")
            embedding_rows = cur.fetchall()
            
            # Process embeddings in chunks
            chunk_size = 5000
            scored_docs = []
            processed = 0
            
            for i in range(0, len(embedding_rows), chunk_size):
                chunk = embedding_rows[i:i+chunk_size]
                for row in chunk:
                    doc_embedding = row['embedding']
                    if isinstance(doc_embedding, str):
                        doc_embedding = json.loads(doc_embedding)
                    
                    cos_sim = cosine_similarity(question_embedding, doc_embedding)
                    
                    if cos_sim >= relevance_threshold:
                        eucl_dist = euclidean_distance(question_embedding, doc_embedding)
                        scored_docs.append({
                            'id': row['id'],
                            'cosine_similarity': cos_sim,
                            'euclidean_distance': eucl_dist
                        })
                
                processed += len(chunk)
                print(f"Processed {processed}/{total_docs} documents ({processed/total_docs:.1%})")
            
            # Sort by relevance
            scored_docs.sort(key=lambda x: -x['cosine_similarity'])
            top_docs = scored_docs[:top_k]
            cur.close()
            conn.close()
        
        if not top_docs:
            print("No documents passed relevance threshold.")
            return
        
        # Retrieve content for top matches
        conn = psycopg2.connect(
            dbname="rag_db",
            user="postgres",
            password="postgres",
            host="localhost",
            connect_timeout=30
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        top_ids = [doc['id'] for doc in top_docs]
        cur.execute(
            "SELECT id, content FROM documents WHERE id = ANY(%s)",
            (top_ids,)
        )
        content_rows = cur.fetchall()
        cur.close()
        conn.close()
        
        # Combine content with scores
        id_to_content = {row['id']: row['content'] for row in content_rows}
        final_docs = []
        for doc in top_docs:
            content = id_to_content.get(doc['id'])
            if content:
                final_docs.append({
                    'id': doc['id'],
                    'content': content,
                    'cosine_similarity': doc['cosine_similarity'],
                    'euclidean_distance': doc.get('euclidean_distance', 0)
                })
        
        if not final_docs:
            print("No documents found for selected IDs.")
            return
        
        # Build context
        context_parts = []
        for idx, doc in enumerate(final_docs):
            truncated_content = doc['content'][:800]
            context_parts.append(
                f"[Doc {idx+1}] (CosSim: {doc['cosine_similarity']:.4f}, "
                f"EuclDist: {doc['euclidean_distance']:.4f})\n{truncated_content}"
            )

        context = "\n\n".join(context_parts)

        # Enhanced prompt with metadata
        prompt = f"""Answer the question using only the context below. If unsure, say 'I don't know'.

Question: {question}

Context (ranked by relevance):
{context}
"""

        answer = generate_answer(prompt)
        print("\nAnswer:", answer)

        # Show sources
        print("\n--- Top Matching Documents ---")
        for doc in final_docs:
            print(f"ID {doc['id']} | "
                  f"Cosine Similarity: {doc['cosine_similarity']:.4f} | "
                  f"Euclidean Distance: {doc['euclidean_distance']:.4f}")

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        print("Failed to generate answer due to an internal error.")

def configure_embedding_parameters():
    """Menu for configuring embedding parameters"""
    global DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K, DEFAULT_VECTOR_SEARCH_LIMIT
    
    print("\nEmbedding Configuration:")
    print(f"1. Relevance Threshold (current: {DEFAULT_RELEVANCE_THRESHOLD:.2f})")
    print(f"2. Top K Results (current: {DEFAULT_TOP_K})")
    print(f"3. Vector Search Limit (current: {DEFAULT_VECTOR_SEARCH_LIMIT})")
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
            new_limit = int(input("Enter new vector search limit: "))
            if new_limit > 0:
                DEFAULT_VECTOR_SEARCH_LIMIT = new_limit
                print("Vector search limit updated.")
            else:
                print("Value must be positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    
    elif choice == "4":
        return
    
    else:
        print("Invalid option.")

def browse_files():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder to Load Files")
    return folder_path

def generate_file_paths(folder_path):
    """Generator yielding file paths one by one with progress tracking"""
    file_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1][1:].lower()
            if ext in ["txt", "csv", "json"]:
                file_count += 1
                yield file_path
    return file_count  # Return total count for progress tracking

def count_files(folder_path):
    """Count supported files without storing paths"""
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1][1:].lower()
            if ext in ["txt", "csv", "json"]:
                count += 1
    return count

def configure_postgres():
    """Apply PostgreSQL configuration optimizations"""
    try:
        conn = psycopg2.connect(
            dbname="rag_db",
            user="postgres",
            password="postgres",
            host="localhost",
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

# Main menu
async def main():
    global embedding_cache, last_cache_refresh
    init_db()
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
        choice = input("Enter your choice (1-8): ")

        if choice == "1":
            ask_inference(
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
                            dbname="rag_db",
                            user="postgres",
                            password="postgres",
                            host="localhost",
                            connect_timeout=5
                        )
                        cur = conn.cursor()
                        for record in records:
                            cur.execute(
                                "INSERT INTO documents (content, tags, embedding) VALUES (%s, %s, %s)",
                                record
                            )
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
        elif choice == "9":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
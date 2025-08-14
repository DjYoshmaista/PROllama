# rag_db.py
import psycopg2
import requests
import asyncpg
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
from async_loader import run_processing
from config import Config
from db import db_manager
from utils import *
from constants import *
import logging
from em_cache import EmbeddingCache
from progress_manager import SimpleProgressTracker
from file_tracker import file_tracker

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_db.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

table_name = Config.TABLE_NAME

# Global embedding cache
embedding_cache = EmbeddingCache()
        
def init_db():
    """Initialize database with schema migration support"""
    try:
        with db_cursor() as (conn, cur):
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
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

# Add data to the database
def add_data():
    global embedding_cache
    content = input("Enter the content to add: ")
    if not content.strip():
        print("Content cannot be empty!")
        return
    
    try:
        # Use simple progress tracker for single operation
        tracker = SimpleProgressTracker("Generating embedding")
        tracker.update(0, 1, "Processing...")
        
        embedding = get_embedding(content)
        tracker.update(1, 1, "Complete")
        
        with db_cursor() as (conn, cur):
            query = f"INSERT INTO {table_name} (content, embedding) VALUES (%s, %s)"
            cur.execute(query, (content, embedding))
            conn.commit()
            
        tracker.complete(1)
        print("Data added successfully.")
        # Reset cache loaded flag to force reload
        embedding_cache.db_loaded = False
    except Exception as e:
        logger.error(f"Error adding data: {str(e)}")
        print("Failed to add data.")

# Query database directly
def query_db():
    try:
        with db_cursor() as (conn, cur):
            query = f"SELECT id, content FROM {table_name}"
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                print(f"ID: {row[0]}, Content: {row[1][:1000]}...")
    except Exception as e:
        logger.error(f"Database query failed: {str(e)}")
        print("Failed to query database.")

# List full contents
def list_contents():
    try:
        with db_cursor() as (conn, cur):
            query = f"SELECT id, content FROM {table_name}"
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                print(f"ID: {row[0]}, Content: {row[1][:200]}...")
    except Exception as e:
        logger.error(f"Failed to list contents: {str(e)}")
        print("Failed to list database contents.")

async def ask_inference_async(
    relevance_threshold=None,
    top_k=None,
    vector_search_limit=None
):
    """Wrapper function for inference pipeline"""
    # Use config defaults if not provided
    relevance_threshold = relevance_threshold or Config.SEARCH_DEFAULTS['relevance_threshold']
    top_k = top_k or Config.SEARCH_DEFAULTS['top_k'] 
    vector_search_limit = vector_search_limit or Config.SEARCH_DEFAULTS['vector_search_limit']
    
    question = get_user_question()
    if not question:
        return

    # Use simple progress tracker for inference
    tracker = SimpleProgressTracker("Processing query")
    tracker.update(0, 4, "Generating question embedding...")
    
    question_embedding = await get_question_embedding(question)
    if not question_embedding:
        return

    tracker.update(1, 4, "Checking cache...")
    use_cache = handle_cache_decision()
    
    tracker.update(2, 4, "Searching for relevant documents...")
    if use_cache and embedding_cache.stats['memory_entries'] > 0:
        top_docs = await perform_gpu_cache_search(
            question_embedding,
            relevance_threshold,
            top_k
        )
    else:
        CHUNK_SIZE = min(vector_search_limit, 500000)
        top_docs = perform_chunked_database_search(
            question_embedding,
            relevance_threshold,
            top_k,
            CHUNK_SIZE
        )
    
    if not top_docs:
        tracker.complete(0)
        return

    tracker.update(3, 4, "Retrieving document contents...")
    final_docs = retrieve_document_contents(top_docs)
    if not final_docs:
        tracker.complete(0)
        return

    tracker.update(4, 4, "Generating answer...")
    generate_and_display_answer(question, final_docs)
    tracker.complete(len(final_docs))

def configure_embedding_parameters():
    """Menu for configuring embedding parameters"""
    print("\nEmbedding Configuration:")
    print(f"1. Relevance Threshold (current: {Config.SEARCH_DEFAULTS['relevance_threshold']:.2f})")
    print(f"2. Top K Results (current: {Config.SEARCH_DEFAULTS['top_k']})")
    print(f"3. Vector Search Limit/Chunk Size (current: {Config.SEARCH_DEFAULTS['vector_search_limit']})")
    print("4. Back to main menu")
    choice = input("Select option: ")
    if choice == "1":
        try:
            new_threshold = float(input("Enter new relevance threshold (0.0-1.0): "))
            if 0.0 <= new_threshold <= 1.0:
                Config.SEARCH_DEFAULTS['relevance_threshold'] = new_threshold
                print("Threshold updated.")
            else:
                print("Invalid value. Must be between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a number.")
    elif choice == "2":
        try:
            new_top_k = int(input("Enter new Top K value: "))
            if new_top_k > 0:
                Config.SEARCH_DEFAULTS['top_k'] = new_top_k
                print("Top K updated.")
            else:
                print("Value must be positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    elif choice == "3":
        try:
            new_limit = int(input("Enter new vector search limit/chunk size: "))
            if new_limit > 0:
                Config.SEARCH_DEFAULTS['vector_search_limit'] = new_limit
                print("Vector search limit/chunk size updated.")
            else:
                print("Value must be positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
    elif choice == "4":
        return
    else:
        print("Invalid option.")

def get_folder_path():
    """Simplified file browser"""
    path = input("Enter folder path (absolute or relative): ").strip()
    path = os.path.expanduser(path)
    
    if not os.path.exists(path):
        print("Path does not exist")
        return None
        
    if not os.path.isdir(path):
        print("Path is not a directory")
        return None
        
    return path

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

def generate_file_paths(folder_path):
    """Generator yielding file paths one by one with progress tracking"""
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1][1:].lower()
            if ext in ["py", "txt", "csv", "json"]:
                yield file_path

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
        with db_cursor() as (conn, cur):
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

def show_file_tracker_stats():
    """Display file tracker statistics"""
    stats = file_tracker.get_processed_files_stats()
    print(f"\n📊 File Processing Statistics:")
    print(f"   Total processed files: {stats['total_files']}")
    print(f"   Total records created: {stats['total_records']}")
    print(f"   Total file size: {stats['total_size_mb']:.2f} MB")
    print(f"   Average records per file: {stats['avg_records_per_file']:.1f}")
    
    # Show recent files
    if file_tracker.processed_files:
        print(f"\n📁 Recently processed files (last 5):")
        recent_files = sorted(
            file_tracker.processed_files.items(), 
            key=lambda x: x[1].processed_at, 
            reverse=True
        )[:5]
        
        for i, (filepath, record) in enumerate(recent_files, 1):
            filename = os.path.basename(filepath)
            print(f"   {i}. {filename:<40} ({record.records_count} records)")

# Main menu
async def main():
    global embedding_cache
    init_db()

    if not validate_embedding_model():
        emb_mdl = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
        print(f"CRITICAL: Embedding model {emb_mdl} not available!")
        return

    from embedding_queue import embedding_queue
    if not embedding_queue.started:
        await embedding_queue.start_workers(concurrency=10)
        logger.info("Starting embedding queue...")

    # Show file tracker stats on startup
    show_file_tracker_stats()

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
        print("9. Show file processing statistics")
        print("10. Clean up missing file records")
        print("11. Exit")
        choice = input("Enter your choice (1-11): ")
        if choice == "1":
            await ask_inference_async()
        elif choice == "2":
            add_data()
        elif choice == "3":
            file_path = filedialog.askopenfilename(title="Select File to Load")
            if file_path:
                file_type = os.path.splitext(file_path)[1][1:].lower()
                try:
                    from load_documents import load_file
                    print(f"Loading file: {file_path} as {file_type}")
                    
                    # Use simple progress tracker for single file
                    tracker = SimpleProgressTracker(f"Loading {os.path.basename(file_path)}")
                    tracker.update(0, 3, "Parsing file...")
                    
                    async with aiohttp.ClientSession() as session:
                        records = await load_file(file_path, file_type, session)
                    
                    tracker.update(1, 3, "Inserting to database...")
                    if records:
                        with db_cursor() as (conn, cur):
                            for i, record in enumerate(records):
                                content = record["content"]
                                tags = record["tags"]  # This is a list
                                embedding = record["embedding"]

                                # Serialize tags list to JSON string
                                tags_json = json.dumps(tags)

                                query = f"INSERT INTO {table_name} (content, tags, embedding) VALUES (%s, %s, %s)"
                                cur.execute(query, (content, tags_json, embedding))
                                
                                # Update progress
                                if i % 10 == 0:
                                    tracker.update(1 + i/len(records), 3, f"Inserting record {i+1}/{len(records)}")
                                
                            conn.commit()
                        
                        tracker.update(2, 3, "Updating file tracker...")
                        # Mark file as processed
                        file_tracker.mark_file_processed(file_path, len(records))
                        file_tracker.save_tracker()
                        
                        tracker.complete(len(records))
                        print(f"Inserted {len(records)} records from file.")
                        # Reset cache loaded flag
                        embedding_cache.db_loaded = False
                    else:
                        tracker.complete(0)
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
                total_processed = await run_processing(file_generator, total_files)
                # Reset cache loaded flag after bulk load
                embedding_cache.db_loaded = False
                print(f"\n✅ Finished! Total documents loaded: {total_processed}")
        elif choice == "5":
            query_db()
        elif choice == "6":
            list_contents()
        elif choice == "7":
            configure_embedding_parameters()
        elif choice == "8":
            configure_postgres()
        elif choice == "9":
            show_file_tracker_stats()
        elif choice == "10":
            missing_files = file_tracker.cleanup_missing_files()
            if missing_files:
                print(f"Cleaned up {len(missing_files)} missing file records")
            else:
                print("No missing files found")
        elif choice == "11":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    asyncio.run(main())
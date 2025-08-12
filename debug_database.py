# debug_database.py - Quick database debugging tool
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from database.operations import db_ops
from database.connection import db_connection
from core.config import config

async def debug_database():
    """Debug database connectivity and vector operations"""
    print("=== Database Debug Tool ===")
    
    # Test basic connectivity
    print("\n1. Testing database connection...")
    try:
        is_connected = db_connection.test_connection()
        print(f"Sync connection: {'✓ OK' if is_connected else '✗ FAILED'}")
        
        is_async_connected = await db_connection.test_async_connection()
        print(f"Async connection: {'✓ OK' if is_async_connected else '✗ FAILED'}")
    except Exception as e:
        print(f"Connection test failed: {e}")
        return
    
    # Check document count
    print("\n2. Checking document count...")
    try:
        doc_count = db_ops.get_document_count()
        print(f"Total documents: {doc_count}")
        
        if doc_count == 0:
            print("No documents found. Please load some data first.")
            return
    except Exception as e:
        print(f"Document count failed: {e}")
        return
    
    # Test vector extension
    print("\n3. Testing vector extension...")
    try:
        with db_connection.get_sync_connection() as (conn, cur):
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            extension = cur.fetchone()
            if extension:
                print("✓ pgvector extension is installed")
            else:
                print("✗ pgvector extension not found")
                print("Please install with: CREATE EXTENSION vector;")
                return
    except Exception as e:
        print(f"Vector extension check failed: {e}")
        return
    
    # Check table structure
    print("\n4. Checking table structure...")
    try:
        with db_connection.get_sync_connection() as (conn, cur):
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{config.database.table_name}'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            
            print(f"Table '{config.database.table_name}' columns:")
            for col_name, col_type in columns:
                print(f"  {col_name}: {col_type}")
            
            # Check if embedding column exists and has correct type
            embedding_col = None
            for col_name, col_type in columns:
                if col_name == 'embedding':
                    embedding_col = col_type
                    break
            
            if embedding_col:
                print(f"✓ Embedding column found: {embedding_col}")
                if 'vector' in embedding_col.lower():
                    print("✓ Embedding column has vector type")
                else:
                    print(f"⚠ Warning: Embedding column type is {embedding_col}, expected vector")
            else:
                print("✗ Embedding column not found")
                return
                
    except Exception as e:
        print(f"Table structure check failed: {e}")
        return
    
    # Test sample embedding query
    print("\n5. Testing sample vector query...")
    try:
        with db_connection.get_sync_connection() as (conn, cur):
            # Get a sample embedding
            cur.execute(f"SELECT embedding FROM {config.database.table_name} LIMIT 1")
            sample = cur.fetchone()
            
            if sample and sample[0]:
                sample_embedding = sample[0]
                print(f"✓ Sample embedding found (type: {type(sample_embedding)})")
                
                # Test vector operations
                if isinstance(sample_embedding, (list, tuple)):
                    embedding_str = '[' + ','.join(map(str, sample_embedding[:5])) + ',...'
                    print(f"✓ Sample embedding format: {embedding_str}")
                elif isinstance(sample_embedding, str):
                    print(f"✓ Sample embedding (string): {sample_embedding[:50]}...")
                
                # Test basic vector query with correct dimensions
                # Create a test vector with 1024 dimensions (same as your embeddings)
                test_vector_1024 = '[' + ','.join(['0.001'] * 1024) + ']'
                cur.execute(f"""
                    SELECT id, 1 - (embedding <=> '{test_vector_1024}'::vector) as similarity
                    FROM {config.database.table_name}
                    LIMIT 1
                """)
                result = cur.fetchone()
                
                if result:
                    print(f"✓ Vector similarity query works: {result}")
                else:
                    print("✗ Vector similarity query returned no results")
                    
            else:
                print("✗ No sample embedding found")
                
    except Exception as e:
        print(f"Vector query test failed: {e}")
        print(f"Error details: {type(e).__name__}: {e}")
        
        # Try to get more specific error info
        if "syntax error" in str(e).lower():
            print("\nPossible issues:")
            print("1. pgvector extension not properly installed")
            print("2. Embedding column not using vector type")
            print("3. Vector syntax not supported in this PostgreSQL version")
        
        return
    
    print("\n=== Debug Complete ===")
    print("Database appears to be properly configured for vector operations.")

if __name__ == "__main__":
    asyncio.run(debug_database())
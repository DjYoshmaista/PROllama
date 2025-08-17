#!/usr/bin/env python3
"""
embedding_test.py - FIXED test script for the embedding system
"""
import os
os.environ['OLLAMA_NUM_PARALLEL'] = '8'
import asyncio
import aiohttp
import json
import logging
import sys
from constants import OLLAMA_API, EMBEDDING_MODEL
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ollama_api():
    """Test basic Ollama API connectivity and model availability"""
    print("🔍 Testing Ollama API...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test 1: Check if Ollama is running
            print("  Checking if Ollama is running...")
            async with session.get(f"{OLLAMA_API}/tags", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    print(f"  ✅ Ollama is running. Found {len(models)} models:")
                    for model in models[:5]:  # Show first 5 models
                        print(f"    - {model}")
                    if len(models) > 5:
                        print(f"    ... and {len(models) - 5} more")
                    
                    # Check if our embedding model is available
                    if EMBEDDING_MODEL in models:
                        print(f"  ✅ Embedding model '{EMBEDDING_MODEL}' is available")
                    else:
                        print(f"  ❌ Embedding model '{EMBEDDING_MODEL}' NOT FOUND")
                        print(f"  Available models: {models}")
                        return False
                else:
                    print(f"  ❌ Ollama API returned status {response.status}")
                    return False
            
            # Test 2: Try to generate an embedding
            print("  Testing embedding generation...")
            test_text = "This is a test sentence for embedding generation."
            
            # FIXED: Use correct endpoint path
            embed_url = f"{OLLAMA_API}/embed"
            print(f"  Making request to: {embed_url}")
            
            async with session.post(
                embed_url,
                json={"model": EMBEDDING_MODEL, "input": test_text},
                timeout=30
            ) as response:
                print(f"  Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"  Response keys: {list(data.keys())}")
                    
                    # Handle both possible response formats
                    embedding = None
                    if "embedding" in data:
                        embedding = data["embedding"]
                    elif "embeddings" in data and data["embeddings"]:
                        embedding = data["embeddings"][0]
                    
                    if embedding:
                        print(f"  ✅ Embedding generated successfully!")
                        print(f"    - Dimension: {len(embedding)}")
                        print(f"    - Expected dimension: {Config.EMBEDDING_DIM}")
                        print(f"    - Sample values: {embedding[:5]}...")
                        
                        if len(embedding) == Config.EMBEDDING_DIM:
                            print(f"  ✅ Embedding dimension matches expected value")
                            return True
                        else:
                            print(f"  ❌ Embedding dimension mismatch!")
                            return False
                    else:
                        print(f"  ❌ No embedding in response: {data}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"  ❌ Embedding API returned status {response.status}")
                    print(f"    Error: {error_text}")
                    return False
                    
    except asyncio.TimeoutError:
        print("  ❌ Timeout connecting to Ollama API")
        return False
    except Exception as e:
        print(f"  ❌ Error testing Ollama API: {e}")
        return False

async def test_embedding_queue():
    """Test the embedding queue system"""
    print("\n🔄 Testing Embedding Queue...")
    
    try:
        # FIXED: Clear any existing queue state first
        if os.path.exists("embedding_queue_state.json"):
            os.remove("embedding_queue_state.json")
            print("  Cleared previous queue state")
        
        # Import the fixed embedding queue
        from embedding_queue import EmbeddingQueue
        
        # Create a fresh instance
        queue = EmbeddingQueue()
        
        print("  Creating test callback function...")
        processed_records = []
        
        async def test_callback(records):
            """Test callback that just stores records"""
            processed_records.extend(records)
            print(f"    Callback received {len(records)} records")
            for record in records:
                print(f"      - Content: {record['content'][:50]}...")
                print(f"      - Tags: {record['tags']}")
                print(f"      - Embedding dim: {len(record['embedding'])}")
        
        print("  Starting embedding queue workers...")
        await queue.start_workers(concurrency=2, insert_callback=test_callback)
        
        if queue.started:
            print("  ✅ Embedding queue started successfully")
        else:
            print("  ❌ Embedding queue failed to start")
            return False
        
        print("  Queuing test items...")
        test_items = [
            ("First test sentence for embedding.", ["test1"]),
            ("Second test sentence with different content.", ["test2"]),
            ("Third and final test sentence.", ["test3"])
        ]
        
        queued_count = 0
        for i, (content, tags) in enumerate(test_items):
            success = await queue.enqueue_for_embedding(
                content=content,
                tags=tags,
                file_path=f"test_file_{i}.txt",
                chunk_index=i
            )
            if success:
                queued_count += 1
                print(f"    ✅ Queued item {i+1}")
            else:
                print(f"    ❌ Failed to queue item {i+1}")
        
        print(f"  Successfully queued {queued_count}/{len(test_items)} items")
        
        # Wait for processing with better monitoring
        print("  Waiting for queue processing...")
        wait_count = 0
        max_wait = 60  # 1 minute timeout
        
        while queue.stats['queue_size'] > 0 and wait_count < max_wait:
            await asyncio.sleep(1)
            wait_count += 1
            if wait_count % 5 == 0:
                stats = queue.stats
                print(f"    Queue status: {stats['queue_size']} remaining, {stats['processed_items']} processed, {stats['failed_items']} failed")
        
        # Check results
        final_stats = queue.stats
        print(f"  Final queue stats:")
        print(f"    - Processed items: {final_stats['processed_items']}")
        print(f"    - Failed items: {final_stats['failed_items']}")
        print(f"    - Queue size: {final_stats['queue_size']}")
        
        print(f"  Callback received {len(processed_records)} records")
        
        # FIXED: Properly stop workers and wait for completion
        print("  Stopping workers and waiting for completion...")
        await queue.stop_workers()
        print("  ✅ Embedding queue stopped")
        
        # Success if we processed all items
        success = (final_stats['processed_items'] == len(test_items) and 
                  len(processed_records) == len(test_items))
        
        if success:
            print("  ✅ All items processed successfully")
        else:
            print(f"  ❌ Processing incomplete: {final_stats['processed_items']}/{len(test_items)} processed")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Error testing embedding queue: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_file_processing():
    """Test file processing pipeline"""
    print("\n📁 Testing File Processing...")
    
    try:
        # Create a test CSV file
        test_file = "test_data.csv"
        test_content = """name,age,description
Alice,25,"A software engineer who loves coding"
Bob,30,"A data scientist working with machine learning"
Charlie,35,"A product manager with technical background"
"""
        
        print(f"  Creating test file: {test_file}")
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Test parsing
        print("  Testing file parsing...")
        import parse_documents
        
        chunk_count = 0
        item_count = 0
        
        for chunk in parse_documents.stream_parse_file(test_file, 'csv', 10):
            chunk_count += 1
            item_count += len(chunk)
            print(f"    Chunk {chunk_count}: {len(chunk)} items")
            
            # Show first item
            if chunk:
                first_item = chunk[0]
                print(f"      Sample item: {first_item['content'][:100]}...")
                print(f"      Tags: {first_item['tags']}")
        
        print(f"  ✅ Parsed {chunk_count} chunks with {item_count} total items")
        
        # Test with embedding queue - FIXED: Ensure queue is ready
        print("  Testing file processing with embedding queue...")
        from embedding_queue import EmbeddingQueue
        from async_loader import database_insert_callback
        
        # Create fresh queue instance for this test
        queue = EmbeddingQueue()
        
        # Start the queue with proper callback
        await queue.start_workers(concurrency=2, insert_callback=database_insert_callback)
        
        # Import the process function
        from async_loader import process_file_with_queue
        
        def progress_callback(file_path, stage, current, total, message):
            print(f"    Progress: {stage} - {message}")
        
        result = await process_file_with_queue(test_file, progress_callback)
        file_path, success, items_queued = result
        
        if success:
            print(f"  ✅ File processing successful: {items_queued} items queued")
            
            # Wait for processing
            print("  Waiting for queue processing...")
            wait_count = 0
            while queue.stats['queue_size'] > 0 and wait_count < 30:
                await asyncio.sleep(1)
                wait_count += 1
                if wait_count % 5 == 0:
                    print(f"    Queue: {queue.stats['queue_size']} remaining")
            
            final_stats = queue.stats
            print(f"  Final processing stats: {final_stats['processed_items']} processed, {final_stats['failed_items']} failed")
        else:
            print(f"  ❌ File processing failed")
        
        # Stop queue
        await queue.stop_workers()
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"  Cleaned up test file")
        
        return success
        
    except Exception as e:
        print(f"  ❌ Error testing file processing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_integration():
    """Test database integration"""
    print("\n🗄️ Testing Database Integration...")
    
    try:
        from db import db_cursor
        from config import Config
        
        # Test database connection
        print("  Testing database connection...")
        with db_cursor() as (conn, cur):
            cur.execute("SELECT 1")
            result = cur.fetchone()
            # FIXED: Handle different cursor result formats
            result_value = result[0] if isinstance(result, (list, tuple)) else result.get(0) or result.get('?column?')
            if result_value == 1:
                print("  ✅ Database connection successful")
            else:
                print(f"  ❌ Database connection failed - unexpected result: {result}")
                return False
        
        # Test table existence
        print(f"  Checking table '{Config.TABLE_NAME}'...")
        with db_cursor() as (conn, cur):
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{Config.TABLE_NAME}'
                )
            """)
            table_exists = cur.fetchone()[0]
            
            if table_exists:
                print(f"  ✅ Table '{Config.TABLE_NAME}' exists")
                
                # Get record count - FIXED: Handle cursor result format
                cur.execute(f"SELECT COUNT(*) FROM {Config.TABLE_NAME}")
                count_result = cur.fetchone()
                count = count_result[0] if isinstance(count_result, (list, tuple)) else count_result.get('count', 0)
                print(f"  📊 Current record count: {count}")
            else:
                print(f"  ❌ Table '{Config.TABLE_NAME}' does not exist")
                return False
        
        # Test vector extension
        print("  Testing pgvector extension...")
        with db_cursor() as (conn, cur):
            cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            vector_ext = cur.fetchone()
            
            if vector_ext:
                print("  ✅ pgvector extension is installed")
            else:
                print("  ❌ pgvector extension not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing database: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_embedding():
    """Simple direct embedding test"""
    print("\n🧪 Testing Direct Embedding Generation...")
    
    try:
        print("  Testing direct API call...")
        
        async with aiohttp.ClientSession() as session:
            test_text = "Simple test for direct embedding"
            
            async with session.post(
                f"{OLLAMA_API}/embed",
                json={"model": EMBEDDING_MODEL, "input": test_text},
                timeout=15
            ) as response:
                print(f"  Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    embedding = data.get("embedding") or (data.get("embeddings")[0] if data.get("embeddings") else None)
                    
                    if embedding and len(embedding) == Config.EMBEDDING_DIM:
                        print(f"  ✅ Direct embedding successful (dim: {len(embedding)})")
                        return True
                    else:
                        print(f"  ❌ Invalid embedding response")
                        return False
                else:
                    error = await response.text()
                    print(f"  ❌ API error: {error}")
                    return False
    except Exception as e:
        print(f"  ❌ Direct embedding test failed: {e}")
        return False

async def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🚀 Starting Comprehensive Embedding System Test")
    print("=" * 60)
    
    tests = [
        ("Direct Embedding", test_simple_embedding),
        ("Ollama API", test_ollama_api),
        ("Database Integration", test_database_integration),
        ("Embedding Queue", test_embedding_queue),
        ("File Processing", test_file_processing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = await test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your embedding system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
        # Provide specific guidance
        if not results.get("Direct Embedding", False):
            print("\n🔧 SUGGESTION: Check your Ollama installation and model availability")
        if not results.get("Embedding Queue", False):
            print("\n🔧 SUGGESTION: There may be API timeout or connection issues")
    
    return passed == total

async def main():
    """Main test function"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "api":
            await test_ollama_api()
        elif test_name == "queue":
            await test_embedding_queue()
        elif test_name == "file":
            await test_file_processing()
        elif test_name == "db":
            await test_database_integration()
        elif test_name == "simple":
            await test_simple_embedding()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: api, queue, file, db, simple")
    else:
        await run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
quick_test.py - Quick test to verify embedding queue is working
"""

import asyncio
import logging
import time

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_embedding_queue_simple():
    """Simple test of the embedding queue"""
    print("🔄 Quick Embedding Queue Test")
    print("=" * 40)
    
    try:
        # Import the queue
        from embedding_queue import EmbeddingQueue
        
        # Create fresh instance
        queue = EmbeddingQueue()
        
        # Simple callback to collect results
        results = []
        
        async def simple_callback(records):
            results.extend(records)
            print(f"✅ Received {len(records)} processed records")
            for record in records:
                print(f"   - Content: {record['content'][:50]}...")
                print(f"   - Embedding length: {len(record['embedding'])}")
        
        # Start workers
        print("Starting workers...")
        await queue.start_workers(concurrency=1, insert_callback=simple_callback)
        
        # Queue some test items
        test_texts = [
            "This is the first test sentence.",
            "Here is another test for embedding generation.",
            "Final test sentence to verify the system works."
        ]
        
        print(f"Queueing {len(test_texts)} test items...")
        for i, text in enumerate(test_texts):
            success = await queue.enqueue_for_embedding(
                content=text,
                tags=[f"test_{i}"],
                file_path=f"test_{i}.txt",
                chunk_index=i
            )
            if success:
                print(f"   ✅ Queued item {i+1}")
            else:
                print(f"   ❌ Failed to queue item {i+1}")
        
        # Wait for processing
        print("Waiting for processing...")
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        while queue.stats['queue_size'] > 0 and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)
            stats = queue.stats
            print(f"   Queue: {stats['queue_size']} items, Processed: {stats['processed_items']}, Failed: {stats['failed_items']}")
        
        # Final results
        final_stats = queue.stats
        print(f"\nFinal Results:")
        print(f"   - Processed: {final_stats['processed_items']}")
        print(f"   - Failed: {final_stats['failed_items']}")
        print(f"   - Received in callback: {len(results)}")
        
        # Stop workers
        await queue.stop_workers()
        
        # Success check
        if final_stats['processed_items'] == len(test_texts) and len(results) == len(test_texts):
            print("\n🎉 SUCCESS: All embeddings generated successfully!")
            return True
        else:
            print(f"\n❌ FAILED: Expected {len(test_texts)} items, got {final_stats['processed_items']} processed, {len(results)} in callback")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_embedding_queue_simple()
    
    if success:
        print("\n✅ Your embedding queue is working correctly!")
        print("You can now use option 4 in the main program.")
    else:
        print("\n❌ There are still issues with the embedding queue.")
        print("Check the debug output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
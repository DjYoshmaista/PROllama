# file_management/streaming_loader.py - Streaming Async Document Loader
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path

from core.config import config
from core.memory import memory_manager
from inference.async_embeddings import async_embedding_service
from file_management.parsers import document_parser
from file_management.chunking import text_splitter, TextChunk
from database.batch_operations import batch_db_ops

logger = logging.getLogger(__name__)

class StreamingAsyncLoader:
    """Streaming async document loader that processes files as they're discovered"""
    
    def __init__(self):
        self.batch_size = 20  # Process more files concurrently
        self.max_concurrent_files = 50  # Higher concurrency for better throughput
        self.progress_interval = 50  # Show progress every N files
    
    async def load_folder_streaming(
        self,
        folder_path: str,
        file_generator: Generator[str, None, None],
        total_files: int,
        enable_chunking: bool = True,
        enable_summarization: bool = False
    ) -> Dict[str, Any]:
        """Stream files and process them in batches with immediate progress"""
        
        logger.info(f"Starting streaming async loading of {total_files} files")
        
        # Initialize services
        print("🔄 Initializing services...")
        await async_embedding_service.start()
        await batch_db_ops.initialize_connection_pool()
        
        print(f"🚀 Starting streaming processing...")
        print(f"📊 Concurrency: {self.max_concurrent_files} files, {self.batch_size} batch size")
        
        # Track progress
        processed_files = 0
        total_chunks = 0
        total_summaries = 0
        failed_files = 0
        start_time = time.time()
        last_progress_time = start_time
        
        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_files)
        
        # Queue for file processing tasks
        pending_tasks = []
        
        try:
            # Process files as they come from the generator
            file_count = 0
            
            async def process_file_with_semaphore(file_path: str):
                async with semaphore:
                    return await self._process_single_file_async(file_path)
            
            # Start processing files in streaming fashion
            for file_path in file_generator:
                file_count += 1
                
                # Create task for this file
                task = asyncio.create_task(process_file_with_semaphore(file_path))
                pending_tasks.append((file_path, task))
                
                # Process completed tasks when we have enough pending or reach batch size
                if len(pending_tasks) >= self.batch_size:
                    completed_count = await self._process_completed_tasks(
                        pending_tasks, processed_files, total_chunks, total_summaries, 
                        failed_files, start_time
                    )
                    
                    # Update counters
                    for result in completed_count:
                        if result:
                            total_chunks += result.get('chunks', 0)
                            total_summaries += result.get('summaries', 0)
                        else:
                            failed_files += 1
                        processed_files += 1
                    
                    # Show progress
                    if processed_files % self.progress_interval == 0 or time.time() - last_progress_time > 10:
                        self._show_progress(processed_files, total_files, total_chunks, 
                                          total_summaries, failed_files, start_time)
                        last_progress_time = time.time()
                    
                    # Clear completed tasks
                    pending_tasks = []
                
                # Early break for testing with large datasets
                if file_count >= 1000:  # Process first 1000 files for testing
                    logger.info(f"Processing first {file_count} files for testing")
                    break
            
            # Process any remaining tasks
            if pending_tasks:
                completed_count = await self._process_completed_tasks(
                    pending_tasks, processed_files, total_chunks, total_summaries, 
                    failed_files, start_time
                )
                
                for result in completed_count:
                    if result:
                        total_chunks += result.get('chunks', 0)
                        total_summaries += result.get('summaries', 0)
                    else:
                        failed_files += 1
                    processed_files += 1
            
            # Final results
            elapsed = time.time() - start_time
            results = {
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "total_summaries": total_summaries,
                "failed_files": failed_files,
                "success_rate": (processed_files - failed_files) / max(processed_files, 1),
                "processing_time": elapsed,
                "files_per_second": processed_files / elapsed if elapsed > 0 else 0
            }
            
            print(f"\n✅ Streaming async loading completed!")
            print(f"📁 Files processed: {processed_files}")
            print(f"📝 Chunks created: {total_chunks}")
            print(f"📑 Summaries created: {total_summaries}")
            print(f"❌ Failed files: {failed_files}")
            print(f"⏱️  Total time: {elapsed:.1f} seconds")
            print(f"⚡ Speed: {results['files_per_second']:.1f} files/second")
            
            return results
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted by user")
            print(f"📁 Files processed so far: {processed_files}")
            print(f"📝 Chunks created: {total_chunks}")
            
            # Cancel pending tasks
            for _, task in pending_tasks:
                if not task.done():
                    task.cancel()
            
            elapsed = time.time() - start_time
            return {
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "total_summaries": total_summaries,
                "failed_files": failed_files,
                "processing_time": elapsed,
                "interrupted": True
            }
            
        except Exception as e:
            logger.error(f"Streaming async loading failed: {e}")
            return {
                "error": str(e),
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "total_summaries": total_summaries,
                "failed_files": failed_files
            }
        
        finally:
            # Cleanup
            print("🔄 Cleaning up services...")
            await async_embedding_service.stop()
            await batch_db_ops.close_connection_pool()
    
    async def _process_completed_tasks(
        self, 
        pending_tasks: List, 
        processed_files: int, 
        total_chunks: int, 
        total_summaries: int,
        failed_files: int,
        start_time: float
    ) -> List:
        """Process completed tasks and return results"""
        
        # Wait for all tasks in this batch to complete
        file_paths = [fp for fp, _ in pending_tasks]
        tasks = [task for _, task in pending_tasks]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        completed_results = []
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {file_path}: {result}")
                completed_results.append(None)
            else:
                completed_results.append(result)
        
        return completed_results
    
    def _show_progress(
        self, 
        processed_files: int, 
        total_files: int, 
        total_chunks: int,
        total_summaries: int, 
        failed_files: int, 
        start_time: float
    ):
        """Show progress update"""
        elapsed = time.time() - start_time
        files_per_sec = processed_files / elapsed if elapsed > 0 else 0
        progress_percent = (processed_files / total_files) * 100 if total_files > 0 else 0
        
        # Create progress bar
        bar_width = 30
        filled = int((processed_files / max(total_files, 1)) * bar_width)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print(f"📊 [{bar}] {processed_files:,}/{total_files:,} ({progress_percent:.1f}%) | "
              f"📝 {total_chunks:,} chunks | ❌ {failed_files:,} failed | "
              f"⚡ {files_per_sec:.1f} files/s | ⏱️  {elapsed:.0f}s")
    
    async def _process_single_file_async(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single file asynchronously"""
        try:
            # Check if file exists and is readable
            if not Path(file_path).exists():
                return None
            
            # Skip very large files to avoid memory issues
            file_size = Path(file_path).stat().st_size
            if file_size > 10 * 1024 * 1024:  # Skip files > 10MB
                logger.debug(f"Skipping large file: {file_path} ({file_size} bytes)")
                return None
            
            # Determine file type
            file_type = Path(file_path).suffix[1:].lower()
            
            # Parse file content
            parsed_content = await self._parse_file_async(file_path, file_type)
            if not parsed_content:
                return None
            
            # Create chunks
            all_chunks = []
            for content_item in parsed_content:
                chunks = self._create_chunks_sync(content_item, file_path)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                return None
            
            # Limit chunks per file to avoid overwhelming the system
            if len(all_chunks) > 100:
                all_chunks = all_chunks[:100]  # Take first 100 chunks
            
            # Generate embeddings for chunks
            chunk_contents = [chunk.content for chunk in all_chunks]
            chunk_embeddings = await async_embedding_service.generate_embeddings_batch(chunk_contents)
            
            # Filter successful embeddings
            valid_chunks = []
            valid_embeddings = []
            
            for chunk, embedding in zip(all_chunks, chunk_embeddings):
                if embedding and len(embedding) == config.embedding.dimension:
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
            
            if not valid_chunks:
                return None
            
            # Insert chunks into database
            chunks_inserted = await batch_db_ops.insert_chunks_batch(valid_chunks, valid_embeddings)
            
            return {
                "chunks": chunks_inserted,
                "summaries": 0
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None
    
    async def _parse_file_async(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse file content asynchronously"""
        try:
            # Run the synchronous parser in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            parsed_content = await loop.run_in_executor(
                None, self._parse_file_sync, file_path, file_type
            )
            return parsed_content
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []
    
    def _parse_file_sync(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse file content synchronously"""
        try:
            parsed_content = []
            chunk_generator = document_parser.parse_file_stream(
                file_path, file_type, chunk_size=5  # Smaller chunks to avoid memory issues
            )
            
            for chunk_batch in chunk_generator:
                parsed_content.extend(chunk_batch)
                # Limit parsed content to avoid memory issues
                if len(parsed_content) > 10:
                    break
            
            return parsed_content
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []
    
    def _create_chunks_sync(self, content_item: Dict[str, Any], file_path: str) -> List[TextChunk]:
        """Create chunks synchronously"""
        content = content_item.get("content", "")
        if not content.strip():
            return []
        
        try:
            chunks = text_splitter.split_text_hierarchical(content, file_path)
            
            # Add tags
            original_tags = content_item.get("tags", [])
            for chunk in chunks:
                if hasattr(chunk, 'tags'):
                    chunk.tags = list(set(original_tags + getattr(chunk, 'tags', [])))
                else:
                    chunk.tags = original_tags
            
            return chunks
        except Exception as e:
            logger.debug(f"Failed to create chunks for {file_path}: {e}")
            return []

# Global streaming async loader instance
streaming_async_loader = StreamingAsyncLoader()
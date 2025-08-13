# file_management/simple_async_loader.py - Simple Pure Async Document Loader
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

class SimpleAsyncLoader:
    """Simple pure async document loader without threading conflicts"""
    
    def __init__(self):
        self.batch_size = 10  # Process files in small batches
        self.chunk_batch_size = 50  # Process chunks in batches
    
    async def load_folder_async(
        self,
        folder_path: str,
        file_generator: Generator[str, None, None],
        total_files: int,
        enable_chunking: bool = True,
        enable_summarization: bool = True
    ) -> Dict[str, Any]:
        """Simple async folder loading with progress tracking"""
        
        logger.info(f"Starting simple async loading of {total_files} files")
        
        # Initialize services
        await async_embedding_service.start()
        await batch_db_ops.initialize_connection_pool()
        
        # Convert generator to list for easier processing
        file_list = list(file_generator)
        total_files = len(file_list)
        
        print(f"\n🚀 Processing {total_files} files with simple async loader...")
        
        # Track progress
        processed_files = 0
        total_chunks = 0
        total_summaries = 0
        failed_files = 0
        start_time = time.time()
        
        try:
            # Process files in batches to avoid overwhelming the system
            for i in range(0, len(file_list), self.batch_size):
                batch = file_list[i:i + self.batch_size]
                
                # Process batch concurrently
                batch_tasks = [self._process_single_file_async(file_path) for file_path in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Update progress
                for file_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {file_path}: {result}")
                        failed_files += 1
                    elif result:
                        total_chunks += result.get('chunks', 0)
                        total_summaries += result.get('summaries', 0)
                    else:
                        failed_files += 1
                    
                    processed_files += 1
                
                # Show progress
                elapsed = time.time() - start_time
                files_per_sec = processed_files / elapsed if elapsed > 0 else 0
                progress_percent = (processed_files / total_files) * 100
                
                print(f"📁 Progress: {processed_files}/{total_files} ({progress_percent:.1f}%) | "
                      f"📝 Chunks: {total_chunks} | 📑 Summaries: {total_summaries} | "
                      f"❌ Failed: {failed_files} | ⚡ {files_per_sec:.1f} files/s")
                
                # Brief pause to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
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
            
            print(f"\n✅ Simple async loading completed!")
            print(f"📁 Files processed: {processed_files}")
            print(f"📝 Chunks created: {total_chunks}")
            print(f"📑 Summaries created: {total_summaries}")
            print(f"❌ Failed files: {failed_files}")
            print(f"⏱️  Total time: {elapsed:.1f} seconds")
            print(f"⚡ Speed: {results['files_per_second']:.1f} files/second")
            
            return results
            
        except Exception as e:
            logger.error(f"Simple async loading failed: {e}")
            return {
                "error": str(e),
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "total_summaries": total_summaries,
                "failed_files": failed_files
            }
        
        finally:
            # Cleanup
            await async_embedding_service.stop()
            await batch_db_ops.close_connection_pool()
    
    async def _process_single_file_async(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single file asynchronously"""
        try:
            # Check if file exists and is readable
            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
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
                logger.warning(f"No valid embeddings generated for {file_path}")
                return None
            
            # Insert chunks into database
            chunks_inserted = await batch_db_ops.insert_chunks_batch(valid_chunks, valid_embeddings)
            
            # For now, skip summaries to simplify and focus on getting embeddings working
            summaries_inserted = 0
            
            logger.info(f"Processed {file_path}: {chunks_inserted} chunks")
            
            return {
                "chunks": chunks_inserted,
                "summaries": summaries_inserted
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
            logger.error(f"Failed to parse {file_path}: {e}")
            return []
    
    def _parse_file_sync(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse file content synchronously"""
        try:
            parsed_content = []
            chunk_generator = document_parser.parse_file_stream(
                file_path, file_type, chunk_size=10
            )
            
            for chunk_batch in chunk_generator:
                parsed_content.extend(chunk_batch)
            
            return parsed_content
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
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
            logger.error(f"Failed to create chunks for {file_path}: {e}")
            return []

# Global simple async loader instance
simple_async_loader = SimpleAsyncLoader()
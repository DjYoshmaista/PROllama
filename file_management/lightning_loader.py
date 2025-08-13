# file_management/lightning_loader.py - Ultra-Fast Lightning Document Loader
import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.config import config
from inference.async_embeddings import async_embedding_service
from file_management.parsers import document_parser
from file_management.chunking import text_splitter, TextChunk
from database.batch_operations import batch_db_ops

logger = logging.getLogger(__name__)

class LightningLoader:
    """Ultra-fast lightning loader for massive datasets"""
    
    def __init__(self):
        self.max_concurrent_files = 50
        self.max_files_to_process = 2000  # Higher limit for lightning mode
        self.supported_extensions = {".py", ".txt", ".csv", ".json"}
        self.skip_dirs = {
            '__pycache__', 'node_modules', '.git', '.venv', 'venv', 
            'build', 'dist', '.pytest_cache', '.mypy_cache', 'target'
        }
    
    async def load_folder_lightning(
        self,
        folder_path: str,
        enable_chunking: bool = True
    ) -> Dict[str, Any]:
        """Lightning-fast loading with minimal overhead"""
        
        logger.info(f"Starting lightning loading from {folder_path}")
        
        # Initialize services
        print("⚡ Lightning Loader - Ultra-Fast Mode")
        print("🔄 Initializing services...")
        await async_embedding_service.start()
        await batch_db_ops.initialize_connection_pool()
        
        # Track progress
        processed_files = 0
        total_chunks = 0
        failed_files = 0
        start_time = time.time()
        
        # Ultra-fast discovery and processing
        semaphore = asyncio.Semaphore(self.max_concurrent_files)
        
        try:
            print(f"⚡ Starting lightning discovery and processing...")
            
            # Create tasks list
            tasks = []
            discovered = 0
            
            # Lightning-fast directory scanning
            for root, dirs, files in os.walk(folder_path):
                # Skip unwanted directories in-place (modifies dirs list)
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in self.skip_dirs]
                
                for file in files:
                    # Skip hidden files immediately
                    if file.startswith('.'):
                        continue
                    
                    # Quick extension check
                    if not any(file.lower().endswith(ext) for ext in self.supported_extensions):
                        continue
                    
                    file_path = os.path.join(root, file)
                    
                    # Minimal checks - just verify file exists and is readable
                    try:
                        if os.access(file_path, os.R_OK):
                            stat_info = os.stat(file_path)
                            # Quick size check
                            if 10 < stat_info.st_size < 5 * 1024 * 1024:  # 10 bytes < size < 5MB
                                # Create processing task immediately
                                task = asyncio.create_task(
                                    self._process_file_with_semaphore(file_path, semaphore)
                                )
                                tasks.append(task)
                                discovered += 1
                                
                                # Process in batches to avoid memory issues
                                if len(tasks) >= 100:
                                    results = await asyncio.gather(*tasks, return_exceptions=True)
                                    processed_files, total_chunks, failed_files = self._update_progress(
                                        results, processed_files, total_chunks, failed_files, start_time
                                    )
                                    tasks = []
                                
                                # Limit for performance
                                if discovered >= self.max_files_to_process:
                                    print(f"⚡ Lightning mode: Processing first {discovered} files")
                                    break
                                    
                    except (OSError, IOError):
                        continue
                
                # Break outer loop if limit reached
                if discovered >= self.max_files_to_process:
                    break
            
            # Process remaining tasks
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                processed_files, total_chunks, failed_files = self._update_progress(
                    results, processed_files, total_chunks, failed_files, start_time
                )
            
            # Final results
            elapsed = time.time() - start_time
            results = {
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "failed_files": failed_files,
                "success_rate": (processed_files - failed_files) / max(processed_files, 1),
                "processing_time": elapsed,
                "files_per_second": processed_files / elapsed if elapsed > 0 else 0
            }
            
            print(f"\\n⚡ Lightning loading completed!")
            print(f"📁 Files processed: {processed_files}")
            print(f"📝 Chunks created: {total_chunks}")
            print(f"❌ Failed files: {failed_files}")
            print(f"⏱️  Total time: {elapsed:.1f} seconds")
            print(f"🚀 Lightning speed: {results['files_per_second']:.1f} files/second")
            
            return results
            
        except Exception as e:
            logger.error(f"Lightning loading failed: {e}")
            return {
                "error": str(e),
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "failed_files": failed_files
            }
        
        finally:
            # Cleanup
            print("🔄 Cleaning up services...")
            await async_embedding_service.stop()
            await batch_db_ops.close_connection_pool()
    
    async def _process_file_with_semaphore(self, file_path: str, semaphore: asyncio.Semaphore):
        """Process a single file with semaphore control"""
        async with semaphore:
            return await self._process_single_file(file_path)
    
    async def _process_single_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single file with minimal overhead"""
        try:
            # Determine file type
            file_type = Path(file_path).suffix[1:].lower()
            
            # Parse file content
            parsed_content = await self._parse_file_lightning(file_path, file_type)
            if not parsed_content:
                return None
            
            # Create chunks
            all_chunks = []
            for content_item in parsed_content[:3]:  # Limit to first 3 content items
                chunks = self._create_chunks_fast(content_item, file_path)
                all_chunks.extend(chunks[:20])  # Limit to 20 chunks per item
            
            if not all_chunks:
                return None
            
            # Generate embeddings
            chunk_contents = [chunk.content for chunk in all_chunks]
            chunk_embeddings = await async_embedding_service.generate_embeddings_batch(chunk_contents)
            
            # Filter valid embeddings
            valid_chunks = []
            valid_embeddings = []
            
            for chunk, embedding in zip(all_chunks, chunk_embeddings):
                if embedding and len(embedding) == config.embedding.dimension:
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
            
            if not valid_chunks:
                return None
            
            # Insert chunks
            chunks_inserted = await batch_db_ops.insert_chunks_batch(valid_chunks, valid_embeddings)
            
            return {
                "chunks": chunks_inserted,
                "summaries": 0
            }
            
        except Exception as e:
            logger.debug(f"Error processing file {file_path}: {e}")
            return None
    
    async def _parse_file_lightning(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Ultra-fast file parsing"""
        try:
            loop = asyncio.get_event_loop()
            parsed_content = await loop.run_in_executor(
                None, self._parse_file_sync, file_path, file_type
            )
            return parsed_content
        except Exception as e:
            return []
    
    def _parse_file_sync(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Synchronous file parsing with minimal overhead"""
        try:
            parsed_content = []
            chunk_generator = document_parser.parse_file_stream(
                file_path, file_type, chunk_size=2  # Very small chunks
            )
            
            for chunk_batch in chunk_generator:
                parsed_content.extend(chunk_batch)
                if len(parsed_content) >= 3:  # Stop after 3 items
                    break
            
            return parsed_content
        except Exception as e:
            return []
    
    def _create_chunks_fast(self, content_item: Dict[str, Any], file_path: str) -> List[TextChunk]:
        """Fast chunk creation"""
        content = content_item.get("content", "")
        if not content.strip() or len(content) < 50:  # Skip very short content
            return []
        
        try:
            chunks = text_splitter.split_text_hierarchical(content, file_path)
            
            # Add tags efficiently
            original_tags = content_item.get("tags", [])
            for chunk in chunks:
                chunk.tags = original_tags
            
            return chunks[:10]  # Limit to 10 chunks max
        except Exception as e:
            return []
    
    def _update_progress(self, results, processed_files, total_chunks, failed_files, start_time):
        """Update progress counters"""
        for result in results:
            if isinstance(result, Exception):
                failed_files += 1
            elif result:
                total_chunks += result.get('chunks', 0)
            else:
                failed_files += 1
            processed_files += 1
        
        # Show progress
        if processed_files % 50 == 0:
            elapsed = time.time() - start_time
            files_per_sec = processed_files / elapsed if elapsed > 0 else 0
            print(f"⚡ Lightning progress: {processed_files} files | {total_chunks} chunks | {files_per_sec:.1f} files/s")
        
        return processed_files, total_chunks, failed_files

# Global lightning loader instance
lightning_loader = LightningLoader()
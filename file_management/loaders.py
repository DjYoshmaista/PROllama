# file_management/loaders.py
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Generator, Optional
from pathlib import Path

from core.config import config
from core.memory import memory_manager
from inference.embeddings import embedding_service
from file_management.parsers import document_parser
from database.operations import db_ops

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles document loading with embedding generation and database insertion"""
    
    def __init__(self):
        self.memory_chunk_size = config.processing.memory_chunk_size
        self.embedding_chunk_size = config.processing.embedding_chunk_size
        self.max_concurrent_chunks = config.processing.max_concurrent_chunks_per_file
    
    async def load_file(self, file_path: str, file_type: Optional[str] = None) -> int:
        """
        Load a single file with streaming processing
        
        Args:
            file_path: Path to file
            file_type: Override file type detection
            
        Returns:
            Number of records successfully inserted
        """
        if not Path(file_path).exists() or Path(file_path).stat().st_size == 0:
            logger.info(f"Skipping empty or non-existent file: {file_path}")
            return 0
        
        total_inserted = 0
        
        try:
            # Determine file type if not provided
            if not file_type:
                file_type = Path(file_path).suffix[1:].lower()
            
            # Process file in streaming chunks
            chunk_generator = document_parser.parse_file_stream(
                file_path, file_type, self.memory_chunk_size
            )
            
            async for chunk in self._process_chunks_async(chunk_generator, file_path):
                if chunk:
                    inserted = await db_ops.insert_documents_batch(chunk)
                    total_inserted += inserted
                    
                    # Memory cleanup periodically
                    if total_inserted % (self.embedding_chunk_size * 5) == 0:
                        memory_manager.force_cleanup_if_needed()
            
            logger.info(f"Loaded {total_inserted} records from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
        
        return total_inserted
    
    async def _process_chunks_async(
        self, 
        chunk_generator: Generator[List[Dict[str, Any]], None, None],
        file_path: str
    ) -> Generator[List[Tuple[str, List[str], List[float]]], None, None]:
        """Process parsed chunks asynchronously with embedding generation"""
        
        for chunk in chunk_generator:
            try:
                processed_chunk = await self._process_single_chunk(chunk, file_path)
                if processed_chunk:
                    yield processed_chunk
            except Exception as e:
                logger.error(f"Error processing chunk from {file_path}: {e}")
    
    async def _process_single_chunk(
        self, 
        chunk: List[Dict[str, Any]], 
        file_path: str
    ) -> List[Tuple[str, List[str], List[float]]]:
        """Process a single chunk: validate, generate embeddings, prepare for DB"""
        
        # Filter valid content
        valid_items = [
            item for item in chunk 
            if item.get("content") and item["content"].strip()
        ]
        
        if not valid_items:
            return []
        
        # Extract content and tags
        contents = [item["content"] for item in valid_items]
        tags_lists = [item.get("tags", []) for item in valid_items]
        
        # Generate embeddings
        embeddings = await embedding_service.generate_embeddings_batch(contents)
        
        # Prepare records for database insertion
        records = []
        failed_count = 0
        
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                failed_count += 1
                continue
            
            # Validate embedding dimensions
            if len(embedding) != config.embedding.dimension:
                failed_count += 1
                logger.warning(f"Invalid embedding dimension for content: {contents[i][:100]}...")
                continue
            
            records.append((
                contents[i],
                tags_lists[i],
                embedding
            ))
        
        if failed_count > 0:
            logger.warning(f"Failed to generate {failed_count}/{len(contents)} embeddings from {file_path}")
        
        return records
    
    async def load_files_batch(
        self, 
        file_paths: List[str], 
        max_concurrent: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Load multiple files concurrently
        
        Args:
            file_paths: List of file paths to load
            max_concurrent: Maximum concurrent file processing
            
        Returns:
            Dictionary mapping file paths to number of records inserted
        """
        max_concurrent = max_concurrent or self.max_concurrent_chunks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def load_with_semaphore(file_path: str) -> Tuple[str, int]:
            async with semaphore:
                count = await self.load_file(file_path)
                return file_path, count
        
        # Create tasks for all files
        tasks = [load_with_semaphore(fp) for fp in file_paths]
        
        # Execute and collect results
        results = {}
        
        for future in asyncio.as_completed(tasks):
            try:
                file_path, count = await future
                results[file_path] = count
            except Exception as e:
                logger.error(f"Error in batch loading: {e}")
        
        return results

class BulkLoader:
    """Handles bulk loading operations with progress tracking"""
    
    def __init__(self, loader: Optional[DocumentLoader] = None):
        self.loader = loader or DocumentLoader()
    
    async def load_from_folder(
        self, 
        folder_path: str, 
        file_generator: Generator[str, None, None],
        total_files: int,
        progress_callback: Optional[callable] = None
    ) -> int:
        """
        Load all files from a folder with progress tracking
        
        Args:
            folder_path: Root folder path
            file_generator: Generator of file paths
            total_files: Total number of files to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Total number of records inserted
        """
        total_records = 0
        processed_files = 0
        failed_files = []
        
        # Start embedding service
        await embedding_service.start()
        
        # Start memory monitoring
        memory_manager.start_monitoring(
            callback=lambda info: logger.debug(f"Memory: {info['percent']:.1f}%")
        )
        
        try:
            # Process files in batches
            batch_size = config.processing.worker_processes * 2
            current_batch = []
            
            for file_path in file_generator:
                current_batch.append(file_path)
                
                if len(current_batch) >= batch_size:
                    # Process current batch
                    batch_results = await self.loader.load_files_batch(current_batch)
                    
                    # Update counters
                    for fp, count in batch_results.items():
                        if count > 0:
                            total_records += count
                        else:
                            failed_files.append(fp)
                        processed_files += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(processed_files, total_files, total_records)
                    
                    # Log progress
                    logger.info(f"Processed {processed_files}/{total_files} files, {total_records} records")
                    
                    # Clear batch
                    current_batch = []
                    
                    # Memory management
                    memory_manager.force_cleanup_if_needed()
            
            # Process remaining files
            if current_batch:
                batch_results = await self.loader.load_files_batch(current_batch)
                
                for fp, count in batch_results.items():
                    if count > 0:
                        total_records += count
                    else:
                        failed_files.append(fp)
                    processed_files += 1
                
                if progress_callback:
                    progress_callback(processed_files, total_files, total_records)
            
            # Final summary
            logger.info(f"Bulk loading completed: {processed_files} files, {total_records} records")
            
            if failed_files:
                logger.warning(f"{len(failed_files)} files failed to process")
                # Save failed files list
                import json
                with open("failed_files.json", "w") as f:
                    json.dump(failed_files, f, indent=2)
            
            return total_records
            
        except Exception as e:
            logger.error(f"Bulk loading failed: {e}")
            return total_records
            
        finally:
            # Cleanup
            memory_manager.stop_monitoring()
            await embedding_service.stop()
    
    async def load_single_file_interactive(self, file_path: str) -> bool:
        """
        Load a single file with interactive feedback
        
        Args:
            file_path: Path to file to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine file type
            file_type = Path(file_path).suffix[1:].lower()
            
            print(f"Loading file: {file_path} (type: {file_type})")
            
            # Start embedding service if needed
            if not embedding_service._started:
                await embedding_service.start()
            
            # Load file
            record_count = await self.loader.load_file(file_path, file_type)
            
            if record_count > 0:
                print(f"Successfully inserted {record_count} records from {file_path}")
                return True
            else:
                print(f"No records inserted from {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            print(f"Failed to load {file_path}: {e}")
            return False

# Global instances
document_loader = DocumentLoader()
bulk_loader = BulkLoader(document_loader)
# file_management/loaders.py - Optimized with Multi-Processing Pipeline
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Generator, Optional, Callable
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Pool, Queue, Manager
import multiprocessing as mp
import time
import gc

from core.config import config
from core.memory import memory_manager
from inference.embeddings import embedding_service
from file_management.parsers import parallel_processor, document_parser
from database.operations import db_ops

# Create prefixed logger for this file
logger = logging.getLogger(__name__)
LOG_PREFIX = "[Loader]"

def _process_file_with_embeddings(args: Tuple[str, str, int]) -> Tuple[str, List[Tuple[str, List[str], List[float]]], int]:
    """
    Worker function that processes a file completely: parsing + embeddings + preparation
    Must be at module level for multiprocessing
    """
    file_path, file_type, chunk_size = args
    
    try:
        # Import here to avoid circular imports in worker process
        import asyncio
        from inference.embeddings import EmbeddingService
        from file_management.parsers import DocumentParser
        
        # Create new instances for this process
        parser = DocumentParser()
        
        # Parse the file
        all_records = []
        for chunk in parser.parse_file_stream(file_path, file_type, chunk_size):
            all_records.extend(chunk)
        
        if not all_records:
            return file_path, [], 0
        
        # Filter valid content
        valid_items = [
            item for item in all_records
            if item.get("content") and item["content"].strip()
        ]
        
        if not valid_items:
            return file_path, [], 0
        
        # Extract content and tags
        contents = [item["content"] for item in valid_items]
        tags_lists = [item.get("tags", []) for item in valid_items]
        
        # Create embedding service for this process
        embedding_service = EmbeddingService()
        
        # Generate embeddings synchronously in worker process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Start embedding service
            loop.run_until_complete(embedding_service.start(concurrency=2))
            
            # Generate embeddings
            embeddings = loop.run_until_complete(
                embedding_service.generate_embeddings_batch(contents)
            )
            
            # Stop embedding service
            loop.run_until_complete(embedding_service.stop())
            
        finally:
            loop.close()
        
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
                continue
            
            records.append((
                contents[i],
                tags_lists[i],
                embedding
            ))
        
        return file_path, records, failed_count
        
    except Exception as e:
        logger.error(f"Worker process error for {file_path}: {e}")
        return file_path, [], 0

class OptimizedDocumentLoader:
    """
    Fully parallel document loader with pipeline architecture
    """
    
    def __init__(self):
        self.memory_chunk_size = config.processing.memory_chunk_size
        self.embedding_chunk_size = config.processing.embedding_chunk_size
        self.max_workers = config.processing.worker_processes
        self.max_concurrent_chunks = config.processing.max_concurrent_chunks
    
    async def load_files_full_parallel(self, file_paths: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Load files using full multi-process pipeline including embeddings
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (total_records, file_results_dict)
        """
        if not file_paths:
            return 0, {}
        
        # Prepare arguments for workers
        worker_args = []
        for file_path in file_paths:
            file_type = Path(file_path).suffix[1:].lower()
            if file_type not in ['txt', 'csv', 'json', 'py']:
                file_type = 'txt'
            worker_args.append((file_path, file_type, self.memory_chunk_size))
        
        total_records = 0
        file_results = {}
        failed_files = []
        
        # Process files in batches to manage memory
        batch_size = self.max_workers * 2
        
        for i in range(0, len(worker_args), batch_size):
            batch = worker_args[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
            
            # Use ProcessPoolExecutor for true parallelism
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files in batch
                future_to_file = {
                    executor.submit(_process_file_with_embeddings, args): args[0]
                    for args in batch
                }
                
                # Collect results and insert to database
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_path_result, records, failed_count = future.result()
                        
                        if records:
                            # Insert to database in main process
                            inserted = await db_ops.insert_documents_batch(records)
                            total_records += inserted
                            file_results[file_path] = inserted
                            
                            if failed_count > 0:
                                logger.warning(f"Failed to generate {failed_count} embeddings for {file_path}")
                        else:
                            file_results[file_path] = 0
                            failed_files.append(file_path)
                            
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        file_results[file_path] = 0
                        failed_files.append(file_path)
            
            # Memory cleanup between batches
            gc.collect()
            memory_manager.force_cleanup_if_needed()
            
            logger.info(f"Batch completed. Total records so far: {total_records}")
        
        if failed_files:
            logger.warning(f"{len(failed_files)} files failed to process")
        
        return total_records, file_results
    
    async def load_files_hybrid_parallel(self, file_paths: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Hybrid approach: parallel parsing, async embeddings, batch database ops
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (total_records, file_results_dict)
        """
        if not file_paths:
            return 0, {}
        
        # Start embedding service
        if not embedding_service._started:
            await embedding_service.start(concurrency=10)
        
        total_records = 0
        file_results = {}
        
        # Step 1: Parse all files in parallel
        logger.info(f"Step 1: Parsing {len(file_paths)} files in parallel")
        start_time = time.time()
        
        parsed_results = parallel_processor.process_mixed_files(file_paths)
        
        parse_time = time.time() - start_time
        logger.info(f"Parsing completed in {parse_time:.2f}s")
        
        # Step 2: Process embeddings and database insertion
        logger.info("Step 2: Generating embeddings and inserting to database")
        
        for file_path, records in parsed_results.items():
            if not records:
                file_results[file_path] = 0
                continue
            
            try:
                # Process in chunks for memory efficiency
                file_total = 0
                for i in range(0, len(records), self.embedding_chunk_size):
                    chunk = records[i:i + self.embedding_chunk_size]
                    processed_chunk = await self._process_chunk_async(chunk, file_path)
                    
                    if processed_chunk:
                        inserted = await db_ops.insert_documents_batch(processed_chunk)
                        file_total += inserted
                
                file_results[file_path] = file_total
                total_records += file_total
                
            except Exception as e:
                logger.error(f"Error processing embeddings for {file_path}: {e}")
                file_results[file_path] = 0
        
        return total_records, file_results
    
    async def _process_chunk_async(
        self, 
        chunk: List[Dict[str, Any]], 
        file_path: str
    ) -> List[Tuple[str, List[str], List[float]]]:
        """Process a chunk asynchronously: embeddings + validation"""
        
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
                continue
            
            records.append((
                contents[i],
                tags_lists[i],
                embedding
            ))
        
        if failed_count > 0:
            logger.warning(f"Failed to generate {failed_count}/{len(contents)} embeddings from {file_path}")
        
        return records
    
    async def load_file_optimized(self, file_path: str, file_type: Optional[str] = None) -> int:
        """
        Load a single file with optimal strategy based on file size
        """
        if not Path(file_path).exists() or Path(file_path).stat().st_size == 0:
            logger.info(f"Skipping empty or non-existent file: {file_path}")
            return 0
        
        file_size = Path(file_path).stat().st_size
        
        # Choose strategy based on file size
        if file_size > 100 * 1024 * 1024:  # 100MB+
            return await self._load_large_file_streaming(file_path, file_type)
        else:
            # Use batch processing for smaller files
            result = await self.load_files_hybrid_parallel([file_path])
            return result[0]
    
    async def _load_large_file_streaming(self, file_path: str, file_type: Optional[str] = None) -> int:
        """Stream process large files with parallel chunk processing"""
        
        if not file_type:
            file_type = Path(file_path).suffix[1:].lower()
        
        total_inserted = 0
        
        try:
            # Use parallel chunk processing for large files
            async for chunk in self._stream_large_file_chunks(file_path, file_type):
                if chunk:
                    inserted = await db_ops.insert_documents_batch(chunk)
                    total_inserted += inserted
                    
                    # Memory cleanup for large files
                    if total_inserted % (self.embedding_chunk_size * 10) == 0:
                        memory_manager.force_cleanup_if_needed()
            
            logger.info(f"Streamed {total_inserted} records from large file {file_path}")
            
        except Exception as e:
            logger.error(f"Error streaming large file {file_path}: {e}")
        
        return total_inserted
    
    async def _stream_large_file_chunks(
        self, 
        file_path: str, 
        file_type: str
    ) -> Generator[List[Tuple[str, List[str], List[float]]], None, None]:
        """Stream large files in parallel-processed chunks"""
        
        # Use the optimized parser for large files
        chunk_generator = document_parser.parse_large_file_parallel(file_path, file_type)
        
        # Process each chunk through the embedding pipeline
        for chunk in chunk_generator:
            if chunk:
                processed_chunk = await self._process_chunk_async(chunk, file_path)
                if processed_chunk:
                    yield processed_chunk

class OptimizedBulkLoader:
    """Enhanced bulk loader with pipeline optimization and progress tracking"""
    
    def __init__(self, loader: Optional[OptimizedDocumentLoader] = None):
        self.loader = loader or OptimizedDocumentLoader()
        self.max_workers = config.processing.worker_processes
    
    async def load_from_folder(
        self, 
        folder_path: str, 
        file_generator: Generator[str, None, None],
        total_files: int,
        progress_callback: Optional[Callable] = None,
        strategy: str = "hybrid"  # "hybrid", "full_parallel", "adaptive"
    ) -> int:
        """
        Public interface for folder loading that maps to the pipeline method
        """
        logger.info(f"{LOG_PREFIX} load_from_folder() called - delegating to load_from_folder_pipeline()")
        return await self.load_from_folder_pipeline(folder_path, file_generator, total_files, progress_callback, strategy)
    
    async def load_from_folder_pipeline(
        self, 
        folder_path: str, 
        file_generator: Generator[str, None, None],
        total_files: int,
        progress_callback: Optional[Callable] = None,
        strategy: str = "hybrid"  # "hybrid", "full_parallel", "adaptive"
    ) -> int:
        """
        Load files using optimized pipeline with selectable strategy
        
        Args:
            folder_path: Root folder path
            file_generator: Generator of file paths
            total_files: Total number of files to process
            progress_callback: Optional callback for progress updates
            strategy: Processing strategy to use
            
        Returns:
            Total number of records inserted
        """
        logger.info(f"{LOG_PREFIX} Starting load_from_folder_pipeline() with strategy: {strategy}")
        pipeline_start = time.time()
        
        total_records = 0
        processed_files = 0
        failed_files = []
        
        # Phase 1: Service Initialization
        print(f"{LOG_PREFIX} Phase 1: Initializing services...")
        logger.info(f"{LOG_PREFIX} Phase 1: Starting service initialization")
        
        init_start = time.time()
        await embedding_service.start(concurrency=15)
        logger.info(f"{LOG_PREFIX} Embedding service started with concurrency=15")
        
        memory_manager.start_monitoring(
            callback=lambda info: logger.debug(f"{LOG_PREFIX} Memory: {info['percent']:.1f}%")
        )
        logger.info(f"{LOG_PREFIX} Memory monitoring started")
        
        init_time = time.time() - init_start
        print(f"{LOG_PREFIX} Services initialized in {init_time:.2f}s")
        logger.info(f"{LOG_PREFIX} Phase 1 completed in {init_time:.2f}s")
        
        try:
            # Phase 2: File Generator Conversion
            print(f"{LOG_PREFIX} Phase 2: Converting file generator to list...")
            logger.info(f"{LOG_PREFIX} Phase 2: Converting generator to list for batch processing")
            
            conversion_start = time.time()
            all_files = list(file_generator)
            conversion_time = time.time() - conversion_start
            
            print(f"{LOG_PREFIX} Converted {len(all_files)} files in {conversion_time:.3f}s")
            logger.info(f"{LOG_PREFIX} File generator converted: {len(all_files)} files in {conversion_time:.3f}s")
            
            # Phase 3: Strategy Selection
            print(f"{LOG_PREFIX} Phase 3: Strategy selection and optimization...")
            logger.info(f"{LOG_PREFIX} Phase 3: Processing strategy selection")
            
            strategy_start = time.time()
            if strategy == "adaptive":
                original_strategy = strategy
                strategy = self._choose_optimal_strategy(all_files)
                strategy_time = time.time() - strategy_start
                
                print(f"{LOG_PREFIX} Adaptive strategy analysis completed in {strategy_time:.3f}s")
                print(f"{LOG_PREFIX} Selected strategy: {strategy}")
                logger.info(f"{LOG_PREFIX} Adaptive strategy selected '{strategy}' (was '{original_strategy}') in {strategy_time:.3f}s")
            else:
                strategy_time = time.time() - strategy_start
                print(f"{LOG_PREFIX} Using specified strategy: {strategy}")
                logger.info(f"{LOG_PREFIX} Using predefined strategy: {strategy}")
            
            # Phase 4: File Processing Pipeline
            print(f"{LOG_PREFIX} Phase 4: Executing file processing pipeline...")
            logger.info(f"{LOG_PREFIX} Phase 4: Starting file processing with {strategy} strategy")
            
            processing_start = time.time()
            
            # Process files using selected strategy
            if strategy == "full_parallel":
                logger.info(f"{LOG_PREFIX} Executing full_parallel strategy")
                total_records, file_results = await self._load_full_parallel_strategy(
                    all_files, progress_callback, total_files
                )
            elif strategy == "hybrid":
                logger.info(f"{LOG_PREFIX} Executing hybrid strategy")
                total_records, file_results = await self._load_hybrid_strategy(
                    all_files, progress_callback, total_files
                )
            else:  # pipeline strategy
                logger.info(f"{LOG_PREFIX} Executing pipeline strategy")
                total_records, file_results = await self._load_pipeline_strategy(
                    all_files, progress_callback, total_files
                )
            
            processing_time = time.time() - processing_start
            print(f"{LOG_PREFIX} Processing pipeline completed in {processing_time:.2f}s")
            logger.info(f"{LOG_PREFIX} Phase 4 completed in {processing_time:.2f}s")
            
            # Phase 5: Results Analysis
            print(f"{LOG_PREFIX} Phase 5: Analyzing results...")
            logger.info(f"{LOG_PREFIX} Phase 5: Analyzing processing results")
            
            analysis_start = time.time()
            
            # Count results
            for file_path, count in file_results.items():
                if count > 0:
                    processed_files += 1
                else:
                    failed_files.append(file_path)
            
            analysis_time = time.time() - analysis_start
            
            # Final summary
            pipeline_time = time.time() - pipeline_start
            success_rate = (processed_files / total_files) * 100 if total_files > 0 else 0
            files_per_sec = total_files / pipeline_time if pipeline_time > 0 else 0
            records_per_sec = total_records / pipeline_time if pipeline_time > 0 else 0
            
            print(f"{LOG_PREFIX} === PIPELINE SUMMARY ===")
            print(f"{LOG_PREFIX} Total pipeline time: {pipeline_time:.2f}s")
            print(f"{LOG_PREFIX} Strategy used: {strategy}")
            print(f"{LOG_PREFIX} Files processed: {processed_files}/{total_files} ({success_rate:.1f}%)")
            print(f"{LOG_PREFIX} Records created: {total_records}")
            print(f"{LOG_PREFIX} Throughput: {files_per_sec:.1f} files/s, {records_per_sec:.1f} records/s")
            
            logger.info(f"{LOG_PREFIX} Bulk loading completed: {processed_files}/{total_files} files "
                       f"({success_rate:.1f}%), {total_records} records, {pipeline_time:.2f}s total, "
                       f"{files_per_sec:.1f} files/s, {records_per_sec:.1f} records/s")
            
            if failed_files:
                logger.warning(f"{LOG_PREFIX} {len(failed_files)} files failed to process")
                print(f"{LOG_PREFIX} Warning: {len(failed_files)} files failed to process")
                
                # Save failed files list
                import json
                failed_files_path = "failed_files.json"
                with open(failed_files_path, "w") as f:
                    json.dump(failed_files, f, indent=2)
                logger.info(f"{LOG_PREFIX} Failed files list saved to: {failed_files_path}")
                print(f"{LOG_PREFIX} Failed files list saved to: {failed_files_path}")
            
            return total_records
            
        except Exception as e:
            pipeline_time = time.time() - pipeline_start
            logger.error(f"{LOG_PREFIX} Bulk loading pipeline failed after {pipeline_time:.2f}s: {e}")
            print(f"{LOG_PREFIX} Pipeline failed after {pipeline_time:.2f}s: {e}")
            return total_records
            
        finally:
            # Phase 6: Cleanup
            print(f"{LOG_PREFIX} Phase 6: Cleaning up services...")
            logger.info(f"{LOG_PREFIX} Phase 6: Starting cleanup process")
            
            cleanup_start = time.time()
            
            # Cleanup
            memory_manager.stop_monitoring()
            logger.info(f"{LOG_PREFIX} Memory monitoring stopped")
            
            await embedding_service.stop()
            logger.info(f"{LOG_PREFIX} Embedding service stopped")
            
            cleanup_time = time.time() - cleanup_start
            total_pipeline_time = time.time() - pipeline_start
            
            print(f"{LOG_PREFIX} Cleanup completed in {cleanup_time:.2f}s")
            print(f"{LOG_PREFIX} Total pipeline time: {total_pipeline_time:.2f}s")
            logger.info(f"{LOG_PREFIX} Cleanup completed in {cleanup_time:.2f}s, total time: {total_pipeline_time:.2f}s")
    
    def _choose_optimal_strategy(self, file_paths: List[str]) -> str:
        """Choose optimal processing strategy based on file characteristics"""
        if not file_paths:
            return "hybrid"
        
        # Analyze file sizes
        total_size = 0
        large_files = 0
        small_files = 0
        
        for file_path in file_paths[:100]:  # Sample first 100 files
            try:
                size = Path(file_path).stat().st_size
                total_size += size
                
                if size > 50 * 1024 * 1024:  # 50MB+
                    large_files += 1
                else:
                    small_files += 1
            except Exception:
                small_files += 1
        
        avg_size = total_size / len(file_paths[:100]) if file_paths else 0
        
        # Decision logic
        if large_files > len(file_paths) * 0.3:  # >30% large files
            return "pipeline"
        elif avg_size < 1024 * 1024:  # Average < 1MB
            return "full_parallel"
        else:
            return "hybrid"
    
    async def _load_full_parallel_strategy(
        self, 
        file_paths: List[str], 
        progress_callback: Optional[Callable],
        total_files: int
    ) -> Tuple[int, Dict[str, int]]:
        """Full parallel processing: everything in worker processes"""
        logger.info("Using full parallel strategy")
        
        batch_size = self.max_workers * 3
        total_records = 0
        all_file_results = {}
        processed_count = 0
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            logger.info(f"Processing parallel batch {i//batch_size + 1}: {len(batch)} files")
            
            batch_records, batch_results = await self.loader.load_files_full_parallel(batch)
            total_records += batch_records
            all_file_results.update(batch_results)
            processed_count += len(batch)
            
            if progress_callback:
                progress_callback(processed_count, total_files, total_records)
            
            # Memory management between batches
            gc.collect()
            memory_manager.force_cleanup_if_needed()
        
        return total_records, all_file_results
    
    async def _load_hybrid_strategy(
        self, 
        file_paths: List[str], 
        progress_callback: Optional[Callable],
        total_files: int
    ) -> Tuple[int, Dict[str, int]]:
        """Enhanced hybrid strategy: fully parallel processing with adaptive batching"""
        logger.info("Using enhanced hybrid strategy with full parallelization")
        
        # Separate files by size for optimal processing with concurrent analysis
        def categorize_file(file_path):
            try:
                size = Path(file_path).stat().st_size
                return ('large' if size > 100 * 1024 * 1024 else 'small', file_path, size)
            except Exception:
                return ('small', file_path, 0)
        
        # Parallelize file categorization
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            categorized_files = list(executor.map(categorize_file, file_paths))
        
        small_files = [fp for cat, fp, size in categorized_files if cat == 'small']
        large_files = [fp for cat, fp, size in categorized_files if cat == 'large']
        
        total_records = 0
        all_file_results = {}
        processed_count = 0
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers * 2)
        
        async def process_file_batch_concurrent(batch, is_large=False):
            """Process a batch of files concurrently"""
            nonlocal total_records, processed_count
            
            async with semaphore:
                if is_large:
                    # Process large files individually but in parallel
                    tasks = []
                    for file_path in batch:
                        task = asyncio.create_task(self.loader.load_file_optimized(file_path))
                        tasks.append((file_path, task))
                    
                    batch_results = {}
                    for file_path, task in tasks:
                        try:
                            file_records = await task
                            total_records += file_records
                            batch_results[file_path] = file_records
                            processed_count += 1
                            
                            if progress_callback:
                                progress_callback(processed_count, total_files, total_records)
                        except Exception as e:
                            logger.error(f"Error processing large file {file_path}: {e}")
                            batch_results[file_path] = 0
                    
                    return batch_results
                else:
                    # Process small files in optimized batches
                    try:
                        batch_records, batch_results = await self.loader.load_files_hybrid_parallel(batch)
                        total_records += batch_records
                        processed_count += len(batch)
                        
                        if progress_callback:
                            progress_callback(processed_count, total_files, total_records)
                        
                        return batch_results
                    except Exception as e:
                        logger.error(f"Error processing small files batch: {e}")
                        return {fp: 0 for fp in batch}
        
        # Process files with concurrent batching
        all_tasks = []
        
        # Process small files in concurrent batches
        if small_files:
            batch_size = max(4, self.max_workers)  # Smaller batches for better parallelization
            for i in range(0, len(small_files), batch_size):
                batch = small_files[i:i + batch_size]
                task = asyncio.create_task(process_file_batch_concurrent(batch, is_large=False))
                all_tasks.append(task)
        
        # Process large files in concurrent individual processing
        if large_files:
            # Process large files in smaller concurrent groups
            large_batch_size = min(3, self.max_workers // 2)  # Limit concurrent large file processing
            for i in range(0, len(large_files), large_batch_size):
                batch = large_files[i:i + large_batch_size]
                task = asyncio.create_task(process_file_batch_concurrent(batch, is_large=True))
                all_tasks.append(task)
        
        # Wait for all concurrent processing to complete
        logger.info(f"Starting concurrent processing of {len(all_tasks)} batches")
        batch_results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Merge all results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
                continue
            if isinstance(result, dict):
                all_file_results.update(result)
        
        logger.info(f"Enhanced hybrid strategy completed: {total_records} records from {processed_count} files")
        return total_records, all_file_results
    
    async def _load_pipeline_strategy(
        self, 
        file_paths: List[str], 
        progress_callback: Optional[Callable],
        total_files: int
    ) -> Tuple[int, Dict[str, int]]:
        """Pipeline strategy: producer-consumer with queues"""
        logger.info("Using pipeline strategy")
        
        # Create queues for pipeline stages
        parse_queue = asyncio.Queue(maxsize=50)
        embed_queue = asyncio.Queue(maxsize=100)
        
        total_records = 0
        all_file_results = {}
        processed_count = 0
        
        # Pipeline stage tasks
        async def file_producer():
            """Produce file paths for processing"""
            for file_path in file_paths:
                await parse_queue.put(file_path)
            
            # Signal completion
            for _ in range(4):  # Number of parser workers
                await parse_queue.put(None)
        
        async def parse_worker():
            """Parse files and put results in embed queue"""
            while True:
                file_path = await parse_queue.get()
                if file_path is None:
                    break
                
                try:
                    # Parse file
                    file_type = Path(file_path).suffix[1:].lower()
                    records = []
                    for chunk in document_parser.parse_file_stream(file_path, file_type, 100):
                        records.extend(chunk)
                    
                    if records:
                        await embed_queue.put((file_path, records))
                    else:
                        await embed_queue.put((file_path, []))
                        
                except Exception as e:
                    logger.error(f"Parse worker error for {file_path}: {e}")
                    await embed_queue.put((file_path, []))
                finally:
                    parse_queue.task_done()
        
        async def embed_and_store_worker():
            """Process embeddings and store to database"""
            nonlocal total_records, processed_count
            
            while True:
                try:
                    # Get parsed data
                    file_path, records = await asyncio.wait_for(embed_queue.get(), timeout=60)
                    
                    if not records:
                        all_file_results[file_path] = 0
                        processed_count += 1
                        if progress_callback:
                            progress_callback(processed_count, total_files, total_records)
                        embed_queue.task_done()
                        continue
                    
                    # Process embeddings
                    processed_chunk = await self.loader._process_chunk_async(records, file_path)
                    
                    if processed_chunk:
                        inserted = await db_ops.insert_documents_batch(processed_chunk)
                        total_records += inserted
                        all_file_results[file_path] = inserted
                    else:
                        all_file_results[file_path] = 0
                    
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count, total_files, total_records)
                    
                except asyncio.TimeoutError:
                    # No more work, exit
                    break
                except Exception as e:
                    logger.error(f"Embed worker error: {e}")
                finally:
                    embed_queue.task_done()
        
        # Start pipeline workers
        tasks = [
            asyncio.create_task(file_producer()),
            *[asyncio.create_task(parse_worker()) for _ in range(4)],
            *[asyncio.create_task(embed_and_store_worker()) for _ in range(3)]
        ]
        
        # Wait for completion
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return total_records, all_file_results
    
    async def load_single_file_interactive(self, file_path: str) -> bool:
        """
        Load a single file with interactive feedback and optimal strategy
        """
        try:
            # Determine file type and size
            file_type = Path(file_path).suffix[1:].lower()
            file_size = Path(file_path).stat().st_size
            size_mb = file_size / (1024 ** 2)
            
            print(f"Loading file: {file_path}")
            print(f"Type: {file_type}, Size: {size_mb:.2f} MB")
            
            # Choose strategy based on size
            if size_mb > 100:
                print("Large file detected - using streaming strategy")
                strategy = "streaming"
            elif size_mb > 10:
                print("Medium file detected - using hybrid strategy")
                strategy = "hybrid"
            else:
                print("Small file detected - using parallel strategy")
                strategy = "parallel"
            
            # Start embedding service if needed
            if not embedding_service._started:
                await embedding_service.start(concurrency=10)
            
            start_time = time.time()
            
            # Load file using optimal strategy
            if strategy == "streaming":
                record_count = await self.loader._load_large_file_streaming(file_path, file_type)
            else:
                record_count = await self.loader.load_file_optimized(file_path, file_type)
            
            load_time = time.time() - start_time
            
            if record_count > 0:
                print(f"Successfully inserted {record_count} records in {load_time:.2f}s")
                print(f"Throughput: {record_count/load_time:.1f} records/second")
                return True
            else:
                print(f"No records inserted from {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            print(f"Failed to load {file_path}: {e}")
            return False

class PerformanceMonitor:
    """Monitor and report performance metrics during bulk loading"""
    
    def __init__(self):
        self.start_time = None
        self.processed_files = 0
        self.total_records = 0
        self.last_update = None
        self.throughput_history = []
    
    def start(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.last_update = self.start_time
    
    def update(self, processed: int, total: int, records: int):
        """Update performance metrics"""
        current_time = time.time()
        
        if self.last_update:
            time_delta = current_time - self.last_update
            files_delta = processed - self.processed_files
            records_delta = records - self.total_records
            
            if time_delta > 0:
                file_throughput = files_delta / time_delta
                record_throughput = records_delta / time_delta
                self.throughput_history.append((file_throughput, record_throughput))
        
        self.processed_files = processed
        self.total_records = records
        self.last_update = current_time
        
        # Calculate overall metrics
        elapsed = current_time - self.start_time
        overall_file_rate = processed / elapsed if elapsed > 0 else 0
        overall_record_rate = records / elapsed if elapsed > 0 else 0
        
        # Estimate completion time
        remaining_files = total - processed
        eta = remaining_files / overall_file_rate if overall_file_rate > 0 else 0
        
        # Print progress
        percent = (processed / total) * 100 if total > 0 else 0
        print(f"\rProgress: {processed}/{total} files ({percent:.1f}%) | "
              f"{records} records | "
              f"{overall_file_rate:.1f} files/s | "
              f"{overall_record_rate:.1f} records/s | "
              f"ETA: {eta/60:.1f}m", end="")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.start_time:
            return {}
        
        elapsed = time.time() - self.start_time
        
        return {
            "total_time_seconds": elapsed,
            "total_files": self.processed_files,
            "total_records": self.total_records,
            "avg_files_per_second": self.processed_files / elapsed if elapsed > 0 else 0,
            "avg_records_per_second": self.total_records / elapsed if elapsed > 0 else 0,
            "avg_records_per_file": self.total_records / self.processed_files if self.processed_files > 0 else 0
        }

# Global instances
optimized_document_loader = OptimizedDocumentLoader()
bulk_loader = OptimizedBulkLoader(optimized_document_loader)
performance_monitor = PerformanceMonitor()
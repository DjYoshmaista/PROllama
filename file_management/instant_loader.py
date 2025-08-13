# file_management/instant_loader.py - Instant Start Document Loader
import asyncio
import logging
import time
import os
import json
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
from datetime import datetime
from tqdm.asyncio import tqdm

from core.config import config
from core.memory import memory_manager
from inference.async_embeddings import async_embedding_service
from file_management.parsers import document_parser
from file_management.chunking import text_splitter, TextChunk
from database.batch_operations import batch_db_ops

# Create prefixed logger for this file
logger = logging.getLogger(__name__)
LOG_PREFIX = "[InstantLoader]"

class InstantAsyncLoader:
    """Instant start async document loader that begins processing immediately"""
    
    def __init__(self):
        self.max_concurrent_files = 30  # Concurrent file processing
        self.discovery_batch_size = 100  # Discover files in batches
        self.max_files_to_process = 1000  # Limit for large datasets
        self.progress_interval = 25  # Show progress every N files
        self.checkpoint_interval = 500  # Checkpoint every N files
        self.checkpoint_file = None  # Will be set during processing
    
    async def load_folder_instant(
        self,
        folder_path: str,
        total_files: int,
        enable_chunking: bool = True,
        enable_summarization: bool = False
    ) -> Dict[str, Any]:
        """Start processing immediately with comprehensive progress tracking and checkpointing"""
        
        logger.info(f"{LOG_PREFIX} Starting instant async loading from {folder_path}")
        print(f"{LOG_PREFIX} Starting instant async loading from {folder_path}")
        
        # Setup checkpointing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = f"checkpoint_{timestamp}.json"
        
        # Try to resume from previous checkpoint
        processed_files, total_chunks, total_summaries, failed_files, processed_file_paths = self._load_checkpoint()
        
        if processed_files > 0:
            print(f"{LOG_PREFIX} 📋 Resuming from checkpoint: {processed_files} files already processed")
            logger.info(f"{LOG_PREFIX} Resumed from checkpoint with {processed_files} files processed")
        
        # Initialize services immediately with detailed logging
        print(f"{LOG_PREFIX} 🔄 Phase 1: Initializing services...")
        logger.info(f"{LOG_PREFIX} Starting service initialization")
        
        service_start = time.time()
        await async_embedding_service.start()
        logger.info(f"{LOG_PREFIX} Async embedding service started")
        
        await batch_db_ops.initialize_connection_pool()
        logger.info(f"{LOG_PREFIX} Database connection pool initialized")
        
        service_time = time.time() - service_start
        print(f"{LOG_PREFIX} ✅ Services initialized in {service_time:.2f}s")
        logger.info(f"{LOG_PREFIX} Service initialization completed in {service_time:.2f}s")
        
        print(f"{LOG_PREFIX} 🚀 Phase 2: Starting instant processing...")
        print(f"{LOG_PREFIX} 📊 Will process up to {self.max_files_to_process} files")
        print(f"{LOG_PREFIX} 💡 Progress updates every {self.progress_interval} files")
        print(f"{LOG_PREFIX} 💾 Checkpoints every {self.checkpoint_interval} files")
        
        start_time = time.time()
        last_progress_time = start_time
        last_checkpoint_time = start_time
        
        # Semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_files)
        logger.info(f"{LOG_PREFIX} Created semaphore with {self.max_concurrent_files} max concurrent files")
        
        # Create a queue for discovered files
        file_queue = asyncio.Queue(maxsize=200)
        logger.info(f"{LOG_PREFIX} Created file queue with maxsize=200")
        
        # Progress tracking
        discovery_active = True
        files_discovered = 0
        embedding_time_total = 0.0
        database_time_total = 0.0
        
        try:
            # Start file discovery in background with enhanced logging
            print(f"{LOG_PREFIX} 🔍 Phase 3: Starting file discovery...")
            logger.info(f"{LOG_PREFIX} Starting background file discovery task")
            
            discovery_task = asyncio.create_task(
                self._discover_files_async(folder_path, file_queue, processed_file_paths)
            )
            
            # Start processing files immediately as they're discovered
            pending_tasks = []
            files_queued = 0
            
            print(f"{LOG_PREFIX} ⚡ Phase 4: Processing files as discovered...")
            
            async def process_file_with_semaphore(file_path: str):
                async with semaphore:
                    return await self._process_single_file_async(file_path)
            
            # Wait for discovery to complete first
            print(f"{LOG_PREFIX} ⏳ Waiting for file discovery to complete...")
            await discovery_task
            
            # Now get all discovered files and process with progress bars
            all_files_to_process = []
            while True:
                try:
                    file_path = await asyncio.wait_for(file_queue.get(), timeout=1.0)
                    if file_path is None:
                        break
                    all_files_to_process.append(file_path)
                except asyncio.TimeoutError:
                    break
            
            total_files_to_process = len(all_files_to_process)
            logger.info(f"{LOG_PREFIX} Starting processing of {total_files_to_process} files")
            
            if total_files_to_process == 0:
                print(f"{LOG_PREFIX} ⚠️ No files to process")
                return {
                    "processed_files": 0,
                    "files_discovered": 0,
                    "total_chunks": 0,
                    "total_summaries": 0,
                    "failed_files": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Initialize progress tracking with separate bars
            from tqdm import tqdm
            import threading
            
            # Create progress bars
            file_pbar = tqdm(total=total_files_to_process, desc="📁 Processing Files", 
                           unit="file", position=0, ncols=100, dynamic_ncols=True)
            chunk_pbar = tqdm(total=0, desc="📝 Creating Chunks", 
                            unit="chunk", position=1, ncols=100, dynamic_ncols=True)
            embed_pbar = tqdm(total=0, desc="🧠 Generating Embeddings", 
                            unit="embed", position=2, ncols=100, dynamic_ncols=True)
            db_pbar = tqdm(total=0, desc="💾 Database Insertions", 
                         unit="record", position=3, ncols=100, dynamic_ncols=True)
            
            # Progress tracking variables
            progress_lock = threading.Lock()
            
            def update_progress_bars(chunks_created, embeddings_generated, records_inserted):
                with progress_lock:
                    if chunks_created > 0:
                        chunk_pbar.total = chunk_pbar.total + chunks_created if chunk_pbar.total else chunks_created
                        chunk_pbar.update(chunks_created)
                    if embeddings_generated > 0:
                        embed_pbar.total = embed_pbar.total + embeddings_generated if embed_pbar.total else embeddings_generated
                        embed_pbar.update(embeddings_generated)
                    if records_inserted > 0:
                        db_pbar.total = db_pbar.total + records_inserted if db_pbar.total else records_inserted
                        db_pbar.update(records_inserted)
            
            try:
                # Process files in batches with progress tracking
                batch_size = 20
                processed_file_paths = set()
                
                for i in range(0, len(all_files_to_process), batch_size):
                    batch = all_files_to_process[i:i + batch_size]
                    batch_start = time.time()
                    
                    # Process batch concurrently
                    tasks = [process_file_with_semaphore(file_path) for file_path in batch]
                    completed_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    batch_time = time.time() - batch_start
                    
                    # Update progress with detailed tracking
                    batch_chunks = 0
                    batch_summaries = 0
                    batch_failed = 0
                    batch_embeddings = 0
                    batch_db_records = 0
                    
                    for j, (file_path, result) in enumerate(zip(batch, completed_results)):
                        processed_files += 1
                        processed_file_paths.add(file_path)
                        file_pbar.update(1)
                        
                        if isinstance(result, Exception):
                            logger.debug(f"{LOG_PREFIX} File processing exception {file_path}: {result}")
                            batch_failed += 1
                            failed_files += 1
                        elif result:
                            chunks = result.get('chunks', 0)
                            batch_chunks += chunks
                            batch_summaries += result.get('summaries', 0)
                            embedding_time_total += result.get('embedding_time', 0.0)
                            database_time_total += result.get('database_time', 0.0)
                            
                            # Assume 1 embedding per chunk and 1 DB record per chunk
                            batch_embeddings += chunks
                            batch_db_records += chunks
                        else:
                            batch_failed += 1
                            failed_files += 1
                    
                    total_chunks += batch_chunks
                    total_summaries += batch_summaries
                    
                    # Update progress bars
                    update_progress_bars(batch_chunks, batch_embeddings, batch_db_records)
                    
                    # Update file progress bar description with stats
                    success_rate = ((processed_files - failed_files) / processed_files * 100) if processed_files > 0 else 0
                    file_pbar.set_description(f"📁 Files (✅{success_rate:.1f}% success)")
                    
                    # Log batch completion
                    files_per_sec = len(batch) / batch_time if batch_time > 0 else 0
                    logger.info(f"{LOG_PREFIX} Batch {i//batch_size + 1}: {len(batch)} files, "
                               f"{batch_chunks} chunks, {batch_failed} failed in {batch_time:.2f}s "
                               f"({files_per_sec:.1f} files/s)")
                    
                    # Checkpoint every N files
                    current_time = time.time()
                    if (processed_files % self.checkpoint_interval == 0 and
                        current_time - last_checkpoint_time > 30):
                        
                        await self._save_checkpoint(
                            processed_files, total_chunks, total_summaries, 
                            failed_files, processed_file_paths
                        )
                        last_checkpoint_time = current_time
                    
                    # Yield control
                    await asyncio.sleep(0.01)
                
            except KeyboardInterrupt:
                print(f"\n{LOG_PREFIX} ⚠️ Processing interrupted by user")
                logger.info(f"{LOG_PREFIX} Processing interrupted by user at {processed_files} files")
                
            finally:
                # Close progress bars
                file_pbar.close()
                chunk_pbar.close()
                embed_pbar.close()
                db_pbar.close()
                print()  # Add newline after progress bars
            
            # Discovery task should already be complete, but ensure cleanup
            if not discovery_task.done():
                logger.info(f"{LOG_PREFIX} Cancelling discovery task")
                discovery_task.cancel()
                try:
                    await discovery_task
                except asyncio.CancelledError:
                    pass
            
            # Save final checkpoint
            await self._save_checkpoint(
                processed_files, total_chunks, total_summaries, 
                failed_files, processed_file_paths, final=True
            )
            
            # Final results with comprehensive metrics
            elapsed = time.time() - start_time
            avg_embedding_time = embedding_time_total / max(processed_files, 1)
            avg_database_time = database_time_total / max(processed_files, 1)
            
            results = {
                "processed_files": processed_files,
                "files_discovered": files_discovered,
                "total_chunks": total_chunks,
                "total_summaries": total_summaries,
                "failed_files": failed_files,
                "success_rate": (processed_files - failed_files) / max(processed_files, 1),
                "processing_time": elapsed,
                "files_per_second": processed_files / elapsed if elapsed > 0 else 0,
                "avg_embedding_time": avg_embedding_time,
                "avg_database_time": avg_database_time,
                "total_embedding_time": embedding_time_total,
                "total_database_time": database_time_total
            }
            
            print(f"\n{LOG_PREFIX} ✅ Instant async loading completed!")
            print(f"{LOG_PREFIX} 📁 Files processed: {processed_files:,}")
            print(f"{LOG_PREFIX} 🔍 Files discovered: {files_discovered:,}")
            print(f"{LOG_PREFIX} 📝 Chunks created: {total_chunks:,}")
            print(f"{LOG_PREFIX} 📑 Summaries created: {total_summaries:,}")
            print(f"{LOG_PREFIX} ❌ Failed files: {failed_files:,}")
            print(f"{LOG_PREFIX} ⏱️  Total time: {elapsed:.1f}s")
            print(f"{LOG_PREFIX} ⚡ Speed: {results['files_per_second']:.1f} files/s")
            print(f"{LOG_PREFIX} 🧠 Avg embedding time: {avg_embedding_time:.2f}s/file")
            print(f"{LOG_PREFIX} 💾 Avg database time: {avg_database_time:.2f}s/file")
            
            logger.info(f"{LOG_PREFIX} Instant async loading completed successfully: "
                       f"{processed_files} files, {total_chunks} chunks, {elapsed:.1f}s")
            
            return results
            
        except KeyboardInterrupt:
            print(f"\n{LOG_PREFIX} ⚠️  Processing interrupted by user")
            print(f"{LOG_PREFIX} 📁 Files processed so far: {processed_files:,}")
            print(f"{LOG_PREFIX} 📝 Chunks created: {total_chunks:,}")
            
            # Save interruption checkpoint
            await self._save_checkpoint(
                processed_files, total_chunks, total_summaries, 
                failed_files, processed_file_paths, interrupted=True
            )
            
            elapsed = time.time() - start_time
            return {
                "processed_files": processed_files,
                "files_discovered": files_discovered,
                "total_chunks": total_chunks,
                "total_summaries": total_summaries,
                "failed_files": failed_files,
                "processing_time": elapsed,
                "interrupted": True
            }
            
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Instant async loading failed: {e}")
            print(f"{LOG_PREFIX} ❌ Error: {e}")
            
            # Save error checkpoint
            await self._save_checkpoint(
                processed_files, total_chunks, total_summaries, 
                failed_files, processed_file_paths, error=str(e)
            )
            
            return {
                "error": str(e),
                "processed_files": processed_files,
                "files_discovered": files_discovered,
                "total_chunks": total_chunks,
                "total_summaries": total_summaries,
                "failed_files": failed_files
            }
        
        finally:
            # Cleanup
            print(f"{LOG_PREFIX} 🔄 Phase 5: Cleaning up services...")
            logger.info(f"{LOG_PREFIX} Starting service cleanup")
            
            cleanup_start = time.time()
            await async_embedding_service.stop()
            logger.info(f"{LOG_PREFIX} Async embedding service stopped")
            
            await batch_db_ops.close_connection_pool()
            logger.info(f"{LOG_PREFIX} Database connection pool closed")
            
            cleanup_time = time.time() - cleanup_start
            print(f"{LOG_PREFIX} ✅ Cleanup completed in {cleanup_time:.2f}s")
            logger.info(f"{LOG_PREFIX} Service cleanup completed in {cleanup_time:.2f}s")
    
    async def _discover_files_async(self, folder_path: str, file_queue: asyncio.Queue, skip_files: set = None):
        """Fully parallelized file discovery with real-time progress bars"""
        discovery_start = time.time()
        discovered = 0
        directories_scanned = 0
        # FIXED: Match discovery.py format (without dots)
        supported_extensions = {"py", "txt", "csv", "json"}
        
        if skip_files is None:
            skip_files = set()
        
        logger.info(f"{LOG_PREFIX} Starting parallelized file discovery from {folder_path}")
        print(f"{LOG_PREFIX} 🔍 Starting parallelized file discovery from {folder_path}")
        
        if skip_files:
            logger.info(f"{LOG_PREFIX} Skipping {len(skip_files)} already processed files")
            print(f"{LOG_PREFIX} 📋 Skipping {len(skip_files)} already processed files")
        
        try:
            # First: Get directory structure in parallel
            print(f"{LOG_PREFIX} 📂 Phase 1: Scanning directory structure...")
            
            directory_queue = asyncio.Queue(maxsize=1000)
            found_directories = []
            
            # Start with the root directory
            await directory_queue.put(folder_path)
            
            # Parallel directory discovery
            async def discover_directories():
                nonlocal directories_scanned
                dirs_found = 0
                
                while True:
                    try:
                        current_dir = await asyncio.wait_for(directory_queue.get(), timeout=1.0)
                        if current_dir is None:
                            break
                            
                        try:
                            # Use thread pool for directory listing
                            loop = asyncio.get_event_loop()
                            entries = await loop.run_in_executor(
                                None, self._list_directory_sync, current_dir
                            )
                            
                            for entry_path, is_dir in entries:
                                if is_dir:
                                    # Filter directories
                                    dir_name = os.path.basename(entry_path)
                                    if not dir_name.startswith('.') and dir_name not in {
                                        '__pycache__', 'node_modules', '.git', '.venv', 'venv', 
                                        'build', 'dist', '.pytest_cache', '.mypy_cache', '.tox', '__MACOSX__'
                                    }:
                                        found_directories.append(entry_path)
                                        await directory_queue.put(entry_path)
                                        dirs_found += 1
                            
                            directories_scanned += 1
                            
                            # Progress updates
                            if directories_scanned % 50 == 0:
                                elapsed = time.time() - discovery_start
                                dir_rate = directories_scanned / elapsed if elapsed > 0 else 0
                                print(f"\r{LOG_PREFIX} 📂 Directories: {directories_scanned:,} scanned, {dirs_found:,} found ({dir_rate:.1f}/s)", end="")
                                
                        except Exception as e:
                            logger.debug(f"{LOG_PREFIX} Error scanning directory {current_dir}: {e}")
                            
                    except asyncio.TimeoutError:
                        # No more directories to process
                        break
                
                await directory_queue.put(None)  # Signal completion
                return found_directories
            
            # Run directory discovery
            all_directories = await discover_directories()
            dir_time = time.time() - discovery_start
            
            print(f"\n{LOG_PREFIX} 📂 Phase 1 Complete: {len(all_directories):,} directories in {dir_time:.2f}s")
            logger.info(f"{LOG_PREFIX} Directory discovery completed: {len(all_directories)} directories")
            
            # Phase 2: Parallel file discovery within directories
            print(f"{LOG_PREFIX} 📁 Phase 2: Discovering files in parallel...")
            
            # Create progress bar for file discovery
            from tqdm import tqdm
            
            file_discovery_semaphore = asyncio.Semaphore(20)  # Limit concurrent directory scans
            
            async def scan_directory_for_files(directory_path):
                async with file_discovery_semaphore:
                    try:
                        loop = asyncio.get_event_loop()
                        files = await loop.run_in_executor(
                            None, self._scan_directory_for_files_sync, directory_path, supported_extensions, skip_files
                        )
                        return files
                    except Exception as e:
                        logger.debug(f"{LOG_PREFIX} Error scanning files in {directory_path}: {e}")
                        return []
            
            # Process directories in batches
            batch_size = 100
            all_files = []
            
            with tqdm(total=len(all_directories), desc="🔍 Scanning directories", 
                     unit="dir", ncols=100, dynamic_ncols=True) as pbar:
                
                for i in range(0, len(all_directories), batch_size):
                    batch = all_directories[i:i + batch_size]
                    
                    # Process batch in parallel
                    tasks = [scan_directory_for_files(dir_path) for dir_path in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Collect results
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.debug(f"{LOG_PREFIX} Batch result error: {result}")
                        elif isinstance(result, list):
                            all_files.extend(result)
                    
                    pbar.update(len(batch))
                    
                    # Show current file count in description
                    pbar.set_description(f"🔍 Found {len(all_files):,} files")
            
            file_discovery_time = time.time() - discovery_start - dir_time
            discovery_rate = len(all_files) / file_discovery_time if file_discovery_time > 0 else 0
            
            print(f"{LOG_PREFIX} 📁 Phase 2 Complete: {len(all_files):,} files in {file_discovery_time:.2f}s ({discovery_rate:.1f} files/s)")
            print(f"{LOG_PREFIX} 🔍 Supported extensions: {sorted(supported_extensions)}")
            logger.info(f"{LOG_PREFIX} File discovery completed: {len(all_files)} files")
            logger.info(f"{LOG_PREFIX} Extension filtering: supported extensions are {sorted(supported_extensions)}")
            
            # Phase 3: Stream files to queue with progress
            print(f"{LOG_PREFIX} 📤 Phase 3: Streaming files to processing queue...")
            
            # Limit files if necessary
            if len(all_files) > self.max_files_to_process:
                print(f"{LOG_PREFIX} 🔒 Limiting to {self.max_files_to_process:,} files (found {len(all_files):,})")
                all_files = all_files[:self.max_files_to_process]
            
            # Stream files to queue with progress
            with tqdm(total=len(all_files), desc="📤 Queuing files", 
                     unit="file", ncols=100, dynamic_ncols=True) as pbar:
                
                for file_path in all_files:
                    await file_queue.put(file_path)
                    discovered += 1
                    pbar.update(1)
                    
                    # Yield control periodically
                    if discovered % 100 == 0:
                        await asyncio.sleep(0.001)
            
            # Signal end of discovery
            await file_queue.put(None)
            
            total_discovery_time = time.time() - discovery_start
            overall_rate = discovered / total_discovery_time if total_discovery_time > 0 else 0
            
            print(f"{LOG_PREFIX} ✅ Discovery Complete: {discovered:,} files ready for processing")
            print(f"{LOG_PREFIX} 📊 Total discovery time: {total_discovery_time:.2f}s ({overall_rate:.1f} files/s)")
            
            logger.info(f"{LOG_PREFIX} Parallel file discovery completed: {discovered} files, "
                       f"{directories_scanned} dirs in {total_discovery_time:.2f}s")
            
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Parallel file discovery failed: {e}")
            print(f"{LOG_PREFIX} ❌ Discovery failed: {e}")
            await file_queue.put(None)  # Signal end even on error
    
    def _list_directory_sync(self, directory_path: str):
        """Synchronously list directory contents"""
        try:
            entries = []
            for entry in os.scandir(directory_path):
                try:
                    entries.append((entry.path, entry.is_dir()))
                except (OSError, IOError):
                    continue
            return entries
        except (OSError, IOError):
            return []
    
    def _scan_directory_for_files_sync(self, directory_path: str, supported_extensions: set, skip_files: set):
        """Synchronously scan directory for supported files"""
        try:
            files = []
            skipped_counts = {"extension": 0, "size": 0, "already_processed": 0, "errors": 0}
            
            for entry in os.scandir(directory_path):
                try:
                    if entry.is_file():
                        file_path = entry.path
                        
                        # Skip already processed files
                        if file_path in skip_files:
                            skipped_counts["already_processed"] += 1
                            continue
                        
                        # Check extension and size (match discovery.py logic)
                        ext = Path(file_path).suffix[1:].lower()  # Remove dot and lowercase
                        if ext in supported_extensions:
                            try:
                                stat_info = entry.stat()
                                if 10 < stat_info.st_size < 10 * 1024 * 1024:  # 10 bytes < size < 10MB
                                    files.append(file_path)
                                else:
                                    skipped_counts["size"] += 1
                            except (OSError, IOError):
                                skipped_counts["errors"] += 1
                                continue
                        else:
                            skipped_counts["extension"] += 1
                                
                except (OSError, IOError):
                    skipped_counts["errors"] += 1
                    continue
            
            # Log filtering results if significant filtering occurred
            total_files = len(files) + sum(skipped_counts.values())
            if total_files > 100:  # Only log for directories with many files
                logger.debug(f"{LOG_PREFIX} Directory {directory_path}: "
                           f"found {len(files)} valid files, skipped {sum(skipped_counts.values())} "
                           f"(ext: {skipped_counts['extension']}, size: {skipped_counts['size']}, "
                           f"processed: {skipped_counts['already_processed']}, errors: {skipped_counts['errors']})")
                    
            return files
        except (OSError, IOError):
            return []
    
    async def _process_completed_batch(self, pending_tasks: List) -> List:
        """Process completed tasks and return results"""
        file_paths = [fp for fp, _ in pending_tasks]
        tasks = [task for _, task in pending_tasks]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        completed_results = []
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.debug(f"Error processing {file_path}: {result}")
                completed_results.append(None)
            else:
                completed_results.append(result)
        
        return completed_results
    
    def _show_enhanced_progress(
        self, 
        processed_files: int,
        discovered_files: int,
        total_files: int, 
        total_chunks: int,
        total_summaries: int, 
        failed_files: int, 
        start_time: float,
        embedding_time_total: float,
        database_time_total: float,
        discovery_active: bool
    ):
        """Show enhanced progress update with detailed metrics"""
        elapsed = time.time() - start_time
        files_per_sec = processed_files / elapsed if elapsed > 0 else 0
        progress_percent = (processed_files / max(total_files, 1)) * 100 if total_files > 0 else 0
        
        # Create progress bar
        bar_width = 40
        filled = int((processed_files / max(total_files, 1)) * bar_width)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Calculate averages
        avg_embedding_time = embedding_time_total / max(processed_files, 1)
        avg_database_time = database_time_total / max(processed_files, 1)
        success_rate = ((processed_files - failed_files) / max(processed_files, 1)) * 100
        
        # Discovery status
        discovery_status = "🔍 Active" if discovery_active else "✅ Complete"
        
        print(f"\n{LOG_PREFIX} ═══ PROGRESS UPDATE ═══")
        print(f"{LOG_PREFIX} [{bar}] {processed_files:,}/{total_files:,} files ({progress_percent:.1f}%)")
        print(f"{LOG_PREFIX} 🔍 Discovery: {discovered_files:,} files {discovery_status}")
        print(f"{LOG_PREFIX} 📝 Chunks: {total_chunks:,} | 📑 Summaries: {total_summaries:,}")
        print(f"{LOG_PREFIX} ✅ Success: {success_rate:.1f}% | ❌ Failed: {failed_files:,}")
        print(f"{LOG_PREFIX} ⚡ Speed: {files_per_sec:.1f} files/s | ⏱️  Elapsed: {elapsed:.0f}s")
        print(f"{LOG_PREFIX} 🧠 Avg Embedding: {avg_embedding_time:.2f}s | 💾 Avg Database: {avg_database_time:.2f}s")
        
        logger.info(f"{LOG_PREFIX} Progress: {processed_files}/{total_files} files, {total_chunks} chunks, "
                   f"{success_rate:.1f}% success, {files_per_sec:.1f} files/s")
    
    async def _process_single_file_async(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Enhanced single file processing with detailed timing"""
        file_start = time.time()
        
        try:
            logger.debug(f"{LOG_PREFIX} Processing file: {file_path}")
            
            # Determine file type
            file_type = Path(file_path).suffix[1:].lower()
            
            # Parse file content with timing
            parse_start = time.time()
            parsed_content = await self._parse_file_async(file_path, file_type)
            parse_time = time.time() - parse_start
            
            if not parsed_content:
                logger.debug(f"{LOG_PREFIX} No content parsed from {file_path}")
                return None
            
            # Create chunks with timing
            chunk_start = time.time()
            all_chunks = []
            for content_item in parsed_content:
                chunks = self._create_chunks_sync(content_item, file_path)
                all_chunks.extend(chunks)
            chunk_time = time.time() - chunk_start
            
            if not all_chunks:
                logger.debug(f"{LOG_PREFIX} No chunks created from {file_path}")
                return None
            
            # Limit chunks per file
            original_chunk_count = len(all_chunks)
            if len(all_chunks) > 50:
                all_chunks = all_chunks[:50]  # Take first 50 chunks
                logger.debug(f"{LOG_PREFIX} Limited chunks from {original_chunk_count} to {len(all_chunks)} for {file_path}")
            
            # Generate embeddings for chunks with timing
            embedding_start = time.time()
            chunk_contents = [chunk.content for chunk in all_chunks]
            chunk_embeddings = await async_embedding_service.generate_embeddings_batch(chunk_contents)
            embedding_time = time.time() - embedding_start
            
            # Filter successful embeddings
            valid_chunks = []
            valid_embeddings = []
            
            for chunk, embedding in zip(all_chunks, chunk_embeddings):
                if embedding and len(embedding) == config.embedding.dimension:
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
            
            if not valid_chunks:
                logger.debug(f"{LOG_PREFIX} No valid embeddings for {file_path}")
                return None
            
            # Insert chunks into database with timing
            database_start = time.time()
            chunks_inserted = await batch_db_ops.insert_chunks_batch(valid_chunks, valid_embeddings)
            database_time = time.time() - database_start
            
            total_file_time = time.time() - file_start
            
            logger.debug(f"{LOG_PREFIX} File processed successfully: {file_path} "
                        f"({chunks_inserted} chunks, {total_file_time:.2f}s total, "
                        f"parse: {parse_time:.2f}s, chunk: {chunk_time:.2f}s, "
                        f"embed: {embedding_time:.2f}s, db: {database_time:.2f}s)")
            
            return {
                "chunks": chunks_inserted,
                "summaries": 0,
                "embedding_time": embedding_time,
                "database_time": database_time,
                "parse_time": parse_time,
                "chunk_time": chunk_time,
                "total_time": total_file_time
            }
            
        except Exception as e:
            file_time = time.time() - file_start
            logger.debug(f"{LOG_PREFIX} Error processing file {file_path} after {file_time:.2f}s: {e}")
            return None
    
    async def _parse_file_async(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse file content asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            parsed_content = await loop.run_in_executor(
                None, self._parse_file_sync, file_path, file_type
            )
            return parsed_content
        except Exception as e:
            return []
    
    def _parse_file_sync(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse file content synchronously"""
        try:
            parsed_content = []
            chunk_generator = document_parser.parse_file_stream(
                file_path, file_type, chunk_size=3  # Smaller chunks
            )
            
            for chunk_batch in chunk_generator:
                parsed_content.extend(chunk_batch)
                if len(parsed_content) > 5:  # Limit to avoid large files
                    break
            
            return parsed_content
        except Exception as e:
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
            return []
    
    def _load_checkpoint(self) -> tuple:
        """Load checkpoint data if available"""
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('checkpoint_') and f.endswith('.json')]
        
        if not checkpoint_files:
            logger.info(f"{LOG_PREFIX} No checkpoint files found, starting fresh")
            return 0, 0, 0, 0, set()
        
        # Use the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        
        try:
            with open(latest_checkpoint, 'r') as f:
                data = json.load(f)
            
            processed_files = data.get('processed_files', 0)
            total_chunks = data.get('total_chunks', 0)
            total_summaries = data.get('total_summaries', 0)
            failed_files = data.get('failed_files', 0)
            processed_file_paths = set(data.get('processed_file_paths', []))
            
            logger.info(f"{LOG_PREFIX} Loaded checkpoint from {latest_checkpoint}: "
                       f"{processed_files} files, {total_chunks} chunks")
            
            return processed_files, total_chunks, total_summaries, failed_files, processed_file_paths
            
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to load checkpoint {latest_checkpoint}: {e}")
            return 0, 0, 0, 0, set()
    
    async def _save_checkpoint(
        self, 
        processed_files: int, 
        total_chunks: int, 
        total_summaries: int,
        failed_files: int, 
        processed_file_paths: set, 
        final: bool = False,
        interrupted: bool = False,
        error: str = None
    ):
        """Save checkpoint data"""
        if not self.checkpoint_file:
            return
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'processed_files': processed_files,
            'total_chunks': total_chunks,
            'total_summaries': total_summaries,
            'failed_files': failed_files,
            'processed_file_paths': list(processed_file_paths),
            'final': final,
            'interrupted': interrupted,
            'error': error
        }
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            status = "final" if final else "interrupted" if interrupted else "error" if error else "regular"
            logger.info(f"{LOG_PREFIX} Checkpoint saved ({status}): {processed_files} files, {total_chunks} chunks")
            
            if final:
                print(f"{LOG_PREFIX} 💾 Final checkpoint saved: {self.checkpoint_file}")
            elif interrupted:
                print(f"{LOG_PREFIX} 💾 Interruption checkpoint saved: {self.checkpoint_file}")
            elif error:
                print(f"{LOG_PREFIX} 💾 Error checkpoint saved: {self.checkpoint_file}")
            elif processed_files % 100 == 0:  # Only print every 100 files to avoid spam
                print(f"{LOG_PREFIX} 💾 Checkpoint saved: {processed_files} files processed")
            
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Failed to save checkpoint: {e}")
            print(f"{LOG_PREFIX} ❌ Failed to save checkpoint: {e}")

# Global instant async loader instance
instant_async_loader = InstantAsyncLoader()
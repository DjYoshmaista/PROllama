# file_management/queue_loader.py - Queue-Based Parallel Document Loader
import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Generator, Callable
from pathlib import Path
from queue import Queue, Empty, Full
import multiprocessing as mp
import psutil

from core.config import config
from core.memory import memory_manager
from inference.async_embeddings import async_embedding_service
from inference.summarization import summarization_service
from file_management.parsers import document_parser
from file_management.chunking import text_splitter, TextChunk
from database.operations import db_ops

logger = logging.getLogger(__name__)

@dataclass
class FileJob:
    """Represents a file processing job"""
    file_path: str
    file_type: str
    job_id: str
    priority: int = 1
    size_bytes: int = 0
    
    def __post_init__(self):
        try:
            self.size_bytes = Path(self.file_path).stat().st_size
        except:
            self.size_bytes = 0

@dataclass
class ChunkJob:
    """Represents a chunk processing job"""
    chunks: List[TextChunk]
    file_path: str
    job_id: str
    priority: int = 1

@dataclass
class ProcessingProgress:
    """Thread-safe progress tracking"""
    def __init__(self):
        self._lock = threading.RLock()
        
        # File processing
        self.total_files = 0
        self.queued_files = 0
        self.processing_files = 0
        self.completed_files = 0
        self.failed_files = 0
        
        # Chunk processing
        self.total_chunks = 0
        self.queued_chunks = 0
        self.processing_chunks = 0
        self.completed_chunks = 0
        self.failed_chunks = 0
        
        # Embedding processing
        self.total_embeddings = 0
        self.queued_embeddings = 0
        self.completed_embeddings = 0
        self.failed_embeddings = 0
        
        # Summary processing
        self.total_summaries = 0
        self.completed_summaries = 0
        
        # Performance metrics
        self.start_time = time.time()
        self.last_update = time.time()
        self.files_per_second = 0.0
        self.chunks_per_second = 0.0
        self.embeddings_per_second = 0.0
    
    def update_file_progress(self, queued: int = 0, processing: int = 0, completed: int = 0, failed: int = 0):
        with self._lock:
            self.queued_files += queued
            self.processing_files += processing
            self.completed_files += completed
            self.failed_files += failed
            self._update_rates()
    
    def update_chunk_progress(self, total: int = 0, queued: int = 0, processing: int = 0, completed: int = 0, failed: int = 0):
        with self._lock:
            self.total_chunks += total
            self.queued_chunks += queued
            self.processing_chunks += processing
            self.completed_chunks += completed
            self.failed_chunks += failed
            self._update_rates()
    
    def update_embedding_progress(self, total: int = 0, queued: int = 0, completed: int = 0, failed: int = 0):
        with self._lock:
            self.total_embeddings += total
            self.queued_embeddings += queued
            self.completed_embeddings += completed
            self.failed_embeddings += failed
            self._update_rates()
    
    def update_summary_progress(self, total: int = 0, completed: int = 0):
        with self._lock:
            self.total_summaries += total
            self.completed_summaries += completed
    
    def _update_rates(self):
        """Update processing rates"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            self.files_per_second = self.completed_files / elapsed
            self.chunks_per_second = self.completed_chunks / elapsed
            self.embeddings_per_second = self.completed_embeddings / elapsed
        
        self.last_update = current_time
    
    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self.start_time
            return {
                # Files
                'total_files': self.total_files,
                'queued_files': self.queued_files,
                'processing_files': self.processing_files,
                'completed_files': self.completed_files,
                'failed_files': self.failed_files,
                
                # Chunks
                'total_chunks': self.total_chunks,
                'queued_chunks': self.queued_chunks,
                'processing_chunks': self.processing_chunks,
                'completed_chunks': self.completed_chunks,
                'failed_chunks': self.failed_chunks,
                
                # Embeddings
                'total_embeddings': self.total_embeddings,
                'queued_embeddings': self.queued_embeddings,
                'completed_embeddings': self.completed_embeddings,
                'failed_embeddings': self.failed_embeddings,
                
                # Summaries
                'total_summaries': self.total_summaries,
                'completed_summaries': self.completed_summaries,
                
                # Performance
                'elapsed_time': elapsed,
                'files_per_second': self.files_per_second,
                'chunks_per_second': self.chunks_per_second,
                'embeddings_per_second': self.embeddings_per_second,
                
                # Completion rates
                'file_completion_rate': (self.completed_files / max(self.total_files, 1)) * 100,
                'chunk_completion_rate': (self.completed_chunks / max(self.total_chunks, 1)) * 100,
                'embedding_completion_rate': (self.completed_embeddings / max(self.total_embeddings, 1)) * 100,
            }

class ProgressDisplayManager:
    """Real-time progress display manager"""
    
    def __init__(self, progress: ProcessingProgress):
        self.progress = progress
        self._running = False
        self._display_thread: Optional[threading.Thread] = None
        self._update_interval = 0.5
    
    def start(self):
        if self._running:
            return
        
        self._running = True
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        
        # Clear screen and prepare for progress display
        print("\n" * 10)  # Reserve space
    
    def stop(self):
        self._running = False
        if self._display_thread:
            self._display_thread.join(timeout=1)
    
    def _display_loop(self):
        """Main display loop"""
        try:
            while self._running:
                self._update_display()
                time.sleep(self._update_interval)
        except Exception as e:
            logger.error(f"Display loop error: {e}")
    
    def _update_display(self):
        """Update the progress display"""
        snapshot = self.progress.get_snapshot()
        
        # Move cursor up and clear
        print('\033[10A', end='')  # Move up 10 lines
        print('\033[J', end='')    # Clear from cursor down
        
        # Header
        print("🚀 " + "="*70)
        print("   QUEUE-BASED PARALLEL DOCUMENT PROCESSING")
        print("="*73)
        
        # File Processing Progress
        file_percent = snapshot['file_completion_rate']
        file_bar = self._create_bar(snapshot['completed_files'], snapshot['total_files'])
        print(f"📁 Files:      {file_bar} {snapshot['completed_files']:4d}/{snapshot['total_files']:4d} ({file_percent:5.1f}%) "
              f"[Q:{snapshot['queued_files']}|P:{snapshot['processing_files']}|F:{snapshot['failed_files']}]")
        
        # Chunk Processing Progress  
        chunk_percent = snapshot['chunk_completion_rate']
        chunk_bar = self._create_bar(snapshot['completed_chunks'], snapshot['total_chunks'])
        print(f"📝 Chunks:     {chunk_bar} {snapshot['completed_chunks']:4d}/{snapshot['total_chunks']:4d} ({chunk_percent:5.1f}%) "
              f"[Q:{snapshot['queued_chunks']}|P:{snapshot['processing_chunks']}|F:{snapshot['failed_chunks']}]")
        
        # Embedding Progress
        embed_percent = snapshot['embedding_completion_rate']
        embed_bar = self._create_bar(snapshot['completed_embeddings'], snapshot['total_embeddings'])
        embed_queue_size = async_embedding_service.queue_size
        print(f"🧠 Embeddings: {embed_bar} {snapshot['completed_embeddings']:4d}/{snapshot['total_embeddings']:4d} ({embed_percent:5.1f}%) "
              f"[Q:{embed_queue_size}|F:{snapshot['failed_embeddings']}]")
        
        # Summary Progress
        summary_percent = (snapshot['completed_summaries'] / max(snapshot['total_summaries'], 1)) * 100
        summary_bar = self._create_bar(snapshot['completed_summaries'], snapshot['total_summaries'])
        print(f"📑 Summaries:  {summary_bar} {snapshot['completed_summaries']:4d}/{snapshot['total_summaries']:4d} ({summary_percent:5.1f}%)")
        
        # Performance Metrics
        print(f"⚡ Performance: {snapshot['files_per_second']:.1f} files/s | "
              f"{snapshot['chunks_per_second']:.1f} chunks/s | "
              f"{snapshot['embeddings_per_second']:.1f} embeddings/s")
        
        # System Resources
        memory_info = memory_manager.get_memory_info()
        cpu_percent = psutil.cpu_percent()
        print(f"💾 Resources:   CPU: {cpu_percent:5.1f}% | Memory: {memory_info['percent']:5.1f}% | "
              f"Elapsed: {snapshot['elapsed_time']:.1f}s")
        
        # Overall Progress
        overall_percent = file_percent
        overall_bar = self._create_bar(snapshot['completed_files'], snapshot['total_files'], width=50)
        print(f"🎯 Overall:    {overall_bar} ({overall_percent:5.1f}%)")
    
    def _create_bar(self, current: int, total: int, width: int = 40) -> str:
        """Create progress bar visualization"""
        if total <= 0:
            return '[' + '█' * width + ']'
        
        filled = int((current / total) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f'[{bar}]'

class QueueBasedParallelLoader:
    """High-performance queue-based parallel document loader"""
    
    def __init__(self):
        # Configuration
        self.max_file_workers = min(mp.cpu_count(), 8)
        self.max_chunk_workers = 4
        self.file_queue_size = 100
        self.chunk_queue_size = 200
        
        # Queues
        self.file_queue: Queue = Queue(maxsize=self.file_queue_size)
        self.chunk_queue: Queue = Queue(maxsize=self.chunk_queue_size)
        
        # Worker pools
        self.file_executor = ThreadPoolExecutor(max_workers=self.max_file_workers)
        self.chunk_executor = ThreadPoolExecutor(max_workers=self.max_chunk_workers)
        
        # State management
        self._running = False
        self.progress = ProcessingProgress()
        self.display_manager = ProgressDisplayManager(self.progress)
        
        # Worker threads
        self.file_workers = []
        self.chunk_workers = []
        
        logger.info(f"Initialized queue-based loader: {self.max_file_workers} file workers, "
                   f"{self.max_chunk_workers} chunk workers")
    
    async def load_folder_parallel(
        self,
        folder_path: str,
        file_generator: Generator[str, None, None],
        total_files: int,
        enable_chunking: bool = True,
        enable_summarization: bool = True
    ) -> Dict[str, Any]:
        """Main parallel loading method with queue-based processing"""
        
        logger.info(f"Starting queue-based parallel loading of {total_files} files")
        
        try:
            # Initialize services
            await async_embedding_service.start()
            
            # Initialize progress
            self.progress.total_files = total_files
            self._running = True
            
            # Start progress display
            self.display_manager.start()
            
            # Start memory monitoring
            memory_manager.start_monitoring()
            
            # Start worker threads
            self._start_workers()
            
            # Fill file queue
            await self._populate_file_queue(file_generator)
            
            # Wait for all processing to complete
            results = await self._wait_for_completion(enable_chunking, enable_summarization)
            
            return results
            
        except Exception as e:
            logger.error(f"Queue-based loading failed: {e}")
            return {
                "error": str(e),
                "processed_files": self.progress.completed_files,
                "total_chunks": self.progress.completed_chunks,
                "total_summaries": self.progress.completed_summaries
            }
        
        finally:
            # Cleanup
            await self._cleanup()
    
    def _start_workers(self):
        """Start all worker threads"""
        # File processing workers
        for i in range(self.max_file_workers):
            worker = threading.Thread(
                target=self._file_worker,
                args=(i,),
                daemon=True,
                name=f"FileWorker-{i}"
            )
            worker.start()
            self.file_workers.append(worker)
        
        # Chunk processing workers
        for i in range(self.max_chunk_workers):
            worker = threading.Thread(
                target=self._chunk_worker,
                args=(i,),
                daemon=True,
                name=f"ChunkWorker-{i}"
            )
            worker.start()
            self.chunk_workers.append(worker)
        
        logger.info(f"Started {len(self.file_workers)} file workers and {len(self.chunk_workers)} chunk workers")
    
    async def _populate_file_queue(self, file_generator: Generator[str, None, None]):
        """Populate the file queue from the generator"""
        queued_files = 0
        
        for file_path in file_generator:
            try:
                # Create file job
                file_type = Path(file_path).suffix[1:].lower()
                priority = self._calculate_priority(file_path, file_type)
                
                job = FileJob(
                    file_path=file_path,
                    file_type=file_type,
                    job_id=f"file_{queued_files}",
                    priority=priority
                )
                
                # Add to queue (non-blocking)
                self.file_queue.put(job, timeout=0.1)
                queued_files += 1
                self.progress.update_file_progress(queued=1)
                
            except Full:
                logger.warning("File queue full, waiting...")
                await asyncio.sleep(0.1)
                # Retry
                try:
                    self.file_queue.put(job, timeout=1)
                    queued_files += 1
                    self.progress.update_file_progress(queued=1)
                except Full:
                    logger.error(f"Failed to queue file: {file_path}")
                    self.progress.update_file_progress(failed=1)
            
            except Exception as e:
                logger.error(f"Error creating job for {file_path}: {e}")
                self.progress.update_file_progress(failed=1)
        
        # Add sentinel values to stop workers
        for _ in range(self.max_file_workers):
            self.file_queue.put(None)
        
        logger.info(f"Queued {queued_files} files for processing")
    
    def _calculate_priority(self, file_path: str, file_type: str) -> int:
        """Calculate processing priority for a file"""
        priority = 1
        
        # Priority by file type
        type_priorities = {
            'py': 5, 'md': 4, 'txt': 3, 'json': 2, 'csv': 1
        }
        priority += type_priorities.get(file_type, 1)
        
        # Priority by size (smaller files first)
        try:
            size = Path(file_path).stat().st_size
            if size < 10000:
                priority += 3
            elif size < 100000:
                priority += 2
            elif size < 1000000:
                priority += 1
        except:
            pass
        
        return priority
    
    def _file_worker(self, worker_id: int):
        """File processing worker thread"""
        logger.debug(f"File worker {worker_id} started")
        
        while self._running:
            try:
                # Get file job
                job = self.file_queue.get(timeout=1)
                
                # Sentinel value
                if job is None:
                    break
                
                self.progress.update_file_progress(queued=-1, processing=1)
                
                # Process file
                success = self._process_file(job)
                
                if success:
                    self.progress.update_file_progress(processing=-1, completed=1)
                else:
                    self.progress.update_file_progress(processing=-1, failed=1)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"File worker {worker_id} error: {e}")
                self.progress.update_file_progress(processing=-1, failed=1)
        
        logger.debug(f"File worker {worker_id} stopped")
    
    def _process_file(self, job: FileJob) -> bool:
        """Process a single file job"""
        try:
            # Check if file exists and is readable
            if not Path(job.file_path).exists():
                logger.warning(f"File not found: {job.file_path}")
                return False
            
            # Parse file content
            parsed_content = self._parse_file_sync(job.file_path, job.file_type)
            if not parsed_content:
                return False
            
            # Create chunks
            all_chunks = []
            for content_item in parsed_content:
                chunks = self._create_chunks_sync(content_item, job.file_path)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                return False
            
            # Create chunk job
            chunk_job = ChunkJob(
                chunks=all_chunks,
                file_path=job.file_path,
                job_id=job.job_id,
                priority=job.priority
            )
            
            # Queue chunks for processing
            try:
                self.chunk_queue.put(chunk_job, timeout=1)
                self.progress.update_chunk_progress(total=len(all_chunks), queued=len(all_chunks))
                return True
            except Full:
                logger.warning(f"Chunk queue full, dropping chunks for {job.file_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error processing file {job.file_path}: {e}")
            return False
    
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
    
    def _chunk_worker(self, worker_id: int):
        """Chunk processing worker thread"""
        logger.debug(f"Chunk worker {worker_id} started")
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self._running:
                try:
                    # Get chunk job
                    job = self.chunk_queue.get(timeout=1)
                    
                    # Sentinel value
                    if job is None:
                        break
                    
                    self.progress.update_chunk_progress(queued=-len(job.chunks), processing=len(job.chunks))
                    
                    # Process chunks
                    success = loop.run_until_complete(self._process_chunks_async(job))
                    
                    if success:
                        self.progress.update_chunk_progress(processing=-len(job.chunks), completed=len(job.chunks))
                    else:
                        self.progress.update_chunk_progress(processing=-len(job.chunks), failed=len(job.chunks))
                
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Chunk worker {worker_id} error: {e}")
                    if 'job' in locals():
                        self.progress.update_chunk_progress(processing=-len(job.chunks), failed=len(job.chunks))
        
        finally:
            loop.close()
            logger.debug(f"Chunk worker {worker_id} stopped")
    
    async def _process_chunks_async(self, job: ChunkJob) -> bool:
        """Process chunks asynchronously"""
        try:
            chunks = job.chunks
            if not chunks:
                return False
            
            # Generate embeddings
            self.progress.update_embedding_progress(total=len(chunks), queued=len(chunks))
            
            chunk_contents = [chunk.content for chunk in chunks]
            embeddings = await async_embedding_service.generate_embeddings_batch(chunk_contents)
            
            # Filter valid embeddings
            valid_chunks = []
            valid_embeddings = []
            
            for chunk, embedding in zip(chunks, embeddings):
                if embedding and len(embedding) == config.embedding.dimension:
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
                    self.progress.update_embedding_progress(queued=-1, completed=1)
                else:
                    self.progress.update_embedding_progress(queued=-1, failed=1)
            
            if not valid_chunks:
                return False
            
            # Insert chunks into database
            inserted = await db_ops.insert_chunks_batch(valid_chunks, valid_embeddings)
            
            # Generate summaries if needed
            if len(valid_chunks) > 1:
                try:
                    self.progress.update_summary_progress(total=len(valid_chunks))
                    summaries = await summarization_service.summarize_chunks(valid_chunks)
                    
                    if summaries:
                        # Generate embeddings for summaries
                        summary_texts = [s.summary for s in summaries]
                        summary_embeddings = await async_embedding_service.generate_embeddings_batch(summary_texts)
                        
                        # Filter and insert summaries
                        valid_summaries = []
                        valid_summary_embeddings = []
                        
                        for summary, embedding in zip(summaries, summary_embeddings):
                            if embedding and len(embedding) == config.embedding.dimension:
                                valid_summaries.append(summary)
                                valid_summary_embeddings.append(embedding)
                        
                        if valid_summaries:
                            await db_ops.insert_summaries_batch(valid_summaries, valid_summary_embeddings)
                            self.progress.update_summary_progress(completed=len(valid_summaries))
                
                except Exception as e:
                    logger.error(f"Summary generation failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            return False
    
    async def _wait_for_completion(self, enable_chunking: bool, enable_summarization: bool) -> Dict[str, Any]:
        """Wait for all processing to complete"""
        
        # Wait for file queue to empty
        while not self.file_queue.empty() or any(w.is_alive() for w in self.file_workers):
            await asyncio.sleep(0.5)
        
        # Signal chunk workers to stop
        for _ in range(self.max_chunk_workers):
            try:
                self.chunk_queue.put(None, timeout=0.1)
            except Full:
                pass
        
        # Wait for chunk queue to empty
        while not self.chunk_queue.empty() or any(w.is_alive() for w in self.chunk_workers):
            await asyncio.sleep(0.5)
        
        # Wait for embedding queue to empty
        max_wait = 30  # seconds
        wait_time = 0
        while async_embedding_service.queue_size > 0 and wait_time < max_wait:
            await asyncio.sleep(1)
            wait_time += 1
        
        # Get final results
        snapshot = self.progress.get_snapshot()
        
        return {
            "processed_files": snapshot['completed_files'],
            "total_chunks": snapshot['completed_chunks'],
            "total_summaries": snapshot['completed_summaries'],
            "total_embeddings": snapshot['completed_embeddings'],
            "failed_files": snapshot['failed_files'],
            "failed_chunks": snapshot['failed_chunks'],
            "failed_embeddings": snapshot['failed_embeddings'],
            "success_rate": snapshot['file_completion_rate'] / 100,
            "processing_time": snapshot['elapsed_time'],
            "files_per_second": snapshot['files_per_second'],
            "chunks_per_second": snapshot['chunks_per_second'],
            "embeddings_per_second": snapshot['embeddings_per_second']
        }
    
    async def _cleanup(self):
        """Cleanup resources"""
        self._running = False
        
        # Stop display
        self.display_manager.stop()
        
        # Stop memory monitoring
        memory_manager.stop_monitoring()
        
        # Wait for workers
        for worker in self.file_workers + self.chunk_workers:
            worker.join(timeout=2)
        
        # Shutdown executors
        self.file_executor.shutdown(wait=True, timeout=5)
        self.chunk_executor.shutdown(wait=True, timeout=5)
        
        # Stop embedding service
        await async_embedding_service.stop()
        
        # Clear queues
        while not self.file_queue.empty():
            try:
                self.file_queue.get_nowait()
            except Empty:
                break
        
        while not self.chunk_queue.empty():
            try:
                self.chunk_queue.get_nowait()
            except Empty:
                break
        
        # Final progress display
        self._display_final_results()
        
        logger.info("Queue-based loader cleanup completed")
    
    def _display_final_results(self):
        """Display final processing results"""
        snapshot = self.progress.get_snapshot()
        
        print("\n" + "="*80)
        print("🎉 QUEUE-BASED PARALLEL LOADING COMPLETED")
        print("="*80)
        print(f"📁 Files:       {snapshot['completed_files']:6d} processed | {snapshot['failed_files']:3d} failed")
        print(f"📝 Chunks:      {snapshot['completed_chunks']:6d} created   | {snapshot['failed_chunks']:3d} failed")
        print(f"🧠 Embeddings:  {snapshot['completed_embeddings']:6d} generated | {snapshot['failed_embeddings']:3d} failed")
        print(f"📑 Summaries:   {snapshot['completed_summaries']:6d} created")
        print(f"⚡ Performance: {snapshot['files_per_second']:.1f} files/s | "
              f"{snapshot['chunks_per_second']:.1f} chunks/s | "
              f"{snapshot['embeddings_per_second']:.1f} embeddings/s")
        print(f"⏱️  Total Time:  {snapshot['elapsed_time']:.1f} seconds")
        print(f"📊 Success Rate: {snapshot['file_completion_rate']:.1f}%")
        print("="*80)

# Global queue-based loader instance
queue_parallel_loader = QueueBasedParallelLoader()
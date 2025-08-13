# file_management/parallel_loader.py - Parallelized Bulk Document Loader
import asyncio
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple, Generator
from pathlib import Path
import threading
from queue import Queue
import json

from core.config import config
from core.memory import memory_manager
from inference.embeddings import embedding_service
from inference.summarization import summarization_service
from file_management.parsers import document_parser
from file_management.chunking import text_splitter, TextChunk
from database.operations import db_ops

logger = logging.getLogger(__name__)

@dataclass
class FileProcessingJob:
    """Represents a file processing job"""
    file_path: str
    file_type: str
    job_id: str
    priority: int = 1
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = self._generate_job_id()
    
    def _generate_job_id(self) -> str:
        """Generate unique job ID based on file path"""
        return hashlib.md5(self.file_path.encode()).hexdigest()[:12]

@dataclass
class ProcessingProgress:
    """Thread-safe progress tracking"""
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    created_chunks: int = 0
    total_summaries: int = 0
    created_summaries: int = 0
    total_embeddings: int = 0
    generated_embeddings: int = 0
    failed_files: int = 0
    duplicate_chunks: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_file_progress(self, increment: int = 1):
        with self._lock:
            self.processed_files += increment
    
    def update_chunk_progress(self, chunks_created: int, total_chunks: int = 0):
        with self._lock:
            self.created_chunks += chunks_created
            if total_chunks > 0:
                self.total_chunks += total_chunks
    
    def update_summary_progress(self, summaries_created: int, total_summaries: int = 0):
        with self._lock:
            self.created_summaries += summaries_created
            if total_summaries > 0:
                self.total_summaries += total_summaries
    
    def update_embedding_progress(self, embeddings_generated: int, total_embeddings: int = 0):
        with self._lock:
            self.generated_embeddings += embeddings_generated
            if total_embeddings > 0:
                self.total_embeddings += total_embeddings
    
    def increment_failures(self, increment: int = 1):
        with self._lock:
            self.failed_files += increment
    
    def increment_duplicates(self, increment: int = 1):
        with self._lock:
            self.duplicate_chunks += increment
    
    def get_snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                'total_files': self.total_files,
                'processed_files': self.processed_files,
                'total_chunks': self.total_chunks,
                'created_chunks': self.created_chunks,
                'total_summaries': self.total_summaries,
                'created_summaries': self.created_summaries,
                'total_embeddings': self.total_embeddings,
                'generated_embeddings': self.generated_embeddings,
                'failed_files': self.failed_files,
                'duplicate_chunks': self.duplicate_chunks
            }

class ContentDeduplicationService:
    """Service for content-based deduplication"""
    
    def __init__(self):
        self._content_hashes: Set[str] = set()
        self._chunk_hashes: Set[str] = set()
        self._file_checksums: Set[str] = set()
        self._lock = threading.RLock()
        self._loaded_existing = False
    
    async def load_existing_content_hashes(self):
        """Load existing content hashes from database"""
        if self._loaded_existing:
            return
        
        with self._lock:
            if self._loaded_existing:
                return
            
            try:
                # Load existing chunk content hashes
                existing_chunks = await self._get_existing_chunk_hashes()
                self._chunk_hashes.update(existing_chunks)
                
                # Load existing file checksums
                existing_files = await self._get_existing_file_checksums()
                self._file_checksums.update(existing_files)
                
                self._loaded_existing = True
                logger.info(f"Loaded {len(self._chunk_hashes)} existing chunk hashes and {len(self._file_checksums)} file checksums")
                
            except Exception as e:
                logger.error(f"Failed to load existing content hashes: {e}")
    
    async def _get_existing_chunk_hashes(self) -> Set[str]:
        """Get content hashes of existing chunks"""
        try:
            # This would need to be implemented in database operations
            # For now, return empty set
            return set()
        except Exception as e:
            logger.error(f"Failed to get existing chunk hashes: {e}")
            return set()
    
    async def _get_existing_file_checksums(self) -> Set[str]:
        """Get checksums of already processed files"""
        try:
            # This would need to be implemented in database operations
            # For now, return empty set
            return set()
        except Exception as e:
            logger.error(f"Failed to get existing file checksums: {e}")
            return set()
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate hash for content deduplication"""
        # Normalize content for consistent hashing
        normalized = ' '.join(content.strip().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def is_file_already_processed(self, file_path: str) -> bool:
        """Check if file was already processed"""
        checksum = self.calculate_file_checksum(file_path)
        if not checksum:
            return False
        
        with self._lock:
            return checksum in self._file_checksums
    
    def is_content_duplicate(self, content: str) -> bool:
        """Check if content is duplicate"""
        content_hash = self.calculate_content_hash(content)
        
        with self._lock:
            return content_hash in self._chunk_hashes
    
    def add_content_hash(self, content: str):
        """Add content hash to prevent future duplicates"""
        content_hash = self.calculate_content_hash(content)
        
        with self._lock:
            self._chunk_hashes.add(content_hash)
    
    def add_file_checksum(self, file_path: str):
        """Add file checksum to prevent reprocessing"""
        checksum = self.calculate_file_checksum(file_path)
        if checksum:
            with self._lock:
                self._file_checksums.add(checksum)
    
    def filter_duplicate_chunks(self, chunks: List[TextChunk]) -> Tuple[List[TextChunk], int]:
        """Filter out duplicate chunks and return unique ones"""
        unique_chunks = []
        duplicates = 0
        
        for chunk in chunks:
            if not self.is_content_duplicate(chunk.content):
                unique_chunks.append(chunk)
                self.add_content_hash(chunk.content)
            else:
                duplicates += 1
        
        return unique_chunks, duplicates

class ProgressDisplayManager:
    """Manages multiple progress bars simultaneously"""
    
    def __init__(self, progress_tracker: ProcessingProgress):
        self.progress = progress_tracker
        self._running = False
        self._display_thread: Optional[threading.Thread] = None
        self._update_interval = 0.5  # Update every 500ms
    
    def start_display(self):
        """Start the progress display thread"""
        if self._running:
            return
        
        self._running = True
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
    
    def stop_display(self):
        """Stop the progress display thread"""
        self._running = False
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=1.0)
    
    def _display_loop(self):
        """Main display loop for progress bars"""
        try:
            while self._running:
                self._update_display()
                time.sleep(self._update_interval)
        except Exception as e:
            logger.error(f"Progress display error: {e}")
    
    def _update_display(self):
        """Update all progress bars"""
        snapshot = self.progress.get_snapshot()
        
        # Clear previous lines (move cursor up and clear)
        print('\033[6A', end='')  # Move cursor up 6 lines
        print('\033[J', end='')   # Clear from cursor to end of screen
        
        # File Processing Progress
        file_percent = (snapshot['processed_files'] / max(snapshot['total_files'], 1)) * 100
        file_bar = self._create_progress_bar(snapshot['processed_files'], snapshot['total_files'], 40)
        print(f"📁 Files:      {file_bar} {snapshot['processed_files']:4d}/{snapshot['total_files']:4d} ({file_percent:5.1f}%)")
        
        # Chunking Progress
        chunk_percent = (snapshot['created_chunks'] / max(snapshot['total_chunks'], 1)) * 100 if snapshot['total_chunks'] > 0 else 0
        chunk_bar = self._create_progress_bar(snapshot['created_chunks'], max(snapshot['total_chunks'], 1), 40)
        print(f"📝 Chunks:     {chunk_bar} {snapshot['created_chunks']:4d}/{snapshot['total_chunks']:4d} ({chunk_percent:5.1f}%)")
        
        # Summarization Progress
        summary_percent = (snapshot['created_summaries'] / max(snapshot['total_summaries'], 1)) * 100 if snapshot['total_summaries'] > 0 else 0
        summary_bar = self._create_progress_bar(snapshot['created_summaries'], max(snapshot['total_summaries'], 1), 40)
        print(f"📑 Summaries:  {summary_bar} {snapshot['created_summaries']:4d}/{snapshot['total_summaries']:4d} ({summary_percent:5.1f}%)")
        
        # Embedding Progress
        embed_percent = (snapshot['generated_embeddings'] / max(snapshot['total_embeddings'], 1)) * 100 if snapshot['total_embeddings'] > 0 else 0
        embed_bar = self._create_progress_bar(snapshot['generated_embeddings'], max(snapshot['total_embeddings'], 1), 40)
        print(f"🧠 Embeddings: {embed_bar} {snapshot['generated_embeddings']:4d}/{snapshot['total_embeddings']:4d} ({embed_percent:5.1f}%)")
        
        # Statistics
        print(f"❌ Failed: {snapshot['failed_files']:3d}  🔄 Duplicates: {snapshot['duplicate_chunks']:3d}  💾 Memory: {memory_manager.get_memory_info()['percent']:.1f}%")
        
        # Overall Progress
        overall_percent = file_percent
        overall_bar = self._create_progress_bar(snapshot['processed_files'], snapshot['total_files'], 50)
        print(f"🚀 Overall:    {overall_bar} ({overall_percent:5.1f}%)")
    
    def _create_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Create a visual progress bar"""
        if total <= 0:
            return '█' * width
        
        filled = int((current / total) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f'[{bar}]'

class ParallelBulkLoader:
    """Fully parallelized bulk document loader with atomic file access"""
    
    def __init__(self):
        self.max_file_workers = 8  # File discovery and access workers
        self.max_doc_workers = 2   # Workers per document processing
        self.deduplication_service = ContentDeduplicationService()
        self.progress_tracker = ProcessingProgress()
        self.progress_display = ProgressDisplayManager(self.progress_tracker)
        
        # Thread-safe job queues
        self.file_queue: Queue = Queue()
        self.processing_queue: Queue = Queue()
        self.embedding_queue: Queue = Queue()
        
        # Thread-safe file access tracking
        self._file_locks: Dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()
        
        # Worker pools
        self.file_executor = ThreadPoolExecutor(max_workers=self.max_file_workers)
        self.processing_executor = ThreadPoolExecutor(max_workers=self.max_file_workers * self.max_doc_workers)
    
    def _get_file_lock(self, file_path: str) -> threading.Lock:
        """Get atomic lock for specific file"""
        with self._locks_mutex:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.Lock()
            return self._file_locks[file_path]
    
    async def load_folder_parallel(
        self,
        folder_path: str,
        file_generator: Generator[str, None, None],
        total_files: int,
        enable_chunking: bool = True,
        enable_summarization: bool = True
    ) -> Dict[str, Any]:
        """
        Main parallel loading method with comprehensive progress tracking
        """
        logger.info(f"Starting parallel bulk loading of {total_files} files")
        
        # Initialize progress tracking
        self.progress_tracker.total_files = total_files
        
        try:
            # Start services
            await embedding_service.start()
            await self.deduplication_service.load_existing_content_hashes()
            
            # Start progress display
            print("\n" + "="*80)
            print("🚀 PARALLEL BULK LOADING - LIVE PROGRESS")
            print("="*80)
            print("\n" * 6)  # Reserve space for progress bars
            self.progress_display.start_display()
            
            # Start memory monitoring
            memory_manager.start_monitoring()
            
            # Create file processing jobs
            jobs = self._create_file_jobs(file_generator, folder_path)
            
            # Execute parallel processing pipeline
            results = await self._execute_parallel_pipeline(
                jobs, enable_chunking, enable_summarization
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel bulk loading failed: {e}")
            return {
                "error": str(e),
                "processed_files": self.progress_tracker.processed_files,
                "created_chunks": self.progress_tracker.created_chunks,
                "created_summaries": self.progress_tracker.created_summaries
            }
        finally:
            # Cleanup
            self.progress_display.stop_display()
            memory_manager.stop_monitoring()
            await embedding_service.stop()
            self.file_executor.shutdown(wait=True)
            self.processing_executor.shutdown(wait=True)
            
            # Final results display
            self._display_final_results()
    
    def _create_file_jobs(self, file_generator: Generator[str, None, None], folder_path: str) -> List[FileProcessingJob]:
        """Create processing jobs from file generator"""
        jobs = []
        
        for file_path in file_generator:
            # Skip already processed files
            if self.deduplication_service.is_file_already_processed(file_path):
                self.progress_tracker.increment_duplicates()
                continue
            
            file_type = Path(file_path).suffix[1:].lower()
            priority = self._calculate_file_priority(file_path, file_type)
            
            job = FileProcessingJob(
                file_path=file_path,
                file_type=file_type,
                priority=priority
            )
            jobs.append(job)
        
        # Sort by priority (higher first)
        jobs.sort(key=lambda x: x.priority, reverse=True)
        return jobs
    
    def _calculate_file_priority(self, file_path: str, file_type: str) -> int:
        """Calculate processing priority for file"""
        priority = 1
        
        # Prioritize by file type
        type_priorities = {
            'py': 5,    # Python files are often important
            'md': 4,    # Documentation
            'txt': 3,   # Text files
            'json': 2,  # Configuration/data
            'csv': 1    # Data files
        }
        priority += type_priorities.get(file_type, 1)
        
        # Prioritize by file size (smaller files first for quick wins)
        try:
            file_size = Path(file_path).stat().st_size
            if file_size < 10000:  # < 10KB
                priority += 2
            elif file_size < 100000:  # < 100KB
                priority += 1
        except:
            pass
        
        return priority
    
    async def _execute_parallel_pipeline(
        self,
        jobs: List[FileProcessingJob],
        enable_chunking: bool,
        enable_summarization: bool
    ) -> Dict[str, Any]:
        """Execute the parallel processing pipeline"""
        
        # Phase 1: Parallel file processing (8 workers)
        file_processing_tasks = []
        semaphore = asyncio.Semaphore(self.max_file_workers)
        
        async def process_file_with_semaphore(job: FileProcessingJob):
            async with semaphore:
                return await self._process_single_file_job(job, enable_chunking, enable_summarization)
        
        # Create tasks for all jobs
        for job in jobs:
            task = asyncio.create_task(process_file_with_semaphore(job))
            file_processing_tasks.append(task)
        
        # Execute all tasks and collect results
        all_results = []
        for task in asyncio.as_completed(file_processing_tasks):
            try:
                result = await task
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"File processing task failed: {e}")
                self.progress_tracker.increment_failures()
        
        # Compile final results
        total_chunks = sum(r.get('chunks', 0) for r in all_results)
        total_summaries = sum(r.get('summaries', 0) for r in all_results)
        success_rate = (len(all_results) / max(len(jobs), 1)) if jobs else 0
        
        return {
            "processed_files": len(all_results),
            "total_chunks": total_chunks,
            "total_summaries": total_summaries,
            "failed_files": self.progress_tracker.failed_files,
            "duplicate_chunks": self.progress_tracker.duplicate_chunks,
            "success_rate": success_rate
        }
    
    async def _process_single_file_job(
        self,
        job: FileProcessingJob,
        enable_chunking: bool,
        enable_summarization: bool
    ) -> Optional[Dict[str, Any]]:
        """Process a single file job with atomic file access"""
        
        # Atomic file access
        file_lock = self._get_file_lock(job.file_path)
        
        with file_lock:
            try:
                # Double-check file hasn't been processed by another worker
                if self.deduplication_service.is_file_already_processed(job.file_path):
                    self.progress_tracker.increment_duplicates()
                    return None
                
                # Mark file as being processed
                self.deduplication_service.add_file_checksum(job.file_path)
                
                # Parse file content
                parsed_content = await self._parse_file_atomic(job.file_path, job.file_type)
                if not parsed_content:
                    self.progress_tracker.increment_failures()
                    return None
                
                # Process content with up to 2 workers per document
                result = await self._process_file_content(
                    parsed_content, job.file_path, enable_chunking, enable_summarization
                )
                
                self.progress_tracker.update_file_progress()
                return result
                
            except Exception as e:
                logger.error(f"Failed to process {job.file_path}: {e}")
                self.progress_tracker.increment_failures()
                return None
    
    async def _parse_file_atomic(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse file content atomically"""
        try:
            parsed_content = []
            
            # Use existing document parser but ensure atomic access
            chunk_generator = document_parser.parse_file_stream(
                file_path, file_type, chunk_size=10
            )
            
            for chunk_batch in chunk_generator:
                parsed_content.extend(chunk_batch)
            
            return parsed_content
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return []
    
    async def _process_file_content(
        self,
        parsed_content: List[Dict[str, Any]],
        file_path: str,
        enable_chunking: bool,
        enable_summarization: bool
    ) -> Dict[str, Any]:
        """Process file content with parallelization (up to 2 workers per document)"""
        
        all_chunks = []
        
        # Phase 1: Create chunks (can be parallelized per content section)
        if enable_chunking and len(parsed_content) > 1:
            # Process content sections in parallel (up to 2 workers)
            chunk_tasks = []
            semaphore = asyncio.Semaphore(self.max_doc_workers)
            
            async def create_chunks_with_semaphore(content_item):
                async with semaphore:
                    return await self._create_chunks_for_content(content_item, file_path)
            
            for content_item in parsed_content:
                task = asyncio.create_task(create_chunks_with_semaphore(content_item))
                chunk_tasks.append(task)
            
            # Collect chunk results
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for result in chunk_results:
                if isinstance(result, list):
                    all_chunks.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Chunk creation failed: {result}")
        else:
            # Sequential processing for single content or disabled chunking
            for content_item in parsed_content:
                chunks = await self._create_chunks_for_content(content_item, file_path)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            return {"chunks": 0, "summaries": 0}
        
        # Update progress
        self.progress_tracker.update_chunk_progress(0, len(all_chunks))
        
        # Phase 2: Deduplication
        unique_chunks, duplicates = self.deduplication_service.filter_duplicate_chunks(all_chunks)
        self.progress_tracker.increment_duplicates(duplicates)
        
        if not unique_chunks:
            return {"chunks": 0, "summaries": 0}
        
        # Phase 3: Generate embeddings (parallel)
        embeddings_created = await self._generate_embeddings_parallel(unique_chunks)
        
        # Phase 4: Create summaries (if enabled)
        summaries_created = 0
        if enable_summarization and len(unique_chunks) > 1:
            summaries_created = await self._create_summaries_parallel(unique_chunks)
        
        return {
            "chunks": embeddings_created,
            "summaries": summaries_created
        }
    
    async def _create_chunks_for_content(
        self, 
        content_item: Dict[str, Any], 
        file_path: str
    ) -> List[TextChunk]:
        """Create chunks for a content item"""
        content = content_item.get("content", "")
        if not content.strip():
            return []
        
        try:
            chunks = text_splitter.split_text_hierarchical(content, file_path)
            
            # Add original tags
            original_tags = content_item.get("tags", [])
            for chunk in chunks:
                if hasattr(chunk, 'tags'):
                    chunk.tags = list(set(original_tags + getattr(chunk, 'tags', [])))
                else:
                    chunk.tags = original_tags
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to create chunks for content from {file_path}: {e}")
            return []
    
    async def _generate_embeddings_parallel(self, chunks: List[TextChunk]) -> int:
        """Generate embeddings for chunks in parallel"""
        if not chunks:
            return 0
        
        try:
            # Update total embeddings count
            self.progress_tracker.update_embedding_progress(0, len(chunks))
            
            # Generate embeddings in batches
            chunk_contents = [chunk.content for chunk in chunks]
            embeddings = await embedding_service.generate_embeddings_batch(chunk_contents)
            
            # Filter valid embeddings
            valid_chunks = []
            valid_embeddings = []
            
            for chunk, embedding in zip(chunks, embeddings):
                if embedding and len(embedding) == config.embedding.dimension:
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
            
            # Update progress
            self.progress_tracker.update_embedding_progress(len(valid_embeddings))
            
            # Insert into database
            if valid_chunks:
                inserted = await db_ops.insert_chunks_batch(valid_chunks, valid_embeddings)
                self.progress_tracker.update_chunk_progress(inserted)
                return inserted
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return 0
    
    async def _create_summaries_parallel(self, chunks: List[TextChunk]) -> int:
        """Create summaries for chunks in parallel"""
        if not chunks:
            return 0
        
        try:
            # Update total summaries count
            self.progress_tracker.update_summary_progress(0, len(chunks))
            
            # Generate summaries
            summaries = await summarization_service.summarize_chunks(chunks)
            
            if not summaries:
                return 0
            
            # Generate embeddings for summaries
            summary_texts = [summary.summary for summary in summaries]
            summary_embeddings = await embedding_service.generate_embeddings_batch(summary_texts)
            
            # Filter valid summaries
            valid_summaries = []
            valid_embeddings = []
            
            for summary, embedding in zip(summaries, summary_embeddings):
                if embedding and len(embedding) == config.embedding.dimension:
                    valid_summaries.append(summary)
                    valid_embeddings.append(embedding)
            
            # Update progress
            self.progress_tracker.update_summary_progress(len(valid_summaries))
            
            # Insert into database
            if valid_summaries:
                inserted = await db_ops.insert_summaries_batch(valid_summaries, valid_embeddings)
                return inserted
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to create summaries: {e}")
            return 0
    
    def _display_final_results(self):
        """Display final processing results"""
        snapshot = self.progress_tracker.get_snapshot()
        
        print("\n" + "="*80)
        print("🎉 PARALLEL BULK LOADING COMPLETED")
        print("="*80)
        print(f"📁 Files Processed:    {snapshot['processed_files']:6d} / {snapshot['total_files']:6d}")
        print(f"📝 Chunks Created:     {snapshot['created_chunks']:6d}")
        print(f"📑 Summaries Created:  {snapshot['created_summaries']:6d}")
        print(f"🧠 Embeddings Generated: {snapshot['generated_embeddings']:6d}")
        print(f"❌ Failed Files:       {snapshot['failed_files']:6d}")
        print(f"🔄 Duplicate Chunks:   {snapshot['duplicate_chunks']:6d}")
        
        success_rate = (snapshot['processed_files'] / max(snapshot['total_files'], 1)) * 100
        print(f"📊 Success Rate:       {success_rate:5.1f}%")
        print("="*80)

# Global parallel bulk loader instance
parallel_bulk_loader = ParallelBulkLoader()
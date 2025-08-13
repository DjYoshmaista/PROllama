# file_management/base_loader.py - Unified Base Loader Architecture
import asyncio
import logging
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Generator, Set, Callable, Tuple
from pathlib import Path
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
from datetime import datetime

from core.config import config
from core.memory import memory_manager

logger = logging.getLogger(__name__)

class LoadingStrategy(Enum):
    """Loading strategy options"""
    STREAMING = "streaming"      # Process files as discovered
    BATCH = "batch"              # Process in batches
    PARALLEL = "parallel"        # Full parallelization
    QUEUE = "queue"              # Queue-based processing
    LIGHTNING = "lightning"      # Ultra-fast minimal processing
    INSTANT = "instant"          # Immediate processing start
    CHUNKED = "chunked"          # Chunk-based processing

@dataclass
class LoadingProgress:
    """Unified progress tracking"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_chunks: int = 0
    created_chunks: int = 0
    total_summaries: int = 0
    created_summaries: int = 0
    total_embeddings: int = 0
    generated_embeddings: int = 0
    duplicate_chunks: int = 0
    
    # Timing metrics
    start_time: float = field(default_factory=time.time)
    discovery_time: float = 0.0
    processing_time: float = 0.0
    embedding_time: float = 0.0
    database_time: float = 0.0
    
    # Performance metrics
    files_per_second: float = 0.0
    chunks_per_second: float = 0.0
    embeddings_per_second: float = 0.0
    
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, **kwargs):
        """Thread-safe update of progress metrics"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    current = getattr(self, key)
                    if isinstance(current, (int, float)):
                        setattr(self, key, current + value)
                    else:
                        setattr(self, key, value)
            self._update_rates()
    
    def _update_rates(self):
        """Update processing rates"""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.files_per_second = self.processed_files / elapsed
            self.chunks_per_second = self.created_chunks / elapsed
            self.embeddings_per_second = self.generated_embeddings / elapsed
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get thread-safe snapshot of current progress"""
        with self._lock:
            return {
                'total_files': self.total_files,
                'processed_files': self.processed_files,
                'failed_files': self.failed_files,
                'created_chunks': self.created_chunks,
                'created_summaries': self.created_summaries,
                'generated_embeddings': self.generated_embeddings,
                'files_per_second': self.files_per_second,
                'elapsed_time': time.time() - self.start_time
            }

@dataclass
class FileProcessingJob:
    """Unified file processing job"""
    file_path: str
    file_type: str
    job_id: str
    priority: int = 1
    size_bytes: int = 0
    
    def __post_init__(self):
        if not self.job_id:
            import hashlib
            self.job_id = hashlib.md5(self.file_path.encode()).hexdigest()[:12]
        if self.size_bytes == 0:
            try:
                self.size_bytes = Path(self.file_path).stat().st_size
            except:
                self.size_bytes = 0

class BaseFileDiscovery:
    """Base file discovery functionality"""
    
    SUPPORTED_EXTENSIONS = {"py", "txt", "csv", "json", "md"}
    SKIP_DIRS = {
        '__pycache__', 'node_modules', '.git', '.venv', 'venv',
        'build', 'dist', '.pytest_cache', '.mypy_cache', 'target',
        '.tox', '__MACOSX__'
    }
    
    def __init__(self, 
                 supported_extensions: Optional[Set[str]] = None,
                 skip_dirs: Optional[Set[str]] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 min_file_size: int = 10):  # 10 bytes
        
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS
        self.skip_dirs = skip_dirs or self.SKIP_DIRS
        self.max_file_size = max_file_size
        self.min_file_size = min_file_size
    
    def discover_files(self, folder_path: str) -> Generator[str, None, None]:
        """Discover files in folder"""
        for root, dirs, files in os.walk(folder_path):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs and not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                
                if self.is_valid_file(file_path):
                    yield file_path
    
    def is_valid_file(self, file_path: str) -> bool:
        """Check if file is valid for processing"""
        try:
            # Check extension
            ext = Path(file_path).suffix[1:].lower()
            if ext not in self.supported_extensions:
                return False
            
            # Check size
            size = os.path.getsize(file_path)
            if not (self.min_file_size < size < self.max_file_size):
                return False
            
            # Check readability
            if not os.access(file_path, os.R_OK):
                return False
            
            return True
            
        except Exception:
            return False
    
    def count_files(self, folder_path: str) -> int:
        """Count valid files in folder"""
        return sum(1 for _ in self.discover_files(folder_path))

class BaseProgressDisplay:
    """Base progress display functionality"""
    
    def __init__(self, progress: LoadingProgress):
        self.progress = progress
        self._running = False
        self._display_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start progress display"""
        if self._running:
            return
        
        self._running = True
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
    
    def stop(self):
        """Stop progress display"""
        self._running = False
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=1.0)
    
    def _display_loop(self):
        """Main display loop"""
        while self._running:
            self._update_display()
            time.sleep(0.5)
    
    def _update_display(self):
        """Update progress display"""
        snapshot = self.progress.get_snapshot()
        
        # Clear previous lines
        print('\033[5A', end='')
        print('\033[J', end='')
        
        # Progress bars
        file_percent = (snapshot['processed_files'] / max(snapshot['total_files'], 1)) * 100
        print(f"📁 Files: {snapshot['processed_files']:,}/{snapshot['total_files']:,} ({file_percent:.1f}%)")
        print(f"📄 Chunks: {snapshot['created_chunks']:,}")
        print(f"🧠 Embeddings: {snapshot['generated_embeddings']:,}")
        print(f"⚡ Speed: {snapshot['files_per_second']:.1f} files/s")
        print(f"⏱️  Elapsed: {snapshot['elapsed_time']:.1f}s")

class BaseCheckpointing:
    """Base checkpointing functionality"""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = None
    
    def create_checkpoint_file(self) -> str:
        """Create new checkpoint file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
        return str(self.checkpoint_file)
    
    def save_checkpoint(self, data: Dict[str, Any]):
        """Save checkpoint data"""
        if not self.checkpoint_file:
            self.create_checkpoint_file()
        
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Checkpoint saved: {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint"""
        try:
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                return None
            
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            with open(latest, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded checkpoint: {latest}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

class BaseLoader(ABC):
    """Abstract base loader with all shared functionality"""
    
    def __init__(self,
                 strategy: LoadingStrategy = LoadingStrategy.BATCH,
                 batch_size: int = 100,
                 max_workers: int = 8,
                 enable_chunking: bool = True,
                 enable_summarization: bool = False,
                 enable_checkpointing: bool = False):
        
        # Configuration
        self.strategy = strategy
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_chunking = enable_chunking
        self.enable_summarization = enable_summarization
        self.enable_checkpointing = enable_checkpointing
        
        # Core components
        self.file_discovery = BaseFileDiscovery()
        self.progress = LoadingProgress()
        self.progress_display = BaseProgressDisplay(self.progress)
        self.checkpointing = BaseCheckpointing() if enable_checkpointing else None
        
        # Thread pools
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = None  # Created on demand
        
        # Service references (will be injected)
        self.embedding_service = None
        self.parser = None
        self.text_splitter = None
        self.summarization_service = None
        self.db_ops = None
        
        logger.info(f"Initialized {self.__class__.__name__} with strategy: {strategy.value}")
    
    async def load_folder(self,
                         folder_path: str,
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Main entry point for folder loading"""
        logger.info(f"Starting folder load: {folder_path} with strategy: {self.strategy.value}")
        
        try:
            # Initialize services
            await self._initialize_services()
            
            # Start progress display
            if self.progress_display:
                self.progress_display.start()
            
            # Load checkpoint if enabled
            if self.enable_checkpointing:
                checkpoint = self.checkpointing.load_latest_checkpoint()
                if checkpoint:
                    self._restore_from_checkpoint(checkpoint)
            
            # Execute strategy-specific loading
            result = await self._execute_strategy(folder_path, progress_callback)
            
            # Save final checkpoint
            if self.enable_checkpointing:
                self._save_checkpoint(final=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Loading failed: {e}")
            if self.enable_checkpointing:
                self._save_checkpoint(error=str(e))
            raise
            
        finally:
            # Cleanup
            await self._cleanup_services()
            if self.progress_display:
                self.progress_display.stop()
            self._display_final_results()
    
    @abstractmethod
    async def _initialize_services(self):
        """Initialize required services"""
        pass
    
    @abstractmethod
    async def _cleanup_services(self):
        """Cleanup services"""
        pass
    
    @abstractmethod
    async def _execute_strategy(self, 
                               folder_path: str,
                               progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute loading strategy"""
        pass
    
    async def _process_single_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single file through the pipeline"""
        try:
            # Parse file
            file_type = Path(file_path).suffix[1:].lower()
            parsed_content = await self._parse_file(file_path, file_type)
            if not parsed_content:
                return None
            
            # Create chunks
            chunks = []
            if self.enable_chunking:
                for content_item in parsed_content:
                    item_chunks = await self._create_chunks(content_item, file_path)
                    chunks.extend(item_chunks)
            
            if not chunks and self.enable_chunking:
                return None
            
            # Generate embeddings
            embeddings = []
            if chunks:
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = await self._generate_embeddings(chunk_texts)
            
            # Create summaries
            summaries = []
            if self.enable_summarization and chunks:
                summaries = await self._create_summaries(chunks)
            
            # Insert into database
            inserted = 0
            if chunks and embeddings:
                inserted = await self._insert_to_database(chunks, embeddings, summaries)
            
            return {
                'chunks': len(chunks),
                'embeddings': len(embeddings),
                'summaries': len(summaries),
                'inserted': inserted
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    @abstractmethod
    async def _parse_file(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Parse file content"""
        pass
    
    @abstractmethod
    async def _create_chunks(self, content: Dict[str, Any], file_path: str) -> List[Any]:
        """Create chunks from content"""
        pass
    
    @abstractmethod
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    async def _create_summaries(self, chunks: List[Any]) -> List[Any]:
        """Create summaries for chunks"""
        pass
    
    @abstractmethod
    async def _insert_to_database(self, chunks: List[Any], embeddings: List[List[float]], summaries: List[Any]) -> int:
        """Insert data into database"""
        pass
    
    def _save_checkpoint(self, final: bool = False, error: str = None):
        """Save checkpoint"""
        if not self.checkpointing:
            return
        
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'progress': self.progress.get_snapshot(),
            'final': final,
            'error': error
        }
        
        self.checkpointing.save_checkpoint(checkpoint_data)
    
    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore from checkpoint"""
        if 'progress' in checkpoint:
            for key, value in checkpoint['progress'].items():
                if hasattr(self.progress, key):
                    setattr(self.progress, key, value)
    
    def _display_final_results(self):
        """Display final results"""
        snapshot = self.progress.get_snapshot()
        
        print("\n" + "="*60)
        print("LOADING COMPLETED")
        print("="*60)
        print(f"Strategy: {self.strategy.value}")
        print(f"Files: {snapshot['processed_files']:,}/{snapshot['total_files']:,}")
        print(f"Chunks: {snapshot['created_chunks']:,}")
        print(f"Embeddings: {snapshot['generated_embeddings']:,}")
        print(f"Failed: {snapshot['failed_files']:,}")
        print(f"Time: {snapshot['elapsed_time']:.1f}s")
        print(f"Speed: {snapshot['files_per_second']:.1f} files/s")
        print("="*60)
# parallel_file_discovery.py
import os
import asyncio
import concurrent.futures
import multiprocessing
from pathlib import Path
import logging
from typing import List, Generator, Tuple
import time

logger = logging.getLogger(__name__)

def validate_file(file_path: str) -> bool:
    """Validate if a file should be processed (worker function)"""
    try:
        # Get file extension
        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in ["py", "txt", "csv", "json"]:
            return False
        
        # Skip very small files
        if os.path.getsize(file_path) < 10:
            return False
            
        # For CSV files, do a quick validation
        if ext == "csv":
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    sample = f.read(1024)
                    # Check if it looks like CSV
                    if not any(sep in sample for sep in [',', '\t', ';', '|']) or len(sample.split('\n')) < 2:
                        return False
            except Exception:
                return False
        
        return True
    except Exception:
        return False

def scan_directory_chunk(args: Tuple[str, int, int]) -> List[str]:
    """Scan a chunk of directories (worker function)"""
    root_path, start_idx, chunk_size = args
    valid_files = []
    
    try:
        # Get all items in directory
        all_items = list(os.scandir(root_path))
        
        # Process only the assigned chunk
        end_idx = min(start_idx + chunk_size, len(all_items))
        chunk_items = all_items[start_idx:end_idx]
        
        for entry in chunk_items:
            try:
                if entry.is_file():
                    if validate_file(entry.path):
                        valid_files.append(entry.path)
                elif entry.is_dir():
                    # Recursively scan subdirectories
                    for sub_file in scan_directory_recursive(entry.path):
                        if validate_file(sub_file):
                            valid_files.append(sub_file)
            except (PermissionError, OSError):
                continue
                
    except Exception as e:
        logger.debug(f"Error scanning directory chunk {root_path}[{start_idx}:{start_idx + chunk_size}]: {e}")
    
    return valid_files

def scan_directory_recursive(directory: str) -> Generator[str, None, None]:
    """Recursively scan directory for files"""
    try:
        for root, dirs, files in os.walk(directory):
            # Filter out hidden directories and common non-data directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    yield os.path.join(root, file)
    except (PermissionError, OSError):
        pass

class ParallelFileDiscovery:
    """Parallelized file discovery system"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
    async def discover_files_async(self, folder_path: str, progress_callback=None) -> List[str]:
        """Asynchronously discover files with progress reporting"""
        logger.info(f"Starting parallel file discovery in {folder_path}")
        start_time = time.time()
        
        # Get initial directory structure
        try:
            root_items = list(os.scandir(folder_path))
            total_dirs = sum(1 for item in root_items if item.is_dir())
            
            if progress_callback:
                progress_callback(0, total_dirs + 1, "Analyzing directory structure...")
            
            # Separate files and directories
            root_files = []
            subdirectories = []
            
            for item in root_items:
                try:
                    if item.is_file():
                        root_files.append(item.path)
                    elif item.is_dir():
                        subdirectories.append(item.path)
                except (PermissionError, OSError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error accessing directory {folder_path}: {e}")
            return []
        
        all_files = []
        
        # Process root files first
        if progress_callback:
            progress_callback(0, total_dirs + 1, "Processing root files...")
            
        loop = asyncio.get_event_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Validate root files in parallel
            if root_files:
                root_tasks = [
                    loop.run_in_executor(executor, validate_file, file_path)
                    for file_path in root_files
                ]
                root_results = await asyncio.gather(*root_tasks, return_exceptions=True)
                
                for file_path, is_valid in zip(root_files, root_results):
                    if is_valid is True:
                        all_files.append(file_path)
            
            # Process subdirectories in parallel chunks
            if subdirectories:
                if progress_callback:
                    progress_callback(1, total_dirs + 1, f"Processing {len(subdirectories)} subdirectories...")
                
                # Create tasks for each subdirectory
                subdir_tasks = []
                for i, subdir in enumerate(subdirectories):
                    task = loop.run_in_executor(
                        executor, 
                        self._scan_subdirectory_worker, 
                        subdir
                    )
                    subdir_tasks.append(task)
                
                # Process subdirectories with progress updates
                completed = 0
                for coro in asyncio.as_completed(subdir_tasks):
                    try:
                        subdir_files = await coro
                        all_files.extend(subdir_files)
                        completed += 1
                        
                        if progress_callback and completed % max(1, len(subdirectories) // 20) == 0:
                            progress_callback(
                                completed + 1, 
                                total_dirs + 1, 
                                f"Processed {completed}/{len(subdirectories)} directories, found {len(all_files)} files"
                            )
                    except Exception as e:
                        logger.debug(f"Subdirectory processing failed: {e}")
                        completed += 1
        
        elapsed = time.time() - start_time
        logger.info(f"File discovery completed: {len(all_files)} files found in {elapsed:.2f}s")
        
        if progress_callback:
            progress_callback(total_dirs + 1, total_dirs + 1, f"Discovery complete: {len(all_files)} files")
        
        return all_files
    
    def _scan_subdirectory_worker(self, subdir_path: str) -> List[str]:
        """Worker function to scan a subdirectory"""
        valid_files = []
        try:
            for file_path in scan_directory_recursive(subdir_path):
                if validate_file(file_path):
                    valid_files.append(file_path)
        except Exception as e:
            logger.debug(f"Error scanning subdirectory {subdir_path}: {e}")
        
        return valid_files
    
    def discover_files_streaming(self, folder_path: str) -> Generator[str, None, None]:
        """Stream files as they are discovered (for very large datasets)"""
        logger.info(f"Starting streaming file discovery in {folder_path}")
        
        # Use a queue to stream results
        import queue
        import threading
        
        file_queue = queue.Queue(maxsize=10000)  # Buffer up to 10k files
        
        def producer():
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit scanning tasks
                    futures = []
                    
                    for root, dirs, files in os.walk(folder_path):
                        # Filter directories
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
                        
                        # Process files in chunks
                        chunk_size = 100
                        for i in range(0, len(files), chunk_size):
                            chunk_files = [os.path.join(root, f) for f in files[i:i + chunk_size]]
                            future = executor.submit(self._validate_file_chunk, chunk_files)
                            futures.append(future)
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            valid_files = future.result()
                            for file_path in valid_files:
                                file_queue.put(file_path)
                        except Exception as e:
                            logger.debug(f"Chunk processing failed: {e}")
                
            except Exception as e:
                logger.error(f"Producer error: {e}")
            finally:
                file_queue.put(None)  # Signal completion
        
        # Start producer thread
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()
        
        # Yield files as they become available
        while True:
            try:
                file_path = file_queue.get(timeout=30)  # 30 second timeout
                if file_path is None:  # End signal
                    break
                yield file_path
            except queue.Empty:
                logger.warning("File discovery timed out")
                break
        
        producer_thread.join(timeout=5)
    
    def _validate_file_chunk(self, file_paths: List[str]) -> List[str]:
        """Validate a chunk of files"""
        valid_files = []
        for file_path in file_paths:
            if validate_file(file_path):
                valid_files.append(file_path)
        return valid_files

# Global instance
file_discovery = ParallelFileDiscovery()
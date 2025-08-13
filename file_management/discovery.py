# file_management/discovery.py - Optimized with Multi-Threading
import os
import logging
from typing import Generator, List, Optional, Set, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import time

# Create prefixed logger for this file
logger = logging.getLogger(__name__)
LOG_PREFIX = "[Discovery]"

class FileDiscovery:
    """Handles parallel file discovery, validation, and path management"""
    
    SUPPORTED_EXTENSIONS = {"py", "txt", "csv", "json"}
    
    def __init__(self, supported_extensions: Optional[Set[str]] = None, max_workers: int = 8):
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS
        self.max_workers = max_workers
        self._results_queue = Queue()
        self._discovery_lock = threading.Lock()
    
    def discover_files_parallel(self, folder_path: str) -> Generator[str, None, None]:
        """
        Parallel file discovery using thread pool for directory traversal
        
        Args:
            folder_path: Root folder to search
            
        Yields:
            File paths for supported files as they're discovered
        """
        logger.info(f"{LOG_PREFIX} Starting parallel file discovery in: {folder_path}")
        discovery_start = time.time()
        
        # Get all subdirectories first
        subdir_start = time.time()
        subdirs = self._get_subdirectories(folder_path)
        subdirs.append(folder_path)  # Include root directory
        subdir_time = time.time() - subdir_start
        
        logger.info(f"{LOG_PREFIX} Found {len(subdirs)} directories to scan in {subdir_time:.3f}s")
        print(f"{LOG_PREFIX} Scanning {len(subdirs)} directories with {self.max_workers} workers...")
        
        files_found = 0
        directories_processed = 0
        
        # Use ThreadPoolExecutor for parallel directory scanning
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            logger.info(f"{LOG_PREFIX} Starting parallel directory scanning with {self.max_workers} workers")
            
            # Submit directory scanning tasks
            future_to_dir = {
                executor.submit(self._scan_directory, subdir): subdir 
                for subdir in subdirs
            }
            
            logger.info(f"{LOG_PREFIX} Submitted {len(future_to_dir)} directory scanning tasks")
            
            # Yield files as they're discovered
            for future in as_completed(future_to_dir):
                directories_processed += 1
                try:
                    files = future.result()
                    dir_path = future_to_dir[future]
                    
                    logger.debug(f"{LOG_PREFIX} Directory {dir_path} yielded {len(files)} files")
                    
                    for file_path in files:
                        files_found += 1
                        yield file_path
                        
                        # Log progress every 100 files
                        if files_found % 100 == 0:
                            logger.info(f"{LOG_PREFIX} Progress: {files_found} files found, {directories_processed}/{len(subdirs)} dirs processed")
                            
                except Exception as e:
                    dir_path = future_to_dir[future]
                    logger.error(f"{LOG_PREFIX} Error scanning directory {dir_path}: {e}")
        
        discovery_time = time.time() - discovery_start
        logger.info(f"{LOG_PREFIX} Parallel file discovery completed: {files_found} files in {discovery_time:.2f}s")
        print(f"{LOG_PREFIX} Discovery complete: {files_found} files found in {discovery_time:.2f}s")
    
    def _get_subdirectories(self, folder_path: str) -> List[str]:
        """Get all subdirectories for parallel processing"""
        subdirs = []
        try:
            for root, dirs, _ in os.walk(folder_path):
                for d in dirs:
                    subdirs.append(os.path.join(root, d))
        except Exception as e:
            logger.error(f"Error getting subdirectories from {folder_path}: {e}")
        return subdirs
    
    def _scan_directory(self, directory: str) -> List[str]:
        """Scan a single directory for supported files (thread-safe)"""
        files = []
        try:
            for item in os.listdir(directory):
                file_path = os.path.join(directory, item)
                if os.path.isfile(file_path) and self.is_supported_file(file_path):
                    files.append(file_path)
        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot access directory {directory}: {e}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
        
        return files
    
    def discover_files(self, folder_path: str) -> Generator[str, None, None]:
        """
        Smart file discovery that chooses between parallel and sequential based on folder size
        
        Args:
            folder_path: Root folder to search
            
        Yields:
            File paths for supported files
        """
        # Quick check of folder size to determine strategy
        dir_count = self._estimate_directory_count(folder_path)
        
        if dir_count > 10:  # Use parallel for larger folder structures
            logger.info(f"Using parallel discovery for {dir_count} directories")
            yield from self.discover_files_parallel(folder_path)
        else:
            logger.info(f"Using sequential discovery for {dir_count} directories")
            yield from self._discover_files_sequential(folder_path)
    
    def _discover_files_sequential(self, folder_path: str) -> Generator[str, None, None]:
        """Original sequential discovery (kept for small folders)"""
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.is_supported_file(file_path):
                        yield file_path
        except Exception as e:
            logger.error(f"Error discovering files in {folder_path}: {e}")
    
    def _estimate_directory_count(self, folder_path: str) -> int:
        """Quickly estimate number of directories for strategy selection"""
        count = 0
        try:
            for root, dirs, _ in os.walk(folder_path):
                count += len(dirs)
                if count > 20:  # Early exit for large structures
                    break
        except Exception:
            pass
        return count
    
    def count_files_parallel(self, folder_path: str) -> int:
        """Count supported files using parallel processing"""
        count = 0
        for _ in self.discover_files_parallel(folder_path):
            count += 1
        return count
    
    def count_files(self, folder_path: str) -> int:
        """Count supported files in a folder (auto-selects strategy)"""
        logger.info(f"{LOG_PREFIX} Starting file count for folder: {folder_path}")
        count_start = time.time()
        
        count = 0
        try:
            for _ in self.discover_files(folder_path):
                count += 1
                
                # Log progress for large counts
                if count % 500 == 0:
                    logger.info(f"{LOG_PREFIX} File counting progress: {count} files so far")
                    
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Error counting files in {folder_path}: {e}")
            
        count_time = time.time() - count_start
        logger.info(f"{LOG_PREFIX} File count completed: {count} files in {count_time:.2f}s")
        print(f"{LOG_PREFIX} Found {count} files in {count_time:.2f}s")
        
        return count
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file has supported extension and is accessible (thread-safe)"""
        try:
            # Check extension
            ext = Path(file_path).suffix[1:].lower()
            if ext not in self.supported_extensions:
                return False
            
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                return False
            
            if not os.access(file_path, os.R_OK):
                return False
            
            # Skip empty files
            if os.path.getsize(file_path) == 0:
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking file {file_path}: {e}")
            return False
    
    def validate_folder_path(self, folder_path: str) -> bool:
        """Validate that folder path exists and is accessible"""
        try:
            # Expand user path
            folder_path = os.path.expanduser(folder_path)
            
            # Check if path exists
            if not os.path.exists(folder_path):
                logger.error(f"Path does not exist: {folder_path}")
                return False
            
            # Check if it's a directory
            if not os.path.isdir(folder_path):
                logger.error(f"Path is not a directory: {folder_path}")
                return False
            
            # Check read permissions
            if not os.access(folder_path, os.R_OK):
                logger.error(f"No read permission for directory: {folder_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating folder path {folder_path}: {e}")
            return False
    
    def get_files_with_info_parallel(self, folder_path: str) -> List[Tuple[str, dict]]:
        """
        Get files with their info using parallel processing
        Returns list of (file_path, file_info) tuples
        """
        files_with_info = []
        
        # Discover files in parallel
        files = list(self.discover_files_parallel(folder_path))
        
        # Get file info in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.get_file_info, file_path): file_path 
                for file_path in files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_info = future.result()
                    files_with_info.append((file_path, file_info))
                except Exception as e:
                    logger.error(f"Error getting info for {file_path}: {e}")
        
        return files_with_info
    
    def get_file_info(self, file_path: str) -> dict:
        """Get detailed information about a file (thread-safe)"""
        try:
            stat = os.stat(file_path)
            return {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 ** 2),
                'extension': Path(file_path).suffix[1:].lower(),
                'modified_time': stat.st_mtime,
                'is_readable': os.access(file_path, os.R_OK)
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {'path': file_path, 'error': str(e)}
    
    def preview_folder_contents(self, folder_path: str, max_files: int = 10) -> List[dict]:
        """Get a preview of files in a folder using parallel processing"""
        preview = []
        
        try:
            files_with_info = self.get_files_with_info_parallel(folder_path)
            
            for i, (file_path, info) in enumerate(files_with_info):
                if i >= max_files:
                    break
                
                rel_path = os.path.relpath(file_path, folder_path)
                info['relative_path'] = rel_path
                preview.append(info)
                
        except Exception as e:
            logger.error(f"Error previewing folder {folder_path}: {e}")
        
        return preview
    
    def get_folder_stats_parallel(self, folder_path: str) -> dict:
        """Get statistics about files in a folder using fully parallel processing with concurrent futures"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'by_extension': {},
            'largest_file': None,
            'smallest_file': None
        }
        
        try:
            # Use parallel file discovery and statistics gathering
            file_paths = list(self.discover_files_parallel(folder_path))
            
            if not file_paths:
                return stats
            
            # Process file info collection in parallel batches
            files_with_info = []
            batch_size = max(100, len(file_paths) // (self.max_workers * 2))
            
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_file = {
                        executor.submit(self.get_file_info, file_path): file_path 
                        for file_path in batch
                    }
                    
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            file_info = future.result()
                            files_with_info.append((file_path, file_info))
                        except Exception as e:
                            logger.error(f"Error getting info for {file_path}: {e}")
            
            # Aggregate statistics in parallel
            largest_size = 0
            smallest_size = float('inf')
            
            # Process aggregation using parallel reduction pattern
            def process_file_batch(batch_info):
                batch_stats = {
                    'count': 0,
                    'size_mb': 0,
                    'extensions': {},
                    'largest': (0, None),
                    'smallest': (float('inf'), None)
                }
                
                for file_path, info in batch_info:
                    if 'error' in info:
                        continue
                        
                    batch_stats['count'] += 1
                    batch_stats['size_mb'] += info.get('size_mb', 0)
                    
                    ext = info.get('extension', 'unknown')
                    batch_stats['extensions'][ext] = batch_stats['extensions'].get(ext, 0) + 1
                    
                    size = info.get('size_bytes', 0)
                    if size > batch_stats['largest'][0]:
                        batch_stats['largest'] = (size, info)
                    
                    if size < batch_stats['smallest'][0]:
                        batch_stats['smallest'] = (size, info)
                
                return batch_stats
            
            # Process in parallel batches
            batch_results = []
            info_batch_size = max(50, len(files_with_info) // self.max_workers)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_futures = []
                for i in range(0, len(files_with_info), info_batch_size):
                    batch = files_with_info[i:i + info_batch_size]
                    if batch:
                        future = executor.submit(process_file_batch, batch)
                        batch_futures.append(future)
                
                for future in as_completed(batch_futures):
                    try:
                        batch_result = future.result()
                        batch_results.append(batch_result)
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
            
            # Merge results
            for batch_result in batch_results:
                stats['total_files'] += batch_result['count']
                stats['total_size_mb'] += batch_result['size_mb']
                
                for ext, count in batch_result['extensions'].items():
                    stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + count
                
                if batch_result['largest'][0] > largest_size:
                    largest_size = batch_result['largest'][0]
                    stats['largest_file'] = batch_result['largest'][1]
                
                if batch_result['smallest'][0] < smallest_size:
                    smallest_size = batch_result['smallest'][0]
                    stats['smallest_file'] = batch_result['smallest'][1]
                    
        except Exception as e:
            logger.error(f"Error getting folder stats for {folder_path}: {e}")
            stats['error'] = str(e)
        
        return stats

class InteractiveFileSelector:
    """Interactive file/folder selection with GUI and CLI fallbacks"""
    
    def __init__(self, discovery: Optional[FileDiscovery] = None):
        self.discovery = discovery or FileDiscovery()
    
    def get_folder_path(self) -> Optional[str]:
        """Alias for browse_for_folder for backward compatibility"""
        return self.browse_for_folder()
        
    def browse_for_folder(self) -> Optional[str]:
        """Browse for folder with GUI fallback to CLI"""
        # Try GUI first
        folder_path = self._gui_folder_browser()
        
        if folder_path:
            return folder_path
        
        # Fallback to manual input
        return self._manual_folder_input()
    
    def _gui_folder_browser(self) -> Optional[str]:
        """Try to use GUI file browser"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            folder_path = filedialog.askdirectory(title="Select Folder to Load Files")
            root.destroy()
            
            if folder_path:
                logger.info(f"Folder selected via GUI: {folder_path}")
                return folder_path
            
        except Exception as e:
            logger.warning(f"GUI file browser not available: {e}")
        
        return None
    
    def _manual_folder_input(self) -> Optional[str]:
        """Get folder path via manual input with validation"""
        print("\n" + "="*60)
        print("GUI file browser is not available")
        print("Please enter the folder path manually")
        print("="*60)
        
        current_dir = os.getcwd()
        print(f"\nCurrent directory: {current_dir}")
        
        while True:
            print("\nOptions:")
            print("1. Enter absolute path (e.g., /home/user/documents)")
            print("2. Enter relative path from current directory")
            print("3. Use current directory")
            print("4. Cancel")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                folder_path = input("Enter absolute folder path: ").strip()
                folder_path = os.path.expanduser(folder_path)
                
            elif choice == "2":
                relative_path = input("Enter relative path from current directory: ").strip()
                folder_path = os.path.join(current_dir, relative_path)
                folder_path = os.path.abspath(folder_path)
                
            elif choice == "3":
                folder_path = current_dir
                print(f"Using current directory: {folder_path}")
                
            elif choice == "4":
                logger.info("Folder selection cancelled by user")
                return None
                
            else:
                print("Invalid option. Please try again.")
                continue
            
            # Validate the folder path
            if self.discovery.validate_folder_path(folder_path):
                # Show preview and confirm using parallel processing
                if self._confirm_folder_selection_parallel(folder_path):
                    return folder_path
            else:
                print(f"\nError: Invalid folder path: {folder_path}")
                retry = input("Would you like to try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
    
    def _confirm_folder_selection_parallel(self, folder_path: str) -> bool:
        """Show folder preview and confirm selection using parallel processing"""
        print("\nAnalyzing folder (using parallel processing)...")
        start_time = time.time()
        
        # Get folder statistics using parallel processing
        stats = self.discovery.get_folder_stats_parallel(folder_path)
        analysis_time = time.time() - start_time
        
        print(f"\nFolder Analysis (completed in {analysis_time:.2f}s):")
        print(f"Total supported files: {stats['total_files']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        
        if stats['by_extension']:
            print("Files by type:")
            for ext, count in stats['by_extension'].items():
                print(f"  .{ext}: {count} files")
        
        # Show preview of files using parallel processing
        preview = self.discovery.preview_folder_contents(folder_path, 10)
        if preview:
            print("\nPreview of files (first 10):")
            for info in preview:
                size_mb = info.get('size_mb', 0)
                rel_path = info.get('relative_path', info.get('name', 'unknown'))
                print(f"  {rel_path} ({size_mb:.2f} MB)")
            
            if stats['total_files'] > 10:
                print(f"  ... and {stats['total_files'] - 10} more files")
        
        # Confirmation
        confirm = input(f"\nProcess {stats['total_files']} file(s) from this directory? (y/n): ").strip().lower()
        return confirm == 'y'

# Global instances
file_discovery = FileDiscovery()
file_selector = InteractiveFileSelector(file_discovery)
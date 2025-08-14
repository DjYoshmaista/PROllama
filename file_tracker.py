# file_tracker.py
import os
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, List, Generator
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FileRecord:
    """Record of a processed file"""
    filepath: str
    size: int
    mtime: float  # modification time
    checksum: str
    processed_at: str
    records_count: int = 0

@dataclass
class FilterStats:
    """Detailed statistics from file filtering operation"""
    total_files: int = 0
    files_to_process: int = 0
    files_skipped: int = 0
    new_files: int = 0
    size_changed: int = 0
    time_changed: int = 0
    content_changed: int = 0
    already_processed: int = 0
    error_files: int = 0
    skip_reasons: Dict[str, str] = None
    
    def __post_init__(self):
        if self.skip_reasons is None:
            self.skip_reasons = {}

class FileTracker:
    """Tracks processed files to avoid reprocessing unchanged files"""
    
    def __init__(self, tracker_file: str = "processed_files.json"):
        self.tracker_file = tracker_file
        self.processed_files: Dict[str, FileRecord] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.load_tracker()
        self._loaded = False
        
    def load_tracker(self):
        """Load existing file tracking data with error recovery"""
        with self._lock:
            # Try to load the main tracker file
            if self._try_load_tracker_file(self.tracker_file):
                return
                
            # Try to load from backup if main file failed
            backup_file = f"{self.tracker_file}.backup"
            if os.path.exists(backup_file):
                logger.warning(f"Main tracker file corrupted, attempting backup recovery")
                if self._try_load_tracker_file(backup_file):
                    # Restore backup as main file
                    try:
                        os.replace(backup_file, self.tracker_file)
                        logger.info("Successfully recovered from backup")
                        return
                    except Exception as e:
                        logger.error(f"Failed to restore backup: {e}")
            
            # If all else fails, start fresh
            logger.info("No valid tracker found, starting fresh")
            self.processed_files = {}
    
    def _try_load_tracker_file(self, filepath: str) -> bool:
        """Try to load a specific tracker file"""
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.processed_files = {
                    path: FileRecord(**record_data) 
                    for path, record_data in data.items()
                }
            logger.info(f"Loaded {len(self.processed_files)} processed file records from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading file tracker from {filepath}: {e}")
            return False
    
    def save_tracker(self):
        """Save file tracking data with atomic operations"""
        with self._lock:
            try:
                # Convert to serializable format
                data = {
                    path: asdict(record) 
                    for path, record in self.processed_files.items()
                }
                
                # Write to temporary file first
                temp_file = f"{self.tracker_file}.tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Create backup of existing tracker
                if os.path.exists(self.tracker_file):
                    backup_file = f"{self.tracker_file}.backup"
                    try:
                        os.replace(self.tracker_file, backup_file)
                    except Exception as e:
                        logger.warning(f"Could not create backup: {e}")
                
                # Atomically replace with new file
                os.replace(temp_file, self.tracker_file)
                
                logger.info(f"Saved {len(self.processed_files)} file records to tracker")
            except Exception as e:
                logger.error(f"Error saving file tracker: {e}")
                # Clean up temp file if it exists
                temp_file = f"{self.tracker_file}.tmp"
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def _calculate_file_checksum(self, filepath: str, chunk_size: int = 8192) -> str:
        """Calculate MD5 checksum of file for change detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate checksum for {filepath}: {e}")
            return ""
    
    def should_process_file(self, filepath: str) -> Tuple[bool, str]:
        """
        Check if file should be processed
        
        Returns:
            Tuple[bool, str]: (should_process, reason_code)
        """
        try:
            # Normalize path
            abs_path = os.path.abspath(filepath)
            
            # Check if file exists
            if not os.path.exists(abs_path):
                return False, "file_not_found"
            
            # Get current file stats
            stat = os.stat(abs_path)
            current_size = stat.st_size
            current_mtime = stat.st_mtime
            
            # Check if we've seen this file before
            with self._lock:
                if abs_path not in self.processed_files:
                    return True, "new_file"
                
                previous_record = self.processed_files[abs_path]
            
            # Check if size changed
            if current_size != previous_record.size:
                return True, "size_changed"
            
            # Check if modification time changed
            if abs(current_mtime - previous_record.mtime) > 1.0:  # 1 second tolerance
                return True, "time_changed"
            
            # For extra safety, check checksum for files that might have changed
            if current_size > 0:  # Only for non-empty files
                current_checksum = self._calculate_file_checksum(abs_path)
                if current_checksum and current_checksum != previous_record.checksum:
                    return True, "content_changed"
            
            return False, "already_processed"
            
        except Exception as e:
            logger.error(f"Error checking file {filepath}: {e}")
            return True, "error_checking"
    
    def mark_file_processed(self, filepath: str, records_count: int = 0):
        """Mark file as processed"""
        try:
            abs_path = os.path.abspath(filepath)
            
            if not os.path.exists(abs_path):
                logger.warning(f"Cannot mark non-existent file as processed: {abs_path}")
                return
            
            stat = os.stat(abs_path)
            checksum = self._calculate_file_checksum(abs_path)
            
            record = FileRecord(
                filepath=abs_path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                checksum=checksum,
                processed_at=datetime.now().isoformat(),
                records_count=records_count
            )
            
            with self._lock:
                self.processed_files[abs_path] = record
            
            logger.debug(f"Marked file as processed: {abs_path} ({records_count} records)")
            
        except Exception as e:
            logger.error(f"Error marking file as processed {filepath}: {e}")
    
    def get_processed_files_stats(self) -> Dict:
        """Get statistics about processed files"""
        with self._lock:
            total_files = len(self.processed_files)
            total_records = sum(record.records_count for record in self.processed_files.values())
            total_size = sum(record.size for record in self.processed_files.values())
            
            return {
                "total_files": total_files,
                "total_records": total_records,
                "total_size_mb": total_size / (1024 * 1024),
                "avg_records_per_file": total_records / total_files if total_files > 0 else 0
            }
    
    def filter_files_to_process(self, file_paths: List[str]) -> Tuple[List[str], List[str], FilterStats]:
        """
        Filter list of files to only those that need processing with detailed stats
        
        Returns:
            Tuple[List[str], List[str], FilterStats]: (files_to_process, skipped_files, detailed_stats)
        """
        files_to_process = []
        skipped_files = []
        stats = FilterStats(total_files=len(file_paths))
        
        for filepath in file_paths:
            should_process, reason = self.should_process_file(filepath)
            if should_process:
                files_to_process.append(filepath)
                # Count specific reason types
                if reason == "new_file":
                    stats.new_files += 1
                elif reason == "size_changed":
                    stats.size_changed += 1
                elif reason == "time_changed":
                    stats.time_changed += 1
                elif reason == "content_changed":
                    stats.content_changed += 1
                elif reason == "error_checking":
                    stats.error_files += 1
            else:
                skipped_files.append(filepath)
                if reason == "already_processed":
                    stats.already_processed += 1
                elif reason == "file_not_found":
                    stats.error_files += 1
                stats.skip_reasons[filepath] = reason
        
        stats.files_to_process = len(files_to_process)
        stats.files_skipped = len(skipped_files)
        
        return files_to_process, skipped_files, stats
    
    def batch_filter_files(self, file_paths: List[str], batch_size: int = 1000) -> Tuple[List[str], FilterStats]:
        """
        Filter files and return both the list and detailed stats (enhanced version)
        
        Args:
            file_paths: List of file paths to filter OR a generator
            batch_size: Size of batches for processing (used for progress tracking)
            
        Returns:
            Tuple[List[str], FilterStats]: (files_to_process, detailed_stats)
        """
        # Handle both list and generator inputs
        if hasattr(file_paths, '__iter__') and not isinstance(file_paths, (list, tuple)):
            # Convert generator to list for processing
            file_paths = list(file_paths)
        
        # Process all files at once but with progress logging
        total_files = len(file_paths)
        logger.info(f"Analyzing {total_files} files for processing needs...")
        
        files_to_process, skipped_files, stats = self.filter_files_to_process(file_paths)
        
        # Log progress information
        if total_files > batch_size:
            logger.info(f"Batch analysis complete: {len(files_to_process)} files need processing")
        
        return files_to_process, stats
    
    def batch_filter_files_generator(self, file_generator: Generator[str, None, None], 
                                   batch_size: int = 100) -> Generator[Tuple[List[str], FilterStats], None, None]:
        """
        Filter files in batches from a generator to only those that need processing
        
        Args:
            file_generator: Generator yielding file paths
            batch_size: Number of files to process in each batch
            
        Yields:
            Tuple[List[str], FilterStats]: (files_to_process_in_batch, batch_stats)
        """
        try:
            batch = []
            batch_num = 1
            
            for filepath in file_generator:
                batch.append(filepath)
                
                if len(batch) >= batch_size:
                    # Process the current batch
                    files_to_process, skipped_files, stats = self.filter_files_to_process(batch)
                    
                    # Add batch information to stats
                    stats.batch_number = batch_num
                    stats.batch_size = len(batch)
                    
                    if files_to_process:  # Only yield if there are files to process
                        yield files_to_process, stats
                    
                    # Log batch statistics
                    logger.info(f"Batch {batch_num}: {len(files_to_process)} to process, {len(skipped_files)} skipped")
                    
                    # Reset for next batch
                    batch = []
                    batch_num += 1
            
            # Process remaining files in the last partial batch
            if batch:
                files_to_process, skipped_files, stats = self.filter_files_to_process(batch)
                stats.batch_number = batch_num
                stats.batch_size = len(batch)
                
                if files_to_process:  # Only yield if there are files to process
                    yield files_to_process, stats
                
                logger.info(f"Final batch {batch_num}: {len(files_to_process)} to process, {len(skipped_files)} skipped")
                
        except Exception as e:
            logger.error(f"Error in batch_filter_files_generator: {e}")
            raise
    
    def ensure_loaded(self):
        """Ensure tracker data is loaded (async-safe)"""
        if not self._loaded:
            with self._lock:
                if not self._loaded:  # Double-check pattern
                    self.load_tracker()
                    self._loaded = True

    def cleanup_missing_files(self):
        """Remove records for files that no longer exist"""
        with self._lock:
            missing_files = []
            for filepath in list(self.processed_files.keys()):
                if not os.path.exists(filepath):
                    missing_files.append(filepath)
                    del self.processed_files[filepath]
            
            if missing_files:
                logger.info(f"Cleaned up {len(missing_files)} missing file records")
                self.save_tracker()
            
            return missing_files
    
    def get_file_info(self, filepath: str) -> Optional[FileRecord]:
        """Get information about a specific file"""
        abs_path = os.path.abspath(filepath)
        with self._lock:
            return self.processed_files.get(abs_path)
    
    def force_reprocess_all(self):
        """Clear all tracking data to force reprocessing of all files"""
        with self._lock:
            backup_count = len(self.processed_files)
            self.processed_files.clear()
            logger.info(f"Cleared {backup_count} file records - all files will be reprocessed")
    
    def get_detailed_stats(self) -> Dict:
        """Get detailed statistics about the tracker state"""
        with self._lock:
            stats = self.get_processed_files_stats()
            
            # Add more detailed information
            if self.processed_files:
                record_counts = [record.records_count for record in self.processed_files.values()]
                stats.update({
                    "min_records_per_file": min(record_counts),
                    "max_records_per_file": max(record_counts),
                    "files_with_no_records": sum(1 for count in record_counts if count == 0),
                    "most_recent_processing": max(record.processed_at for record in self.processed_files.values()),
                    "oldest_processing": min(record.processed_at for record in self.processed_files.values())
                })
            
            return stats

# Global file tracker instance
file_tracker = FileTracker()
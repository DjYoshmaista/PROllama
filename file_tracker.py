# file_tracker.py
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
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

class FileTracker:
    """Tracks processed files to avoid reprocessing unchanged files"""
    
    def __init__(self, tracker_file: str = "processed_files.json"):
        self.tracker_file = tracker_file
        self.processed_files: Dict[str, FileRecord] = {}
        self.load_tracker()
        
    def load_tracker(self):
        """Load existing file tracking data"""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_files = {
                        path: FileRecord(**record_data) 
                        for path, record_data in data.items()
                    }
                logger.info(f"Loaded {len(self.processed_files)} processed file records")
            except Exception as e:
                logger.error(f"Error loading file tracker: {e}")
                self.processed_files = {}
        else:
            logger.info("No existing file tracker found, starting fresh")
    
    def save_tracker(self):
        """Save file tracking data"""
        try:
            # Convert to serializable format
            data = {
                path: asdict(record) 
                for path, record in self.processed_files.items()
            }
            
            # Create backup of existing tracker
            if os.path.exists(self.tracker_file):
                backup_file = f"{self.tracker_file}.backup"
                os.rename(self.tracker_file, backup_file)
            
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.processed_files)} file records to tracker")
        except Exception as e:
            logger.error(f"Error saving file tracker: {e}")
    
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
            Tuple[bool, str]: (should_process, reason)
        """
        try:
            # Normalize path
            abs_path = os.path.abspath(filepath)
            
            # Check if file exists
            if not os.path.exists(abs_path):
                return False, "File does not exist"
            
            # Get current file stats
            stat = os.stat(abs_path)
            current_size = stat.st_size
            current_mtime = stat.st_mtime
            
            # Check if we've seen this file before
            if abs_path not in self.processed_files:
                return True, "New file"
            
            previous_record = self.processed_files[abs_path]
            
            # Check if size changed
            if current_size != previous_record.size:
                return True, f"Size changed: {previous_record.size} -> {current_size}"
            
            # Check if modification time changed
            if abs(current_mtime - previous_record.mtime) > 1.0:  # 1 second tolerance
                return True, f"Modified: {datetime.fromtimestamp(previous_record.mtime)} -> {datetime.fromtimestamp(current_mtime)}"
            
            # For extra safety, check checksum for files that might have changed
            if current_size > 0:  # Only for non-empty files
                current_checksum = self._calculate_file_checksum(abs_path)
                if current_checksum and current_checksum != previous_record.checksum:
                    return True, f"Content changed (checksum mismatch)"
            
            return False, f"Already processed ({previous_record.processed_at})"
            
        except Exception as e:
            logger.error(f"Error checking file {filepath}: {e}")
            return True, f"Error checking file, will process: {e}"
    
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
            
            self.processed_files[abs_path] = record
            logger.debug(f"Marked file as processed: {abs_path} ({records_count} records)")
            
        except Exception as e:
            logger.error(f"Error marking file as processed {filepath}: {e}")
    
    def get_processed_files_stats(self) -> Dict:
        """Get statistics about processed files"""
        total_files = len(self.processed_files)
        total_records = sum(record.records_count for record in self.processed_files.values())
        total_size = sum(record.size for record in self.processed_files.values())
        
        return {
            "total_files": total_files,
            "total_records": total_records,
            "total_size_mb": total_size / (1024 * 1024),
            "avg_records_per_file": total_records / total_files if total_files > 0 else 0
        }
    
    def filter_files_to_process(self, file_paths: list) -> Tuple[list, list, Dict]:
        """
        Filter list of files to only those that need processing
        
        Returns:
            Tuple[list, list, Dict]: (files_to_process, skipped_files, stats)
        """
        files_to_process = []
        skipped_files = []
        skip_reasons = {}
        
        for filepath in file_paths:
            should_process, reason = self.should_process_file(filepath)
            if should_process:
                files_to_process.append(filepath)
            else:
                skipped_files.append(filepath)
                skip_reasons[filepath] = reason
        
        stats = {
            "total_files": len(file_paths),
            "files_to_process": len(files_to_process),
            "files_skipped": len(skipped_files),
            "skip_reasons": skip_reasons
        }
        
        return files_to_process, skipped_files, stats
    
    def cleanup_missing_files(self):
        """Remove records for files that no longer exist"""
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
        return self.processed_files.get(abs_path)

# Global file tracker instance
file_tracker = FileTracker()
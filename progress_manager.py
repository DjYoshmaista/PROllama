# progress_manager.py
import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
import os
import sys
from datetime import datetime

@dataclass
class ProgressStats:
    """Statistics for progress tracking"""
    files_processed: int = 0
    data_pieces_created: int = 0
    embeddings_created: int = 0
    embeddings_queued: int = 0
    files_total: int = 0
    data_pieces_total: int = 0
    current_file: str = ""
    start_time: float = field(default_factory=time.time)
    
class MultiProgressManager:
    """Manages multiple asynchronous progress bars in terminal"""
    
    def __init__(self):
        self.stats = ProgressStats()
        self.running = False
        self.display_thread = None
        self.lock = threading.Lock()
        self.terminal_height = self._get_terminal_height()
        
    def _get_terminal_height(self):
        """Get terminal height for proper display"""
        try:
            return os.get_terminal_size().lines
        except OSError:
            return 24  # Default fallback
    
    def start(self, total_files: int, estimated_data_pieces: int = None):
        """Start the progress display"""
        with self.lock:
            self.stats.files_total = total_files
            self.stats.data_pieces_total = estimated_data_pieces or (total_files * 10)  # Estimate
            self.stats.start_time = time.time()
            self.running = True
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def stop(self):
        """Stop the progress display"""
        with self.lock:
            self.running = False
        
        if self.display_thread:
            self.display_thread.join(timeout=1.0)
        
        # Clear progress bars and show final stats
        self._clear_progress_area()
        self._show_final_stats()
    
    def update_files(self, processed: int, current_file: str = ""):
        """Update file processing progress"""
        with self.lock:
            self.stats.files_processed = processed
            if current_file:
                self.stats.current_file = current_file
    
    def update_data_pieces(self, created: int, total: int = None):
        """Update data pieces progress"""
        with self.lock:
            self.stats.data_pieces_created = created
            if total is not None:
                self.stats.data_pieces_total = total
    
    def update_embeddings(self, created: int, queued: int = 0):
        """Update embeddings progress"""
        with self.lock:
            self.stats.embeddings_created = created
            self.stats.embeddings_queued = queued
    
    def increment_files(self, current_file: str = ""):
        """Increment file counter"""
        with self.lock:
            self.stats.files_processed += 1
            if current_file:
                self.stats.current_file = current_file
    
    def increment_data_pieces(self, count: int = 1):
        """Increment data pieces counter"""
        with self.lock:
            self.stats.data_pieces_created += count
    
    def increment_embeddings(self, count: int = 1):
        """Increment embeddings counter"""
        with self.lock:
            self.stats.embeddings_created += count
    
    def _display_loop(self):
        """Main display loop running in separate thread"""
        # Move cursor to bottom and reserve space for progress bars
        print("\n" * 6, end="", flush=True)
        
        while self.running:
            self._render_progress_bars()
            time.sleep(0.1)  # Update 10 times per second
    
    def _render_progress_bars(self):
        """Render all progress bars"""
        with self.lock:
            stats = ProgressStats(
                files_processed=self.stats.files_processed,
                data_pieces_created=self.stats.data_pieces_created,
                embeddings_created=self.stats.embeddings_created,
                embeddings_queued=self.stats.embeddings_queued,
                files_total=self.stats.files_total,
                data_pieces_total=self.stats.data_pieces_total,
                current_file=self.stats.current_file,
                start_time=self.stats.start_time
            )
        
        # Calculate elapsed time and rates
        elapsed = time.time() - stats.start_time
        files_rate = stats.files_processed / elapsed if elapsed > 0 else 0
        data_rate = stats.data_pieces_created / elapsed if elapsed > 0 else 0
        embed_rate = stats.embeddings_created / elapsed if elapsed > 0 else 0
        
        # Move cursor up to overwrite previous progress bars
        print("\033[6A", end="")
        
        # Progress bar 1: Files processed
        files_bar = self._create_progress_bar(
            stats.files_processed, 
            stats.files_total,
            f"Files Processed ({files_rate:.1f}/s)",
            50
        )
        print(f"\033[K{files_bar}")
        
        # Progress bar 2: Data pieces created
        data_bar = self._create_progress_bar(
            stats.data_pieces_created,
            stats.data_pieces_total,
            f"Data Pieces ({data_rate:.1f}/s)",
            50
        )
        print(f"\033[K{data_bar}")
        
        # Progress bar 3: Embeddings created + queue info
        embed_bar = self._create_progress_bar(
            stats.embeddings_created,
            stats.data_pieces_created if stats.data_pieces_created > 0 else stats.data_pieces_total,
            f"Embeddings ({embed_rate:.1f}/s) [Queue: {stats.embeddings_queued}]",
            50
        )
        print(f"\033[K{embed_bar}")
        
        # Progress bar 4: Overall progress (based on files)
        overall_bar = self._create_progress_bar(
            stats.files_processed,
            stats.files_total,
            f"Overall Progress",
            50
        )
        print(f"\033[K{overall_bar}")
        
        # Current file info
        current_file_display = stats.current_file
        if len(current_file_display) > 60:
            current_file_display = "..." + current_file_display[-57:]
        print(f"\033[KCurrent: {current_file_display}")
        
        # Status line with timing info
        status = f"Elapsed: {elapsed:.1f}s | Est. Remaining: {self._estimate_remaining(stats, elapsed)}"
        print(f"\033[K{status}")
        
        sys.stdout.flush()
    
    def _create_progress_bar(self, current: int, total: int, label: str, width: int = 50) -> str:
        """Create a single progress bar string"""
        if total <= 0:
            percentage = 0
            filled_length = 0
        else:
            percentage = min(100, (current / total) * 100)
            filled_length = int(width * current // total) if total > 0 else 0
        
        bar = "█" * filled_length + "░" * (width - filled_length)
        return f"{label:<25} |{bar}| {current:>6}/{total:<6} ({percentage:5.1f}%)"
    
    def _estimate_remaining(self, stats: ProgressStats, elapsed: float) -> str:
        """Estimate remaining time"""
        if stats.files_processed == 0 or elapsed == 0:
            return "Unknown"
        
        files_rate = stats.files_processed / elapsed
        remaining_files = stats.files_total - stats.files_processed
        
        if files_rate > 0:
            remaining_seconds = remaining_files / files_rate
            if remaining_seconds < 60:
                return f"{remaining_seconds:.1f}s"
            elif remaining_seconds < 3600:
                return f"{remaining_seconds/60:.1f}m"
            else:
                return f"{remaining_seconds/3600:.1f}h"
        return "Unknown"
    
    def _clear_progress_area(self):
        """Clear the progress display area"""
        print("\033[6A", end="")  # Move up 6 lines
        for _ in range(6):
            print("\033[K")  # Clear each line
    
    def _show_final_stats(self):
        """Show final statistics"""
        elapsed = time.time() - self.stats.start_time
        print(f"\n📊 Processing Complete!")
        print(f"   Files Processed: {self.stats.files_processed}/{self.stats.files_total}")
        print(f"   Data Pieces: {self.stats.data_pieces_created}")
        print(f"   Embeddings: {self.stats.embeddings_created}")
        print(f"   Total Time: {elapsed:.2f}s")
        print(f"   Average Rate: {self.stats.files_processed/elapsed:.2f} files/s")

class SimpleProgressTracker:
    """Simple progress tracker for single file operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        
    def update(self, current: int, total: int, detail: str = ""):
        """Update progress for single operation"""
        elapsed = time.time() - self.start_time
        percentage = (current / total) * 100 if total > 0 else 0
        rate = current / elapsed if elapsed > 0 else 0
        
        bar_width = 40
        filled_length = int(bar_width * current // total) if total > 0 else 0
        bar = "█" * filled_length + "░" * (bar_width - filled_length)
        
        print(f"\r{self.operation_name}: |{bar}| {current}/{total} ({percentage:.1f}%) [{rate:.1f}/s] {detail}", 
              end="", flush=True)
    
    def complete(self, total_processed: int):
        """Mark operation as complete"""
        elapsed = time.time() - self.start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"\n✅ {self.operation_name} complete: {total_processed} items in {elapsed:.2f}s ({rate:.1f}/s)")

# Global progress manager instance
progress_manager = MultiProgressManager()
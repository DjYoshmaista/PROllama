# progress_manager.py
import threading
import time
import sys
import os
from dataclasses import dataclass, field

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
        self.lock = threading.RLock()
        self.last_display = ""  # Store last display to avoid unnecessary redraws
        self.progress_lines = 6  # Number of lines for progress display
        self.update_interval = 0.1  # Update every 500ms instead of 100ms
        
    def start(self, total_files: int, estimated_data_pieces: int = None):
        """Start the progress display"""
        with self.lock:
            self.stats.files_total = total_files
            self.stats.data_pieces_total = estimated_data_pieces or (total_files * 10)
            self.stats.start_time = time.time()
            self.running = True
            self.last_display = ""
        
        # Reserve space for progress bars
        print("\n" * (self.progress_lines - 1))
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
    
    def stop(self):
        """Stop the progress display"""
        with self.lock:
            self.running = False
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        # Move cursor past progress area and show final stats
        print(f"\n" * 2)
        self._show_final_stats()
    
    def update_files(self, processed: int, current_file: str = ""):
        """Update file processing progress"""
        with self.lock:
            self.stats.files_processed = processed
            if current_file:
                # Truncate long filenames
                if len(current_file) > 50:
                    self.stats.current_file = "..." + current_file[-47:]
                else:
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
                if len(current_file) > 50:
                    self.stats.current_file = "..." + current_file[-47:]
                else:
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
        while self.running:
            try:
                self._render_progress_bars()
                time.sleep(self.update_interval)
            except Exception as e:
                # Don't let display errors crash the thread
                pass
    
    def _render_progress_bars(self):
        """Render all progress bars"""
        # Take a snapshot of stats to avoid lock contention
        with self.lock:
            current_stats = ProgressStats(
                files_processed=self.stats.files_processed,
                data_pieces_created=self.stats.data_pieces_created,
                embeddings_created=self.stats.embeddings_created,
                embeddings_queued=self.stats.embeddings_queued,
                files_total=self.stats.files_total,
                data_pieces_total=self.stats.data_pieces_total,
                current_file=self.stats.current_file,
                start_time=self.stats.start_time
            )
        
        # Build the complete display
        display_lines = self._build_display(current_stats)
        new_display = "\n".join(display_lines)
        
        # Only update if display actually changed
        if new_display != self.last_display:
            self.last_display = new_display
            
            # Move cursor up to overwrite previous display
            sys.stdout.write(f"\033[{self.progress_lines}A")
            
            # Write each line, clearing to end of line
            for line in display_lines:
                sys.stdout.write(f"\033[K{line}\n")
            
            sys.stdout.flush()
    
    def _build_display(self, stats: ProgressStats) -> list:
        """Build the complete progress display"""
        elapsed = time.time() - stats.start_time
        
        # Calculate rates
        files_rate = stats.files_processed / elapsed if elapsed > 0 else 0
        data_rate = stats.data_pieces_created / elapsed if elapsed > 0 else 0
        embed_rate = stats.embeddings_created / elapsed if elapsed > 0 else 0
        
        lines = []
        
        # Files progress bar
        files_bar = self._create_progress_bar(
            stats.files_processed,
            stats.files_total,
            f"Files ({files_rate:.1f}/s)",
            40
        )
        lines.append(files_bar)
        
        # Data pieces progress bar
        data_bar = self._create_progress_bar(
            stats.data_pieces_created,
            stats.data_pieces_total,
            f"Data Pieces ({data_rate:.1f}/s)",
            40
        )
        lines.append(data_bar)
        
        # Embeddings progress bar
        embed_total = max(stats.data_pieces_created, stats.data_pieces_total)
        embed_bar = self._create_progress_bar(
            stats.embeddings_created,
            embed_total,
            f"Embeddings ({embed_rate:.1f}/s)",
            40
        )
        lines.append(embed_bar)
        
        # Overall progress
        overall_bar = self._create_progress_bar(
            stats.files_processed,
            stats.files_total,
            "Overall Progress",
            40
        )
        lines.append(overall_bar)
        
        # Current file and queue info
        current_info = f"Current: {stats.current_file or 'None'} | Queue: {stats.embeddings_queued}"
        if len(current_info) > 80:
            current_info = current_info[:77] + "..."
        lines.append(current_info)
        
        # Timing info
        remaining_estimate = self._estimate_remaining(stats, elapsed)
        timing_info = f"Elapsed: {elapsed:.1f}s | Est. Remaining: {remaining_estimate}"
        lines.append(timing_info)
        
        return lines
    
    def _create_progress_bar(self, current: int, total: int, label: str, width: int = 40) -> str:
        """Create a single progress bar string"""
        if total <= 0:
            percentage = 0
            filled_length = 0
        else:
            percentage = min(100, (current / total) * 100)
            filled_length = int(width * current // total) if total > 0 else 0
        
        # Use simpler ASCII characters for better compatibility
        filled_char = "█"
        empty_char = "░"
        
        try:
            bar = filled_char * filled_length + empty_char * (width - filled_length)
        except:
            # Fallback to basic characters if Unicode fails
            bar = "=" * filled_length + "-" * (width - filled_length)
        
        return f"{label:<20} |{bar}| {current:>6}/{total:<6} ({percentage:5.1f}%)"
    
    def _estimate_remaining(self, stats: ProgressStats, elapsed: float) -> str:
        """Estimate remaining time"""
        if stats.files_processed == 0 or elapsed == 0:
            return "Unknown"
        
        files_rate = stats.files_processed / elapsed
        remaining_files = stats.files_total - stats.files_processed
        
        if files_rate > 0 and remaining_files > 0:
            remaining_seconds = remaining_files / files_rate
            if remaining_seconds < 60:
                return f"{remaining_seconds:.0f}s"
            elif remaining_seconds < 3600:
                return f"{remaining_seconds/60:.1f}m"
            else:
                return f"{remaining_seconds/3600:.1f}h"
        return "Complete" if remaining_files <= 0 else "Unknown"
    
    def _show_final_stats(self):
        """Show final statistics"""
        elapsed = time.time() - self.stats.start_time
        print("=" * 60)
        print("📊 Processing Complete!")
        print(f"   Files Processed: {self.stats.files_processed:,}/{self.stats.files_total:,}")
        print(f"   Data Pieces: {self.stats.data_pieces_created:,}")
        print(f"   Embeddings: {self.stats.embeddings_created:,}")
        print(f"   Total Time: {elapsed:.2f}s")
        if elapsed > 0:
            print(f"   Average Rate: {self.stats.files_processed/elapsed:.2f} files/s")
        print("=" * 60)

class SimpleProgressTracker:
    """Simple progress tracker for single file operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_update = 0
        self.update_threshold = 0.1  # Only update display every 100ms
        
    def update(self, current: int, total: int, detail: str = ""):
        """Update progress for single operation"""
        now = time.time()
        
        # Throttle updates to avoid flickering
        if now - self.last_update < self.update_threshold and current < total:
            return
        
        self.last_update = now
        elapsed = now - self.start_time
        percentage = (current / total) * 100 if total > 0 else 0
        rate = current / elapsed if elapsed > 0 else 0
        
        bar_width = 30
        filled_length = int(bar_width * current // total) if total > 0 else 0
        
        try:
            bar = "█" * filled_length + "░" * (bar_width - filled_length)
        except:
            # Fallback for systems that don't support Unicode
            bar = "=" * filled_length + "-" * (bar_width - filled_length)
        
        # Truncate detail if too long
        if len(detail) > 30:
            detail = detail[:27] + "..."
        
        status_line = f"\r{self.operation_name}: |{bar}| {current:,}/{total:,} ({percentage:.1f}%) [{rate:.1f}/s] {detail}"
        
        # Ensure we don't exceed terminal width
        try:
            terminal_width = os.get_terminal_size().columns
            if len(status_line) > terminal_width:
                status_line = status_line[:terminal_width-3] + "..."
        except:
            # If we can't get terminal size, truncate to safe length
            if len(status_line) > 120:
                status_line = status_line[:117] + "..."
        
        print(status_line, end="", flush=True)
    
    def complete(self, total_processed: int):
        """Mark operation as complete"""
        elapsed = time.time() - self.start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"\n✅ {self.operation_name} complete: {total_processed:,} items in {elapsed:.2f}s ({rate:.1f}/s)")

# Global progress manager instance
progress_manager = MultiProgressManager()
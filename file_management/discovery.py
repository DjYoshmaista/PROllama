# file_management/discovery.py
import os
import logging
from typing import Generator, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)

class FileDiscovery:
    """Handles file discovery, validation, and path management"""
    
    SUPPORTED_EXTENSIONS = {"py", "txt", "csv", "json"}
    
    def __init__(self, supported_extensions: Optional[Set[str]] = None):
        self.supported_extensions = supported_extensions or self.SUPPORTED_EXTENSIONS
    
    def discover_files(self, folder_path: str) -> Generator[str, None, None]:
        """
        Generator that yields supported file paths from a folder
        
        Args:
            folder_path: Root folder to search
            
        Yields:
            File paths for supported files
        """
        try:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.is_supported_file(file_path):
                        yield file_path
        except Exception as e:
            logger.error(f"Error discovering files in {folder_path}: {e}")
    
    def count_files(self, folder_path: str) -> int:
        """Count supported files in a folder"""
        count = 0
        try:
            for _ in self.discover_files(folder_path):
                count += 1
        except Exception as e:
            logger.error(f"Error counting files in {folder_path}: {e}")
        return count
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file has supported extension and is accessible"""
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
                logger.debug(f"Skipping empty file: {file_path}")
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
    
    def get_file_info(self, file_path: str) -> dict:
        """Get detailed information about a file"""
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
        """Get a preview of files in a folder"""
        preview = []
        
        try:
            file_gen = self.discover_files(folder_path)
            for i, file_path in enumerate(file_gen):
                if i >= max_files:
                    break
                
                info = self.get_file_info(file_path)
                rel_path = os.path.relpath(file_path, folder_path)
                info['relative_path'] = rel_path
                preview.append(info)
                
        except Exception as e:
            logger.error(f"Error previewing folder {folder_path}: {e}")
        
        return preview
    
    def get_folder_stats(self, folder_path: str) -> dict:
        """Get statistics about files in a folder"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'by_extension': {},
            'largest_file': None,
            'smallest_file': None
        }
        
        try:
            largest_size = 0
            smallest_size = float('inf')
            
            for file_path in self.discover_files(folder_path):
                info = self.get_file_info(file_path)
                
                stats['total_files'] += 1
                stats['total_size_mb'] += info.get('size_mb', 0)
                
                ext = info.get('extension', 'unknown')
                stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
                
                size = info.get('size_bytes', 0)
                if size > largest_size:
                    largest_size = size
                    stats['largest_file'] = info
                
                if size < smallest_size:
                    smallest_size = size
                    stats['smallest_file'] = info
                    
        except Exception as e:
            logger.error(f"Error getting folder stats for {folder_path}: {e}")
            stats['error'] = str(e)
        
        return stats

class InteractiveFileSelector:
    """Interactive file/folder selection with GUI and CLI fallbacks"""
    
    def __init__(self, discovery: Optional[FileDiscovery] = None):
        self.discovery = discovery or FileDiscovery()
    
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
                # Show preview and confirm
                if self._confirm_folder_selection(folder_path):
                    return folder_path
            else:
                print(f"\nError: Invalid folder path: {folder_path}")
                retry = input("Would you like to try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
    
    def _confirm_folder_selection(self, folder_path: str) -> bool:
        """Show folder preview and confirm selection"""
        # Get folder statistics
        stats = self.discovery.get_folder_stats(folder_path)
        
        print(f"\nFolder Analysis:")
        print(f"Total supported files: {stats['total_files']}")
        print(f"Total size: {stats['total_size_mb']:.2f} MB")
        
        if stats['by_extension']:
            print("Files by type:")
            for ext, count in stats['by_extension'].items():
                print(f"  .{ext}: {count} files")
        
        # Show preview of files
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
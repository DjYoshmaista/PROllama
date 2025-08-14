# file_processing_utils.py - File handling and validation utilities
import os
import logging
from file_tracker import file_tracker

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ["py", "txt", "csv", "json"]

def get_folder_path():
    """Simplified file browser"""
    path = input("Enter folder path (absolute or relative): ").strip()
    path = os.path.expanduser(path)
    
    if not os.path.exists(path):
        print("Path does not exist")
        return None
        
    if not os.path.isdir(path):
        print("Path is not a directory")
        return None
        
    return path

async def browse_files():
    """Browse for files with fallback to manual input if GUI unavailable"""
    try:
        # Try to use GUI file browser
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="Select Folder to Load Files")
        root.destroy()  # Clean up the tkinter root window
        
        if folder_path:
            logger.info(f"Folder selected via GUI: {folder_path}")
            return folder_path
        else:
            logger.info("No folder selected via GUI dialog")
            return None
            
    except Exception as e:
        # GUI not available (SSH, no display, etc.)
        logger.warning(f"GUI file browser not available: {e}")
        print("\n" + "="*60)
        print("GUI file browser is not available (no display detected)")
        print("Please enter the folder path manually")
        print("="*60)
        
        return await get_manual_folder_path()

async def get_manual_folder_path():
    """Get folder path via manual input with validation"""
    while True:
        print("\nOptions:")
        print("1. Enter absolute path (e.g., /home/user/documents)")
        print("2. Enter relative path from current directory")
        print("3. Use current directory")
        print("4. Cancel")
        
        # Show current working directory for reference
        current_dir = os.getcwd()
        print(f"\nCurrent directory: {current_dir}")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            # Absolute path
            folder_path = input("Enter absolute folder path: ").strip()
            # Expand user home directory if ~ is used
            folder_path = os.path.expanduser(folder_path)
            
        elif choice == "2":
            # Relative path
            relative_path = input("Enter relative path from current directory: ").strip()
            folder_path = os.path.join(current_dir, relative_path)
            folder_path = os.path.abspath(folder_path)  # Convert to absolute
            
        elif choice == "3":
            # Use current directory
            folder_path = current_dir
            print(f"Using current directory: {folder_path}")
            
        elif choice == "4":
            # Cancel
            logger.info("Folder selection cancelled by user")
            return None
            
        else:
            print("Invalid option. Please try again.")
            continue
        
        # Validate the folder path
        if await validate_folder_path(folder_path):
            return folder_path
        else:
            print(f"\nError: Invalid folder path: {folder_path}")
            retry = input("Would you like to try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None

async def validate_folder_path(folder_path):
    """Enhanced folder validation with file tracker integration"""
    try:
        # Check if path exists
        if not os.path.exists(folder_path):
            logger.error(f"Path does not exist: {folder_path}")
            print(f"Error: Path does not exist: {folder_path}")
            return False
        
        # Check if it's a directory
        if not os.path.isdir(folder_path):
            logger.error(f"Path is not a directory: {folder_path}")
            print(f"Error: Path is not a directory: {folder_path}")
            return False
        
        # Check if we have read permissions
        if not os.access(folder_path, os.R_OK):
            logger.error(f"No read permission for directory: {folder_path}")
            print(f"Error: No read permission for directory: {folder_path}")
            return False
        
        # Count supported files in the directory
        all_files = collect_supported_files(folder_path)
        # Count supported files in the directory
        supported_extensions = ["py", "txt", "csv", "json"]
        file_count = 0
        sample_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1][1:].lower()
                if ext in supported_extensions:
                    file_count += 1
                    if len(sample_files) < 10:  # Keep sample for preview
                        rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                        sample_files.append(rel_path)        
        if not all_files:
            logger.warning(f"No supported files found in: {folder_path}")
            print(f"\nWarning: No supported files found in: {folder_path}")
            print(f"Supported file types: {', '.join(SUPPORTED_EXTENSIONS)}")
            
            # Ask if user wants to continue anyway
            continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
            if continue_anyway != 'y':
                return False

        elif len(all_files) != file_count:
            logger.warning(f"Some files in the directory are of unsupported types and will be ignored.")
            print(f"\nWarning: Some files in the directory are of unsupported types and will be ignored.")
            print(f"Supported file types: {', '.join(SUPPORTED_EXTENSIONS)}")
            
        # Use file tracker to determine which files need processing
        files_to_process, filter_stats = file_tracker.batch_filter_files(all_files)
        
        print(f"\nFile Analysis Results:")
        print(f"  Total files found: {len(all_files)}")
        print(f"  Files needing processing: {len(files_to_process)}")
        print(f"    - New files: {filter_stats.get('new_files', 0)}")
        print(f"    - Changed files: {filter_stats.get('size_changed', 0) + filter_stats.get('time_changed', 0) + filter_stats.get('content_changed', 0)}")
        print(f"  Files already processed: {filter_stats.get('already_processed', 0)}")
        
        if len(files_to_process) == 0:
            print("\n✅ All files in this directory have already been processed!")
            print("No new processing is needed.")
            choice = input("Would you like to force reprocessing of all files? (y/n): ").strip().lower()
            if choice != 'y':
                return False
            # If forcing reprocess, return all files
            return _confirm_processing(all_files, folder_path, force_reprocess=True)
        else:
            print(f"\nFound {file_count} supported file(s) in the directory")
            logger.info(f"Found {file_count} supported files in: {folder_path}")
        
        # Show a preview of files that will be processed
        if sample_files:
            print("\nPreview of files to be processed (first 10):")
            for file_path in sample_files:
                print(f"  - {file_path}")
            
            if file_count > 10:
                print(f"  ... and {file_count - 10} more file(s)")
        
        # Final confirmation
        confirm = input(f"\nProcess {file_count} file(s) from this directory? (y/n): ").strip().lower()
        if confirm != 'y':
            logger.info("User cancelled after folder validation")
            return False
        
        logger.info(f"Folder path validated successfully: {folder_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating folder path: {e}")
        print(f"Error validating folder path: {e}")
        return False
        return _confirm_processing(files_to_process, folder_path)
        
    except Exception as e:
        logger.error(f"Error validating folder path: {e}")
        print(f"Error validating folder path: {e}")
        return False

def _confirm_processing(files_to_process, folder_path, force_reprocess=False):
    """Confirm processing with user"""
    # Show a preview of files that will be processed
    if files_to_process:
        print(f"\nPreview of files to be processed (first 10):")
        for i, filepath in enumerate(files_to_process[:10]):
            rel_path = os.path.relpath(filepath, folder_path)
            print(f"  {i+1}. {rel_path}")
        
        if len(files_to_process) > 10:
            print(f"  ... and {len(files_to_process) - 10} more file(s)")
    
    # Final confirmation
    if force_reprocess:
        confirm = input(f"\nForce reprocess all {len(files_to_process)} file(s) from this directory? (y/n): ").strip().lower()
    else:
        confirm = input(f"\nProcess {len(files_to_process)} file(s) from this directory? (y/n): ").strip().lower()
        
    if confirm != 'y':
        logger.info("User cancelled after folder validation")
        return False
    
    logger.info(f"Folder path validated successfully: {folder_path}")
    return True

def collect_supported_files(folder_path):
    """Collect all supported files from directory"""
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1][1:].lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_files.append(os.path.join(root, file))
    return all_files

def generate_file_paths_filtered(folder_path):
    """Generator yielding only files that need processing"""
    all_files = collect_supported_files(folder_path)
    
    # Filter through file tracker
    files_to_process, _ = file_tracker.batch_filter_files(all_files)
    
    # Yield filtered files
    for file_path in files_to_process:
        yield file_path

def generate_file_paths(folder_path):
    """Generator yielding all supported file paths"""
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1][1:].lower()
            if ext in SUPPORTED_EXTENSIONS:
                yield file_path

def count_files_filtered(folder_path):
    """Count files that actually need processing"""
    all_files = collect_supported_files(folder_path)
    files_to_process, _ = file_tracker.batch_filter_files(all_files)
    return len(files_to_process)

def count_files(folder_path):
    """Count all supported files"""
    return len(collect_supported_files(folder_path))

def get_file_processing_stats(folder_path):
    """Get detailed statistics about file processing needs"""
    all_files = collect_supported_files(folder_path)
    
    if not all_files:
        return {
            'total_files': 0,
            'files_to_process': 0,
            'already_processed': 0,
            'filter_stats': {}
        }
    
    files_to_process, filter_stats = file_tracker.batch_filter_files(all_files)
    
    return {
        'total_files': len(all_files),
        'files_to_process': len(files_to_process),
        'already_processed': len(all_files) - len(files_to_process),
        'filter_stats': filter_stats,
        'file_list': files_to_process
    }

def force_reprocess_files():
    """Menu option to force reprocessing of specific files or directories"""
    print("\nForce Reprocessing Options:")
    print("1. Reprocess specific file")
    print("2. Reprocess entire directory")
    print("3. Clear all file tracking data")
    print("4. Back to main menu")
    
    choice = input("Select option: ").strip()
    
    if choice == "1":
        filepath = input("Enter file path to reprocess: ").strip()
        if os.path.exists(filepath):
            # Remove from tracker and process
            abs_path = os.path.abspath(filepath)
            if abs_path in file_tracker.processed_files:
                del file_tracker.processed_files[abs_path]
                file_tracker.save_tracker(force=True)
                print(f"Removed {filepath} from tracker. It will be reprocessed next time.")
            else:
                print(f"File {filepath} was not in the tracker.")
        else:
            print("File does not exist.")
    
    elif choice == "2":
        folder_path = input("Enter directory path to reprocess: ").strip()
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Remove all files in this directory from tracker
            removed_count = 0
            folder_path = os.path.abspath(folder_path)
            
            for filepath in list(file_tracker.processed_files.keys()):
                if filepath.startswith(folder_path):
                    del file_tracker.processed_files[filepath]
                    removed_count += 1
            
            file_tracker.save_tracker(force=True)
            print(f"Removed {removed_count} files from tracker. Directory will be reprocessed next time.")
        else:
            print("Directory does not exist.")
    
    elif choice == "3":
        confirm = input("⚠️  This will clear ALL file tracking data. Are you sure? (y/n): ").strip().lower()
        if confirm == 'y':
            file_tracker.processed_files.clear()
            if hasattr(file_tracker, 'pending_saves'):
                file_tracker.pending_saves.clear()
            file_tracker.save_tracker(force=True)
            print("All file tracking data cleared.")
    
    elif choice == "4":
        return
    else:
        print("Invalid option.")

def show_file_tracker_stats():
    """Enhanced file tracker statistics display"""
    stats = file_tracker.get_processed_files_stats()
    print(f"\n📊 File Processing Statistics:")
    print(f"   Total processed files: {stats['total_files']}")
    print(f"   Total records created: {stats['total_records']}")
    print(f"   Total file size: {stats['total_size_mb']:.2f} MB")
    print(f"   Average records per file: {stats['avg_records_per_file']:.1f}")
    
    # Add timing stats if available
    if 'total_processing_time' in stats:
        print(f"   Total processing time: {stats['total_processing_time']:.2f} seconds")
        print(f"   Average time per file: {stats['avg_time_per_file']:.2f} seconds")
        print(f"   Average time per record: {stats['avg_time_per_record']:.3f} seconds")
    
    if stats.get('pending_saves', 0) > 0:
        print(f"   ⚠️  Pending saves: {stats['pending_saves']} files")
    
    # Show recent files
    recent_files = file_tracker.get_recent_files(5) if hasattr(file_tracker, 'get_recent_files') else []
    if recent_files:
        print(f"\n📁 Recently processed files (last 5):")
        for i, (filepath, record) in enumerate(recent_files, 1):
            filename = os.path.basename(filepath)
            processing_time = getattr(record, 'processing_time', 0)
            print(f"   {i}. {filename:<40} ({record.records_count} records, {processing_time:.2f}s)")
    else:
        # Fallback for older file tracker versions
        if file_tracker.processed_files:
            print(f"\n📁 Recently processed files (last 5):")
            recent_files = sorted(
                file_tracker.processed_files.items(), 
                key=lambda x: x[1].processed_at, 
                reverse=True
            )[:5]
            
            for i, (filepath, record) in enumerate(recent_files, 1):
                filename = os.path.basename(filepath)
                print(f"   {i}. {filename:<40} ({record.records_count} records)")

def validate_single_file(file_path):
    """Validate a single file for processing"""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        if not os.path.isfile(file_path):
            return False, "Path is not a file"
        
        if not os.access(file_path, os.R_OK):
            return False, "No read permission"
        
        ext = os.path.splitext(file_path)[1][1:].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}"
        
        # Check with file tracker
        should_process, reason = file_tracker.should_process_file(file_path)
        
        return True, f"Ready for processing ({'will reprocess' if not should_process else 'new/changed file'})"
        
    except Exception as e:
        return False, f"Validation error: {e}"
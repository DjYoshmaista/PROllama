# progress_tracker.py
from async_loader import run_processing

async def track_progress(file_generator, total_files):
    """Track progress of file processing with memory-efficient approach"""
    print(f"Starting processing for {total_files} files")
    # Use the new run_processing API that accepts a generator
    return await run_processing(file_generator, total_files)
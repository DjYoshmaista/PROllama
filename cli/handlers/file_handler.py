# cli/handlers/file_handler.py
import logging
import time
from database.cache import embedding_cache
from file_management.discovery import file_selector, file_discovery
from file_management.loaders import bulk_loader

# Create prefixed logger for this file
logger = logging.getLogger(__name__)
LOG_PREFIX = "[CLI/FileHandler]"

class FileHandler:
    """Handles file loading operations"""
    
    async def handle_load_single_file(self):
        """Handle single file loading"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="Select File to Load")
            root.destroy()
            
            if file_path:
                success = await bulk_loader.load_single_file_interactive(file_path)
                if success:
                    # Invalidate cache after loading
                    embedding_cache.invalidate()
                    print("File loaded successfully.")
                else:
                    print("Failed to load file.")
            else:
                print("No file selected.")
                
        except Exception as e:
            logger.error(f"File loading error: {e}")
            print(f"Failed to load file: {e}")
    
    async def handle_load_folder(self):
        """Handle folder loading with NEW chunked processing system"""
        print(f"{LOG_PREFIX} Starting menu option 4: Load documents from folder")
        logger.info(f"{LOG_PREFIX} Initiating NEW chunked folder processing")
        
        start_time = time.time()
        
        # Step 1: Folder Selection
        print(f"{LOG_PREFIX} 📂 Step 1: Folder Selection")
        logger.info(f"{LOG_PREFIX} Launching folder browser...")
        
        folder_selection_start = time.time()
        folder_path = file_selector.browse_for_folder()
        folder_selection_time = time.time() - folder_selection_start
        
        if not folder_path:
            print(f"{LOG_PREFIX} No folder selected, operation cancelled")
            logger.info(f"{LOG_PREFIX} User cancelled folder selection")
            return
        
        print(f"{LOG_PREFIX} ✅ Selected folder: {folder_path}")
        logger.info(f"{LOG_PREFIX} Folder selected in {folder_selection_time:.2f}s: {folder_path}")
        
        # Step 2: Quick Discovery Preview
        print(f"{LOG_PREFIX} 🔍 Step 2: Quick Discovery Preview")
        logger.info(f"{LOG_PREFIX} Getting preview of folder contents...")
        
        discovery_start = time.time()
        total_files = file_discovery.count_files(folder_path)
        discovery_time = time.time() - discovery_start
        
        if total_files == 0:
            print(f"{LOG_PREFIX} No supported files found in the selected folder")
            logger.warning(f"{LOG_PREFIX} No supported files found in folder: {folder_path}")
            return
        
        print(f"{LOG_PREFIX} ✅ Found {total_files:,} supported files in {discovery_time:.2f}s")
        logger.info(f"{LOG_PREFIX} Discovery results: {total_files} files, {discovery_time:.2f}s")
        
        # Step 3: User Confirmation
        print(f"{LOG_PREFIX} ⚠️  Step 3: User Confirmation")
        print(f"{LOG_PREFIX} This will process {total_files:,} files using the NEW chunked processing system:")
        print(f"{LOG_PREFIX} • Generator-based file discovery in chunks")
        print(f"{LOG_PREFIX} • Parallel text processing, embedding generation, and database insertion")
        print(f"{LOG_PREFIX} • Comprehensive metrics after each chunk")
        
        if not self._confirm_folder_loading():
            print(f"{LOG_PREFIX} Loading cancelled by user")
            logger.info(f"{LOG_PREFIX} User cancelled loading operation")
            return
        
        logger.info(f"{LOG_PREFIX} User confirmed chunked processing of {total_files} files")
        
        # Step 4: Execute NEW Chunked Processing System
        print(f"\n{LOG_PREFIX} 🚀 Step 4: Executing NEW Chunked Processing System")
        await self._execute_chunked_processing(folder_path)
        
        total_time = time.time() - start_time
        print(f"\n{LOG_PREFIX} 🏁 FOLDER PROCESSING COMPLETE!")
        print(f"{LOG_PREFIX} Total operation time: {total_time:.2f}s")
        print(f"{LOG_PREFIX} Returning to main menu...")
        logger.info(f"{LOG_PREFIX} Complete chunked folder processing finished in {total_time:.2f}s")
    
    def _confirm_folder_loading(self) -> bool:
        """Confirm folder loading with user"""
        confirm = input("Proceed with loading? (y/N): ").strip().lower()
        return confirm == 'y'
    
    async def _execute_chunked_processing(self, folder_path: str):
        """Execute the NEW chunked processing system"""
        logger.info(f"{LOG_PREFIX} Starting NEW chunked processing system")
        
        try:
            # Import the new chunked processor
            from file_management.chunked_processor import chunked_processor
            
            # Configure processor settings
            print(f"{LOG_PREFIX} 🔧 Configuring chunked processor...")
            chunked_processor.chunk_size = 500  # Process 500 files per chunk
            chunked_processor.max_workers = 8   # Use 8 parallel workers
            
            logger.info(f"{LOG_PREFIX} Chunked processor configured: chunk_size=500, max_workers=8")
            
            # Execute the chunked processing
            print(f"{LOG_PREFIX} 🎯 Starting chunked processing pipeline...")
            logger.info(f"{LOG_PREFIX} Executing chunked processor on folder: {folder_path}")
            
            # Process the entire folder using the new system
            overall_metrics = await chunked_processor.process_folder(folder_path)
            
            # Invalidate cache after successful processing
            print(f"\n{LOG_PREFIX} 🔄 Step 5: Cache Invalidation")
            logger.info(f"{LOG_PREFIX} Starting cache invalidation process")
            
            cache_start = time.time()
            embedding_cache.invalidate()
            cache_time = time.time() - cache_start
            
            print(f"{LOG_PREFIX} ✅ Cache invalidated in {cache_time:.3f}s")
            logger.info(f"{LOG_PREFIX} Cache invalidation completed in {cache_time:.3f}s")
            
            # Display final success message
            print(f"\n{LOG_PREFIX} 🎉 CHUNKED PROCESSING SUCCESSFULLY COMPLETED!")
            print(f"{LOG_PREFIX} • Processed {overall_metrics.total_files_processed:,} files")
            print(f"{LOG_PREFIX} • Created {overall_metrics.total_chunks_created:,} chunks")
            print(f"{LOG_PREFIX} • Generated {overall_metrics.total_embeddings_generated:,} embeddings")
            print(f"{LOG_PREFIX} • Inserted {overall_metrics.total_database_inserts:,} database records")
            print(f"{LOG_PREFIX} • Processed {overall_metrics.total_disk_space_processed / (1024*1024*1024):.2f} GB of data")
            print(f"{LOG_PREFIX} • Total time: {overall_metrics.total_processing_time:.1f}s")
            print(f"{LOG_PREFIX} • Success rate: {(overall_metrics.total_files_processed/max(overall_metrics.total_files_discovered,1)*100):.1f}%")
            
            logger.info(f"{LOG_PREFIX} Chunked processing completed successfully: "
                       f"{overall_metrics.total_files_processed} files, "
                       f"{overall_metrics.total_chunks_created} chunks, "
                       f"{overall_metrics.total_processing_time:.1f}s")
            
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Chunked processing failed: {e}")
            print(f"\n{LOG_PREFIX} ❌ Chunked processing failed: {e}")
            raise
    
    def select_file_with_dialog(self) -> str:
        """Open file selection dialog"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(title="Select File to Load")
            root.destroy()
            return file_path
        except ImportError:
            print("GUI file dialog not available. Please enter file path manually.")
            return input("Enter file path: ").strip()
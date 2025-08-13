#!/usr/bin/env python3
"""
Chunked File Processing System
Redesigned pipeline with generator-based discovery, parallel processing, and comprehensive metrics
"""
import asyncio
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Tuple, Set
from dataclasses import dataclass
from queue import Queue
import psutil

# Core imports
from file_management.parsers import document_parser
from file_management.chunking import text_chunker
from inference.summarization import summarizer
from inference.async_embeddings import async_embedding_service
from database.batch_operations import batch_db_ops

# Setup logging
logger = logging.getLogger(__name__)
LOG_PREFIX = "[ChunkedProcessor]"

@dataclass
class ProcessingMetrics:
    """Metrics for each processing chunk"""
    chunk_id: int
    files_discovered: int
    files_processed: int
    files_failed: int
    chunks_created: int
    summaries_created: int
    embeddings_generated: int
    database_inserts: int
    discovery_time: float
    processing_time: float
    embedding_time: float
    database_time: float
    total_time: float
    disk_space_processed: int  # bytes
    memory_peak: float  # MB
    cpu_usage: float  # percentage

@dataclass
class OverallMetrics:
    """Overall processing metrics"""
    total_files_discovered: int = 0
    total_files_processed: int = 0
    total_files_failed: int = 0
    total_chunks_created: int = 0
    total_summaries_created: int = 0
    total_embeddings_generated: int = 0
    total_database_inserts: int = 0
    total_processing_time: float = 0.0
    total_disk_space_processed: int = 0  # bytes
    chunk_metrics: List[ProcessingMetrics] = None
    
    def __post_init__(self):
        if self.chunk_metrics is None:
            self.chunk_metrics = []

class FileDiscoveryGenerator:
    """Generator-based file discovery system"""
    
    def __init__(self, chunk_size: int = 500, supported_extensions: Optional[Set[str]] = None):
        self.chunk_size = chunk_size
        self.supported_extensions = supported_extensions or {"py", "txt", "csv", "json"}
        self.skip_dirs = {
            '__pycache__', 'node_modules', '.git', '.venv', 'venv', 
            'build', 'dist', '.pytest_cache', '.mypy_cache', 'target'
        }
        
    def discover_files(self, root_path: str) -> Iterator[Tuple[int, List[str]]]:
        """
        Generator that yields chunks of file paths
        Returns: (chunk_id, list_of_file_paths)
        """
        logger.info(f"{LOG_PREFIX} Starting file discovery in: {root_path}")
        print(f"{LOG_PREFIX} 🔍 Discovering files in chunks of {self.chunk_size}...")
        
        chunk_id = 0
        current_chunk = []
        total_discovered = 0
        
        start_time = time.time()
        
        try:
            for root, dirs, files in os.walk(root_path):
                # Skip unwanted directories
                dirs[:] = [d for d in dirs if d not in self.skip_dirs]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Filter by extension
                    ext = Path(file_path).suffix[1:].lower()
                    if ext not in self.supported_extensions:
                        continue
                    
                    # Filter by size (10 bytes < size < 10MB)
                    try:
                        file_size = os.path.getsize(file_path)
                        if not (10 < file_size < 10 * 1024 * 1024):
                            continue
                    except (OSError, IOError):
                        continue
                    
                    current_chunk.append(file_path)
                    total_discovered += 1
                    
                    # Yield chunk when it reaches the desired size
                    if len(current_chunk) >= self.chunk_size:
                        discovery_time = time.time() - start_time
                        
                        print(f"{LOG_PREFIX} 📦 Chunk {chunk_id}: {len(current_chunk)} files "
                              f"(discovered {total_discovered:,} total in {discovery_time:.1f}s)")
                        
                        yield chunk_id, current_chunk.copy()
                        chunk_id += 1
                        current_chunk.clear()
            
            # Yield remaining files in final chunk
            if current_chunk:
                discovery_time = time.time() - start_time
                
                print(f"{LOG_PREFIX} 📦 Final Chunk {chunk_id}: {len(current_chunk)} files "
                      f"(discovered {total_discovered:,} total in {discovery_time:.1f}s)")
                
                yield chunk_id, current_chunk.copy()
                
        except Exception as e:
            logger.error(f"{LOG_PREFIX} File discovery failed: {e}")
            raise
        
        discovery_time = time.time() - start_time
        print(f"{LOG_PREFIX} ✅ Discovery complete: {total_discovered:,} files in {discovery_time:.1f}s")
        logger.info(f"{LOG_PREFIX} File discovery completed: {total_discovered} files in {discovery_time:.1f}s")

class ChunkedFileProcessor:
    """Main chunked file processing system"""
    
    def __init__(self, chunk_size: int = 500, max_workers: int = 8):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.discovery_generator = FileDiscoveryGenerator(chunk_size)
        self.overall_metrics = OverallMetrics()
        
        # Initialize services
        self.text_chunker = text_chunker
        self.summarizer = summarizer
        self.embedding_service = async_embedding_service
        self.batch_db = batch_db_ops
        
    async def process_single_file(self, file_path: str) -> Optional[Dict]:
        """Process a single file through the complete pipeline"""
        try:
            # Parse file content
            content = document_parser.parse_single_file(file_path)
            if not content:
                return None
            
            # Create chunks
            chunks = self.text_chunker.create_chunks(content)
            if not chunks:
                return None
            
            # Generate summary
            summary = await self.summarizer.create_summary_async(content)
            
            file_size = os.path.getsize(file_path)
            
            return {
                'file_path': file_path,
                'content': content,
                'chunks': chunks,
                'summary': summary,
                'file_size': file_size,
                'chunk_count': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Error processing file {file_path}: {e}")
            return None
    
    async def process_file_chunk(self, chunk_id: int, file_paths: List[str]) -> ProcessingMetrics:
        """Process a chunk of files with full parallelization"""
        chunk_start = time.time()
        
        print(f"\n{LOG_PREFIX} 🚀 Processing Chunk {chunk_id}: {len(file_paths)} files")
        logger.info(f"{LOG_PREFIX} Starting chunk {chunk_id} with {len(file_paths)} files")
        
        # Initialize metrics
        metrics = ProcessingMetrics(
            chunk_id=chunk_id,
            files_discovered=len(file_paths),
            files_processed=0,
            files_failed=0,
            chunks_created=0,
            summaries_created=0,
            embeddings_generated=0,
            database_inserts=0,
            discovery_time=0.0,
            processing_time=0.0,
            embedding_time=0.0,
            database_time=0.0,
            total_time=0.0,
            disk_space_processed=0,
            memory_peak=0.0,
            cpu_usage=0.0
        )
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Step 1: Parallel file processing (parsing, chunking, summarization)
        processing_start = time.time()
        print(f"{LOG_PREFIX} 📝 Step 1: Parsing, chunking, and summarizing {len(file_paths)} files...")
        
        # Process files in parallel
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(file_path):
            async with semaphore:
                return await self.process_single_file(file_path)
        
        # Execute parallel processing
        tasks = [process_with_semaphore(fp) for fp in file_paths]
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        successful_results = []
        for i, result in enumerate(processed_results):
            if isinstance(result, Exception):
                logger.error(f"{LOG_PREFIX} Failed to process {file_paths[i]}: {result}")
                metrics.files_failed += 1
            elif result is not None:
                successful_results.append(result)
                metrics.files_processed += 1
                metrics.chunks_created += result['chunk_count']
                metrics.summaries_created += 1
                metrics.disk_space_processed += result['file_size']
            else:
                metrics.files_failed += 1
        
        metrics.processing_time = time.time() - processing_start
        
        print(f"{LOG_PREFIX} ✅ Step 1 Complete: {metrics.files_processed}/{len(file_paths)} files processed "
              f"({metrics.chunks_created} chunks, {metrics.summaries_created} summaries) in {metrics.processing_time:.1f}s")
        
        if not successful_results:
            metrics.total_time = time.time() - chunk_start
            return metrics
        
        # Step 2: Parallel embedding generation
        embedding_start = time.time()
        print(f"{LOG_PREFIX} 🧠 Step 2: Generating embeddings for {metrics.chunks_created} chunks...")
        
        # Collect all chunks for embedding
        all_chunks = []
        for result in successful_results:
            for chunk in result['chunks']:
                all_chunks.append({
                    'file_path': result['file_path'],
                    'chunk_text': chunk['text'],
                    'chunk_metadata': chunk
                })
        
        # Generate embeddings in parallel
        embedding_tasks = []
        for chunk_data in all_chunks:
            task = self.embedding_service.generate_embedding_async(chunk_data['chunk_text'])
            embedding_tasks.append((chunk_data, task))
        
        # Wait for embeddings with proper error handling
        embedding_results = []
        for chunk_data, task in embedding_tasks:
            try:
                embedding = await task
                if embedding is not None:
                    chunk_data['embedding'] = embedding
                    embedding_results.append(chunk_data)
                    metrics.embeddings_generated += 1
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Embedding generation failed for chunk: {e}")
        
        metrics.embedding_time = time.time() - embedding_start
        
        print(f"{LOG_PREFIX} ✅ Step 2 Complete: {metrics.embeddings_generated}/{metrics.chunks_created} embeddings generated "
              f"in {metrics.embedding_time:.1f}s")
        
        # Step 3: Parallel database insertion
        if embedding_results:
            db_start = time.time()
            print(f"{LOG_PREFIX} 💾 Step 3: Inserting {len(embedding_results)} chunks into database...")
            
            # Prepare data for batch insertion
            documents_data = []
            for chunk_data in embedding_results:
                doc_data = {
                    'file_path': chunk_data['file_path'],
                    'content': chunk_data['chunk_text'],
                    'embedding': chunk_data['embedding'],
                    'metadata': chunk_data['chunk_metadata']
                }
                documents_data.append(doc_data)
            
            # Insert in parallel batches
            try:
                inserted_count = await self.batch_db.insert_documents_parallel(documents_data)
                metrics.database_inserts = inserted_count
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Database insertion failed: {e}")
                metrics.database_inserts = 0
            
            metrics.database_time = time.time() - db_start
            
            print(f"{LOG_PREFIX} ✅ Step 3 Complete: {metrics.database_inserts} documents inserted "
                  f"in {metrics.database_time:.1f}s")
        
        # Final metrics calculation
        metrics.total_time = time.time() - chunk_start
        metrics.memory_peak = max(initial_memory, process.memory_info().rss / 1024 / 1024)
        metrics.cpu_usage = process.cpu_percent()
        
        # Display chunk summary
        self._display_chunk_metrics(metrics)
        
        return metrics
    
    def _display_chunk_metrics(self, metrics: ProcessingMetrics):
        """Display comprehensive metrics for a completed chunk"""
        print(f"\n{LOG_PREFIX} 📊 CHUNK {metrics.chunk_id} METRICS:")
        print(f"{LOG_PREFIX} ═══════════════════════════════════════")
        print(f"{LOG_PREFIX} 📁 Files: {metrics.files_processed}/{metrics.files_discovered} processed "
              f"({metrics.files_failed} failed)")
        print(f"{LOG_PREFIX} 📝 Chunks: {metrics.chunks_created} created")
        print(f"{LOG_PREFIX} 📑 Summaries: {metrics.summaries_created} generated")
        print(f"{LOG_PREFIX} 🧠 Embeddings: {metrics.embeddings_generated} generated")
        print(f"{LOG_PREFIX} 💾 Database: {metrics.database_inserts} documents inserted")
        print(f"{LOG_PREFIX} 💽 Data: {metrics.disk_space_processed / (1024*1024):.1f} MB processed")
        print(f"{LOG_PREFIX} ⏱️  Times: Processing={metrics.processing_time:.1f}s, "
              f"Embedding={metrics.embedding_time:.1f}s, DB={metrics.database_time:.1f}s")
        print(f"{LOG_PREFIX} 🖥️  Resources: Memory={metrics.memory_peak:.1f}MB, CPU={metrics.cpu_usage:.1f}%")
        print(f"{LOG_PREFIX} ⚡ Total: {metrics.total_time:.1f}s "
              f"({metrics.files_processed/metrics.total_time:.1f} files/s)")
        print(f"{LOG_PREFIX} ═══════════════════════════════════════")
    
    async def process_folder(self, folder_path: str) -> OverallMetrics:
        """Main entry point: process entire folder using chunked approach"""
        print(f"\n{LOG_PREFIX} 🎯 STARTING CHUNKED FOLDER PROCESSING")
        print(f"{LOG_PREFIX} 📂 Folder: {folder_path}")
        print(f"{LOG_PREFIX} 📦 Chunk Size: {self.chunk_size} files")
        print(f"{LOG_PREFIX} 👥 Max Workers: {self.max_workers}")
        print(f"{LOG_PREFIX} {'='*60}")
        
        overall_start = time.time()
        logger.info(f"{LOG_PREFIX} Starting chunked folder processing: {folder_path}")
        
        try:
            # Process each chunk generated by discovery
            for chunk_id, file_paths in self.discovery_generator.discover_files(folder_path):
                # Process this chunk
                chunk_metrics = await self.process_file_chunk(chunk_id, file_paths)
                
                # Update overall metrics
                self.overall_metrics.total_files_discovered += chunk_metrics.files_discovered
                self.overall_metrics.total_files_processed += chunk_metrics.files_processed
                self.overall_metrics.total_files_failed += chunk_metrics.files_failed
                self.overall_metrics.total_chunks_created += chunk_metrics.chunks_created
                self.overall_metrics.total_summaries_created += chunk_metrics.summaries_created
                self.overall_metrics.total_embeddings_generated += chunk_metrics.embeddings_generated
                self.overall_metrics.total_database_inserts += chunk_metrics.database_inserts
                self.overall_metrics.total_disk_space_processed += chunk_metrics.disk_space_processed
                self.overall_metrics.chunk_metrics.append(chunk_metrics)
            
            # Calculate total processing time
            self.overall_metrics.total_processing_time = time.time() - overall_start
            
            # Display final summary
            self._display_final_summary()
            
            return self.overall_metrics
            
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Folder processing failed: {e}")
            print(f"{LOG_PREFIX} ❌ Processing failed: {e}")
            raise
    
    def _display_final_summary(self):
        """Display comprehensive final summary"""
        m = self.overall_metrics
        
        print(f"\n{LOG_PREFIX} 🏁 FINAL PROCESSING SUMMARY")
        print(f"{LOG_PREFIX} {'='*70}")
        print(f"{LOG_PREFIX} 📁 Total Files Discovered: {m.total_files_discovered:,}")
        print(f"{LOG_PREFIX} ✅ Total Files Processed: {m.total_files_processed:,}")
        print(f"{LOG_PREFIX} ❌ Total Files Failed: {m.total_files_failed:,}")
        print(f"{LOG_PREFIX} 📝 Total Chunks Created: {m.total_chunks_created:,}")
        print(f"{LOG_PREFIX} 📑 Total Summaries: {m.total_summaries_created:,}")
        print(f"{LOG_PREFIX} 🧠 Total Embeddings: {m.total_embeddings_generated:,}")
        print(f"{LOG_PREFIX} 💾 Total DB Inserts: {m.total_database_inserts:,}")
        print(f"{LOG_PREFIX} 💽 Total Data Processed: {m.total_disk_space_processed / (1024*1024*1024):.2f} GB")
        print(f"{LOG_PREFIX} ⏱️  Total Processing Time: {m.total_processing_time:.1f}s")
        print(f"{LOG_PREFIX} ⚡ Overall Speed: {m.total_files_processed/m.total_processing_time:.1f} files/s")
        print(f"{LOG_PREFIX} 📊 Success Rate: {(m.total_files_processed/max(m.total_files_discovered,1)*100):.1f}%")
        print(f"{LOG_PREFIX} 📦 Chunks Processed: {len(m.chunk_metrics)}")
        print(f"{LOG_PREFIX} {'='*70}")
        
        # Chunk-by-chunk breakdown
        if m.chunk_metrics:
            print(f"{LOG_PREFIX} 📈 CHUNK BREAKDOWN:")
            for chunk_metric in m.chunk_metrics:
                rate = chunk_metric.files_processed / chunk_metric.total_time if chunk_metric.total_time > 0 else 0
                print(f"{LOG_PREFIX}   Chunk {chunk_metric.chunk_id}: "
                      f"{chunk_metric.files_processed} files, "
                      f"{chunk_metric.chunks_created} chunks, "
                      f"{chunk_metric.total_time:.1f}s ({rate:.1f} files/s)")
        
        logger.info(f"{LOG_PREFIX} Chunked processing completed successfully: "
                   f"{m.total_files_processed} files, {m.total_chunks_created} chunks, "
                   f"{m.total_processing_time:.1f}s")

# Global instance
chunked_processor = ChunkedFileProcessor()
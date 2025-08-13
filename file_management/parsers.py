# file_management/parsers.py - Optimized with Multi-Processing
import json
import csv
import os
import ast
import logging
import ijson
from typing import Generator, List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, Manager, Queue
import multiprocessing as mp
from functools import partial
import time

# Create prefixed logger for this file
logger = logging.getLogger(__name__)
LOG_PREFIX = "[Parser]"

def _parse_file_worker(args: Tuple[str, str, int]) -> List[Dict[str, Any]]:
    """
    Worker function for multiprocessing file parsing
    Must be at module level for pickle compatibility
    """
    file_path, file_type, chunk_size = args
    parser = DocumentParser(max_chunk_size=1000)
    
    try:
        # Collect all chunks for this file
        all_records = []
        for chunk in parser.parse_file_stream(file_path, file_type, chunk_size):
            all_records.extend(chunk)
        return all_records
    except Exception as e:
        logger.error(f"Worker error parsing {file_path}: {e}")
        return []

def _parse_chunk_worker(args: Tuple[str, str, int, int]) -> List[Dict[str, Any]]:
    """
    Worker function for parsing file chunks in parallel
    """
    file_path, file_type, start_pos, chunk_size = args
    
    try:
        # This is a simplified chunk parser - would need file-type specific logic
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(start_pos)
            content = f.read(chunk_size)
            
            if content:
                return [{
                    "content": content,
                    "tags": [file_type, file_path, f"chunk_{start_pos}"]
                }]
    except Exception as e:
        logger.error(f"Chunk worker error for {file_path} at {start_pos}: {e}")
    
    return []

class DocumentParser:
    """Multi-process document parser with streaming support"""
    
    def __init__(self, max_chunk_size: int = 1000, max_workers: int = None):
        self.max_chunk_size = max_chunk_size
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.parsers = {
            'txt': self._parse_txt_stream,
            'csv': self._parse_csv_stream,
            'json': self._parse_json_stream,
            'py': self._parse_python_stream
        }
    
    def parse_files_parallel(self, file_paths: List[str], file_types: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parse multiple files in parallel using process pool
        
        Args:
            file_paths: List of file paths to parse
            file_types: Optional list of file types (auto-detected if None)
            
        Returns:
            Dictionary mapping file paths to their parsed records
        """
        if not file_paths:
            logger.info(f"{LOG_PREFIX} No files to parse, returning empty results")
            return {}
        
        logger.info(f"{LOG_PREFIX} Starting parallel parsing of {len(file_paths)} files with {self.max_workers} workers")
        parse_start = time.time()
        
        # Prepare arguments for workers
        if file_types is None:
            file_types = [None] * len(file_paths)
        
        # Determine file types
        type_start = time.time()
        worker_args = []
        for file_path, file_type in zip(file_paths, file_types):
            if not file_type:
                ext = Path(file_path).suffix[1:].lower()
                file_type = ext if ext in self.parsers else 'txt'
            
            worker_args.append((file_path, file_type, 100))  # chunk_size
        
        type_time = time.time() - type_start
        logger.info(f"{LOG_PREFIX} File type determination completed in {type_time:.3f}s")
        
        results = {}
        files_processed = 0
        total_records = 0
        
        # Use ProcessPoolExecutor for CPU-intensive parsing
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            logger.info(f"{LOG_PREFIX} Submitting {len(worker_args)} parsing tasks to process pool")
            
            # Submit all files for processing
            future_to_file = {
                executor.submit(_parse_file_worker, args): args[0] 
                for args in worker_args
            }
            
            logger.info(f"{LOG_PREFIX} All parsing tasks submitted, waiting for completion")
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                files_processed += 1
                
                try:
                    parsed_records = future.result()
                    results[file_path] = parsed_records
                    total_records += len(parsed_records)
                    
                    logger.debug(f"{LOG_PREFIX} Parsed {len(parsed_records)} records from {file_path}")
                    
                    # Log progress every 10 files
                    if files_processed % 10 == 0:
                        elapsed = time.time() - parse_start
                        files_per_sec = files_processed / elapsed if elapsed > 0 else 0
                        logger.info(f"{LOG_PREFIX} Progress: {files_processed}/{len(file_paths)} files, "
                                   f"{total_records} records, {files_per_sec:.1f} files/s")
                                   
                except Exception as e:
                    logger.error(f"{LOG_PREFIX} Error parsing {file_path}: {e}")
                    results[file_path] = []
        
        parse_time = time.time() - parse_start
        files_per_sec = len(file_paths) / parse_time if parse_time > 0 else 0
        records_per_sec = total_records / parse_time if parse_time > 0 else 0
        
        logger.info(f"{LOG_PREFIX} Parallel parsing completed: {len(file_paths)} files, {total_records} records, "
                   f"{parse_time:.2f}s, {files_per_sec:.1f} files/s, {records_per_sec:.1f} records/s")
        
        return results
    
    def parse_large_file_parallel(self, file_path: str, file_type: Optional[str] = None) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Parse large files using parallel chunk processing
        
        Args:
            file_path: Path to large file
            file_type: File type (auto-detected if None)
            
        Yields:
            Chunks of parsed records
        """
        if not os.path.exists(file_path):
            return
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return
        
        # Determine file type
        if not file_type:
            ext = Path(file_path).suffix[1:].lower()
            file_type = ext if ext in self.parsers else 'txt'
        
        # For very large files, use chunk-based parallel processing
        if file_size > 50 * 1024 * 1024:  # 50MB threshold
            yield from self._parse_large_file_chunks(file_path, file_type, file_size)
        else:
            # For smaller files, use regular streaming
            yield from self.parse_file_stream(file_path, file_type, 100)
    
    def _parse_large_file_chunks(self, file_path: str, file_type: str, file_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Parse large files by splitting into chunks and processing in parallel"""
        chunk_size = 1024 * 1024  # 1MB chunks
        num_chunks = (file_size + chunk_size - 1) // chunk_size
        
        # Create worker arguments for each chunk
        worker_args = [
            (file_path, file_type, i * chunk_size, min(chunk_size, file_size - i * chunk_size))
            for i in range(num_chunks)
        ]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(_parse_chunk_worker, args): i 
                for i, args in enumerate(worker_args)
            }
            
            # Collect results in order
            chunk_results = {}
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    chunk_records = future.result()
                    chunk_results[chunk_index] = chunk_records
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index} of {file_path}: {e}")
                    chunk_results[chunk_index] = []
            
            # Yield results in order
            for i in range(num_chunks):
                if i in chunk_results and chunk_results[i]:
                    yield chunk_results[i]
    
    def parse_file_stream(
        self, 
        file_path: str, 
        file_type: Optional[str] = None,
        chunk_size: int = 100
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Stream parse single file (fallback for process-pool compatibility)
        """
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.warning(f"File is missing or empty: {file_path}")
            return
        
        # Determine file type
        if not file_type:
            ext = Path(file_path).suffix[1:].lower()
            file_type = ext if ext in self.parsers else 'txt'
        
        # Get appropriate parser
        parser_func = self.parsers.get(file_type, self._parse_txt_stream)
        
        try:
            yield from parser_func(file_path, chunk_size)
        except Exception as e:
            logger.error(f"Error parsing {file_path} as {file_type}: {e}")
            # Fallback to text parsing
            yield from self._parse_txt_stream(file_path, chunk_size)
    
    def _parse_txt_stream(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream text files by paragraphs or blocks"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                current_chunk = []
                buffer = ""
                
                for line in f:
                    buffer += line
                    
                    # Break on empty lines (paragraph boundaries) or buffer size
                    if line.strip() == "" or len(buffer) > 8000:
                        if buffer.strip():
                            current_chunk.append({
                                "content": buffer.strip(),
                                "tags": ["txt_paragraph", file_path]
                            })
                            buffer = ""
                            
                            if len(current_chunk) >= chunk_size:
                                yield current_chunk
                                current_chunk = []
                    
                    # Safety net for very long lines
                    if len(buffer) > 10000:
                        current_chunk.append({
                            "content": buffer[:10000],
                            "tags": ["txt_long_line", file_path]
                        })
                        buffer = buffer[10000:]
                
                # Add remaining content
                if buffer.strip():
                    current_chunk.append({
                        "content": buffer.strip(),
                        "tags": ["txt_final", file_path]
                    })
                
                if current_chunk:
                    yield current_chunk
                    
        except Exception as e:
            logger.error(f"Text parsing failed for {file_path}: {e}")
            # Fallback: read in blocks
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    current_chunk = []
                    while True:
                        content = f.read(8192)
                        if not content:
                            break
                        current_chunk.append({
                            "content": content,
                            "tags": ["txt_fallback_block", file_path]
                        })
                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []
                    if current_chunk:
                        yield current_chunk
            except Exception as fallback_e:
                logger.error(f"Text fallback also failed for {file_path}: {fallback_e}")
    
    def _parse_csv_stream(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream CSV files row by row with parallel-friendly approach"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Detect CSV dialect
                sample = f.read(10240).replace('\0', '')
                f.seek(0)
                
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    has_header = sniffer.has_header(sample)
                except Exception:
                    dialect = 'excel'
                    has_header = False
                
                reader = csv.reader(f, dialect)
                
                # Process headers
                headers = []
                if has_header:
                    try:
                        raw_headers = next(reader)
                        headers = self._clean_csv_headers(raw_headers)
                    except Exception as e:
                        logger.warning(f"Error reading CSV headers in {file_path}: {e}")
                
                if not headers:
                    # Determine column count from first row
                    try:
                        first_row = next(reader)
                        headers = [f"column_{i}" for i in range(len(first_row))]
                        # Put the row back by seeking and re-reading
                        f.seek(0)
                        reader = csv.reader(f, dialect)
                        if has_header:
                            next(reader, None)
                    except Exception:
                        headers = ["column_0"]
                
                current_chunk = []
                base_tags = headers + [file_path]
                
                for i, row in enumerate(reader):
                    try:
                        cleaned_values = [str(cell).strip() if cell is not None else "" for cell in row]
                        
                        # Create row dictionary
                        row_dict = {}
                        for j, value in enumerate(cleaned_values):
                            if j < len(headers):
                                key = headers[j]
                            else:
                                key = f"extra_col_{j}"
                            row_dict[key] = value
                        
                        content = json.dumps(row_dict, ensure_ascii=False)
                        current_chunk.append({
                            "content": content,
                            "tags": base_tags.copy()
                        })
                        
                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []
                            
                    except Exception as e:
                        logger.warning(f"Error processing CSV row {i} in {file_path}: {e}")
                
                if current_chunk:
                    yield current_chunk
                    
        except Exception as e:
            logger.error(f"CSV parsing failed for {file_path}: {e}")
            # Fallback to text
            yield from self._parse_txt_stream(file_path, chunk_size)
    
    def _clean_csv_headers(self, raw_headers: List[str]) -> List[str]:
        """Clean and deduplicate CSV headers"""
        processed_headers = []
        seen = set()
        
        for h in raw_headers:
            clean_h = str(h).strip() if h is not None else ""
            original_h = clean_h
            counter = 1
            
            while clean_h in seen:
                clean_h = f"{original_h}_{counter}"
                counter += 1
            
            seen.add(clean_h)
            processed_headers.append(clean_h)
        
        return processed_headers
    
    def _parse_json_stream(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream JSON files with support for arrays and objects"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Check if it's a JSON array
                first_char = f.read(1)
                base_tags = [file_path]
                
                if first_char == '[':
                    # Stream JSON array using ijson
                    f.seek(0)
                    yield from self._parse_json_array_stream(f, base_tags, chunk_size)
                else:
                    # Single JSON object or scalar
                    f.seek(0)
                    data = json.load(f)
                    yield from self._parse_json_data(data, base_tags, chunk_size)
                    
        except Exception as e:
            logger.error(f"JSON parsing failed for {file_path}: {e}")
            yield from self._parse_txt_stream(file_path, chunk_size)
    
    def _parse_json_array_stream(self, file_obj, base_tags: List[str], chunk_size: int):
        """Stream large JSON arrays using ijson"""
        try:
            parser = ijson.parse(file_obj)
            current_chunk = []
            current_object = {}
            object_depth = 0
            in_object = False
            key = None
            
            for prefix, event, value in parser:
                if event == 'start_map':
                    if object_depth == 0:
                        current_object = {}
                    object_depth += 1
                    in_object = True
                    
                elif event == 'end_map':
                    object_depth -= 1
                    if object_depth == 0 and in_object:
                        content = json.dumps(current_object, ensure_ascii=False)
                        tags = base_tags.copy() + list(current_object.keys())[:10]
                        
                        current_chunk.append({
                            "content": content,
                            "tags": tags
                        })
                        
                        in_object = False
                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []
                            
                elif in_object and event == 'map_key':
                    key = value
                    
                elif in_object and event in ['string', 'number', 'boolean', 'null']:
                    current_object[key] = value
            
            if current_chunk:
                yield current_chunk
                
        except Exception as e:
            logger.error(f"JSON array streaming failed: {e}")
    
    def _parse_json_data(self, data: Any, base_tags: List[str], chunk_size: int):
        """Parse JSON data (non-streaming)"""
        records = []
        
        if isinstance(data, list):
            for item in data:
                content = json.dumps(item, ensure_ascii=False)
                item_tags = base_tags.copy()
                if isinstance(item, dict):
                    item_tags.extend(list(item.keys())[:10])
                records.append({"content": content, "tags": item_tags})
                
        elif isinstance(data, dict):
            content = json.dumps(data, ensure_ascii=False)
            tags = base_tags.copy() + list(data.keys())[:10]
            records.append({"content": content, "tags": tags})
            
        else:
            content = json.dumps(data, ensure_ascii=False)
            records.append({"content": content, "tags": base_tags.copy()})
        
        # Yield in chunks
        for i in range(0, len(records), chunk_size):
            yield records[i:i + chunk_size]
    
    def _parse_python_stream(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Parse Python files using AST with parallel-friendly approach"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code, filename=file_path)
            current_chunk = []
            parent_stack = []
            
            def process_node(node):
                nonlocal current_chunk
                
                if isinstance(node, ast.Module):
                    parent_stack.append("__main__")
                    
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    element_type = "class" if isinstance(node, ast.ClassDef) else "function"
                    element_name = node.name
                    
                    element_info = {
                        "type": element_type,
                        "name": element_name,
                        "full_name": ".".join(parent_stack + [element_name]),
                        "parent_path": list(parent_stack),
                        "line_number": getattr(node, 'lineno', -1),
                        "docstring": ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None
                    }
                    
                    # Add source segment if possible
                    try:
                        if hasattr(ast, 'get_source_segment'):
                            element_info["source_segment"] = ast.get_source_segment(source_code, node)
                    except Exception:
                        pass
                    
                    content = json.dumps(element_info, ensure_ascii=False)
                    current_chunk.append({
                        "content": content,
                        "tags": [element_type, file_path]
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
                    
                    # Process nested elements
                    parent_stack.append(element_name)
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            yield from process_node(child)
                    parent_stack.pop()
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = {
                        "type": "import",
                        "names": [],
                        "parent_path": list(parent_stack),
                        "line_number": getattr(node, 'lineno', -1)
                    }
                    
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_info["names"].append({
                                "name": alias.name,
                                "asname": alias.asname
                            })
                    elif isinstance(node, ast.ImportFrom):
                        import_info["module"] = node.module
                        for alias in node.names:
                            import_info["names"].append({
                                "name": alias.name,
                                "asname": alias.asname
                            })
                    
                    content = json.dumps(import_info, ensure_ascii=False)
                    current_chunk.append({
                        "content": content,
                        "tags": ["import", file_path]
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
            
            # Start processing
            yield from process_node(tree)
            
            if current_chunk:
                yield current_chunk
                
        except SyntaxError as e:
            logger.warning(f"Python syntax error in {file_path}: {e}")
            yield from self._parse_txt_stream(file_path, chunk_size)
            
        except Exception as e:
            logger.error(f"Python parsing failed for {file_path}: {e}")
            yield from self._parse_txt_stream(file_path, chunk_size)


class ParallelDocumentProcessor:
    """High-level coordinator for parallel document processing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.parser = DocumentParser(max_workers=self.max_workers)
    
    def process_files_batch(self, file_paths: List[str], batch_size: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process files in batches with optimal parallelization strategy
        
        Args:
            file_paths: List of file paths to process
            batch_size: Size of processing batches (auto-calculated if None)
            
        Returns:
            Dictionary mapping file paths to parsed records
        """
        if not file_paths:
            return {}
        
        # Calculate optimal batch size based on file count and workers
        if batch_size is None:
            batch_size = max(1, len(file_paths) // (self.max_workers * 2))
        
        all_results = {}
        
        # Process files in batches to manage memory
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
            
            batch_results = self.parser.parse_files_parallel(batch)
            all_results.update(batch_results)
            
            # Memory cleanup between batches
            import gc
            gc.collect()
        
        return all_results
    
    def process_mixed_files(self, file_paths: List[str], size_threshold: int = 50 * 1024 * 1024) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process mixed file sizes with adaptive strategy
        
        Args:
            file_paths: List of file paths
            size_threshold: Threshold for large file processing (bytes)
            
        Returns:
            Dictionary mapping file paths to parsed records
        """
        if not file_paths:
            return {}
        
        # Separate files by size
        small_files = []
        large_files = []
        
        for file_path in file_paths:
            try:
                file_size = os.path.getsize(file_path)
                if file_size > size_threshold:
                    large_files.append(file_path)
                else:
                    small_files.append(file_path)
            except Exception as e:
                logger.error(f"Error checking size of {file_path}: {e}")
                small_files.append(file_path)  # Default to small file processing
        
        results = {}
        
        # Process small files in parallel batches
        if small_files:
            logger.info(f"Processing {len(small_files)} small files in parallel")
            small_results = self.parser.parse_files_parallel(small_files)
            results.update(small_results)
        
        # Process large files individually with internal parallelization
        if large_files:
            logger.info(f"Processing {len(large_files)} large files with chunk parallelization")
            for file_path in large_files:
                try:
                    file_records = []
                    for chunk in self.parser.parse_large_file_parallel(file_path):
                        file_records.extend(chunk)
                    results[file_path] = file_records
                    logger.info(f"Processed large file {file_path}: {len(file_records)} records")
                except Exception as e:
                    logger.error(f"Error processing large file {file_path}: {e}")
                    results[file_path] = []
        
        return results

# Global parser instances
document_parser = DocumentParser()
parallel_processor = ParallelDocumentProcessor()
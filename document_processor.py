# document_processor.py - Unified Document Processing
import os
import json
import csv
import ast
import logging
import asyncio
import hashlib
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
from core_config import config

# Conditional imports with fallbacks
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    logging.warning("ijson not available, falling back to standard JSON parsing for large files")

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

logger = logging.getLogger(__name__)

@dataclass
class FileRecord:
    """Record of a processed file with comprehensive metadata"""
    filepath: str
    size: int
    mtime: float
    checksum: str
    processed_at: str
    records_count: int = 0
    processing_time: float = 0.0
    file_type: str = ""
    error_message: str = ""

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

class TokenOverlapProcessor:
    """Handles intelligent token overlap between document chunks"""
    
    def __init__(self):
        self.overlap_tokens = config.FILE_CONFIG['token_overlap_size']
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation - 1 token ≈ 4 characters for most languages"""
        return len(text) // 4
    
    @staticmethod
    def get_token_boundaries(text: str, target_tokens: int) -> int:
        """Find character position closest to target token count at word boundaries"""
        estimated_chars = target_tokens * 4
        if estimated_chars >= len(text):
            return len(text)
        
        target_pos = min(estimated_chars, len(text))
        
        # Look for word boundary within ±50 characters
        for offset in range(50):
            # Check forward
            pos = target_pos + offset
            if pos < len(text) and text[pos].isspace():
                return pos
            
            # Check backward
            pos = target_pos - offset
            if pos > 0 and text[pos].isspace():
                return pos
        
        return target_pos
    
    def create_overlapping_chunks(self, content: str, chunk_size_tokens: int = 500) -> List[str]:
        """Split content into overlapping chunks with intelligent boundaries"""
        if not content.strip():
            return []
        
        total_tokens = self.estimate_tokens(content)
        
        if total_tokens <= chunk_size_tokens:
            return [content]
        
        chunks = []
        start_pos = 0
        
        while start_pos < len(content):
            end_tokens = chunk_size_tokens
            end_pos = self.get_token_boundaries(content[start_pos:], end_tokens)
            end_pos += start_pos
            
            chunk = content[start_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            if end_pos >= len(content):
                break
            
            # Calculate overlap for next chunk
            overlap_chars = self.get_token_boundaries(
                content[start_pos:end_pos], self.overlap_tokens
            )
            
            start_pos = max(start_pos + 1, end_pos - overlap_chars)
        
        return chunks

class FileTracker:
    """Comprehensive file tracking with change detection and state persistence"""
    
    def __init__(self, tracker_file: str = "processed_files.json"):
        self.tracker_file = tracker_file
        self.processed_files: Dict[str, FileRecord] = {}
        self._lock = threading.RLock()
        self.load_tracker()
        self._loaded = True
        
    def load_tracker(self):
        """Load existing file tracking data with error recovery"""
        with self._lock:
            if self._try_load_tracker_file(self.tracker_file):
                return
                
            # Try backup recovery
            backup_file = f"{self.tracker_file}.backup"
            if os.path.exists(backup_file):
                logger.warning("Main tracker file corrupted, attempting backup recovery")
                if self._try_load_tracker_file(backup_file):
                    try:
                        os.replace(backup_file, self.tracker_file)
                        logger.info("Successfully recovered from backup")
                        return
                    except Exception as e:
                        logger.error(f"Failed to restore backup: {e}")
            
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
            logger.info(f"Loaded {len(self.processed_files)} processed file records")
            return True
        except Exception as e:
            logger.error(f"Error loading file tracker from {filepath}: {e}")
            return False
    
    def save_tracker(self, force: bool = False):
        """Save file tracking data with atomic operations"""
        with self._lock:
            try:
                data = {
                    path: asdict(record) 
                    for path, record in self.processed_files.items()
                }
                
                # Write to temporary file first
                temp_file = f"{self.tracker_file}.tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Create backup
                if os.path.exists(self.tracker_file):
                    backup_file = f"{self.tracker_file}.backup"
                    try:
                        os.replace(self.tracker_file, backup_file)
                    except Exception as e:
                        logger.warning(f"Could not create backup: {e}")
                
                # Atomically replace
                os.replace(temp_file, self.tracker_file)
                logger.debug(f"Saved {len(self.processed_files)} file records")
                
            except Exception as e:
                logger.error(f"Error saving file tracker: {e}")
                # Cleanup
                temp_file = f"{self.tracker_file}.tmp"
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def _calculate_file_checksum(self, filepath: str, chunk_size: int = 8192) -> str:
        """Calculate MD5 checksum for change detection"""
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
        """Determine if file needs processing with detailed reasoning"""
        try:
            abs_path = os.path.abspath(filepath)
            
            if not os.path.exists(abs_path):
                return False, "file_not_found"
            
            stat = os.stat(abs_path)
            current_size = stat.st_size
            current_mtime = stat.st_mtime
            
            with self._lock:
                if abs_path not in self.processed_files:
                    return True, "new_file"
                
                previous_record = self.processed_files[abs_path]
            
            # Size change check
            if current_size != previous_record.size:
                return True, "size_changed"
            
            # Modification time check (1 second tolerance)
            if abs(current_mtime - previous_record.mtime) > 1.0:
                return True, "time_changed"
            
            # Content change check for non-empty files
            if current_size > 0:
                current_checksum = self._calculate_file_checksum(abs_path)
                if current_checksum and current_checksum != previous_record.checksum:
                    return True, "content_changed"
            
            return False, "already_processed"
            
        except Exception as e:
            logger.error(f"Error checking file {filepath}: {e}")
            return True, "error_checking"
    
    def mark_file_processed(self, filepath: str, records_count: int = 0, 
                          processing_time: float = 0.0, file_type: str = "",
                          error_message: str = ""):
        """Mark file as processed with comprehensive metadata"""
        try:
            abs_path = os.path.abspath(filepath)
            
            if not os.path.exists(abs_path):
                logger.warning(f"Cannot mark non-existent file: {abs_path}")
                return
            
            stat = os.stat(abs_path)
            checksum = self._calculate_file_checksum(abs_path)
            
            record = FileRecord(
                filepath=abs_path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                checksum=checksum,
                processed_at=datetime.now().isoformat(),
                records_count=records_count,
                processing_time=processing_time,
                file_type=file_type,
                error_message=error_message
            )
            
            with self._lock:
                self.processed_files[abs_path] = record
            
            logger.debug(f"Marked file processed: {abs_path} ({records_count} records)")
            
        except Exception as e:
            logger.error(f"Error marking file processed {filepath}: {e}")
    
    def batch_filter_files(self, file_paths: List[str]) -> Tuple[List[str], FilterStats]:
        """Filter files efficiently with detailed statistics"""
        files_to_process = []
        stats = FilterStats(total_files=len(file_paths))
        
        for filepath in file_paths:
            should_process, reason = self.should_process_file(filepath)
            if should_process:
                files_to_process.append(filepath)
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
                if reason == "already_processed":
                    stats.already_processed += 1
                elif reason == "file_not_found":
                    stats.error_files += 1
                stats.skip_reasons[filepath] = reason
        
        stats.files_to_process = len(files_to_process)
        stats.files_skipped = len(file_paths) - len(files_to_process)
        
        return files_to_process, stats
    
    def get_processed_files_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        with self._lock:
            if not self.processed_files:
                return {
                    "total_files": 0,
                    "total_records": 0,
                    "total_size_mb": 0,
                    "avg_records_per_file": 0,
                    "total_processing_time": 0,
                    "avg_time_per_file": 0
                }
            
            total_files = len(self.processed_files)
            total_records = sum(r.records_count for r in self.processed_files.values())
            total_size = sum(r.size for r in self.processed_files.values())
            total_time = sum(r.processing_time for r in self.processed_files.values())
            
            return {
                "total_files": total_files,
                "total_records": total_records,
                "total_size_mb": total_size / (1024 * 1024),
                "avg_records_per_file": total_records / total_files,
                "total_processing_time": total_time,
                "avg_time_per_file": total_time / total_files,
                "avg_time_per_record": total_time / total_records if total_records > 0 else 0
            }
    
    def cleanup_missing_files(self) -> List[str]:
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

class DocumentParser:
    """Unified document parser supporting multiple formats with streaming"""
    
    def __init__(self):
        self.token_processor = TokenOverlapProcessor()
        self.supported_extensions = config.FILE_CONFIG['supported_extensions']
        self.max_file_size = config.FILE_CONFIG['max_file_size']
    
    def stream_parse_file(self, file_path: str, file_type: str = None, 
                         chunk_size: int = None) -> Generator[List[Dict[str, Any]], None, None]:
        """Unified streaming parser with intelligent format detection"""
        chunk_size = chunk_size or config.CHUNK_SIZES['file_processing']
        
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.warning(f"File not found or empty: {file_path}")
            return
        
        # Validate file size
        if os.path.getsize(file_path) > self.max_file_size:
            logger.warning(f"File too large: {file_path}")
            yield [{"content": f"File too large: {file_path}", "tags": ["error", "oversized"]}]
            return
        
        # Determine file type
        ext = os.path.splitext(file_path)[1][1:].lower()
        file_type = file_type or ext or 'txt'
        
        try:
            if file_type == 'csv':
                yield from self._parse_csv(file_path, chunk_size)
            elif file_type == 'json':
                yield from self._parse_json(file_path, chunk_size)
            elif file_type == 'py':
                yield from self._parse_python(file_path, chunk_size)
            else:
                yield from self._parse_text(file_path, chunk_size)
                
        except Exception as e:
            logger.error(f"Error parsing {file_path} ({file_type}): {e}", exc_info=True)
            # Fallback to text parsing
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read(1000000)  # Limit size
                yield [{"content": content, "tags": [f"{file_type}_fallback", file_path]}]
            except Exception as fallback_e:
                logger.error(f"Fallback parsing failed for {file_path}: {fallback_e}")
                yield [{"content": f"Failed to parse: {file_path}", "tags": ["parse_error"]}]
    
    def _parse_csv(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Enhanced CSV parser with dialect detection and large cell handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                # Sample for dialect detection
                sample = f.read(10240).replace('\0', '')
                f.seek(0)
                
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    has_header = sniffer.has_header(sample)
                except Exception:
                    logger.debug(f"CSV sniffing failed for {file_path}, using defaults")
                    dialect = 'excel'
                    has_header = False
                
                reader = csv.reader(f, dialect)
                
                # Process headers
                headers = []
                if has_header:
                    try:
                        raw_headers = next(reader)
                        seen = set()
                        for h in raw_headers:
                            clean_h = str(h).strip() if h is not None else ""
                            original_h = clean_h
                            counter = 1
                            while clean_h in seen:
                                clean_h = f"{original_h}_{counter}"
                                counter += 1
                            seen.add(clean_h)
                            headers.append(clean_h)
                    except Exception as e:
                        logger.warning(f"Error reading CSV header: {e}")
                        headers = []
                
                if not headers:
                    # Determine column count from first row
                    try:
                        f.seek(0)
                        temp_reader = csv.reader(f, dialect)
                        if has_header:
                            next(temp_reader, None)
                        first_row = next(temp_reader, None)
                        if first_row:
                            headers = [f"column_{i}" for i in range(len(first_row))]
                        f.seek(0)
                        reader = csv.reader(f, dialect)
                        if has_header:
                            next(reader, None)
                    except Exception:
                        headers = ["column_0"]
                
                current_chunk = []
                base_tags = headers + [file_path, "csv"]
                
                for i, row in enumerate(reader):
                    try:
                        cleaned_values = [str(cell).strip() if cell else "" for cell in row]
                        row_dict = {}
                        
                        for j, value in enumerate(cleaned_values):
                            key = headers[j] if j < len(headers) else f"extra_col_{j}"
                            row_dict[key] = value
                        
                        content = json.dumps(row_dict, ensure_ascii=False)
                        
                        # Handle large content with overlap
                        if self.token_processor.estimate_tokens(content) > 500:
                            chunks = self.token_processor.create_overlapping_chunks(content, 500)
                            for chunk_idx, chunk in enumerate(chunks):
                                chunk_tags = base_tags + [f"csv_large_row_{i}_chunk_{chunk_idx}"]
                                current_chunk.append({
                                    "content": chunk,
                                    "tags": chunk_tags,
                                    "file_path": file_path,
                                    "chunk_index": len(current_chunk)
                                })
                        else:
                            current_chunk.append({
                                "content": content,
                                "tags": base_tags.copy(),
                                "file_path": file_path,
                                "chunk_index": len(current_chunk)
                            })
                        
                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []
                            
                    except Exception as e:
                        logger.warning(f"CSV row {i} error in {file_path}: {e}")
                
                if current_chunk:
                    yield current_chunk
                    
        except Exception as e:
            logger.warning(f"CSV parsing failed for {file_path}: {e}")
            # Fallback to text parsing
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000000)
                yield [{"content": content, "tags": ["csv_fallback", file_path]}]
    
    def _parse_json(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Enhanced JSON parser with streaming support for large files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                first_char = f.read(1)
                
                # Use streaming parser for large JSON arrays
                if first_char == '[' and HAS_IJSON:
                    f.seek(0)
                    parser = ijson.parse(f)
                    current_chunk = []
                    current_object = {}
                    object_depth = 0
                    in_object = False
                    key = None
                    base_tags = [file_path, "json"]
                    
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
                                tags = base_tags.copy() + [str(k) for k in current_object.keys()]
                                
                                # Handle large objects with overlap
                                if self.token_processor.estimate_tokens(content) > 500:
                                    chunks = self.token_processor.create_overlapping_chunks(content, 500)
                                    for chunk_idx, chunk in enumerate(chunks):
                                        chunk_tags = tags + [f"json_large_object_chunk_{chunk_idx}"]
                                        current_chunk.append({
                                            "content": chunk,
                                            "tags": chunk_tags,
                                            "file_path": file_path,
                                            "chunk_index": len(current_chunk)
                                        })
                                else:
                                    current_chunk.append({
                                        "content": content,
                                        "tags": tags,
                                        "file_path": file_path,
                                        "chunk_index": len(current_chunk)
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
                else:
                    # Standard JSON parsing
                    f.seek(0)
                    data = json.load(f)
                    base_tags = [file_path, "json"]
                    records_to_yield = []
                    
                    if isinstance(data, list):
                        for item in data:
                            try:
                                content = json.dumps(item, ensure_ascii=False)
                                item_tags = base_tags.copy()
                                if isinstance(item, dict):
                                    item_tags.extend([str(k) for k in item.keys()])
                                
                                # Apply overlap splitting if needed
                                if self.token_processor.estimate_tokens(content) > 500:
                                    chunks = self.token_processor.create_overlapping_chunks(content, 500)
                                    for chunk_idx, chunk in enumerate(chunks):
                                        chunk_tags = item_tags + [f"json_array_item_chunk_{chunk_idx}"]
                                        records_to_yield.append({
                                            "content": chunk,
                                            "tags": chunk_tags,
                                            "file_path": file_path,
                                            "chunk_index": len(records_to_yield)
                                        })
                                else:
                                    records_to_yield.append({
                                        "content": content,
                                        "tags": item_tags,
                                        "file_path": file_path,
                                        "chunk_index": len(records_to_yield)
                                    })
                            except Exception as e:
                                logger.warning(f"JSON array item error in {file_path}: {e}")
                        
                        # Yield in chunks
                        for i in range(0, len(records_to_yield), chunk_size):
                            yield records_to_yield[i:i + chunk_size]
                    
                    elif isinstance(data, dict):
                        content = json.dumps(data, ensure_ascii=False)
                        tags = base_tags.copy() + [str(k) for k in data.keys()]
                        
                        # Apply overlap splitting if needed
                        if self.token_processor.estimate_tokens(content) > 500:
                            chunks = self.token_processor.create_overlapping_chunks(content, 500)
                            chunk_records = []
                            for chunk_idx, chunk in enumerate(chunks):
                                chunk_tags = tags + [f"json_object_chunk_{chunk_idx}"]
                                chunk_records.append({
                                    "content": chunk,
                                    "tags": chunk_tags,
                                    "file_path": file_path,
                                    "chunk_index": chunk_idx
                                })
                            yield chunk_records
                        else:
                            yield [{
                                "content": content,
                                "tags": tags,
                                "file_path": file_path,
                                "chunk_index": 0
                            }]
                    else:
                        # Scalar value
                        content = json.dumps(data, ensure_ascii=False)
                        yield [{
                            "content": content,
                            "tags": base_tags.copy(),
                            "file_path": file_path,
                            "chunk_index": 0
                        }]
                        
        except Exception as e:
            logger.warning(f"JSON parsing failed for {file_path}: {e}")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000000)
                yield [{"content": content, "tags": [file_path, "json_fallback"]}]
    
    def _parse_python(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Enhanced Python parser with AST analysis and overlap handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                source_code = f.read()
            
            file_context = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path)
            }
            
            tree = ast.parse(source_code, filename=file_path)
            current_chunk = []
            parent_stack = []
            
            def get_source_segment(node):
                """Extract source code for AST node with fallbacks"""
                try:
                    if hasattr(ast, 'get_source_segment'):
                        segment = ast.get_source_segment(source_code, node)
                        if segment:
                            return segment
                    
                    # Manual line extraction fallback
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        lines = source_code.splitlines()
                        start_line = max(0, node.lineno - 1)
                        end_line = min(len(lines), getattr(node, 'end_lineno', node.lineno))
                        return '\n'.join(lines[start_line:end_line])
                    
                    return f"# Source for {getattr(node, 'name', type(node).__name__)} not available"
                except Exception as e:
                    logger.debug(f"Source segment extraction failed: {e}")
                    return f"# Error extracting source: {e}"
            
            def process_node(node):
                """Recursively process AST nodes with overlap splitting"""
                nonlocal current_chunk
                
                element_info = None
                source_segment = get_source_segment(node)
                
                if isinstance(node, ast.Module):
                    element_info = {
                        "type": "module",
                        "name": "__main__",
                        "full_name": file_context['file_path'],
                        "parent_path": [],
                        "source_segment": source_segment,
                        "line_number": 1,
                        "docstring": ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None
                    }
                    parent_stack.append("__main__")
                
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    element_type = "class" if isinstance(node, ast.ClassDef) else (
                        "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                    )
                    element_name = node.name
                    
                    element_info = {
                        "type": element_type,
                        "name": element_name,
                        "full_name": ".".join(parent_stack + [element_name]),
                        "parent_path": list(parent_stack),
                        "source_segment": source_segment,
                        "line_number": getattr(node, 'lineno', -1),
                        "docstring": ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None
                    }
                    
                    content = json.dumps(element_info, ensure_ascii=False)
                    
                    # Apply overlap splitting for large code segments
                    if self.token_processor.estimate_tokens(content) > 500:
                        chunks = self.token_processor.create_overlapping_chunks(content, 500)
                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_tags = [element_type, file_context['file_path'], f"{element_name}_chunk_{chunk_idx}"]
                            current_chunk.append({
                                "content": chunk,
                                "tags": chunk_tags,
                                "file_path": file_path,
                                "chunk_index": len(current_chunk)
                            })
                    else:
                        current_chunk.append({
                            "content": content,
                            "tags": [element_type, file_context['file_path']],
                            "file_path": file_path,
                            "chunk_index": len(current_chunk)
                        })
                    
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
                    
                    # Process nested elements
                    parent_stack.append(element_name)
                    for child_node in ast.iter_child_nodes(node):
                        if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            yield from process_node(child_node)
                    parent_stack.pop()
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = {
                        "type": "import",
                        "names": [],
                        "parent_path": list(parent_stack),
                        "line_number": getattr(node, 'lineno', -1),
                        "source_segment": source_segment
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
                        "tags": ["import", file_context['file_path']],
                        "file_path": file_path,
                        "chunk_index": len(current_chunk)
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
            
            # Process the AST
            yield from process_node(tree)
            
            if current_chunk:
                yield current_chunk
                
        except SyntaxError as e:
            logger.warning(f"Python syntax error in {file_path}: {e}")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000000)
                yield [{"content": content, "tags": ["py_syntax_error", file_path]}]
        except Exception as e:
            logger.warning(f"Python parsing failed for {file_path}: {e}")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000000)
                yield [{"content": content, "tags": ["py_parse_fallback", file_path]}]
    
    def _parse_text(self, file_path: str, chunk_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """Enhanced text parser with intelligent overlap"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                full_content = f.read()
            
            if not full_content.strip():
                return
            
            # Split into overlapping chunks
            overlapping_chunks = self.token_processor.create_overlapping_chunks(full_content, 500)
            
            current_chunk = []
            for chunk_idx, chunk_content in enumerate(overlapping_chunks):
                if chunk_content.strip():
                    current_chunk.append({
                        "content": chunk_content.strip(),
                        "tags": ["txt_chunk", file_path, f"chunk_{chunk_idx}"],
                        "file_path": file_path,
                        "chunk_index": chunk_idx
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
            
            if current_chunk:
                yield current_chunk
                
        except Exception as e:
            logger.warning(f"Text parsing failed for {file_path}: {e}")
            # Fallback with basic overlap
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    current_chunk = []
                    block_size = 8192
                    overlap_size = 512
                    previous_block = ""
                    block_idx = 0
                    
                    while True:
                        content = f.read(block_size)
                        if not content:
                            break
                        
                        if previous_block:
                            full_block = previous_block[-overlap_size:] + content
                        else:
                            full_block = content
                        
                        current_chunk.append({
                            "content": full_block,
                            "tags": ["txt_fallback_block", file_path, f"block_{block_idx}"],
                            "file_path": file_path,
                            "chunk_index": block_idx
                        })
                        
                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []
                        
                        previous_block = content
                        block_idx += 1
                    
                    if current_chunk:
                        yield current_chunk
            except Exception as fallback_e:
                logger.error(f"Fallback text parsing failed for {file_path}: {fallback_e}")
                yield [{"content": f"Failed to parse: {file_path}", "tags": ["parse_error"]}]

class FileProcessor:
    """High-level file processing coordinator"""
    
    def __init__(self):
        self.parser = DocumentParser()
        self.tracker = FileTracker()
        self.supported_extensions = config.FILE_CONFIG['supported_extensions']
    
    async def process_file(self, file_path: str, progress_callback=None) -> Tuple[str, bool, int, float]:
        """Process a single file with comprehensive tracking"""
        start_time = time.time()
        file_type = os.path.splitext(file_path)[1][1:].lower()
        processed_records = 0
        
        def report_progress(stage, current=0, total=1, message=""):
            if progress_callback and callable(progress_callback):
                try:
                    progress_callback(file_path, stage, current, total, message)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
        
        try:
            # Check if processing needed
            should_process, reason = self.tracker.should_process_file(file_path)
            if not should_process:
                report_progress("skipped", 1, 1, f"Skipped: {reason}")
                return (file_path, True, 0, 0.0)
            
            report_progress("parsing", 0, 1, "Starting file parsing")
            
            # Parse file and collect records
            all_records = []
            chunk_count = 0
            
            for chunk in self.parser.stream_parse_file(file_path, file_type, 
                                                     config.CHUNK_SIZES['file_processing']):
                if not chunk:
                    continue
                
                chunk_count += 1
                all_records.extend(chunk)
                processed_records = len(all_records)
                
                report_progress("parsing", chunk_count, chunk_count + 1, 
                              f"Parsed {processed_records} records")
            
            processing_time = time.time() - start_time
            
            # Mark as processed
            self.tracker.mark_file_processed(
                file_path, 
                processed_records, 
                processing_time, 
                file_type
            )
            
            report_progress("complete", 1, 1, f"Processed {processed_records} records")
            logger.info(f"Successfully processed {file_path}: {processed_records} records in {processing_time:.2f}s")
            
            return (file_path, True, processed_records, processing_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            self.tracker.mark_file_processed(
                file_path, 
                0, 
                processing_time, 
                file_type, 
                error_msg
            )
            
            report_progress("error", 0, 1, f"Error: {error_msg[:100]}")
            logger.error(f"Error processing {file_path}: {e}")
            
            return (file_path, False, 0, processing_time)
    
    def discover_files(self, directory_path: str, recursive: bool = True) -> Tuple[List[str], FilterStats]:
        """Discover and filter files for processing"""
        try:
            all_files = []
            
            if recursive:
                for root, dirs, files in os.walk(directory_path):
                    # Filter out hidden directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') and 
                             d not in ['__pycache__', 'node_modules', '.git']]
                    
                    for file in files:
                        if not file.startswith('.'):
                            file_path = os.path.join(root, file)
                            ext = os.path.splitext(file)[1][1:].lower()
                            if ext in self.supported_extensions:
                                all_files.append(file_path)
            else:
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)
                    if os.path.isfile(item_path) and not item.startswith('.'):
                        ext = os.path.splitext(item)[1][1:].lower()
                        if ext in self.supported_extensions:
                            all_files.append(item_path)
            
            # Filter using tracker
            files_to_process, filter_stats = self.tracker.batch_filter_files(all_files)
            
            logger.info(f"File discovery complete: {len(all_files)} total, "
                       f"{len(files_to_process)} need processing")
            
            return files_to_process, filter_stats
            
        except Exception as e:
            logger.error(f"File discovery failed for {directory_path}: {e}")
            return [], FilterStats()
    
    def get_file_browser_path(self) -> Optional[str]:
        """Get file/folder path using GUI browser with CLI fallback"""
        try:
            if HAS_GUI:
                root = tk.Tk()
                root.withdraw()
                folder_path = filedialog.askdirectory(title="Select Folder to Process")
                root.destroy()
                
                if folder_path:
                    logger.info(f"Folder selected via GUI: {folder_path}")
                    return folder_path
            
            # CLI fallback
            return self._get_cli_path()
            
        except Exception as e:
            logger.warning(f"GUI browser failed: {e}")
            return self._get_cli_path()
    
    def _get_cli_path(self) -> Optional[str]:
        """Get path via CLI input with validation"""
        print("\n" + "="*60)
        print("FILE/FOLDER SELECTION")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Enter absolute path")
            print("2. Enter relative path from current directory")
            print("3. Use current directory")
            print("4. Cancel")
            
            current_dir = os.getcwd()
            print(f"\nCurrent directory: {current_dir}")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                path = input("Enter absolute path: ").strip()
                path = os.path.expanduser(path)
            elif choice == "2":
                relative = input("Enter relative path: ").strip()
                path = os.path.abspath(os.path.join(current_dir, relative))
            elif choice == "3":
                path = current_dir
            elif choice == "4":
                return None
            else:
                print("Invalid option.")
                continue
            
            if self._validate_path(path):
                return path
            else:
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
    
    def _validate_path(self, path: str) -> bool:
        """Validate file/directory path"""
        try:
            if not os.path.exists(path):
                print(f"Error: Path does not exist: {path}")
                return False
            
            if not os.access(path, os.R_OK):
                print(f"Error: No read permission: {path}")
                return False
            
            if os.path.isfile(path):
                # Validate single file
                ext = os.path.splitext(path)[1][1:].lower()
                if ext not in self.supported_extensions:
                    print(f"Error: Unsupported file type: {ext}")
                    print(f"Supported types: {', '.join(self.supported_extensions)}")
                    return False
                print(f"File validated: {path}")
                return True
            
            elif os.path.isdir(path):
                # Count supported files
                file_count = 0
                for root, _, files in os.walk(path):
                    for file in files:
                        ext = os.path.splitext(file)[1][1:].lower()
                        if ext in self.supported_extensions:
                            file_count += 1
                
                if file_count == 0:
                    print(f"Warning: No supported files found in {path}")
                    print(f"Supported types: {', '.join(self.supported_extensions)}")
                    continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
                    return continue_anyway == 'y'
                
                print(f"Directory validated: {file_count} supported files found")
                return True
            
            else:
                print(f"Error: Path is neither file nor directory: {path}")
                return False
                
        except Exception as e:
            print(f"Error validating path: {e}")
            return False

# Global instances
file_processor = FileProcessor()
file_tracker = file_processor.tracker
document_parser = file_processor.parser

# Legacy compatibility functions
def stream_parse_file(file_path: str, file_type: str = None, chunk_size: int = None):
    """Legacy compatibility function"""
    return document_parser.stream_parse_file(file_path, file_type, chunk_size)

def generate_file_paths(folder_path: str):
    """Legacy compatibility function"""
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1][1:].lower()
            if ext in config.FILE_CONFIG['supported_extensions']:
                yield file_path

def count_files(folder_path: str) -> int:
    """Legacy compatibility function"""
    return len(list(generate_file_paths(folder_path)))
# parse_documents.py
import json
import csv
import os
import ast
import logging
import re
from config import Config
from typing import List, Generator, Dict, Any, Optional

# FIXED: Conditional import with fallback for ijson
try:
    import ijson  # For streaming JSON parsing
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    logging.warning("ijson not available, falling back to standard JSON parsing for large files")

logger = logging.getLogger(__name__)

class TokenOverlapProcessor:
    """Handles token overlap between chunks"""
    
    OVERLAP_TOKENS = 32
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation - 1 token ≈ 4 characters for most languages"""
        return len(text) // 4
    
    @staticmethod
    def get_token_boundaries(text: str, target_tokens: int) -> int:
        """Find character position closest to target token count"""
        estimated_chars = target_tokens * 4
        if estimated_chars >= len(text):
            return len(text)
        
        # Try to break at word boundaries near the target
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
        
        # Fall back to exact position if no word boundary found
        return target_pos
    
    @classmethod
    def create_overlapping_chunks(cls, content: str, chunk_size_tokens: int = 500) -> List[str]:
        """
        Split content into overlapping chunks
        
        Args:
            content: Text content to split
            chunk_size_tokens: Target size for each chunk in tokens
            
        Returns:
            List of overlapping text chunks
        """
        if not content.strip():
            return []
        
        total_tokens = cls.estimate_tokens(content)
        
        # If content is smaller than one chunk, return as-is
        if total_tokens <= chunk_size_tokens:
            return [content]
        
        chunks = []
        start_pos = 0
        
        while start_pos < len(content):
            # Calculate end position for this chunk
            end_tokens = chunk_size_tokens
            end_pos = cls.get_token_boundaries(content[start_pos:], end_tokens)
            end_pos += start_pos
            
            # Extract chunk
            chunk = content[start_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)
            
            # If this is the last chunk, break
            if end_pos >= len(content):
                break
            
            # Calculate overlap for next chunk
            overlap_chars = cls.get_token_boundaries(content[start_pos:end_pos], cls.OVERLAP_TOKENS)
            
            # Start next chunk with overlap
            start_pos = max(start_pos + 1, end_pos - overlap_chars)
        
        return chunks

# --- Wrapper function ---
def stream_parse_file(file_path, file_type=None, chunk_size=None):
    """Unified streaming parser with token overlap"""
    chunk_size = chunk_size or Config.CHUNK_SIZES['file_processing']
    
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return

    ext = os.path.splitext(file_path)[1][1:].lower()
    file_type = file_type or ext or 'txt'
    
    try:
        if file_type == 'csv':
            yield from stream_parse_csv(file_path, chunk_size)
        elif file_type == 'json':
            yield from stream_parse_json(file_path, chunk_size)
        elif file_type == 'py':
            yield from stream_parse_py(file_path, chunk_size)
        else:
            yield from stream_parse_txt(file_path, chunk_size)

    except Exception as e:
        logger.error(f"Error during streaming parse of {file_path} ({file_type}): {e}", exc_info=True)
        # Final fallback: yield the file content as a single text chunk
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(1000000) # Limit size
            yield [{"content": content, "tags": [f"{file_type}_fallback", file_path]}]
        except Exception as fallback_e:
            logger.error(f"Fallback text read also failed for {file_path}: {fallback_e}")
            yield [{"content": f"Failed to parse or read file: {file_path}", "tags": ["parse_error", file_path]}]

def stream_parse_csv(file_path, chunk_size=100):
    """
    Streaming CSV parser with token overlap for large cells
    Each row becomes a record, but large cells are split with overlap
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(10240).replace('\0', '')
            f.seek(0)
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                has_header = sniffer.has_header(sample)
            except Exception as sniff_e:
                logger.debug(f"CSV sniffing failed for {file_path}, using default 'excel': {sniff_e}")
                dialect = 'excel'
                has_header = False

            reader = csv.reader(f, dialect)
            raw_headers = []
            if has_header:
                try:
                    raw_headers = next(reader)
                except Exception as header_e:
                    logger.warning(f"Error reading CSV header in {file_path}: {header_e}")
                    raw_headers = []

            # Process headers: clean and ensure uniqueness
            processed_headers = []
            if raw_headers:
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
            else:
                # Determine number of columns from the first data row
                try:
                    f_temp = open(file_path, 'r', encoding='utf-8', errors='replace')
                    temp_reader = csv.reader(f_temp, dialect)
                    if has_header:
                        next(temp_reader, None)
                    first_data_row = next(temp_reader, None)
                    f_temp.close()
                    if first_data_row is not None:
                        num_cols = len(first_data_row)
                    else:
                        num_cols = 0
                except Exception as peek_e:
                    logger.warning(f"Error peeking for CSV column count in {file_path}: {peek_e}")
                    num_cols = 0
                processed_headers = [f"column_{i}" for i in range(num_cols)]
                f.seek(0)
                reader = csv.reader(f, dialect)
                if has_header:
                    next(reader, None)

            current_chunk = []
            base_tags = processed_headers + [file_path]

            for i, row in enumerate(reader):
                try:
                    cleaned_values = [str(cell).strip() if cell is not None else "" for cell in row]
                    row_dict = {}
                    for j, value in enumerate(cleaned_values):
                        if j < len(processed_headers):
                            key = processed_headers[j]
                        else:
                            key = f"extra_col_{j}"
                        row_dict[key] = value

                    # Create base content
                    content = json.dumps(row_dict, ensure_ascii=False)
                    
                    # Check if content is very large and needs overlap splitting
                    if TokenOverlapProcessor.estimate_tokens(content) > 500:
                        # Split large CSV rows with overlap
                        chunks = TokenOverlapProcessor.create_overlapping_chunks(content, 500)
                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_tags = base_tags + [f"csv_large_row_{i}_chunk_{chunk_idx}"]
                            current_chunk.append({
                                "content": chunk,
                                "tags": chunk_tags
                            })
                    else:
                        current_chunk.append({
                            "content": content,
                            "tags": base_tags.copy()
                        })

                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
                except Exception as e:
                    logger.warning(f"Row {i} error in {file_path}: {str(e)}")

            if current_chunk:
                yield current_chunk
    except Exception as e:
        logger.warning(f"CSV stream parse failed for {file_path}: {str(e)}")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(1000000)
            yield [{"content": content, "tags": ["csv_fallback_text_streaming", file_path]}]

def stream_parse_json(file_path, chunk_size=100):
    """
    Streaming JSON parser with token overlap for large objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            first_char = f.read(1)
            if first_char == '[' and HAS_IJSON:
                f.seek(0)
                parser = ijson.parse(f)
                current_chunk = []
                current_object = {}
                object_depth = 0
                in_object = False
                key = None
                base_tags = [file_path]
                
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
                            
                            # Check if object is large and needs overlap splitting
                            if TokenOverlapProcessor.estimate_tokens(content) > 500:
                                chunks = TokenOverlapProcessor.create_overlapping_chunks(content, 500)
                                for chunk_idx, chunk in enumerate(chunks):
                                    chunk_tags = tags + [f"json_large_object_chunk_{chunk_idx}"]
                                    current_chunk.append({"content": chunk, "tags": chunk_tags})
                            else:
                                current_chunk.append({"content": content, "tags": tags})
                            
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
                # Single JSON object or scalar, or ijson not available
                f.seek(0)
                data = json.load(f)
                base_tags = [file_path]
                records_to_yield = []
                
                if isinstance(data, list):
                    for item in data:
                        try:
                            content = json.dumps(item, ensure_ascii=False)
                            item_tags = base_tags.copy()
                            if isinstance(item, dict):
                                item_tags.extend([str(k) for k in item.keys()])
                            
                            # Apply overlap splitting if needed
                            if TokenOverlapProcessor.estimate_tokens(content) > 500:
                                chunks = TokenOverlapProcessor.create_overlapping_chunks(content, 500)
                                for chunk_idx, chunk in enumerate(chunks):
                                    chunk_tags = item_tags + [f"json_array_item_chunk_{chunk_idx}"]
                                    records_to_yield.append({"content": chunk, "tags": chunk_tags})
                            else:
                                records_to_yield.append({"content": content, "tags": item_tags})
                        except Exception as item_e:
                            logger.warning(f"Error serializing item in JSON array {file_path}: {item_e}")

                    # Yield records in chunks
                    for i in range(0, len(records_to_yield), chunk_size):
                         yield records_to_yield[i:i + chunk_size]

                elif isinstance(data, dict):
                    content = json.dumps(data, ensure_ascii=False)
                    tags = base_tags.copy() + [str(k) for k in data.keys()]
                    
                    # Apply overlap splitting if needed
                    if TokenOverlapProcessor.estimate_tokens(content) > 500:
                        chunks = TokenOverlapProcessor.create_overlapping_chunks(content, 500)
                        chunk_records = []
                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_tags = tags + [f"json_object_chunk_{chunk_idx}"]
                            chunk_records.append({"content": chunk, "tags": chunk_tags})
                        yield chunk_records
                    else:
                        yield [{"content": content, "tags": tags}]
                else: # Scalar
                    content = json.dumps(data, ensure_ascii=False)
                    tags = base_tags.copy()
                    yield [{"content": content, "tags": tags}]
    except Exception as e:
        logger.warning(f"JSON stream parse failed for {file_path}: {str(e)}")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(1000000)
            yield [{"content": content, "tags": [file_path, "json_fallback_text"]}]

def stream_parse_py(file_path, chunk_size=100):
    """
    Streaming Python file parser with token overlap for large functions/classes
    """
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
            """Get the source code segment for a node with overlap handling"""
            try:
                if hasattr(ast, 'get_source_segment'):
                    segment = ast.get_source_segment(source_code, node)
                    if segment:
                        return segment
                
                # Fallback: extract lines manually
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    lines = source_code.splitlines()
                    start_line = max(0, node.lineno - 1)
                    end_line = min(len(lines), getattr(node, 'end_lineno', node.lineno))
                    return '\n'.join(lines[start_line:end_line])
                
                return f"# Source for {getattr(node, 'name', type(node).__name__)} not available"
            except Exception as e:
                logger.debug(f"Could not get source segment for node in {file_path}: {e}")
                return f"# Error getting source segment: {e}"

        def process_node(node):
            """Recursively process AST nodes with overlap splitting"""
            nonlocal current_chunk

            element_info = None
            source_segment = get_source_segment(node)
            source_lines = source_code.splitlines()

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
                element_type = "class" if isinstance(node, ast.ClassDef) else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
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

                # Convert to JSON content
                content = json.dumps(element_info, ensure_ascii=False)
                
                # Apply overlap splitting if the source segment is large
                if TokenOverlapProcessor.estimate_tokens(content) > 500:
                    chunks = TokenOverlapProcessor.create_overlapping_chunks(content, 500)
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_tags = [element_type, file_context['file_path'], f"{element_name}_chunk_{chunk_idx}"]
                        current_chunk.append({"content": chunk, "tags": chunk_tags})
                else:
                    current_chunk.append({
                        "content": content,
                        "tags": [element_type, file_context['file_path']]
                    })

                # Check if chunk is full and yield
                if len(current_chunk) >= chunk_size:
                    yield current_chunk
                    current_chunk = []

                # Manage parent stack for nested elements
                parent_stack.append(element_name)
                # Process children (methods inside class, nested functions etc.)
                for child_node in ast.iter_child_nodes(node):
                     if isinstance(child_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                         yield from process_node(child_node)
                parent_stack.pop()

            # Add logic for other important nodes if needed (e.g., imports for context)
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
                     "tags": ["import", file_context['file_path']]
                 })
                 if len(current_chunk) >= chunk_size:
                     yield current_chunk
                     current_chunk = []

        # Start processing from the root (Module)
        yield from process_node(tree)

        # Yield any remaining items in the current chunk after traversal
        if current_chunk:
            yield current_chunk

    except SyntaxError as se:
        logger.warning(f"Python syntax error in {file_path}: {se}")
        # Fallback: Treat as text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
             content = f.read(1000000)
             yield [{"content": content, "tags": ["py_syntax_error", file_path]}]
    except FileNotFoundError:
        logger.warning(f"Python file not found: {file_path}")
        yield [{"content": f"File not found: {file_path}", "tags": ["py_file_not_found", file_path]}]
    except Exception as e:
        logger.warning(f"Python parse failed for {file_path}: {str(e)}", exc_info=True)
        # Fallback: treat as text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(1000000)
            yield [{"content": content, "tags": ["py_parse_fallback", file_path]}]

def stream_parse_txt(file_path, chunk_size=100):
    """
    Streaming TXT parser with token overlap between chunks
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Read entire file content
            full_content = f.read()
        
        if not full_content.strip():
            return
        
        # Split content into overlapping chunks
        overlapping_chunks = TokenOverlapProcessor.create_overlapping_chunks(full_content, 500)
        
        current_chunk = []
        for chunk_idx, chunk_content in enumerate(overlapping_chunks):
            if chunk_content.strip():
                current_chunk.append({
                    "content": chunk_content.strip(),
                    "tags": ["txt_chunk", file_path, f"chunk_{chunk_idx}"]
                })
                
                if len(current_chunk) >= chunk_size:
                    yield current_chunk
                    current_chunk = []
        
        # Yield any remaining items
        if current_chunk:
            yield current_chunk

    except Exception as e:
        logger.warning(f"TXT parse failed for {file_path}: {str(e)}")
        # Fallback: read in fixed-size blocks with overlap
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
                    
                    # Add overlap from previous block
                    if previous_block:
                        full_block = previous_block[-overlap_size:] + content
                    else:
                        full_block = content
                    
                    current_chunk.append({
                        "content": full_block, 
                        "tags": ["txt_fallback_block", file_path, f"block_{block_idx}"]
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
                    
                    previous_block = content
                    block_idx += 1
                
                if current_chunk:
                    yield current_chunk
        except Exception as fallback_e:
            logger.error(f"Fallback TXT parsing also failed for {file_path}: {fallback_e}")
            yield [{"content": f"Failed to parse file: {file_path}", "tags": ["parse_error", file_path]}]
# file_management/parsers.py
import json
import csv
import os
import ast
import logging
import ijson
from typing import Generator, List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentParser:
    """Unified document parser with streaming support"""
    
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.parsers = {
            'txt': self._parse_txt_stream,
            'csv': self._parse_csv_stream,
            'json': self._parse_json_stream,
            'py': self._parse_python_stream
        }
    
    def parse_file_stream(
        self, 
        file_path: str, 
        file_type: Optional[str] = None,
        chunk_size: int = 100
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Unified streaming parser that determines file type and delegates to appropriate parser
        
        Args:
            file_path: Path to file
            file_type: Override file type detection
            chunk_size: Number of records per chunk
            
        Yields:
            Chunks of parsed records
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
        """Stream CSV files row by row"""
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
        """Parse Python files using AST"""
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

# Global parser instance
document_parser = DocumentParser()
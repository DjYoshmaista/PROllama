# parse_documents.py
import json
import csv
import os
import ast
import logging
from config import Config

# FIXED: Conditional import with fallback for ijson
try:
    import ijson  # For streaming JSON parsing
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    logging.warning("ijson not available, falling back to standard JSON parsing for large files")

logger = logging.getLogger(__name__)

# --- Wrapper function ---
def stream_parse_file(file_path, file_type=None, chunk_size=None):
    """Unified streaming parser"""
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
    """Streaming CSV parser with generators.
       Stores each row as a JSON object string to preserve structure.
       Tags include the original headers/column names for that file.
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
                    # Use a temporary reader to peek
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
                # Reset main reader state
                f.seek(0)
                reader = csv.reader(f, dialect)
                if has_header:
                    next(reader, None) # Skip header again

            current_chunk = []
            # Tags for all rows from this file will be the processed headers + file path
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

                    content = json.dumps(row_dict, ensure_ascii=False)
                    current_chunk.append({
                        "content": content,
                        "tags": base_tags.copy() # Include processed headers as tags
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
    """Streaming JSON parser with generators.
       Parses structure, captures hierarchy, and yields chunks of elements.
       Tags include top-level keys for objects.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # FIXED: Check if ijson is available before using it
            first_char = f.read(1)
            if first_char == '[' and HAS_IJSON:
                f.seek(0)
                parser = ijson.parse(f)
                current_chunk = []
                current_object = {}
                object_depth = 0
                in_object = False
                key = None
                # Base tags for items in this file
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
                            # Content is the serialized object
                            content = json.dumps(current_object, ensure_ascii=False)
                            # Tags are file path + object's top-level keys
                            tags = base_tags.copy() + [str(k) for k in current_object.keys()]
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
                # Reuse logic from non-streaming parse_json, but yield in chunks
                base_tags = [file_path]
                records_to_yield = []
                if isinstance(data, list):
                    for item in data:
                        try:
                            content = json.dumps(item, ensure_ascii=False)
                            item_tags = base_tags.copy()
                            if isinstance(item, dict):
                                item_tags.extend([str(k) for k in item.keys()])
                            records_to_yield.append({"content": content, "tags": item_tags})
                        except Exception as item_e:
                            logger.warning(f"Error serializing item in JSON array {file_path}: {item_e}")

                    # Yield records in chunks
                    for i in range(0, len(records_to_yield), chunk_size):
                         yield records_to_yield[i:i + chunk_size]

                elif isinstance(data, dict):
                    content = json.dumps(data, ensure_ascii=False)
                    tags = base_tags.copy() + [str(k) for k in data.keys()]
                    yield [{"content": content, "tags": tags}]
                else: # Scalar
                    content = json.dumps(data, ensure_ascii=False)
                    tags = base_tags.copy()
                    yield [{"content": content, "tags": tags}]
    except Exception as e:
        logger.warning(f"JSON stream parse failed for {file_path}: {str(e)}")
        # Fallback: treat as text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(1000000)
            yield [{"content": content, "tags": [file_path, "json_fallback_text"]}]

def stream_parse_py(file_path, chunk_size=100):
    """
    Streaming Python (.py) file parser using AST.
    Parses structure, captures hierarchy, and yields chunks of elements.
    Each element includes its parent path for tracing.
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
        # Stack to keep track of the current nesting (e.g., ['ClassName', 'method_name'])
        parent_stack = []

        def get_source_segment(node):
            """Get the source code segment for a node."""
            try:
                if hasattr(ast, 'get_source_segment'):
                    return ast.get_source_segment(source_code, node)
                else:
                    # Fallback: reconstruct or use basic info
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        lines = source_code.splitlines()
                        if 1 <= node.lineno <= len(lines):
                            sig_line = lines[node.lineno - 1] # lineno is 1-based
                            indent = ""
                            if sig_line.startswith((' ', '\t')):
                                 for char in sig_line:
                                     if char in (' ', '\t'):
                                         indent += char
                                     else:
                                         break
                            return f"{indent}# ... (code for {node.name}) ..."
                    return f"# Source segment for {type(node).__name__} not available"
            except Exception as e:
                logger.debug(f"Could not get source segment for node in {file_path}: {e}")
                return f"# Error getting source segment: {e}"

        def get_indentation(node, source_lines):
            """Determine the indentation string for a node."""
            try:
                if 1 <= node.lineno <= len(source_lines):
                    line = source_lines[node.lineno - 1]
                    indent = ""
                    for char in line:
                        if char in (' ', '\t'):
                            indent += char
                        else:
                            break
                    return indent
                return ""
            except:
                 return ""

        def process_node(node):
            """Recursively process AST nodes."""
            nonlocal current_chunk

            element_info = None
            source_segment = get_source_segment(node)
            source_lines = source_code.splitlines()

            if isinstance(node, ast.Module):
                # The module itself represents the file
                element_info = {
                    "type": "module",
                    "name": "__main__",
                    "full_name": file_context['file_path'],
                    "parent_path": [],
                    "source_segment": source_segment,
                    "line_number": 1,
                    "indentation": "",
                    "docstring": ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None
                }
                parent_stack.append("__main__")

            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                element_type = "class" if isinstance(node, ast.ClassDef) else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function")
                element_name = node.name

                # Determine indentation
                indent_str = get_indentation(node, source_lines)

                element_info = {
                    "type": element_type,
                    "name": element_name,
                    "full_name": ".".join(parent_stack + [element_name]),
                    "parent_path": list(parent_stack),
                    "source_segment": source_segment,
                    "line_number": getattr(node, 'lineno', -1),
                    "indentation": indent_str,
                    "docstring": ast.get_docstring(node) if hasattr(ast, 'get_docstring') else None
                }

                # Add to chunk
                current_chunk.append({
                    "content": json.dumps(element_info, ensure_ascii=False),
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
                 current_chunk.append({
                     "content": json.dumps(import_info, ensure_ascii=False),
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
    Streaming TXT parser with generators.
    Attempts to split text into meaningful chunks (e.g., paragraphs).
    Yields chunks of documents.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            current_chunk = []
            buffer = ""
            for line in f:
                buffer += line
                # Heuristic: Assume paragraphs are separated by blank lines
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
                
                # Safety net for extremely long lines
                if len(buffer) > 10000:
                    current_chunk.append({
                        "content": buffer.strip()[:10000],
                        "tags": ["txt_long_line", file_path]
                    })
                    buffer = buffer[10000:]

            # Add the last piece if it exists
            if buffer.strip():
                 current_chunk.append({
                    "content": buffer.strip(),
                    "tags": ["txt_final", file_path]
                })

            # Yield any remaining items in the current chunk
            if current_chunk:
                yield current_chunk

    except Exception as e:
        logger.warning(f"TXT parse failed for {file_path}: {str(e)}")
        # Fallback: read in fixed-size blocks
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            current_chunk = []
            while True:
                content = f.read(8192)
                if not content:
                    break
                current_chunk.append({"content": content, "tags": ["txt_fallback_block", file_path]})
                if len(current_chunk) >= chunk_size:
                    yield current_chunk
                    current_chunk = []
            if current_chunk:
                yield current_chunk
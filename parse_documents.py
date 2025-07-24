# parse_documents.py
import json
import csv
import os
import re
import logging
import ijson  # For streaming JSON parsing

logger = logging.getLogger()
MAX_CHUNK_SIZE = 1000  # Maximum records per chunk for streaming

def parse_file(file_path, file_type):
    try:
        if not os.path.exists(file_path):
            return []
        if os.path.getsize(file_path) == 0:
            return []
        
        if file_type == 'txt':
            return parse_txt(file_path)
        elif file_type == 'csv':
            return safe_parse_csv(file_path)
        elif file_type == 'json':
            return parse_json(file_path)
        else:
            return []
    except Exception as e:
        logger.error(f"Parse error ({file_path}): {e}")
        return []

def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read(1000000)  # Limit to 1MB
    return [{"content": content, "tags": []}]

def safe_parse_csv(file_path):
    """Robust CSV parsing with multiple fallback strategies"""
    try:
        # Attempt 1: Standard CSV parsing
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Skip problematic characters
            sample = f.read(10240).replace('\0', '')
            f.seek(0)
            
            # Try to detect dialect
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                has_header = sniffer.has_header(sample)
            except:
                dialect = 'excel'
                has_header = False
                
            reader = csv.reader(f, dialect)
            headers = []
            if has_header:
                try:
                    headers = next(reader)
                except:
                    headers = []
            
            data = []
            for i, row in enumerate(reader):
                try:
                    # Clean and convert all values to strings
                    cleaned_row = [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    content = " ".join(cleaned_row)
                    data.append({
                        "content": content,
                        "tags": headers.copy()
                    })
                except Exception as e:
                    logger.warning(f"Row {i} error in {file_path}: {str(e)}")
            return data
    except Exception as e:
        logger.warning(f"CSV parse failed for {file_path}: {str(e)}")
        # Final fallback: Treat as text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(1000000)
            return [{"content": content, "tags": ["csv_fallback"]}]

def parse_json(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    
    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, dict):
                content = json.dumps(item, ensure_ascii=False)
                tags = [str(k) for k in item.keys()]
            else:
                content = str(item)
                tags = []
            result.append({"content": content, "tags": tags})
        return result
    elif isinstance(data, dict):
        content = json.dumps(data, ensure_ascii=False)
        tags = [str(k) for k in data.keys()]
        return [{"content": content, "tags": tags}]
    else:
        return [{"content": str(data), "tags": []}]

def stream_parse_csv(file_path, chunk_size=100):
    """Streaming CSV parser with generators"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Skip problematic characters
            sample = f.read(10240).replace('\0', '')
            f.seek(0)
            
            # Try to detect dialect
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                has_header = sniffer.has_header(sample)
            except:
                dialect = 'excel'
                has_header = False
                
            reader = csv.reader(f, dialect)
            headers = []
            if has_header:
                try:
                    headers = next(reader)
                except:
                    headers = []
            
            current_chunk = []
            for i, row in enumerate(reader):
                try:
                    cleaned_row = [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    content = " ".join(cleaned_row)
                    current_chunk.append({
                        "content": content,
                        "tags": headers.copy()
                    })
                    
                    if len(current_chunk) >= chunk_size:
                        yield current_chunk
                        current_chunk = []
                except Exception as e:
                    logger.warning(f"Row {i} error in {file_path}: {str(e)}")
            
            if current_chunk:
                yield current_chunk
    except Exception as e:
        logger.warning(f"CSV parse failed for {file_path}: {str(e)}")
        # Fallback to text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            while True:
                content = f.read(8192)
                if not content:
                    break
                yield [{"content": content, "tags": ["csv_fallback"]}]

def stream_parse_json(file_path, chunk_size=100):
    """Streaming JSON parser with generators"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Handle JSON arrays
            if f.read(1) == '[':
                f.seek(0)
                parser = ijson.parse(f)
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
                            tags = list(current_object.keys())
                            current_chunk.append({"content": content, "tags": tags})
                            in_object = False
                            
                            if len(current_chunk) >= chunk_size:
                                yield current_chunk
                                current_chunk = []
                    elif in_object and event == 'map_key':
                        key = value
                    elif in_object and event in ['string', 'number', 'boolean']:
                        current_object[key] = value
                
                if current_chunk:
                    yield current_chunk
            else:
                # Single JSON object
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    for i in range(0, len(data), chunk_size):
                        chunk = data[i:i+chunk_size]
                        records = []
                        for item in chunk:
                            if isinstance(item, dict):
                                content = json.dumps(item, ensure_ascii=False)
                                tags = list(item.keys())
                            else:
                                content = str(item)
                                tags = []
                            records.append({"content": content, "tags": tags})
                        yield records
                elif isinstance(data, dict):
                    content = json.dumps(data, ensure_ascii=False)
                    tags = list(data.keys())
                    yield [{"content": content, "tags": tags}]
                else:
                    yield [{"content": str(data), "tags": []}]
    except Exception as e:
        logger.warning(f"JSON parse failed for {file_path}: {str(e)}")
        # Fallback: treat as text
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(1000000)
            yield [{"content": content, "tags": []}]
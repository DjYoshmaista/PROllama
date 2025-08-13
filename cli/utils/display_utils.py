# cli/utils/display_utils.py
from typing import Dict, List, Any
import textwrap

class DisplayUtils:
    """Utilities for formatting and displaying information"""
    
    @staticmethod
    def print_header(title: str, width: int = 60, char: str = "="):
        """Print a formatted header"""
        print(f"\n{char * width}")
        print(title.center(width))
        print(char * width)
    
    @staticmethod
    def print_subheader(title: str, width: int = 60, char: str = "-"):
        """Print a formatted subheader"""
        print(f"\n{char * width}")
        print(title)
        print(char * width)
    
    @staticmethod
    def print_section(title: str, content: str, indent: int = 2):
        """Print a formatted section with content"""
        print(f"\n{title}:")
        indented_content = textwrap.indent(content, " " * indent)
        print(indented_content)
    
    @staticmethod
    def print_key_value_pairs(data: Dict[str, Any], indent: int = 2):
        """Print key-value pairs in a formatted way"""
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            indented_key = " " * indent + f"{formatted_key}:"
            print(f"{indented_key} {value}")
    
    @staticmethod
    def print_numbered_list(items: List[str], start: int = 1):
        """Print a numbered list of items"""
        for i, item in enumerate(items, start):
            print(f"{i}. {item}")
    
    @staticmethod
    def print_bulleted_list(items: List[str], bullet: str = "•"):
        """Print a bulleted list of items"""
        for item in items:
            print(f"{bullet} {item}")
    
    @staticmethod
    def print_table(headers: List[str], rows: List[List[Any]], max_width: int = 20):
        """Print a simple table"""
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Limit column widths
        col_widths = [min(width, max_width) for width in col_widths]
        
        # Print headers
        header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
        print(header_row)
        print("-" * len(header_row))
        
        # Print rows
        for row in rows:
            formatted_row = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)
                    if len(cell_str) > max_width:
                        cell_str = cell_str[:max_width-3] + "..."
                    formatted_row.append(cell_str.ljust(col_widths[i]))
            print(" | ".join(formatted_row))
    
    @staticmethod
    def print_progress_bar(current: int, total: int, width: int = 50, prefix: str = "", suffix: str = ""):
        """Print a progress bar"""
        if total == 0:
            percent = 100
        else:
            percent = (current / total) * 100
        
        filled_length = int(width * current // total) if total > 0 else width
        bar = "█" * filled_length + "░" * (width - filled_length)
        
        print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="", flush=True)
    
    @staticmethod
    def print_status(message: str, status: str = "INFO", timestamp: bool = False):
        """Print a status message with optional timestamp"""
        from datetime import datetime
        
        status_symbols = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "PROCESSING": "⏳"
        }
        
        symbol = status_symbols.get(status.upper(), "•")
        
        if timestamp:
            time_str = datetime.now().strftime("%H:%M:%S")
            print(f"[{time_str}] {symbol} {message}")
        else:
            print(f"{symbol} {message}")
    
    @staticmethod
    def print_document_preview(doc_id: int, content: str, tags: List[str] = None, 
                             created_at: str = None, preview_length: int = 200):
        """Print a formatted document preview"""
        print(f"\n📄 Document ID: {doc_id}")
        
        if created_at:
            print(f"📅 Created: {created_at}")
        
        if tags:
            tags_str = ", ".join(tags[:5])  # Show first 5 tags
            if len(tags) > 5:
                tags_str += f" (+{len(tags) - 5} more)"
            print(f"🏷️  Tags: {tags_str}")
        
        # Content preview
        if len(content) > preview_length:
            preview = content[:preview_length] + "..."
        else:
            preview = content
        
        print(f"📝 Content: {preview}")
    
    @staticmethod
    def print_search_results(results: List[Dict[str, Any]]):
        """Print formatted search results"""
        if not results:
            print("No results found.")
            return
        
        DisplayUtils.print_header(f"Search Results ({len(results)} found)")
        
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            if 'similarity' in result:
                print(f"Similarity: {result['similarity']:.4f}")
            
            if 'id' in result:
                print(f"ID: {result['id']}")
            
            if 'content_preview' in result:
                print(f"Preview: {result['content_preview']}")
            
            if 'tags' in result and result['tags']:
                tags_str = ", ".join(result['tags'][:3])
                if len(result['tags']) > 3:
                    tags_str += f" (+{len(result['tags']) - 3} more)"
                print(f"Tags: {tags_str}")
    
    @staticmethod
    def clear_screen():
        """Clear the terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
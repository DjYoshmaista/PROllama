# security_utils.py
import os
import re
import logging
from pathlib import Path
from typing import Union, List, Optional

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

class InputValidator:
    """Validates user inputs to prevent security issues"""
    
    # Dangerous file extensions that should never be processed
    DANGEROUS_EXTENSIONS = {
        'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'vbe', 'js', 'jse',
        'ws', 'wsf', 'wsc', 'wsh', 'ps1', 'ps1xml', 'ps2', 'ps2xml', 'psc1',
        'psc2', 'msh', 'msh1', 'msh2', 'mshxml', 'msh1xml', 'msh2xml'
    }
    
    # Safe file extensions for processing
    SAFE_EXTENSIONS = {'txt', 'csv', 'json', 'py', 'md', 'rst', 'log'}
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Maximum path length
    MAX_PATH_LENGTH = 4096
    
    @classmethod
    def validate_file_path(cls, file_path: Union[str, Path]) -> str:
        """
        Validate file path for security issues
        
        Args:
            file_path: Path to validate
            
        Returns:
            Normalized absolute path
            
        Raises:
            SecurityError: If path is unsafe
        """
        if not file_path:
            raise SecurityError("Empty file path provided")
        
        # Convert to string and normalize
        path_str = str(file_path).strip()
        
        # Check path length
        if len(path_str) > cls.MAX_PATH_LENGTH:
            raise SecurityError(f"Path too long: {len(path_str)} characters")
        
        # Check for path traversal attempts
        if '..' in path_str or path_str.startswith('/'):
            # Allow absolute paths but validate them carefully
            pass
        
        try:
            # Normalize and resolve the path
            normalized_path = os.path.abspath(os.path.expanduser(path_str))
        except Exception as e:
            raise SecurityError(f"Invalid path format: {e}")
        
        # Check if path exists
        if not os.path.exists(normalized_path):
            raise SecurityError(f"Path does not exist: {normalized_path}")
        
        # Check file extension if it's a file
        if os.path.isfile(normalized_path):
            cls._validate_file_extension(normalized_path)
        else:
            raise SecurityError(f"Path is not a file: {normalized_path}")

        # Check file extension if it's a file
        if os.path.isfile(normalized_path):
            cls._validate_file_extension(normalized_path)
            cls._validate_file_size(normalized_path)
        
        return normalized_path
    
    @classmethod
    def _validate_file_extension(cls, file_path: str):
        """Validate file extension for safety"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip('.')
        
        if ext in cls.DANGEROUS_EXTENSIONS:
            raise SecurityError(f"Dangerous file extension: .{ext}")
        
        if ext not in cls.SAFE_EXTENSIONS:
            logger.warning(f"Unusual file extension: .{ext} for {file_path}")
    
    @classmethod
    def _validate_file_size(cls, file_path: str):
        """Validate file size"""
        try:
            size = os.path.getsize(file_path)
            if size > cls.MAX_FILE_SIZE:
                raise SecurityError(f"File too large: {size} bytes (max: {cls.MAX_FILE_SIZE})")
        except OSError as e:
            raise SecurityError(f"Cannot check file size: {e}")
    
    @classmethod
    def validate_directory_path(cls, dir_path: Union[str, Path]) -> str:
        """
        Validate directory path for security issues
        
        Args:
            dir_path: Directory path to validate
            
        Returns:
            Normalized absolute path
            
        Raises:
            SecurityError: If path is unsafe
        """
        if not dir_path:
            raise SecurityError("Empty directory path provided")
        
        path_str = str(dir_path).strip()
        
        # Check path length
        if len(path_str) > cls.MAX_PATH_LENGTH:
            raise SecurityError(f"Path too long: {len(path_str)} characters")
        
        try:
            # Normalize and resolve the path
            normalized_path = os.path.abspath(os.path.expanduser(path_str))
        except Exception as e:
            raise SecurityError(f"Invalid directory path format: {e}")
        
        # Check if directory exists
        if not os.path.exists(normalized_path):
            raise SecurityError(f"Directory does not exist: {normalized_path}")
        
        if not os.path.isdir(normalized_path):
            raise SecurityError(f"Path is not a directory: {normalized_path}")
        
        # Check read permissions
        if not os.access(normalized_path, os.R_OK):
            raise SecurityError(f"No read permission for directory: {normalized_path}")
        
        return normalized_path
    
    @classmethod
    def validate_content_string(cls, content: str, max_length: int = 1000000) -> str:
        """
        Validate content string for basic safety
        
        Args:
            content: Content string to validate
            max_length: Maximum allowed length
            
        Returns:
            Validated content string
            
        Raises:
            SecurityError: If content is unsafe
        """
        if not isinstance(content, str):
            raise SecurityError("Content must be a string")
        
        if len(content) > max_length:
            raise SecurityError(f"Content too long: {len(content)} characters (max: {max_length})")
        
        # Check for potential injection patterns (basic check)
        dangerous_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected in content: {pattern}")
        
        return content
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename for safe storage
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        if not filename:
            return "unnamed_file"
        
        # Remove directory separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        # Ensure it's not empty
        if not sanitized.strip():
            return "unnamed_file"
        
        return sanitized.strip()
    
    @classmethod
    def validate_tags_list(cls, tags: List[str], max_tags: int = 50, max_tag_length: int = 100) -> List[str]:
        """
        Validate list of tags
        
        Args:
            tags: List of tag strings
            max_tags: Maximum number of tags allowed
            max_tag_length: Maximum length per tag
            
        Returns:
            Validated and sanitized tags list
            
        Raises:
            SecurityError: If tags are invalid
        """
        if not isinstance(tags, list):
            raise SecurityError("Tags must be a list")
        
        if len(tags) > max_tags:
            raise SecurityError(f"Too many tags: {len(tags)} (max: {max_tags})")
        
        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                continue  # Skip non-string tags
            
            # Sanitize tag
            sanitized_tag = re.sub(r'[<>"\']', '', str(tag).strip())
            
            if len(sanitized_tag) > max_tag_length:
                sanitized_tag = sanitized_tag[:max_tag_length]
            
            if sanitized_tag and sanitized_tag not in validated_tags:
                validated_tags.append(sanitized_tag)
        
        return validated_tags

    @staticmethod
    def validate_text_content(content: str, max_length: int = 1000000) -> str:
        """Validate and sanitize text content"""
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        
        if len(content) == 0:
            raise ValueError("Content cannot be empty")
        
        if len(content) > max_length:
            raise ValueError(f"Content too long: {len(content)} > {max_length}")
        
        # Remove null bytes and other dangerous characters
        sanitized = content.replace('\x00', '')
        
        return sanitized
    
    @staticmethod
    def validate_database_identifier(identifier: str) -> str:
        """Validate database table/column identifiers"""
        if not isinstance(identifier, str):
            raise ValueError("Identifier must be a string")
        
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
            raise ValueError(f"Invalid database identifier: {identifier}")
        
        if len(identifier) > 63:  # PostgreSQL limit
            raise ValueError(f"Identifier too long: {identifier}")
        
        # Check for SQL keywords (basic check)
        sql_keywords = {
            'select', 'insert', 'update', 'delete', 'drop', 'create', 
            'alter', 'grant', 'revoke', 'union', 'join', 'where'
        }
        
        if identifier.lower() in sql_keywords:
            raise ValueError(f"Identifier cannot be SQL keyword: {identifier}")
        
        return identifier
    
    @staticmethod
    def validate_tags(tags) -> List[str]:
        """Validate and sanitize tags"""
        if not isinstance(tags, (list, tuple)):
            if isinstance(tags, str):
                tags = [tags]
            else:
                raise ValueError("Tags must be a list or string")
        
        validated_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                tag = str(tag)
            
            # Sanitize tag
            sanitized_tag = re.sub(r'[^\w\-_\.]', '_', tag)[:100]  # Limit length
            
            if sanitized_tag and sanitized_tag not in validated_tags:
                validated_tags.append(sanitized_tag)
        
        return validated_tags[:50]  # Limit number of tags
    
# Global validator instance
input_validator = InputValidator()

class PathSecurityValidator:
    """Validates file paths for security issues"""
    
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'\.\.\\',  # Windows directory traversal
        r'^/etc/',  # System directories (Unix)
        r'^/proc/',  # Process filesystem
        r'^/sys/',  # System filesystem
        r'^C:\\Windows\\',  # Windows system directory
        r'^C:\\Program Files\\',  # Windows program files
    ]
    
    @classmethod
    def validate_path(cls, file_path: str, allowed_extensions: Optional[List[str]] = None) -> bool:
        """
        Validate file path for security issues
        
        Args:
            file_path: Path to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            bool: True if path is safe
            
        Raises:
            SecurityError: If path contains dangerous patterns
        """
        try:
            # Normalize the path
            normalized_path = os.path.normpath(file_path)
            
            # Check for dangerous patterns
            for pattern in cls.DANGEROUS_PATTERNS:
                if re.search(pattern, normalized_path, re.IGNORECASE):
                    raise SecurityError(f"Dangerous path pattern detected: {pattern}")
            
            # Check for null bytes
            if '\x00' in file_path:
                raise SecurityError("Null byte detected in path")
            
            # Validate file extension if provided
            if allowed_extensions:
                ext = Path(file_path).suffix.lower().lstrip('.')
                if ext not in [e.lower() for e in allowed_extensions]:
                    raise SecurityError(f"File extension '{ext}' not allowed")
            
            # Check if path is within allowed directories (relative paths only)
            if os.path.isabs(normalized_path):
                # For absolute paths, ensure they don't access system directories
                abs_path = Path(normalized_path).resolve()
                
                # Block access to system directories
                dangerous_dirs = ['/etc', '/proc', '/sys', '/root', 
                                '/usr/bin', '/usr/sbin', '/bin', '/sbin']
                
                for dangerous_dir in dangerous_dirs:
                    if str(abs_path).startswith(dangerous_dir):
                        raise SecurityError(f"Access to system directory not allowed: {dangerous_dir}")
            
            return True
            
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            logger.error(f"Path validation error: {e}")
            raise SecurityError(f"Path validation failed: {e}")
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove dangerous characters
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        # Ensure it doesn't start with dangerous patterns
        if sanitized.startswith('.'):
            sanitized = '_' + sanitized[1:]
        
        return sanitized

class DatabaseSecurityMixin:
    """Security utilities for database operations"""
    
    @staticmethod
    def escape_like_pattern(pattern: str) -> str:
        """Escape LIKE pattern for PostgreSQL"""
        # Escape special LIKE characters
        escaped = pattern.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        return escaped
    
    @staticmethod
    def validate_limit_offset(limit: Optional[int] = None, offset: Optional[int] = None) -> tuple:
        """Validate and sanitize LIMIT/OFFSET parameters"""
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError("Limit must be a non-negative integer")
            if limit > 10000:  # Reasonable upper bound
                logger.warning(f"Large limit requested: {limit}, capping to 10000")
                limit = 10000
        
        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise ValueError("Offset must be a non-negative integer")
            if offset > 1000000:  # Reasonable upper bound
                raise ValueError("Offset too large")
        
        return limit, offset

# Global security validator instances
path_validator = PathSecurityValidator()
db_security = DatabaseSecurityMixin()
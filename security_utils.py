# security_utils.py - Security utilities for RAGdb
import os
import re
import logging
from pathlib import Path
from typing import Optional, List
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Custom security-related exception"""
    pass

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

class InputValidator:
    """Input validation utilities"""
    
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
input_validator = InputValidator()
db_security = DatabaseSecurityMixin()
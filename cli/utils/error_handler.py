# cli/utils/error_handler.py
import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling for CLI operations"""
    
    @staticmethod
    def handle_error(error: Exception, context: str = "", user_message: Optional[str] = None):
        """Handle an error with logging and user notification"""
        # Log the full error with stack trace
        logger.error(f"Error in {context}: {error}", exc_info=True)
        
        # Show user-friendly message
        if user_message:
            print(f"❌ {user_message}")
        else:
            print(f"❌ An error occurred: {str(error)}")
    
    @staticmethod
    def handle_validation_error(field_name: str, value: Any, expected: str):
        """Handle validation errors"""
        message = f"Invalid {field_name}: '{value}'. Expected {expected}."
        print(f"⚠️ {message}")
        logger.warning(f"Validation error - {message}")
    
    @staticmethod
    def handle_file_error(file_path: str, operation: str, error: Exception):
        """Handle file-related errors"""
        context = f"File {operation}: {file_path}"
        user_message = f"Failed to {operation} file '{file_path}': {str(error)}"
        ErrorHandler.handle_error(error, context, user_message)
    
    @staticmethod
    def handle_database_error(operation: str, error: Exception):
        """Handle database-related errors"""
        context = f"Database {operation}"
        user_message = f"Database operation failed: {str(error)}"
        ErrorHandler.handle_error(error, context, user_message)
    
    @staticmethod
    def handle_embedding_error(operation: str, error: Exception):
        """Handle embedding service errors"""
        context = f"Embedding {operation}"
        user_message = f"Embedding operation failed: {str(error)}"
        ErrorHandler.handle_error(error, context, user_message)
    
    @staticmethod
    def handle_network_error(operation: str, error: Exception):
        """Handle network-related errors"""
        context = f"Network {operation}"
        user_message = f"Network operation failed. Please check your connection: {str(error)}"
        ErrorHandler.handle_error(error, context, user_message)

def safe_execute(operation_name: str = "", user_message: Optional[str] = None):
    """Decorator for safe execution of operations with error handling"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = operation_name or f"{func.__name__}"
                ErrorHandler.handle_error(e, context, user_message)
                return None
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = operation_name or f"{func.__name__}"
                ErrorHandler.handle_error(e, context, user_message)
                return None
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'async' in str(func.__code__.co_flags):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class FileOperationError(Exception):
    """Custom exception for file operation errors"""
    pass

class DatabaseOperationError(Exception):
    """Custom exception for database operation errors"""
    pass

class EmbeddingServiceError(Exception):
    """Custom exception for embedding service errors"""
    pass

class NetworkOperationError(Exception):
    """Custom exception for network operation errors"""
    pass
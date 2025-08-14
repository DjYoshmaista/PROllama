# error_handling.py - Centralized error handling and logging
import logging
import traceback
import functools
import asyncio
import time
from typing import Any, Callable, Optional, Dict, Type
from contextlib import contextmanager
import psycopg2
import asyncpg

# Custom exceptions
class RAGdbError(Exception):
    """Base exception for RAGdb"""
    pass

class DatabaseError(RAGdbError):
    """Database-related errors"""
    pass

class ProcessingError(RAGdbError):
    """File processing errors"""
    pass

class EmbeddingError(RAGdbError):
    """Embedding generation errors"""
    pass

class ValidationError(RAGdbError):
    """Input validation errors"""
    pass

class ConfigurationError(RAGdbError):
    """Configuration-related errors"""
    pass

# Error mapping
POSTGRES_ERROR_MAPPING = {
    '23505': 'Duplicate key violation',
    '23503': 'Foreign key violation',
    '42P01': 'Table does not exist',
    '42703': 'Column does not exist',
    '53300': 'Too many connections',
    '57014': 'Query cancelled',
    '08003': 'Connection does not exist',
    '08006': 'Connection failure',
}

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def handle_database_error(self, error: Exception, operation: str) -> DatabaseError:
        """Handle database-specific errors"""
        if isinstance(error, (psycopg2.Error, asyncpg.PostgresError)):
            error_code = getattr(error, 'pgcode', None)
            error_msg = POSTGRES_ERROR_MAPPING.get(error_code, str(error))
            
            self.logger.error(f"Database error in {operation}: {error_msg} (code: {error_code})")
            return DatabaseError(f"{operation} failed: {error_msg}")
        
        self.logger.error(f"Unexpected database error in {operation}: {error}")
        return DatabaseError(f"{operation} failed: {str(error)}")
    
    def handle_processing_error(self, error: Exception, file_path: str) -> ProcessingError:
        """Handle file processing errors"""
        error_key = f"processing_{file_path}"
        self._track_error(error_key)
        
        self.logger.error(f"Processing error for {file_path}: {error}")
        return ProcessingError(f"Failed to process {file_path}: {str(error)}")
    
    def handle_embedding_error(self, error: Exception, text_preview: str = "") -> EmbeddingError:
        """Handle embedding generation errors"""
        preview = text_preview[:50] + "..." if len(text_preview) > 50 else text_preview
        
        self.logger.error(f"Embedding error for text '{preview}': {error}")
        return EmbeddingError(f"Failed to generate embedding: {str(error)}")
    
    def _track_error(self, error_key: str):
        """Track error frequency for rate limiting"""
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = time.time()
        
        # Log warning if error is frequent
        if self.error_counts[error_key] > 10:
            self.logger.warning(f"Frequent errors detected for {error_key}: {self.error_counts[error_key]} occurrences")
    
    def should_retry(self, error: Exception, attempt: int, max_attempts: int = 3) -> bool:
        """Determine if operation should be retried"""
        if attempt >= max_attempts:
            return False
        
        # Retry on transient database errors
        if isinstance(error, (psycopg2.OperationalError, asyncpg.ConnectionFailureError)):
            return True
        
        # Retry on timeout errors
        if isinstance(error, asyncio.TimeoutError):
            return True
        
        # Don't retry on validation or logic errors
        if isinstance(error, (ValidationError, ValueError, TypeError)):
            return False
        
        return True
    
    def get_retry_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay"""
        return min(base_delay * (2 ** attempt), 30.0)  # Max 30 seconds

# Global error handler
error_handler = ErrorHandler()

def handle_errors(
    default_return=None,
    reraise: bool = False,
    log_level: int = logging.ERROR,
    operation_name: Optional[str] = None
):
    """Decorator for standardized error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                op_name = operation_name or func.__name__
                logger = logging.getLogger(func.__module__)
                logger.log(log_level, f"Error in {op_name}: {e}", exc_info=True)
                
                if reraise:
                    raise
                return default_return
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                op_name = operation_name or func.__name__
                logger = logging.getLogger(func.__module__)
                logger.log(log_level, f"Error in {op_name}: {e}", exc_info=True)
                
                if reraise:
                    raise
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    backoff_factor: float = 2.0
):
    """Decorator for retrying failed operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    if not error_handler.should_retry(e, attempt, max_attempts):
                        break
                    
                    delay = error_handler.get_retry_delay(attempt, base_delay)
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s")
                    
                    await asyncio.sleep(delay)
            
            # All attempts failed
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        break
                    
                    if not error_handler.should_retry(e, attempt, max_attempts):
                        break
                    
                    delay = error_handler.get_retry_delay(attempt, base_delay)
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s")
                    
                    time.sleep(delay)
            
            # All attempts failed
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@contextmanager
def error_context(operation: str, **context):
    """Context manager for error handling with additional context"""
    logger = logging.getLogger(__name__)
    try:
        logger.debug(f"Starting operation: {operation}", extra=context)
        yield
        logger.debug(f"Completed operation: {operation}")
    except Exception as e:
        logger.error(f"Error in operation '{operation}': {e}", extra=context, exc_info=True)
        raise

class PerformanceMonitor:
    """Monitor performance and detect anomalies"""
    
    def __init__(self):
        self.operation_times: Dict[str, list] = {}
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Monitor operation performance"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._record_operation_time(operation_name, duration)
            
            # Log slow operations
            if duration > 10.0:  # More than 10 seconds
                self.logger.warning(f"Slow operation detected: {operation_name} took {duration:.2f}s")
    
    def _record_operation_time(self, operation: str, duration: float):
        """Record operation timing for analysis"""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        times = self.operation_times[operation]
        times.append(duration)
        
        # Keep only last 100 measurements
        if len(times) > 100:
            times.pop(0)
    
    def get_performance_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation"""
        times = self.operation_times.get(operation, [])
        if not times:
            return {}
        
        return {
            'count': len(times),
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'total': sum(times)
        }

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Utility functions
def log_system_info():
    """Log system information for debugging"""
    import psutil
    import sys
    
    logger = logging.getLogger(__name__)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    
    memory = psutil.virtual_memory()
    logger.info(f"Memory: {memory.available / (1024**3):.1f}GB available / {memory.total / (1024**3):.1f}GB total")
    
    disk = psutil.disk_usage('.')
    logger.info(f"Disk space: {disk.free / (1024**3):.1f}GB free / {disk.total / (1024**3):.1f}GB total")
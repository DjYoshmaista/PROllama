# cli/utils/__init__.py
"""
CLI Utils Package

This package contains utility modules for the CLI:
- input_utils: Utilities for handling and validating user input
- display_utils: Utilities for formatting and displaying information
- error_handler: Centralized error handling and custom exceptions

These utilities are used across different handlers to provide consistent
input handling, output formatting, and error management.
"""

from .input_utils import InputUtils
from .display_utils import DisplayUtils
from .error_handler import (
    ErrorHandler, 
    safe_execute,
    ValidationError,
    FileOperationError,
    DatabaseOperationError,
    EmbeddingServiceError,
    NetworkOperationError
)

__all__ = [
    'InputUtils',
    'DisplayUtils',
    'ErrorHandler',
    'safe_execute',
    'ValidationError',
    'FileOperationError', 
    'DatabaseOperationError',
    'EmbeddingServiceError',
    'NetworkOperationError'
]
# cli/handlers/__init__.py
"""
CLI Handlers Package

This package contains all the specialized handlers for different CLI operations:
- InferenceHandler: Handles question-answering and inference operations
- DataHandler: Handles manual data addition and management
- FileHandler: Handles file loading operations (single files and folders)
- DatabaseHandler: Handles database queries and management operations
- ConfigHandler: Handles system configuration and settings
- SystemHandler: Handles system information and monitoring

Each handler is responsible for a specific domain of functionality, keeping
the main CLI interface clean and organized.
"""

from .inference_handler import InferenceHandler
from .data_handler import DataHandler
from .file_handler import FileHandler
from .database_handler import DatabaseHandler
from .config_handler import ConfigHandler
from .system_handler import SystemHandler

__all__ = [
    'InferenceHandler',
    'DataHandler', 
    'FileHandler',
    'DatabaseHandler',
    'ConfigHandler',
    'SystemHandler'
]
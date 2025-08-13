# cli/__init__.py
"""
CLI Package for RAG Database System

This package provides a command-line interface for the RAG (Retrieval-Augmented Generation) 
database system. It's organized into several modules:

Main Interface:
- interface.py: Main entry point and CLI orchestration

Core Components:
- menu.py: Menu display and navigation
- cleanup.py: Resource cleanup and management

Handlers (cli/handlers/):
- inference_handler.py: Question-answering operations
- data_handler.py: Data addition and management
- file_handler.py: File loading operations
- database_handler.py: Database queries and management
- config_handler.py: System configuration
- system_handler.py: System information and monitoring

Utilities (cli/utils/):
- input_utils.py: Input handling and validation
- display_utils.py: Output formatting and display
- error_handler.py: Error handling and exceptions

Usage:
    from cli import cli
    await cli.run()
"""

from .interface import cli, CLIInterface
from .menu import MenuDisplay
from .cleanup import CleanupManager

__all__ = [
    'cli',
    'CLIInterface', 
    'MenuDisplay',
    'CleanupManager'
]
# debug_logging_setup.py - Use this to enable DEBUG logging for embedding issues
import logging
import sys

def setup_debug_logging():
    """Set up DEBUG logging specifically for embedding-related modules"""
    
    # Create a more detailed formatter for debugging
    debug_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Set up console handler with DEBUG level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(debug_formatter)
    
    # Set up file handler for debug logs
    debug_file_handler = logging.FileHandler("embedding_debug.log", mode='w', encoding='utf-8')
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(debug_formatter)
    
    # Configure specific loggers for embedding-related modules
    embedding_loggers = [
        'embedding_queue',
        'async_loader', 
        'embedding_service',
        'load_documents',
        'parse_documents'
    ]
    
    for logger_name in embedding_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.addHandler(debug_file_handler)
        logger.propagate = False  # Don't propagate to root logger to avoid duplicates
    
    # Also set the root logger to INFO to see general progress
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    print("DEBUG logging enabled for embedding modules")
    print("Debug logs will be written to: embedding_debug.log")
    print("Enabled debug logging for modules:", ', '.join(embedding_loggers))

if __name__ == "__main__":
    setup_debug_logging()
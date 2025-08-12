# main.py
"""
Refactored RAG Database System - Main Entry Point

This is the new modular entry point that replaces the monolithic rag_db.py
"""
import asyncio
import multiprocessing
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli.interface import cli

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    try:
        # Set multiprocessing start method for compatibility
        multiprocessing.set_start_method('spawn', force=True)
        
        # Run the CLI interface
        asyncio.run(cli.run())
        
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
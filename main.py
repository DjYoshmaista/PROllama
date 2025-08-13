# main.py - Enhanced RAG Database System Entry Point
"""
Enhanced RAG Database System with Queue-Based Parallel Processing

Latest Features:
- Queue-based parallel document processing (50-embedding background queue)
- Intelligent text chunking with 32-token overlaps
- Cross-referenced chunks for context reconstruction
- Automatic summarization with importance scoring
- Multi-query expansion (5 variations per question)
- Hierarchical search (summaries first, then detailed chunks)
- High-performance batch database operations
- Real-time progress monitoring and statistics
"""

import asyncio
import multiprocessing
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cli.enhanced_interface import enhanced_cli

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_rag_system.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Enhanced main entry point"""
    try:
        # Set multiprocessing start method for compatibility
        multiprocessing.set_start_method('spawn', force=True)
        
        # Display startup banner
        print("="*80)
        print("🚀 ENHANCED QUEUE-BASED RAG DATABASE SYSTEM")
        print("="*80)
        print("🔄 Queue-Based Processing: 50-Embedding Background Queue | 8 Workers")
        print("📊 Database: PostgreSQL + pgvector | Batch Operations | Connection Pooling")
        print("🧠 Embedding: GPU-Accelerated Ollama | Parallel Generation")
        print("📝 Processing: Intelligent Chunking | Auto-Summarization | Deduplication")
        print("🔍 Search: Multi-Query Expansion | Hierarchical (Summaries → Chunks)")
        print("⚡ Performance: Real-time Progress | Memory Management | Optimization")
        print("="*80)
        
        # Run the enhanced CLI interface
        asyncio.run(enhanced_cli.run())
        
    except KeyboardInterrupt:
        print("\n👋 Gracefully shutting down enhanced system...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
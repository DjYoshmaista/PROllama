# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PROllama is an Enhanced RAG (Retrieval Augmented Generation) Database System that combines PostgreSQL with pgvector, Ollama LLMs, and intelligent document processing. The system provides semantic search capabilities with multi-query expansion, hierarchical search (summaries → chunks), and automatic text summarization.

## Development Environment Setup

### Initial Setup
```bash
# Run the setup script to install dependencies and configure the system
bash setup.sh

# Activate the virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Main entry point
python main.py

# Debug database setup (optional)
python debug_database.py
```

### Required System Dependencies
- PostgreSQL with pgvector extension
- Ollama with these models:
  - `dengcao/Qwen3-Embed-0.6B:Q8_0` (embeddings)
  - `qwen3:8b` (inference)
  - `dengcao/Qwen3-Reranker-4B:Q5_K_M` (reranking)
- Python 3.x with venv support

### Environment Configuration
The system requires a `.env` file with:
```
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
TABLE_NAME=documents
```

## Architecture Overview

### Core Components

**CLI Interface** (`cli/interface.py`)
- Enhanced menu-driven interface with 10 main options
- Handles document loading, querying, database management
- Supports both single file and bulk folder processing

**Database Layer** (`database/`)
- `operations.py`: Enhanced operations with deduplication and chunking support
- `connection.py`: PostgreSQL connection management
- `cache.py`: Embedding caching for performance

**Document Processing** (`file_management/`)
- `chunking.py`: Intelligent text chunking with 32-token overlaps and cross-references
- `loaders.py`: Document loading with parallel processing support
- `parsers.py`: Support for .txt, .py, .csv, .json file types
- `discovery.py`: File system discovery and selection

**Inference Engine** (`inference/`)
- `engine.py`: Core semantic search with multiple optimization strategies
- `multi_query.py`: Multi-query expansion (5 variations per question)
- `embeddings.py`: Embedding generation via Ollama
- `summarization.py`: Automatic summarization with importance scoring

**Core Utilities** (`core/`)
- `config.py`: Configuration management from environment variables
- `memory.py`: Memory management and monitoring
- `vector_ops.py`: Vector similarity operations

### Database Schema

The system uses a hierarchical structure:
- **Chunks Table**: Individual text chunks with embeddings and cross-references
- **Summaries Table**: Auto-generated summaries for faster initial search
- **Checksums Table**: Content deduplication tracking

### Search Strategy

1. **Multi-Query Expansion**: Generates 5 variations of each user query
2. **Hierarchical Search**: Searches summaries first, then detailed chunks
3. **Context Reconstruction**: Uses chunk overlaps and cross-references
4. **Similarity Thresholds**: Configurable relevance filtering

## File Processing Pipeline

1. **Discovery**: File system scanning with type filtering
2. **Parsing**: Content extraction based on file type
3. **Chunking**: Intelligent text segmentation with overlaps
4. **Embedding**: Vector generation via Ollama embedding model
5. **Storage**: PostgreSQL with pgvector indexing
6. **Summarization**: Automatic summary generation for large chunks

## Memory and Performance

- Parallel processing for document loading and embedding generation
- GPU acceleration when available (CUDA/ROCm)
- Memory monitoring and management
- Embedding caching for repeated queries
- Database-level vector similarity for large datasets

## Common Workflow

1. **Setup**: Run `bash setup.sh` to configure system
2. **Load Documents**: Use Menu Option 4 (bulk folder) or 3 (single file)
3. **Query**: Use Menu Option 1 for semantic search with RAG
4. **Monitor**: Menu Option 9 for system information and performance
5. **Manage**: Menu Option 8 for database operations and maintenance

## Key Features

- **Intelligent Chunking**: 32-token overlaps with hierarchical relationships
- **Multi-Query Search**: 5 query variations for comprehensive results
- **Auto-Summarization**: Importance-scored summaries for faster search
- **Cross-Referenced Context**: Maintains document structure relationships
- **Deduplication**: Content-based deduplication using SHA-256 hashes
- **Performance Optimization**: Multiple search strategies based on dataset size
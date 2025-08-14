# config.py
import os
import logging
from constants import *
from typing import Dict, Any, ClassVar

logger = logging.getLogger(__name__)

class Config:
    # Database configuration
    EMBEDDING_DIM = EMB_DIM
    DB_CONFIG = {
        'dbname': DB_NAME,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': DB_HOST
    }
    TABLE_NAME = TABLE_NAME
    
    # Processing configuration
    CHUNK_SIZES = {
        'memory_safe': 100,
        'embedding_batch': 100,
        'insert_batch': 500,
        'file_processing': 100
    }
    
    # Search configuration
    SEARCH_DEFAULTS = {
        'relevance_threshold': 0.3,
        'top_k': 25,
        'vector_search_limit': 999999999
    }
    
    # Performance configuration
    MAX_CONCURRENT_REQUESTS = 200
    MAX_CACHE_SIZE = 20000000
    CACHE_REFRESH_INTERVAL = 300
    MEMORY_CLEANUP_THRESHOLD = 85
    MAX_CONCURRENT_CHUNKS = 32
    WORKER_PROCESSES = min(8, os.cpu_count())  # Now works with os import
    
    # PostgreSQL optimization
    PG_OPTIMIZATION_SETTINGS = {
        "statement_timeout": "300000",
        "work_mem": "16MB", 
        "maintenance_work_mem": "512MB"
    }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration values"""
        try:
            # Validate dimensions
            if cls.EMBEDDING_DIM <= 0:
                logger.error(f"Invalid embedding dimension: {cls.EMBEDDING_DIM}")
                return False
            
            # Validate chunk sizes
            for key, value in cls.CHUNK_SIZES.items():
                if not isinstance(value, int) or value <= 0:
                    logger.error(f"Invalid chunk size for {key}: {value}")
                    return False
            
            # Validate search defaults
            threshold = cls.SEARCH_DEFAULTS['relevance_threshold']
            if not 0.0 <= threshold <= 1.0:
                logger.error(f"Invalid relevance threshold: {threshold}")
                return False
            
            if cls.SEARCH_DEFAULTS['top_k'] <= 0:
                logger.error(f"Invalid top_k: {cls.SEARCH_DEFAULTS['top_k']}")
                return False
                
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration values at runtime with validation"""
        for k, v in kwargs.items():
            try:
                if hasattr(cls, k):
                    # Validate before setting
                    if k == 'EMBEDDING_DIM' and (not isinstance(v, int) or v <= 0):
                        raise ValueError(f"Invalid embedding dimension: {v}")
                    setattr(cls, k, v)
                    logger.info(f"Updated {k} to {v}")
                elif k in cls.SEARCH_DEFAULTS:
                    if k == 'relevance_threshold' and not 0.0 <= v <= 1.0:
                        raise ValueError(f"Invalid relevance threshold: {v}")
                    if k == 'top_k' and (not isinstance(v, int) or v <= 0):
                        raise ValueError(f"Invalid top_k: {v}")
                    cls.SEARCH_DEFAULTS[k] = v
                    logger.info(f"Updated search default {k} to {v}")
                elif k in cls.CHUNK_SIZES:
                    if not isinstance(v, int) or v <= 0:
                        raise ValueError(f"Invalid chunk size for {k}: {v}")
                    cls.CHUNK_SIZES[k] = v
                    logger.info(f"Updated chunk size {k} to {v}")
                else:
                    logger.warning(f"Unknown configuration key: {k}")
            except Exception as e:
                logger.error(f"Failed to update {k}: {e}")
                raise
    
    @classmethod
    def get_db_connection_string(cls) -> str:
        """Get formatted database connection string"""
        return (f"postgresql://{cls.DB_CONFIG['user']}:{cls.DB_CONFIG['password']}"
                f"@{cls.DB_CONFIG['host']}:{cls.DB_CONFIG.get('port', '5432')}"
                f"/{cls.DB_CONFIG['dbname']}")

# Validate configuration on module load
if not Config.validate_config():
    raise RuntimeError("Configuration validation failed")
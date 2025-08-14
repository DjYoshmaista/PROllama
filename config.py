# config.py
from constants import *

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
    WORKER_PROCESSES = min(8, os.cpu_count())
    
    # PostgreSQL optimization
    PG_OPTIMIZATION_SETTINGS = {
        "statement_timeout": "300000",
        "work_mem": "16MB", 
        "maintenance_work_mem": "512MB"
    }
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration values at runtime"""
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            elif k in cls.SEARCH_DEFAULTS:
                cls.SEARCH_DEFAULTS[k] = v
            elif k in cls.CHUNK_SIZES:
                cls.CHUNK_SIZES[k] = v
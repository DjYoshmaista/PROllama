# core/config.py
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    name: str
    user: str
    password: str
    host: str
    table_name: str
    
    @classmethod
    def from_env(cls):
        return cls(
            name=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD'),
            host=os.environ.get('DB_HOST'),
            table_name=os.environ.get('TABLE_NAME')
        )
    
    @property
    def connection_string(self) -> str:
        return f"postgres://{self.user}:{self.password}@{self.host}/{self.name}"

@dataclass
class EmbeddingConfig:
    model: str
    dimension: int
    api_url: str
    max_concurrent_requests: int = 200
    
    @classmethod
    def from_env(cls):
        return cls(
            model="dengcao/Qwen3-Embedding-0.6B:Q8_0",
            dimension=int(os.environ.get('EMB_DIM', 1024)),
            api_url="http://localhost:11434/api",
            max_concurrent_requests=200
        )

@dataclass
class InferenceConfig:
    relevance_threshold: float = 0.3
    top_k: int = 25
    vector_search_limit: int = 999999999
    max_cache_size: int = 20000000
    cache_refresh_interval: int = 300
    generation_model: str = "qwen3:8b"

@dataclass
class ProcessingConfig:
    max_concurrent_chunks_per_file: int = 8
    memory_chunk_size: int = 100
    embedding_chunk_size: int = 100
    insert_chunk_size: int = 500
    memory_cleanup_threshold: int = 85
    max_concurrent_chunks: int = 32
    worker_processes: int = min(8, os.cpu_count())
    pool_recycle_after: int = 200

class Config:
    """Centralized configuration manager"""
    
    def __init__(self):
        self.database = DatabaseConfig.from_env()
        self.embedding = EmbeddingConfig.from_env()
        self.inference = InferenceConfig()
        self.processing = ProcessingConfig()
        self._validate()
    
    def _validate(self):
        """Validate configuration"""
        required_fields = [
            self.database.name, self.database.user, 
            self.database.password, self.database.host,
            self.database.table_name
        ]
        if not all(required_fields):
            raise ValueError("Missing required database configuration")
    
    def update_inference_params(self, **kwargs):
        """Update inference parameters"""
        for key, value in kwargs.items():
            if hasattr(self.inference, key):
                setattr(self.inference, key, value)

# Global configuration instance
config = Config()
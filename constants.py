# constants.py
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

def get_env_var(key: str, default: Optional[str] = None, required: bool = True) -> str:
    """Safely get environment variable with validation"""
    value = os.environ.get(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable {key} not set")
    return value

def get_int_env_var(key: str, default: Optional[int] = None, required: bool = True) -> int:
    """Safely get integer environment variable with validation"""
    value_str = get_env_var(key, str(default) if default is not None else None, required)
    try:
        return int(value_str) if value_str else default
    except ValueError:
        raise ValueError(f"Environment variable {key} must be a valid integer, got: {value_str}")

# Core configuration with validation
EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
EMB_DIM = get_int_env_var('EMB_DIM', 1024)
DB_NAME = get_env_var('DB_NAME', 'ragdb')
DB_USER = get_env_var('DB_USER', 'postgres')
DB_PASSWORD = get_env_var('DB_PASSWORD', 'password')
DB_HOST = get_env_var('DB_HOST', 'localhost')
TABLE_NAME = get_env_var('TABLE_NAME', 'rag_db_code')

# API endpoints
OLLAMA_API = get_env_var('OLLAMA_API', "http://localhost:11434/api", required=False)
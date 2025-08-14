# constants.py
import os
from dotenv import load_dotenv

load_dotenv()

# Core configuration
EMBEDDING_MODEL = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
EMB_DIM = int(os.environ.get('EMB_DIM'))
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
TABLE_NAME = os.environ.get('TABLE_NAME')

# API endpoints
OLLAMA_API = "http://localhost:11434/api"
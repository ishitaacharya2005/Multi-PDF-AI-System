from __future__ import annotations

from src.utils.helpers import load_config_file

_CONFIG = load_config_file()

OLLAMA_MODEL = _CONFIG.get("OLLAMA_MODEL", "llama3.2")
EMBEDDING_MODEL = _CONFIG.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_PATH = _CONFIG.get("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE = int(_CONFIG.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(_CONFIG.get("CHUNK_OVERLAP", 50))
TOP_K = int(_CONFIG.get("TOP_K", 5))
BM25_TOP_K = int(_CONFIG.get("BM25_TOP_K", 10))
RRF_K = int(_CONFIG.get("RRF_K", 60))

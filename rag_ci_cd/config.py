from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"
CHUNK_CACHE = DATA_DIR / "chunks"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_data"

# Embedding
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Retrieval
HYBRID_TOP_K = 40
RERANK_TOP_K = 15
FINAL_TOP_K = 15
RRF_K = 60

# Generation
MAIN_MODEL = "llama3.2:3b"
BACKUP_MODEL = "qwen2.5:1.5b"
VISION_MODEL = "llama3.2:3b"

# Chunking
CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP_CHARS = 100

# Paths
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_CACHE.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

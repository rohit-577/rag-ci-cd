# Financial Document RAG System

Production-grade retrieval-augmented generation system for financial documents: PDFs, CSVs, and TXT files. Built for correctness, retrieval quality, source grounding, and testability.

## Architecture

```
docs/  →  Ingestion  →  Chunking  →  Indexing  →  Retrieval  →  Reranking  →  Generation
                                                (dense+BM25)  (cross-encoder)  (local LLM)
```

- **Ingestion**: Layout-aware parsers for PDF (pypdf), CSV (dict-based), and TXT (paragraph-based)
- **Chunking**: Structure-aware contextual chunking with section titles, page numbers, table headers, document metadata
- **Indexing**: Persistent index (JSON) with BM25 lexical index + dense embeddings (BGE-M3)
- **Retrieval**: Hybrid dense + BM25 with Reciprocal Rank Fusion (RRF) and deduplication
- **Reranking**: Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Routing**: Rule-based query classifier (simple vs complex) for model and depth selection
- **Generation**: Grounded answer generation via local LLM (qwen3:14b / llama3.2:3b) with forced citations
- **Evaluation**: Local harness with retrieval recall, grounding, confidence, and citation metrics

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

Requires:
- Python 3.12+
- Ollama with `qwen3:14b`, `llama3.2:3b` models
- CUDA-capable GPU (recommended) or CPU

## Usage

### 1. Build the index

```bash
python3 -m rag_ci_cd.cli index
```

Processes all files in `docs/` (PDF, CSV, TXT), chunks them, generates embeddings, and saves the index.

### 2. Start the API server

```bash
python3 -m rag_ci_cd.cli serve
```

FastAPI service at `http://localhost:6565`.

### 3. Query

```bash
curl -X POST http://localhost:6565/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What stability metrics are in the documents?", "top_k": 10, "rerank": true}'
```

### 4. Run evaluation

```bash
python3 -m rag_ci_cd.cli eval
```

### 5. Run tests

```bash
python3 -m pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + indexed chunk count |
| POST | `/query` | Ask a question about the documents |
| GET | `/documents` | List indexed documents with chunk counts |

### POST /query

Request:
```json
{
  "query": "What is the revenue trend?",
  "top_k": 10,
  "rerank": true,
  "route": null
}
```

Response:
```json
{
  "query": "What is the revenue trend?",
  "answer": "...",
  "citations": [{"chunk_id": "...", "filename": "...", "excerpt": "..."}],
  "confidence": 0.9,
  "sufficiency": "sufficient",
  "route": "complex",
  "retrieval_time_ms": 150.0,
  "generation_time_ms": 2500.0
}
```

## Project Structure

```
rag_ci_cd/
  models/           # Pydantic schemas
  ingestion/        # PDF/CSV/TXT parsers
  chunking/         # Contextual chunking
  indexing/         # Persistent index store
  embeddings/       # BGE-M3 embedding model
  retrieval/        # Dense + BM25 + hybrid RRF
  reranking/        # Cross-encoder reranker
  routing/          # Query classifier
  generation/       # LLM-based answer generation
  evaluation/       # Metrics + evaluation harness
  service/          # FastAPI application
  cli.py            # CLI entry points
tests/              # 74+ tests
```

## Key Design Decisions

- **Hybrid retrieval** (dense + BM25) with RRF fusion ensures both semantic understanding and exact term matching—critical for financial symbols, numbers, and dates.
- **Cross-encoder reranking** adds a second stage that jointly scores query-document pairs, improving precision on top of bi-encoder retrieval.
- **Contextual chunking** preserves document structure (page numbers, section titles, table headers) so each chunk carries its provenance.
- **Query routing** saves cost/time on simple lookups while allocating full resources for complex, multi-document questions.
- **Grounded generation** with confidence scoring and explicit abstention prevents hallucination of financial figures.

# Financial Document RAG System

Multi-stage retrieval-augmented generation pipeline for querying PDFs, CSVs, and TXT files — with automated CI/CD, Docker packaging, and a full test suite.

## How It Works

```
Query
  │
  ▼
┌──────────────┐     ┌────────────────┐     ┌───────────┐     ┌────────────┐
│   Routing    │────▶│   Retrieval    │────▶│ Reranking │────▶│ Generation │
│ simple/complex│     │ dense + BM25   │     │cross-encod│     │ Ollama LLM │
└──────────────┘     └────────────────┘     └───────────┘     └────────────┘
                          │                                        │
                     BGE-M3 embeds                          citations + confidence
```

| Stage | What it does |
|-------|-------------|
| **Routing** | Classifies query as `simple` (10 chunks, fast model) or `complex` (30 chunks, capable model) |
| **Retrieval** | Searches index using dense embeddings (BGE-M3) + BM25 lexical match, fused via Reciprocal Rank Fusion |
| **Reranking** | Cross-encoder re-scores top chunks for precision |
| **Generation** | Local Ollama LLM generates grounded answer with source citations and confidence score |

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) with models pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama pull qwen2.5:1.5b
  ```

### 1. Install

```bash
git clone https://github.com/rohit-577/rag-ci-cd.git
cd rag-ci-cd
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Build the Index

Processes all 30 documents in `docs/` (PDF, CSV, TXT), chunks them, generates embeddings, and saves the index.

```bash
make index
```

### 3. Start the API

```bash
make serve
# API runs at http://localhost:6565
```

### 4. Query

```bash
curl -X POST http://localhost:6565/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of India?", "top_k": 10}'
```

### 5. Run Tests

```bash
make test              # 141 unit tests (~30s)
make test-integration  # 20 end-to-end tests (requires Ollama, ~15min)
```

### 6. Docker

```bash
make docker-build
make docker-run
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + chunk count |
| `POST` | `/query` | Ask a question about the documents |
| `GET` | `/documents` | List all indexed documents |

**POST /query request:**
```json
{
  "query": "Compare revenue trends across documents",
  "top_k": 15,
  "rerank": true,
  "route": null
}
```

**Response:**
```json
{
  "query": "Compare revenue trends across documents",
  "answer": "According to the documents...",
  "citations": [
    {"chunk_id": "...", "filename": "DOC-1_ALBERT.txt", "excerpt": "..."}
  ],
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
├── models/           # Pydantic schemas (Document, Chunk, Answer)
├── ingestion/        # PDF/CSV/TXT parsers + factory
├── chunking/         # Structure-aware contextual chunker
├── embeddings/       # BGE-M3 embedding model
├── indexing/         # Persistent JSON + numpy index store
├── retrieval/        # Dense + BM25 + hybrid RRF fusion
├── reranking/        # Cross-encoder reranker
├── routing/          # Query classifier (simple/complex)
├── generation/       # Ollama LLM with citation extraction
├── evaluation/       # Metrics + evaluation harness
├── service/          # FastAPI server
├── cli.py            # CLI entry points
└── __main__.py       # python -m rag_ci_cd

tests/
├── test_*.py         # 9 unit test files (141 tests)
├── test_integration.py  # 20 gold-set integration tests
└── gold_set.json     # End-to-end test questions

docs/                 # 30 documents (10 TXT, 10 CSV, 10 PDF)
.github/workflows/    # CI/CD pipeline
Dockerfile            # Multi-stage Docker build
Makefile              # Dev shortcuts
```

## CI/CD Pipeline

Automated on every push/PR to `main`:

```
┌─────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Lint   │────▶│  Unit Tests  │────▶│ Integration Tests│────▶│ Docker Build+Push│
│ ruff    │     │ 141 tests    │     │ 20 tests + Ollama│     │ GHCR             │
└─────────┘     └──────────────┘     └─────────────────┘     └──────────────────┘
    │                │                       │                         │
  ~30s             ~30s                    ~25min                    ~3min
 PRs only        PRs + push              push to main              push to main
```

| Job | Triggers | What it does |
|-----|----------|-------------|
| **Lint** | Every push/PR | `ruff check` + `ruff format --check` |
| **Unit Tests** | Every push/PR | 141 tests (no Ollama needed) |
| **Integration Tests** | Push to `main` only | Pulls Ollama models, builds index, runs all 167 tests |
| **Docker Build+Push** | After integration passes | Builds multi-stage image, pushes to `ghcr.io` |

### Make Targets

| Command | Description |
|---------|-------------|
| `make lint` | Check code style |
| `make format` | Auto-fix style |
| `make test` | Run 141 unit tests |
| `make test-integration` | Run all 167 tests |
| `make index` | Build search index |
| `make serve` | Start API on `:6565` |
| `make eval` | Run evaluation harness |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run container |

## Test Suite

**141 unit tests** covering every module:
- Ingestion: PDF, CSV, TXT parsing + factory dispatch
- Chunking: structure preservation, metadata, determinism
- Retrieval: dense, lexical, hybrid RRF, index store
- Reranking: cross-encoder scoring and ordering
- Routing: query classification, model/depth config
- Generation: citation extraction, confidence parsing
- Evaluation: recall, grounding, metric computation
- Service: FastAPI health, query, documents endpoints

**20 integration tests** (gold set):
- 10 easy (single-doc factual extraction)
- 6 hard (multi-doc comparison, filtering, counting)
- 4 extreme (corpus-level aggregation, max operations)
- Auto-skips on Ollama timeout

## Requirements

- Python 3.12+
- Ollama (`llama3.2:3b`, `qwen2.5:1.5b`)
- CUDA GPU recommended (works on CPU, slower)
- ~4GB disk for models + index

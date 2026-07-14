from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_ci_cd.chunking.contextual_chunker import ContextualChunker
from rag_ci_cd.config import DOCS_DIR, INDEX_DIR
from rag_ci_cd.embeddings.model import EmbeddingModel
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.ingestion.factory import get_parser
from rag_ci_cd.models.answers import AnswerRequest, AnswerResponse
from rag_ci_cd.pipeline import answer_query

store: IndexStore | None = None


def build_index() -> IndexStore:
    global store
    s = IndexStore(INDEX_DIR)

    if s.size == 0:
        loaded = s.load()
        if not loaded:
            print("Building index from docs/ ...")
            emb = EmbeddingModel()
            pdfs = sorted(DOCS_DIR.glob("*.pdf"))
            csvs = sorted(DOCS_DIR.glob("*.csv"))
            txts = sorted(DOCS_DIR.glob("*.txt"))
            all_files = pdfs + csvs + txts

            for path in all_files:
                try:
                    parser = get_parser(path)
                    doc = parser.parse(path)
                    chunker = ContextualChunker()
                    chunks = chunker.chunk(doc)
                    if not chunks:
                        continue
                    texts = [c.content for c in chunks]
                    embeddings = emb.embed_chunks(texts, batch_size=4)
                    s.add_chunks(chunks, embeddings)
                    print(f"  Indexed {path.name} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"  Skipped {path.name}: {e}")

            s.save()
            print(f"Index built: {s.size} total chunks")
    else:
        print(f"Index already loaded: {s.size} chunks")

    store = s
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_index()
    yield


app = FastAPI(
    title="Financial Document RAG System",
    description="Multi-stage retrieval-augmented generation system for financial documents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    global store
    return {
        "status": "ok",
        "chunks_indexed": store.size if store else 0,
    }


@app.post("/query", response_model=AnswerResponse)
async def query(request: AnswerRequest):
    global store
    if store is None or store.size == 0:
        raise HTTPException(status_code=503, detail="Index not built yet. No documents indexed.")
    return answer_query(store, request)


@app.get("/documents")
async def list_documents():
    global store
    if store is None:
        return {"documents": []}
    seen: dict[str, dict] = {}
    for chunk in store.chunks:
        if chunk.doc_id not in seen:
            seen[chunk.doc_id] = {
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "doc_type": chunk.doc_type.value,
                "ticker": chunk.ticker,
                "year": chunk.year,
                "chunks": 0,
            }
        seen[chunk.doc_id]["chunks"] += 1
    return {"documents": list(seen.values())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_ci_cd.service.app:app", host="0.0.0.0", port=6565, reload=False)

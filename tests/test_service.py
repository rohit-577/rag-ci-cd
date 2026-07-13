from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_ci_cd.chunking.contextual_chunker import ContextualChunker
from rag_ci_cd.embeddings.model import EmbeddingModel
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.ingestion.factory import get_parser


@pytest.fixture
def test_store(tmp_path) -> IndexStore:
    store = IndexStore(tmp_path)
    docs_root = Path(__file__).resolve().parent.parent / "docs"
    samples = [
        docs_root / "DOC-1_INTC_2016_ID556661.csv",
        docs_root / "DOC-103_TSLA_2019_ID679016.txt",
    ]
    emb = EmbeddingModel()
    for sp in samples:
        if not sp.exists():
            continue
        parser = get_parser(sp)
        doc = parser.parse(sp)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        texts = [c.content for c in chunks]
        embeddings = emb.embed_chunks(texts, batch_size=4)
        store.add_chunks(chunks, embeddings)
    return store


class TestServiceHealth:
    def test_health_endpoint(self, test_store: IndexStore):
        from rag_ci_cd.service.app import app

        # Override the global store for testing
        import rag_ci_cd.service.app as svc_app
        svc_app.store = test_store

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["chunks_indexed"] > 0


class TestServiceQuery:
    def test_query_endpoint(self, test_store: IndexStore):
        from rag_ci_cd.service.app import app
        import rag_ci_cd.service.app as svc_app
        svc_app.store = test_store

        client = TestClient(app)
        response = client.post("/query", json={
            "query": "stability metric",
            "top_k": 5,
            "rerank": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "stability metric"
        assert len(data["citations"]) > 0

    def test_query_no_index(self):
        from rag_ci_cd.service.app import app
        import rag_ci_cd.service.app as svc_app
        svc_app.store = None

        client = TestClient(app)
        response = client.post("/query", json={"query": "test"})
        assert response.status_code == 503


class TestServiceDocuments:
    def test_documents_endpoint(self, test_store: IndexStore):
        from rag_ci_cd.service.app import app
        import rag_ci_cd.service.app as svc_app
        svc_app.store = test_store

        client = TestClient(app)
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) > 0
        for doc in data["documents"]:
            assert "doc_id" in doc
            assert "chunks" in doc

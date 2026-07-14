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
        docs_root / "DOC-11_STUDENTS_2024_011.csv",
        docs_root / "DOC-1_ALBERT_2024_001.txt",
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


@pytest.mark.integration
class TestServiceHealth:
    def test_health_endpoint(self, test_store: IndexStore):
        import rag_ci_cd.service.app as svc_app
        from rag_ci_cd.service.app import app

        svc_app.store = test_store

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["chunks_indexed"] > 0


@pytest.mark.integration
class TestServiceQuery:
    def test_query_endpoint(self, test_store: IndexStore):
        import rag_ci_cd.service.app as svc_app
        from rag_ci_cd.service.app import app

        svc_app.store = test_store

        client = TestClient(app)
        response = client.post(
            "/query",
            json={
                "query": "student marks",
                "top_k": 5,
                "rerank": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "student marks"
        assert len(data["citations"]) > 0

    def test_query_no_index(self):
        import rag_ci_cd.service.app as svc_app
        from rag_ci_cd.service.app import app

        svc_app.store = None

        client = TestClient(app)
        response = client.post("/query", json={"query": "test"})
        assert response.status_code == 503

    def test_query_response_structure(self, test_store: IndexStore):
        import rag_ci_cd.service.app as svc_app
        from rag_ci_cd.service.app import app

        svc_app.store = test_store

        client = TestClient(app)
        response = client.post(
            "/query",
            json={"query": "Einstein", "top_k": 3, "rerank": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert "sufficiency" in data
        assert "retrieval_time_ms" in data
        assert "generation_time_ms" in data


@pytest.mark.integration
class TestServiceDocuments:
    def test_documents_endpoint(self, test_store: IndexStore):
        import rag_ci_cd.service.app as svc_app
        from rag_ci_cd.service.app import app

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

    def test_documents_have_correct_fields(self, test_store: IndexStore):
        import rag_ci_cd.service.app as svc_app
        from rag_ci_cd.service.app import app

        svc_app.store = test_store

        client = TestClient(app)
        response = client.get("/documents")
        data = response.json()
        for doc in data["documents"]:
            assert "doc_id" in doc
            assert "filename" in doc
            assert "doc_type" in doc
            assert "chunks" in doc
            assert doc["chunks"] > 0

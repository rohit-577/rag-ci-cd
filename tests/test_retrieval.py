from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rag_ci_cd.chunking.contextual_chunker import ContextualChunker
from rag_ci_cd.embeddings.model import EmbeddingModel
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.ingestion.factory import get_parser
from rag_ci_cd.retrieval.dense import dense_retrieve
from rag_ci_cd.retrieval.hybrid import hybrid_retrieve
from rag_ci_cd.retrieval.lexical import lexical_retrieve


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory) -> IndexStore:
    tmp = tmp_path_factory.mktemp("index")
    store = IndexStore(tmp)
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


class TestDenseRetrieval:
    def test_returns_results(self, indexed_store: IndexStore):
        results = dense_retrieve(indexed_store, "stability metric")
        assert len(results) > 0

    def test_results_have_scores(self, indexed_store: IndexStore):
        results = dense_retrieve(indexed_store, "stability metric")
        for r in results:
            assert r.dense_score > 0

    def test_results_have_metadata(self, indexed_store: IndexStore):
        results = dense_retrieve(indexed_store, "neural architecture")
        for r in results:
            assert r.chunk_id
            assert r.filename
            assert r.content


class TestLexicalRetrieval:
    def test_returns_results(self, indexed_store: IndexStore):
        results = lexical_retrieve(indexed_store, "auto-scaling cluster")
        assert len(results) > 0

    def test_bm25_scores_positive(self, indexed_store: IndexStore):
        results = lexical_retrieve(indexed_store, "HNSW indexing")
        for r in results:
            assert r.lexical_score > 0

    def test_lexical_finds_exact_terms(self, indexed_store: IndexStore):
        results = lexical_retrieve(indexed_store, "SYSTEM_MEMORY_BLOB")
        if results:
            found = any("SYSTEM MEMORY BLOB" in r.content for r in results)
            assert found


class TestHybridRetrieval:
    def test_returns_results(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "stability metric negative")
        assert len(result.chunks) > 0

    def test_hybrid_has_scores(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "auto-scaling availability zones")
        for c in result.chunks[:5]:
            assert c.hybrid_score > 0

    def test_hybrid_deduplicates(self, indexed_store: IndexStore):
        query = "Deploying an auto-scaling cluster with a raft consensus model"
        result = hybrid_retrieve(indexed_store, query, top_k=30)
        contents = [c.content for c in result.chunks]
        assert len(contents) == len(set(c.lower().strip() for c in contents))

    def test_retrieval_time_measured(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "database server crash")
        assert result.retrieval_time_ms > 0

    def test_returns_metadata(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "quarterly revenue")
        for c in result.chunks[:3]:
            assert c.filename
            assert c.doc_id


class TestBM25Impl:
    def test_bm25_tokenize(self):
        from rag_ci_cd.indexing.store import _tokenize
        assert _tokenize("Hello World") == ["hello", "world"]
        assert _tokenize("") == []

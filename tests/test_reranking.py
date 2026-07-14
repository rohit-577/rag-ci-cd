from __future__ import annotations

from pathlib import Path

import pytest

from rag_ci_cd.chunking.contextual_chunker import ContextualChunker
from rag_ci_cd.embeddings.model import EmbeddingModel
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.ingestion.factory import get_parser
from rag_ci_cd.reranking.reranker import Reranker
from rag_ci_cd.retrieval.hybrid import hybrid_retrieve


@pytest.fixture(scope="module")
def tiny_store(tmp_path_factory) -> IndexStore:
    tmp = tmp_path_factory.mktemp("rerank_idx")
    store = IndexStore(tmp)
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


class TestReranker:
    def test_reranker_returns_fewer_results(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "student marks", top_k=20)
        assert len(result.chunks) >= 2
        reranker = Reranker()
        reranked = reranker.rerank_result(result, top_k=3)
        assert len(reranked.chunks) <= 3

    def test_reranker_assigns_scores(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "Einstein physics", top_k=10)
        reranker = Reranker()
        reranked = reranker.rerank(result.query, result.chunks, top_k=10)
        for c in reranked:
            assert c.rerank_score is not None

    def test_reranker_maintains_metadata(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "Mathematics grade", top_k=10)
        reranker = Reranker()
        reranked = reranker.rerank(result.query, result.chunks, top_k=10)
        for c in reranked:
            assert c.chunk_id
            assert c.filename
            assert c.content

    def test_reranker_empty_input(self):
        reranker = Reranker()
        result = reranker.rerank("test", [])
        assert result == []

    def test_reranker_method_indicator(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "Alice Johnson", top_k=10)
        reranker = Reranker()
        reranked = reranker.rerank_result(result, top_k=3)
        assert "reranked" in reranked.method

    def test_reranker_increases_retrieval_time(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "student marks grade", top_k=10)
        original_time = result.retrieval_time_ms
        reranker = Reranker()
        reranked = reranker.rerank_result(result, top_k=5)
        assert reranked.retrieval_time_ms >= original_time

    def test_reranker_rerank_scores_are_ordered(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "Einstein relativity Nobel", top_k=10)
        reranker = Reranker()
        reranked = reranker.rerank(result.query, result.chunks, top_k=10)
        scores = [c.rerank_score for c in reranked if c.rerank_score is not None]
        assert scores == sorted(scores, reverse=True)

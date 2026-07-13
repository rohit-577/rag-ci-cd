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


class TestReranker:
    def test_reranker_returns_fewer_results(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "stability metric", top_k=20)
        assert len(result.chunks) >= 5
        reranker = Reranker()
        reranked = reranker.rerank_result(result, top_k=5)
        assert len(reranked.chunks) <= 5

    def test_reranker_assigns_scores(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "auto-scaling cluster", top_k=10)
        reranker = Reranker()
        reranked = reranker.rerank(result.query, result.chunks, top_k=10)
        for c in reranked:
            assert c.rerank_score is not None

    def test_reranker_maintains_metadata(self, tiny_store: IndexStore):
        result = hybrid_retrieve(tiny_store, "neural architecture", top_k=10)
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
        result = hybrid_retrieve(tiny_store, "database server", top_k=10)
        reranker = Reranker()
        reranked = reranker.rerank_result(result, top_k=5)
        assert "reranked" in reranked.method

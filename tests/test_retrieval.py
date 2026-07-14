from __future__ import annotations

from pathlib import Path

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
        docs_root / "DOC-11_STUDENTS_2024_011.csv",
        docs_root / "DOC-1_ALBERT_2024_001.txt",
        docs_root / "DOC-12_FRUITS_2024_012.csv",
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
        results = dense_retrieve(indexed_store, "student marks")
        assert len(results) > 0

    def test_results_have_scores(self, indexed_store: IndexStore):
        results = dense_retrieve(indexed_store, "student marks")
        for r in results:
            assert r.dense_score > 0

    def test_results_have_metadata(self, indexed_store: IndexStore):
        results = dense_retrieve(indexed_store, "Einstein physics")
        for r in results:
            assert r.chunk_id
            assert r.filename
            assert r.content

    def test_top_k_limits_results(self, indexed_store: IndexStore):
        results = dense_retrieve(indexed_store, "fruit calories", top_k=3)
        assert len(results) <= 3


class TestLexicalRetrieval:
    def test_returns_results(self, indexed_store: IndexStore):
        results = lexical_retrieve(indexed_store, "Alice Johnson Mathematics")
        assert len(results) > 0

    def test_bm25_scores_positive(self, indexed_store: IndexStore):
        results = lexical_retrieve(indexed_store, "Einstein relativity")
        for r in results:
            assert r.lexical_score > 0

    def test_lexical_finds_exact_terms(self, indexed_store: IndexStore):
        results = lexical_retrieve(indexed_store, "Albert Einstein")
        if results:
            found = any("Albert Einstein" in r.content for r in results)
            assert found

    def test_lexical_finds_csv_content(self, indexed_store: IndexStore):
        results = lexical_retrieve(indexed_store, "Bob Smith Physics")
        if results:
            found = any("Bob Smith" in r.content for r in results)
            assert found


class TestHybridRetrieval:
    def test_returns_results(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "student mathematics marks")
        assert len(result.chunks) > 0

    def test_hybrid_has_scores(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "fruit calories origin")
        for c in result.chunks[:5]:
            assert c.hybrid_score > 0

    def test_hybrid_deduplicates(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "student marks grade", top_k=30)
        contents = [c.content for c in result.chunks]
        assert len(contents) == len(set(c.lower().strip() for c in contents))

    def test_retrieval_time_measured(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "Einstein Nobel Prize")
        assert result.retrieval_time_ms > 0

    def test_returns_metadata(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "Apple fruit red")
        for c in result.chunks[:3]:
            assert c.filename
            assert c.doc_id

    def test_retrieval_result_has_method(self, indexed_store: IndexStore):
        result = hybrid_retrieve(indexed_store, "student grade marks")
        assert result.method is not None
        assert len(result.method) > 0

    def test_retrieval_result_has_query(self, indexed_store: IndexStore):
        query = "what are the student names"
        result = hybrid_retrieve(indexed_store, query)
        assert result.query == query


class TestBM25Impl:
    def test_bm25_tokenize(self):
        from rag_ci_cd.indexing.store import _tokenize

        assert _tokenize("Hello World") == ["hello", "world"]
        assert _tokenize("") == []

    def test_bm25_tokenize_expands_hyphens(self):
        from rag_ci_cd.indexing.store import _tokenize

        tokens = _tokenize("cross-encoder model")
        assert "cross" in tokens
        assert "encoder" in tokens

    def test_bm25_tokenize_strips_punctuation(self):
        from rag_ci_cd.indexing.store import _tokenize

        tokens = _tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens


class TestIndexStore:
    def test_add_and_retrieve_chunks(self, indexed_store: IndexStore):
        assert indexed_store.size > 0

    def test_get_chunk_by_id(self, indexed_store: IndexStore):
        chunks = indexed_store.chunks
        if chunks:
            chunk = indexed_store.get_chunk(chunks[0].chunk_id)
            assert chunk is not None
            assert chunk.chunk_id == chunks[0].chunk_id

    def test_get_nonexistent_chunk(self, indexed_store: IndexStore):
        chunk = indexed_store.get_chunk("nonexistent_id")
        assert chunk is None

    def test_embeddings_shape(self, indexed_store: IndexStore):
        emb = indexed_store.embeddings
        assert emb is not None
        assert emb.shape[0] == indexed_store.size
        assert emb.shape[1] > 0

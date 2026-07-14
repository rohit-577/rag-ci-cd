from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_ci_cd.chunking.contextual_chunker import ContextualChunker
from rag_ci_cd.config import DOCS_DIR
from rag_ci_cd.embeddings.model import EmbeddingModel
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.ingestion.factory import get_parser
from rag_ci_cd.models.answers import AnswerRequest
from rag_ci_cd.pipeline import answer_query


def _load_gold_set() -> list[dict]:
    gold_path = Path(__file__).parent / "gold_set.json"
    return json.loads(gold_path.read_text())


@pytest.fixture(scope="module")
def full_store(tmp_path_factory) -> IndexStore:
    tmp = tmp_path_factory.mktemp("integration_index")
    store = IndexStore(tmp)
    emb = EmbeddingModel()
    chunker = ContextualChunker()

    all_files = sorted(DOCS_DIR.glob("*.txt")) + sorted(DOCS_DIR.glob("*.csv")) + sorted(DOCS_DIR.glob("*.pdf"))

    for path in all_files:
        parser = get_parser(path)
        doc = parser.parse(path)
        chunks = chunker.chunk(doc)
        if not chunks:
            continue
        texts = [c.content for c in chunks]
        embeddings = emb.embed_chunks(texts, batch_size=4)
        store.add_chunks(chunks, embeddings)

    return store


@pytest.mark.integration
class TestGoldSet:
    """Run every question in gold_set.json through the full RAG pipeline."""

    @pytest.mark.parametrize(
        "example",
        _load_gold_set(),
        ids=[e["id"] for e in _load_gold_set()],
    )
    def test_gold_set_question(self, full_store: IndexStore, example: dict):
        request = AnswerRequest(
            query=example["question"],
            top_k=15,
            rerank=True,
        )
        response = answer_query(full_store, request)

        answer_text = response.answer
        answer_normalized = answer_text.replace(",", "").lower()
        expected = example["expected_terms"]
        found = [
            term
            for term in expected
            if term.lower() in answer_normalized or term.lower() in answer_text.lower()
        ]

        assert len(found) > 0, (
            f"Question: {example['question']}\n"
            f"Expected any of: {expected}\n"
            f"Answer: {answer_text[:300]}\n"
            f"Matched: {found}"
        )

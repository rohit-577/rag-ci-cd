from rag_ci_cd.evaluation.harness import EvalExample
from rag_ci_cd.evaluation.metrics import (
    answer_grounding,
    compute_answer_metrics,
    compute_retrieval_metrics,
    retrieval_recall,
)
from rag_ci_cd.models.answers import Answer, Citation
from rag_ci_cd.models.retrieval import RetrievedChunk


class TestRetrievalRecall:
    def _chunk(self, chunk_id: str) -> RetrievedChunk:
        return RetrievedChunk(chunk_id=chunk_id, doc_id="d", filename="f", content="x")

    def test_perfect_recall(self):
        chunks = [self._chunk("a")]
        assert retrieval_recall(chunks, {"a"}) == 1.0

    def test_zero_recall(self):
        chunks = [self._chunk("a")]
        assert retrieval_recall(chunks, {"b"}) == 0.0

    def test_partial_recall(self):
        chunks = [self._chunk("a"), self._chunk("b")]
        assert retrieval_recall(chunks, {"a", "c"}) == 0.5

    def test_empty_relevant(self):
        chunks = [self._chunk("a")]
        assert retrieval_recall(chunks, set()) == 0.0

    def test_empty_chunks(self):
        assert retrieval_recall([], {"a"}) == 0.0

    def test_full_recall_multiple(self):
        chunks = [self._chunk("a"), self._chunk("b"), self._chunk("c")]
        assert retrieval_recall(chunks, {"a", "b", "c"}) == 1.0


class TestAnswerGrounding:
    def test_sufficient(self):
        ans = Answer(query="q", answer="a", confidence=0.9, sufficiency="sufficient")
        assert answer_grounding(ans) == 1.0

    def test_partial(self):
        ans = Answer(query="q", answer="a", confidence=0.5, sufficiency="partial")
        assert answer_grounding(ans) == 0.5

    def test_insufficient(self):
        ans = Answer(query="q", answer="a", confidence=0.1, sufficiency="insufficient")
        assert answer_grounding(ans) == 0.0


class TestComputeRetrievalMetrics:
    def test_perfect_metrics(self):
        chunks = [
            RetrievedChunk(chunk_id="a", doc_id="d", filename="f", content="x"),
            RetrievedChunk(chunk_id="b", doc_id="d", filename="f", content="y"),
        ]
        metrics = compute_retrieval_metrics(chunks, {"a", "b"})
        assert metrics["recall"] == 1.0
        assert metrics["num_retrieved"] == 2
        assert metrics["num_relevant"] == 2

    def test_partial_metrics(self):
        chunks = [
            RetrievedChunk(chunk_id="a", doc_id="d", filename="f", content="x"),
            RetrievedChunk(chunk_id="b", doc_id="d", filename="f", content="y"),
        ]
        metrics = compute_retrieval_metrics(chunks, {"a", "c"})
        assert metrics["recall"] == 0.5

    def test_empty_relevant_set(self):
        chunks = [RetrievedChunk(chunk_id="a", doc_id="d", filename="f", content="x")]
        metrics = compute_retrieval_metrics(chunks, set())
        assert metrics["recall"] == 0.0


class TestComputeAnswerMetrics:
    def test_sufficient_answer(self):
        ans = Answer(
            query="q",
            answer="a",
            citations=[Citation(chunk_id="c1", filename="f", excerpt="e")],
            confidence=0.9,
            sufficiency="sufficient",
        )
        metrics = compute_answer_metrics(ans)
        assert metrics["confidence"] == 0.9
        assert metrics["grounding"] == 1.0
        assert metrics["citation_count"] == 1
        assert metrics["citation_precision"] == 1.0

    def test_insufficient_answer(self):
        ans = Answer(query="q", answer="a", confidence=0.1, sufficiency="insufficient")
        metrics = compute_answer_metrics(ans)
        assert metrics["grounding"] == 0.0
        assert metrics["citation_count"] == 0
        assert metrics["citation_precision"] == 0.0


class TestEvalExample:
    def test_default_construction(self):
        ex = EvalExample(query="test query")
        assert ex.query == "test query"
        assert ex.relevant_chunk_ids == set()
        assert ex.expected_answer_terms == []

    def test_with_expected_terms(self):
        ex = EvalExample(query="q", expected_answer_terms=["revenue", "$10M"])
        assert "revenue" in ex.expected_answer_terms

    def test_with_relevant_ids(self):
        ex = EvalExample(query="q", relevant_chunk_ids=["a", "b"])
        assert "a" in ex.relevant_chunk_ids
        assert "b" in ex.relevant_chunk_ids

    def test_with_description(self):
        ex = EvalExample(query="q", description="test description")
        assert ex.description == "test description"

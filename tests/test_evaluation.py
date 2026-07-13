from rag_ci_cd.evaluation.harness import EvalExample
from rag_ci_cd.evaluation.metrics import retrieval_recall, answer_grounding
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


class TestEvalExample:
    def test_default_construction(self):
        ex = EvalExample(query="test query")
        assert ex.query == "test query"
        assert ex.relevant_chunk_ids == set()
        assert ex.expected_answer_terms == []

    def test_with_expected_terms(self):
        ex = EvalExample(query="q", expected_answer_terms=["revenue", "$10M"])
        assert "revenue" in ex.expected_answer_terms

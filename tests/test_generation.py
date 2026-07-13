from rag_ci_cd.generation.generator import _parse_confidence_sufficiency, _extract_citations, answer_to_response
from rag_ci_cd.models.answers import Answer, Citation
from rag_ci_cd.models.retrieval import RetrievedChunk, RetrievalResult
from rag_ci_cd.routing.router import Route


class TestConfidenceParsing:
    def test_insufficient_detection(self):
        text = "I cannot answer this question from the available documents."
        conf, suff = _parse_confidence_sufficiency(text)
        assert suff == "insufficient"
        assert conf < 0.5

    def test_sufficient_detection(self):
        text = "The revenue was $10 million [Source: doc.pdf]."
        conf, suff = _parse_confidence_sufficiency(text)
        assert suff == "sufficient"
        assert conf > 0.5

    def test_partial_detection(self):
        text = "The revenue may have been around $10 million, but I am unsure."
        conf, suff = _parse_confidence_sufficiency(text)
        assert suff == "partial"


class TestCitationExtraction:
    def test_citation_from_referenced_filename(self):
        chunks = [
            RetrievedChunk(
                chunk_id="abc123",
                doc_id="DOC-1",
                filename="report.pdf",
                content="The revenue was $10 million.",
                page_number=5,
            )
        ]
        result = RetrievalResult(query="test", chunks=chunks)
        text = "According to report.pdf, the revenue was $10 million."
        citations = _extract_citations(text, result)
        assert len(citations) > 0
        assert citations[0].filename == "report.pdf"

    def test_citations_empty_when_no_reference(self):
        chunks = [
            RetrievedChunk(
                chunk_id="abc",
                doc_id="DOC-1",
                filename="doc.csv",
                content="Some data here.",
            )
        ]
        result = RetrievalResult(query="test", chunks=chunks)
        text = "The answer does not reference the document."
        citations = _extract_citations(text, result)
        # Should still add a fallback citation
        assert len(citations) > 0

    def test_fallback_citation(self):
        chunks = [
            RetrievedChunk(
                chunk_id="def456",
                doc_id="DOC-2",
                filename="filing.pdf",
                content="Net income: $5M.",
                page_number=3,
                ticker="ACME",
                year=2024,
            )
        ]
        result = RetrievalResult(query="test", chunks=chunks)
        text = "No explicit reference."
        citations = _extract_citations(text, result)
        assert any(c.filename == "filing.pdf" for c in citations)


class TestAnswerToResponse:
    def test_conversion(self):
        answer = Answer(
            query="What is revenue?",
            answer="$10M",
            citations=[Citation(chunk_id="c1", filename="doc.pdf", excerpt="revenue $10M")],
            confidence=0.9,
            sufficiency="sufficient",
        )
        resp = answer_to_response(answer, route=Route.SIMPLE, retrieval_time_ms=100.0, generation_time_ms=200.0)
        assert resp.query == "What is revenue?"
        assert resp.answer == "$10M"
        assert resp.route == "simple"
        assert resp.retrieval_time_ms == 100.0
        assert resp.generation_time_ms == 200.0
        assert len(resp.citations) == 1

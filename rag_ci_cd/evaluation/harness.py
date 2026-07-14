from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_ci_cd.evaluation.metrics import compute_answer_metrics, compute_retrieval_metrics
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.models.answers import Answer, AnswerRequest
from rag_ci_cd.pipeline import answer_query


class EvalExample:
    def __init__(
        self,
        query: str,
        relevant_chunk_ids: list[str] | None = None,
        expected_answer_terms: list[str] | None = None,
        description: str = "",
    ):
        self.query = query
        self.relevant_chunk_ids = set(relevant_chunk_ids or [])
        self.expected_answer_terms = expected_answer_terms or []
        self.description = description


GOLD_SET: list[EvalExample] = [
    EvalExample(
        query="What is the stability metric for SYS_LOG_466?",
        description="Simple factual extraction from CSV log data",
    ),
    EvalExample(
        query="Auto-scaling cluster raft consensus availability zones",
        description="Lexical match test for ARCH_NOTE content",
    ),
    EvalExample(
        query="HNSW indexing algorithm",
        description="Technical term retrieval test",
    ),
    EvalExample(
        query="Neural architecture causal attention patterns",
        description="Dense retrieval test for semantically similar content",
    ),
    EvalExample(
        query="SYSTEM MEMORY BLOB",
        description="TXT document header retrieval",
    ),
    EvalExample(
        query="Compare different stability metrics",
        description="Complex multi-document comparison query",
    ),
]


def run_evaluation(
    store: IndexStore,
    gold_set: list[EvalExample] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if gold_set is None:
        gold_set = GOLD_SET

    results: list[dict[str, Any]] = []
    for example in gold_set:
        request = AnswerRequest(query=example.query, top_k=top_k, rerank=True)
        response = answer_query(store, request)

        retrieval_metrics = compute_retrieval_metrics([], example.relevant_chunk_ids)
        ans = Answer(
            query=response.query,
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            sufficiency=response.sufficiency,
            reasoning=response.reasoning,
        )
        answer_metrics = compute_answer_metrics(ans)

        term_hits = 0
        for term in example.expected_answer_terms:
            if term.lower() in response.answer.lower():
                term_hits += 1

        results.append(
            {
                "query": example.query,
                "description": example.description,
                "answer": response.answer,
                "route": response.route,
                "confidence": response.confidence,
                "sufficiency": response.sufficiency,
                "citation_count": len(response.citations),
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                "expected_terms_found": term_hits,
                "expected_terms_total": len(example.expected_answer_terms),
                "answer_metrics": answer_metrics,
                "retrieval_metrics": retrieval_metrics,
            }
        )

    return results


def print_evaluation(results: list[dict[str, Any]]) -> None:
    print("=" * 80)
    print("RAG SYSTEM EVALUATION REPORT")
    print("=" * 80)

    total = len(results)
    sufficient = sum(1 for r in results if r["sufficiency"] == "sufficient")
    partial = sum(1 for r in results if r["sufficiency"] == "partial")
    insufficient = sum(1 for r in results if r["sufficiency"] == "insufficient")
    avg_confidence = sum(r["confidence"] for r in results) / max(total, 1)
    avg_retrieval_ms = sum(r["retrieval_time_ms"] for r in results) / max(total, 1)
    avg_generation_ms = sum(r["generation_time_ms"] for r in results) / max(total, 1)
    total_citations = sum(r["citation_count"] for r in results)

    print(f"\nTotal queries: {total}")
    print(f"Sufficient: {sufficient}  |  Partial: {partial}  |  Insufficient: {insufficient}")
    print(f"Average confidence: {avg_confidence:.2f}")
    print(f"Average retrieval time: {avg_retrieval_ms:.0f}ms")
    print(f"Average generation time: {avg_generation_ms:.0f}ms")
    print(f"Total citations: {total_citations}")
    print()

    for r in results:
        print(f"--- [{r['route']}] {r['query'][:60]} ---")
        print(f"  Answer: {r['answer'][:120]}...")
        print(f"  Confidence: {r['confidence']} | Sufficiency: {r['sufficiency']} | Citations: {r['citation_count']}")
        print()


def save_evaluation(results: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Evaluation saved to {path}")

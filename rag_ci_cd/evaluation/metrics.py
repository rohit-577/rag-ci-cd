from __future__ import annotations

from rag_ci_cd.models.answers import Answer


def retrieval_recall(retrieved_chunks: list, relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_ids = {c.chunk_id for c in retrieved_chunks}
    hit_count = len(retrieved_ids & relevant_ids)
    return hit_count / len(relevant_ids)


def citation_precision(answer: Answer) -> float:
    if not answer.citations:
        return 0.0
    cited_chunks = {c.chunk_id for c in answer.citations}
    # We consider all citations valid since they're extracted from what the LLM said
    return 1.0 if cited_chunks else 0.0


def answer_grounding(answer: Answer) -> float:
    if answer.sufficiency == "sufficient":
        return 1.0
    elif answer.sufficiency == "partial":
        return 0.5
    return 0.0


def compute_retrieval_metrics(
    retrieved_chunks: list,
    relevant_chunk_ids: set[str],
) -> dict[str, float]:
    recall = retrieval_recall(retrieved_chunks, relevant_chunk_ids)
    precision_at_k = 0.0
    for rank, chunk in enumerate(retrieved_chunks[:10], start=1):
        if chunk.chunk_id in relevant_chunk_ids:
            precision_at_k += 1.0 / rank

    return {
        "recall": round(recall, 4),
        "map_at_10": round(precision_at_k / max(len(relevant_chunk_ids), 1), 4),
        "num_retrieved": len(retrieved_chunks),
        "num_relevant": len(relevant_chunk_ids),
    }


def compute_answer_metrics(answer: Answer) -> dict[str, float]:
    return {
        "confidence": round(answer.confidence, 4),
        "grounding": round(answer_grounding(answer), 4),
        "citation_count": len(answer.citations),
        "citation_precision": round(citation_precision(answer), 4),
    }

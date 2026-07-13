from __future__ import annotations

from rag_ci_cd.models.retrieval import RetrievedChunk


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    result: list[RetrievedChunk] = []
    for c in chunks:
        key = _normalize(c.content)
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result

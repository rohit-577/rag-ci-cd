from __future__ import annotations

import re
import time

from rag_ci_cd.config import RRF_K
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.models.retrieval import RetrievedChunk, RetrievalResult
from rag_ci_cd.retrieval.dedup import deduplicate
from rag_ci_cd.retrieval.dense import dense_retrieve
from rag_ci_cd.retrieval.lexical import lexical_retrieve

_DOC_ID_PATTERN = re.compile(r'\bDOC-(\d+)\b')

KNOWN_TICKERS = {"AAPL", "AMD", "AMZN", "GOOGL", "INTC", "META", "MSFT", "NFLX", "NVDA", "TSLA"}
_MAX_YEARS = 12  # 2015-2026


def _extract_tickers(query: str) -> list[str]:
    words = re.findall(r"[A-Z]{2,5}", query.upper())
    return [w for w in words if w in KNOWN_TICKERS]


def _extract_year(query: str) -> int | None:
    years = re.findall(r"\b(20\d{2})\b", query)
    return int(years[0]) if years else None


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank)


def _rrf_merge(dense_results: list[RetrievedChunk], lexical_results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    rrf_scores: dict[str, tuple[RetrievedChunk, float]] = {}

    for rank, chunk in enumerate(dense_results):
        score = _rrf_score(rank)
        rrf_scores[chunk.chunk_id] = (chunk, score)

    for rank, chunk in enumerate(lexical_results):
        existing_score = rrf_scores.get(chunk.chunk_id, (chunk, 0.0))[1]
        new_score = existing_score + _rrf_score(rank)
        rrf_scores[chunk.chunk_id] = (chunk, new_score)

    scored: list[tuple[float, RetrievedChunk]] = []
    for chunk_id, (chunk, score) in rrf_scores.items():
        chunk.hybrid_score = score
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    merged = [chunk for _, chunk in scored]
    merged = deduplicate(merged)

    for rank, chunk in enumerate(merged):
        chunk.rank = rank + 1

    return merged


def _ensure_year_diversity(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    seen_years: set[int] = set()
    deferred: list[RetrievedChunk] = []

    for c in chunks:
        if c.year is not None and c.year not in seen_years:
            selected.append(c)
            seen_years.add(c.year)
            if len(selected) >= top_k:
                return selected
        else:
            deferred.append(c)

    filler = [c for c in deferred if c not in selected]
    selected.extend(filler)
    return selected[:top_k]


def _get_all_chunks_for_doc(store: IndexStore, doc_id: str, top_k: int) -> list[RetrievedChunk]:
    chunks_list = []
    for idx, chunk in enumerate(store.chunks):
        if chunk.doc_id == doc_id:
            chunks_list.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                content=chunk.content,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                table_headers=chunk.table_headers,
                ticker=chunk.ticker,
                year=chunk.year,
                doc_type=chunk.doc_type.value if chunk.doc_type else None,
                chunk_type=chunk.chunk_type.value if chunk.chunk_type else None,
                dense_score=1.0,
                lexical_score=1.0,
                hybrid_score=1.0,
            ))
    return chunks_list[:top_k]


def hybrid_retrieve(
    store: IndexStore,
    query: str,
    top_k: int = 20,
) -> RetrievalResult:
    start = time.time()
    tickers = _extract_tickers(query)
    query_year = _extract_year(query)

    doc_match = _DOC_ID_PATTERN.search(query)
    if doc_match:
        target_doc_id = f"DOC-{doc_match.group(1)}"
        all_target = _get_all_chunks_for_doc(store, target_doc_id, top_k * 2)
        if all_target:
            for rank, chunk in enumerate(all_target):
                chunk.rank = rank + 1
            elapsed = (time.time() - start) * 1000
            return RetrievalResult(
                query=query,
                chunks=all_target[:top_k],
                retrieval_time_ms=round(elapsed, 2),
                method="focused_doc",
            )

    if len(tickers) > 1:
        non_ticker_terms = re.sub(r"\b[A-Z]{2,5}\b", "", query).strip()
        if not non_ticker_terms:
            non_ticker_terms = "stability metric"
        per_ticker_k = max(top_k, 15)
        all_dense: list[RetrievedChunk] = []
        all_lexical: list[RetrievedChunk] = []
        for ticker in tickers:
            sub_q = f"{non_ticker_terms} {ticker}"
            if query_year:
                sub_q += f" {query_year}"
            all_dense.extend(dense_retrieve(store, sub_q, top_k=per_ticker_k))
            all_lexical.extend(lexical_retrieve(store, sub_q, top_k=per_ticker_k))
        merged = _rrf_merge(all_dense, all_lexical)
        elapsed = (time.time() - start) * 1000
        return RetrievalResult(
            query=query,
            chunks=merged[:top_k],
            retrieval_time_ms=round(elapsed, 2),
            method="hybrid_rrf_multi_ticker",
        )

    dense_results = dense_retrieve(store, query, top_k=top_k * 2)
    lexical_results = lexical_retrieve(store, query, top_k=top_k * 2)
    merged = _rrf_merge(dense_results, lexical_results)

    elapsed = (time.time() - start) * 1000

    return RetrievalResult(
        query=query,
        chunks=merged[:top_k],
        retrieval_time_ms=round(elapsed, 2),
        method="hybrid_rrf",
    )

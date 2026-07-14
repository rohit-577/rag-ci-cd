from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from rag_ci_cd.models.answers import Answer, AnswerResponse, Citation
from rag_ci_cd.models.retrieval import RetrievalResult
from rag_ci_cd.routing.router import Route, get_model_for_route


def _call_ollama(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode()
    req = Request("http://localhost:11434/api/generate", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urlopen(req, timeout=300)
        result = json.loads(resp.read().decode())
        return result.get("response", "")
    except (URLError, json.JSONDecodeError, OSError) as e:
        return f"[Error generating response: {e}]"


_SYSTEM_PROMPT = """You are a precise financial document analyst. Your job is to answer questions using ONLY the evidence provided in the context below.

RULES:
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. After EVERY important factual claim, cite the source in brackets like [Source: filename, page N] or [Source: filename, Section: title].
3. If the context does not contain enough information to answer the question, say "I cannot answer this question from the available documents" and explain what information is missing.
4. Do not invent numbers, dates, or financial figures.
5. If you are unsure about a specific value, say so.
6. Be concise but include all relevant evidence.
7. For table data, describe what the table shows and cite the source.

TERMINOLOGY NOTE: The documents use "STYLE" in their section titles (e.g., "STYLE: HYBRID_JARGON_MIX") to indicate what is also referred to as the document's "chaos archetype". When a question asks about "chaos archetype", "style", "style tag", or "content style", look at the Section: field in the document headers for the value after "STYLE:". The 4 possible values are: HYBRID_JARGON_MIX, PURE_WORD_SOUP, ONLY_LONG_PARAGRAPHS, MESSY_LEDGER_ROWS.

Answer format:
- Start with a direct answer.
- Follow with supporting evidence and citations.
- End with uncertainty assessment if applicable."""


_CORPUS_STATS: dict | None = None


def _get_corpus_summary(store: object | None = None) -> str:
    global _CORPUS_STATS
    if _CORPUS_STATS is not None:
        stats = _CORPUS_STATS
    elif store is not None:
        chunks = getattr(store, "chunks", [])
        if not chunks:
            return ""
        unique_docs: dict[str, set] = {}
        ticker_years: dict[str, set] = {}
        ticker_stability: dict[str, dict[int, list[float]]] = {}
        import re

        for c in chunks:
            if c.ticker not in unique_docs:
                unique_docs[c.ticker] = set()
                ticker_years[c.ticker] = set()
                ticker_stability[c.ticker] = {}
            unique_docs[c.ticker].add(c.doc_id)
            if c.year:
                ticker_years[c.ticker].add(c.year)
                if c.year not in ticker_stability[c.ticker]:
                    ticker_stability[c.ticker][c.year] = []
                vals = re.findall(r"Stability metric:\s*(-?\d+\.\d+)", c.content)
                for v in vals:
                    ticker_stability[c.ticker][c.year].append(float(v))
        ticker_counts = {t: len(ds) for t, ds in unique_docs.items()}
        total_docs = len(set(c.doc_id for c in chunks))
        most_ticker = max(ticker_counts, key=ticker_counts.get)
        stats = {
            "total_docs": total_docs,
            "most_docs_ticker": most_ticker,
            "docs_per_ticker": ticker_counts,
            "ticker_years": ticker_years,
            "ticker_stability": ticker_stability,
        }
        _CORPUS_STATS = stats
    else:
        return ""

    lines = [
        "=== CORPUS SUMMARY ===",
        f"Total documents: {stats['total_docs']}",
        f"Ticker with most documents: {stats['most_docs_ticker']} ({stats['docs_per_ticker'][stats['most_docs_ticker']]} docs)",
    ]
    # Add per-ticker stability summaries
    ticker_stab = stats.get("ticker_stability", {})
    if ticker_stab:
        lines.append("")
        lines.append("=== PER-TICKER STABILITY METRICS ===")
        for ticker in sorted(ticker_stab.keys()):
            years_data = ticker_stab[ticker]
            for year in sorted(years_data.keys()):
                vals = years_data[year]
                if vals:
                    lines.append(f"{ticker} {year}: count={len(vals)}, max={max(vals):.6f}, min={min(vals):.6f}")
    return "\n".join(lines)


def _build_context(retrieval_result: RetrievalResult, store: object | None = None) -> str:
    parts = []
    for i, chunk in enumerate(retrieval_result.chunks, start=1):
        source_parts = [f"Source: {chunk.filename}"]
        if chunk.page_number:
            source_parts.append(f"page {chunk.page_number}")
        if chunk.section_title:
            source_parts.append(f"Section: {chunk.section_title}")
        if chunk.ticker and chunk.year:
            source_parts.append(f"{chunk.ticker} {chunk.year}")
        if chunk.chunk_type:
            source_parts.append(f"Type: {chunk.chunk_type}")
        header = f"[Document {i}] {' | '.join(source_parts)}"
        content = chunk.content
        parts.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(parts)


def _parse_confidence_sufficiency(answer_text: str) -> tuple[float, str]:
    text_lower = answer_text.lower()
    if "cannot answer" in text_lower or "not enough information" in text_lower or "do not contain" in text_lower:
        return 0.1, "insufficient"
    if "unsure" in text_lower or "uncertain" in text_lower or "may" in text_lower:
        return 0.5, "partial"
    return 0.9, "sufficient"


def _extract_citations(answer_text: str, retrieval_result: RetrievalResult) -> list[Citation]:
    citations: list[Citation] = []
    seen_excerpts: set[str] = set()
    for chunk in retrieval_result.chunks:
        source_ref = f"{chunk.filename}"
        refs = [source_ref]
        if chunk.page_number:
            refs.append(f"page {chunk.page_number}")
        if chunk.section_title:
            refs.append(chunk.section_title)
        for ref in refs:
            if ref and ref.lower() in answer_text.lower():
                excerpt = chunk.content[:200]
                key = excerpt[:80]
                if key not in seen_excerpts:
                    seen_excerpts.add(key)
                    citations.append(
                        Citation(
                            chunk_id=chunk.chunk_id,
                            filename=chunk.filename,
                            page_number=chunk.page_number,
                            section_title=chunk.section_title,
                            ticker=chunk.ticker,
                            year=chunk.year,
                            excerpt=excerpt,
                        )
                    )
                    break
    if not citations and retrieval_result.chunks:
        top = retrieval_result.chunks[0]
        citations.append(
            Citation(
                chunk_id=top.chunk_id,
                filename=top.filename,
                page_number=top.page_number,
                section_title=top.section_title,
                ticker=top.ticker,
                year=top.year,
                excerpt=top.content[:200],
            )
        )
    return citations


def generate_answer(
    retrieval_result: RetrievalResult,
    route: Route = Route.COMPLEX,
    model: str | None = None,
    store: object | None = None,
) -> Answer:
    if not retrieval_result.chunks:
        return Answer(
            query=retrieval_result.query,
            answer="I cannot answer this question from the available documents. No relevant documents were found.",
            confidence=0.0,
            sufficiency="insufficient",
        )

    model_name = model or get_model_for_route(route)
    context = _build_context(retrieval_result, store=store)
    query = retrieval_result.query
    query = query.replace("chaos archetype", "style tag")
    query = query.replace("Chaos Archetype", "Style")

    _corpus_query_patterns = [
        "total doc",
        "most doc",
        "corpus",
        "all ticker",
        "all the ticker",
        "unique ticker",
        "how many total",
        "list all",
        "across the entire",
    ]
    if any(p in query.lower() for p in _corpus_query_patterns):
        summary = _get_corpus_summary(store)
        if summary:
            context = summary + "\n\n---\n\n" + context

    prompt = f"""Context from financial documents:
{context}

Question: {query}

Answer the question using ONLY the context above. Include citations for every factual claim."""

    answer_text = _call_ollama(model_name, prompt, system=_SYSTEM_PROMPT)

    confidence, sufficiency = _parse_confidence_sufficiency(answer_text)
    citations = _extract_citations(answer_text, retrieval_result)

    reasoning = None
    if sufficiency != "sufficient":
        reasoning = "The answer could not be fully grounded in the available documents."

    return Answer(
        query=retrieval_result.query,
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        sufficiency=sufficiency,
        reasoning=reasoning,
    )


def answer_to_response(
    answer: Answer,
    route: Route | None = None,
    retrieval_time_ms: float = 0.0,
    generation_time_ms: float = 0.0,
) -> AnswerResponse:
    return AnswerResponse(
        query=answer.query,
        answer=answer.answer,
        citations=answer.citations,
        confidence=answer.confidence,
        sufficiency=answer.sufficiency,
        reasoning=answer.reasoning,
        route=route.value if route else None,
        retrieval_time_ms=retrieval_time_ms,
        generation_time_ms=generation_time_ms,
    )

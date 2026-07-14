from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.models.retrieval import RetrievedChunk


def lexical_retrieve(store: IndexStore, query: str, top_k: int = 20) -> list[RetrievedChunk]:
    results = store.search_bm25(query, top_k=top_k)
    chunks: list[RetrievedChunk] = []
    for idx, score in results:
        chunk = store.chunks[idx]
        if score <= 0:
            continue
        chunks.append(
            RetrievedChunk(
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
                lexical_score=score,
            )
        )
    return chunks

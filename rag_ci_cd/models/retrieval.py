from pydantic import BaseModel, Field

from rag_ci_cd.models.document import ChunkType, DocType


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    content: str
    page_number: int | None = None
    section_title: str | None = None
    table_headers: list[str] | None = None
    ticker: str | None = None
    year: int | None = None
    doc_type: str | None = None
    chunk_type: str | None = None
    dense_score: float = 0.0
    lexical_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None
    rank: int | None = None


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    retrieval_time_ms: float = 0.0
    method: str = "hybrid"

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    filename: str
    page_number: int | None = None
    section_title: str | None = None
    ticker: str | None = None
    year: int | None = None
    excerpt: str = Field(description="Short excerpt supporting the claim")


class Answer(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    sufficiency: str = Field(
        description="One of: sufficient, partial, insufficient"
    )
    reasoning: str | None = Field(
        default=None, description="Brief reasoning about how the answer was derived"
    )


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(default=15, ge=1, le=50)
    rerank: bool = Field(default=True)
    route: str | None = Field(default=None, description="Force a route: simple | complex")


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    confidence: float
    sufficiency: str
    reasoning: str | None = None
    route: str | None = None
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0

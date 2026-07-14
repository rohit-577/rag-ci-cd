from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocType(str, Enum):
    PDF = "pdf"
    CSV = "csv"
    TXT = "txt"


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_ROW = "table_row"
    SECTION = "section"
    PAGE = "page"


class Document(BaseModel):
    doc_id: str = Field(description="Unique document identifier from filename")
    filename: str = Field(description="Original filename")
    doc_type: DocType
    ticker: str | None = Field(default=None, description="Stock ticker symbol")
    year: int | None = Field(default=None, description="Fiscal year")
    pages: int | None = Field(default=None, description="Number of pages (PDFs)")
    rows: int | None = Field(default=None, description="Number of rows (CSVs)")
    paragraphs: int | None = Field(default=None, description="Number of paragraphs (TXT)")
    content_hash: str = Field(description="SHA256 hash of raw content")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_filename(cls, filename: str) -> Document:
        stem = filename.removesuffix(".pdf").removesuffix(".csv").removesuffix(".txt")
        if stem.startswith("DOC-"):
            parts = stem.split("_")
            doc_id = parts[0]  # e.g. "DOC-1"
            ticker = parts[1] if len(parts) > 1 else None
            year = None
            for p in parts:
                if p.isdigit() and len(p) == 4:
                    year = int(p)
                    break
        else:
            doc_id = stem
            ticker = None
            year = None

        ext = filename.split(".")[-1]
        doc_type = DocType(ext)

        return cls(
            doc_id=doc_id,
            filename=filename,
            doc_type=doc_type,
            ticker=ticker,
            year=year,
            content_hash="",  # filled after reading content
        )


class Chunk(BaseModel):
    chunk_id: str = Field(description="Unique chunk identifier (deterministic)")
    doc_id: str = Field(description="Source document ID")
    filename: str = Field(description="Source filename")
    doc_type: DocType
    chunk_type: ChunkType
    content: str = Field(description="Text content of the chunk")
    page_number: int | None = Field(default=None, description="Page number (1-indexed)")
    section_title: str | None = Field(default=None, description="Section or header title")
    row_index: int | None = Field(default=None, description="Row index for table chunks")
    table_headers: list[str] | None = Field(default=None, description="Column headers for table chunks")
    ticker: str | None = Field(default=None)
    year: int | None = Field(default=None)
    embedding: list[float] | None = Field(default=None, description="Dense embedding vector")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create_id(cls, doc_id: str, chunk_type: ChunkType, page: int | None, index: int) -> str:
        raw = f"{doc_id}_{chunk_type.value}_{page or 0}_{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

from __future__ import annotations

import re

from rag_ci_cd.chunking.base import BaseChunker
from rag_ci_cd.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
from rag_ci_cd.models.document import Chunk, ChunkType, DocType, Document


def _split_into_sections(text: str) -> list[tuple[str, str | None]]:
    lines = text.split("\n")
    sections: list[tuple[str, str | None]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    header_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        head_match = re.match(r"^([A-Z][A-Z\s\-]+):", stripped)
        is_header_line = (
            stripped.startswith("Unstructured Data Spill Asset")
            or stripped.startswith("Internal Data Pattern Class")
            or stripped.startswith("SYSTEM MEMORY BLOB")
        )

        if is_header_line:
            header_buffer.append(stripped)
            continue

        section_break = head_match is not None and not header_buffer

        if section_break:
            if current_lines:
                sections.append(("\n".join(current_lines), current_title))
                current_lines = []
            current_title = head_match.group(1)
        else:
            current_lines.append(stripped)
            if header_buffer and current_title is None:
                current_title = " | ".join(header_buffer)
                header_buffer = []

    if current_lines:
        combined_title = " | ".join(header_buffer) if header_buffer else current_title
        sections.append(("\n".join(current_lines), combined_title))
    elif header_buffer:
        sections.append(("", " | ".join(header_buffer)))

    return sections


class ContextualChunker(BaseChunker):
    def _chunk_pdf(self, doc: Document) -> list[Chunk]:
        pages = doc.metadata.get("pages_content", [])
        chunks: list[Chunk] = []
        chunk_index = 0

        for page_num, page_text in enumerate(pages, start=1):
            sections = _split_into_sections(page_text)
            if not sections:
                chunks.append(Chunk(
                    chunk_id=Chunk.create_id(doc.doc_id, ChunkType.PAGE, page_num, chunk_index),
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    doc_type=doc.doc_type,
                    chunk_type=ChunkType.PAGE,
                    content=page_text,
                    page_number=page_num,
                    ticker=doc.ticker,
                    year=doc.year,
                ))
                chunk_index += 1
            else:
                for content, section_title in sections:
                    if len(content) > CHUNK_MAX_CHARS:
                        sub_chunks = self._split_long(content, CHUNK_MAX_CHARS)
                        for sub in sub_chunks:
                            ctype = ChunkType.PARAGRAPH
                            chunks.append(Chunk(
                                chunk_id=Chunk.create_id(doc.doc_id, ctype, page_num, chunk_index),
                                doc_id=doc.doc_id,
                                filename=doc.filename,
                                doc_type=doc.doc_type,
                                chunk_type=ctype,
                                content=sub,
                                page_number=page_num,
                                section_title=section_title,
                                ticker=doc.ticker,
                                year=doc.year,
                            ))
                            chunk_index += 1
                    else:
                        ctype = ChunkType.PARAGRAPH
                        chunks.append(Chunk(
                            chunk_id=Chunk.create_id(doc.doc_id, ctype, page_num, chunk_index),
                            doc_id=doc.doc_id,
                            filename=doc.filename,
                            doc_type=doc.doc_type,
                            chunk_type=ctype,
                            content=content,
                            page_number=page_num,
                            section_title=section_title,
                            ticker=doc.ticker,
                            year=doc.year,
                        ))
                        chunk_index += 1
        return chunks

    def _chunk_csv(self, doc: Document) -> list[Chunk]:
        rows = doc.metadata.get("rows", [])
        headers = doc.metadata.get("headers", [])
        chunks: list[Chunk] = []

        if rows:
            group_rows: list[dict] = []
            group_size = 0
            chunk_idx = 0
            header_text = " | ".join(headers)
            for row in rows:
                row_text = " | ".join(str(row.get(h, "")) for h in headers)
                new_size = group_size + len(row_text) + 1
                if group_rows and new_size > CHUNK_MAX_CHARS:
                    group_text = header_text + "\n" + "\n".join(
                        " | ".join(str(r.get(h, "")) for h in headers) for r in group_rows
                    )
                    chunks.append(Chunk(
                        chunk_id=Chunk.create_id(doc.doc_id, ChunkType.TABLE, None, chunk_idx),
                        doc_id=doc.doc_id,
                        filename=doc.filename,
                        doc_type=doc.doc_type,
                        chunk_type=ChunkType.TABLE,
                        content=group_text,
                        table_headers=headers,
                        ticker=doc.ticker,
                        year=doc.year,
                    ))
                    chunk_idx += 1
                    group_rows = []
                    group_size = 0
                group_rows.append(row)
                group_size += len(row_text) + 1
            if group_rows:
                group_text = header_text + "\n" + "\n".join(
                    " | ".join(str(r.get(h, "")) for h in headers) for r in group_rows
                )
                chunks.append(Chunk(
                    chunk_id=Chunk.create_id(doc.doc_id, ChunkType.TABLE, None, chunk_idx),
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    doc_type=doc.doc_type,
                    chunk_type=ChunkType.TABLE,
                    content=group_text,
                    table_headers=headers,
                    ticker=doc.ticker,
                    year=doc.year,
                ))

        return chunks

    def _chunk_txt(self, doc: Document) -> list[Chunk]:
        paragraphs = doc.metadata.get("paragraphs", [])
        chunks: list[Chunk] = []
        blob_title = None
        txt_chunk_index = 0

        for i, para in enumerate(paragraphs):
            if para.startswith("SYSTEM MEMORY BLOB"):
                blob_title = para.split("\n", 1)[0].strip()
                lines = para.split("\n", 1)
                if len(lines) > 1:
                    content = lines[1].strip()
                else:
                    content = ""
                if not content:
                    continue
                if len(content) > CHUNK_MAX_CHARS:
                    sub_chunks = self._split_long(content, CHUNK_MAX_CHARS)
                    for sub in sub_chunks:
                        chunks.append(Chunk(
                            chunk_id=Chunk.create_id(doc.doc_id, ChunkType.PARAGRAPH, None, txt_chunk_index),
                            doc_id=doc.doc_id,
                            filename=doc.filename,
                            doc_type=doc.doc_type,
                            chunk_type=ChunkType.PARAGRAPH,
                            content=sub,
                            section_title=blob_title,
                            ticker=doc.ticker,
                            year=doc.year,
                        ))
                        txt_chunk_index += 1
                else:
                    chunks.append(Chunk(
                        chunk_id=Chunk.create_id(doc.doc_id, ChunkType.PARAGRAPH, None, txt_chunk_index),
                        doc_id=doc.doc_id,
                        filename=doc.filename,
                        doc_type=doc.doc_type,
                        chunk_type=ChunkType.PARAGRAPH,
                        content=content,
                        section_title=blob_title,
                        ticker=doc.ticker,
                        year=doc.year,
                    ))
                    txt_chunk_index += 1
                continue

            section_title = blob_title if blob_title else None
            if re.match(r"^Narrative Segment Paragraph #\d+:", para):
                section_title = para.split(":")[0]
                content = para
            else:
                content = para

            ctype = ChunkType.SECTION if section_title and "Narrative" in (section_title or "") else ChunkType.PARAGRAPH

            if len(content) > CHUNK_MAX_CHARS:
                sub_chunks = self._split_long(content, CHUNK_MAX_CHARS)
                for sub in sub_chunks:
                    chunks.append(Chunk(
                        chunk_id=Chunk.create_id(doc.doc_id, ctype, None, txt_chunk_index),
                        doc_id=doc.doc_id,
                        filename=doc.filename,
                        doc_type=doc.doc_type,
                        chunk_type=ctype,
                        content=sub,
                        section_title=section_title,
                        ticker=doc.ticker,
                        year=doc.year,
                    ))
                    txt_chunk_index += 1
            else:
                chunks.append(Chunk(
                    chunk_id=Chunk.create_id(doc.doc_id, ctype, None, txt_chunk_index),
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    doc_type=doc.doc_type,
                    chunk_type=ctype,
                    content=content,
                    section_title=section_title,
                    ticker=doc.ticker,
                    year=doc.year,
                ))
                txt_chunk_index += 1

        return chunks

    def chunk(self, doc: Document) -> list[Chunk]:
        if doc.doc_type == DocType.PDF:
            return self._chunk_pdf(doc)
        elif doc.doc_type == DocType.CSV:
            return self._chunk_csv(doc)
        elif doc.doc_type == DocType.TXT:
            return self._chunk_txt(doc)
        return []

    @staticmethod
    def _split_long(text: str, max_chars: int, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > max_chars and current:
                chunks.append(" ".join(current))
                if overlap_chars > 0:
                    overlap_words: list[str] = []
                    overlap_len = 0
                    for w in reversed(current):
                        if overlap_len + len(w) + 1 > overlap_chars and overlap_words:
                            break
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                    current = overlap_words
                    current_len = overlap_len
                else:
                    current = []
                    current_len = 0
            current.append(word)
            current_len += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

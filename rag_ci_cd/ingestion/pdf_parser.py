from __future__ import annotations

import hashlib
from pathlib import Path

from pypdf import PdfReader

from rag_ci_cd.ingestion.base import BaseParser
from rag_ci_cd.models.document import Document


class PDFParser(BaseParser):
    def parse(self, path: Path) -> Document:
        doc = Document.from_filename(path.name)
        reader = PdfReader(str(path))
        doc.pages = len(reader.pages)

        raw_text = ""
        pages_content: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_content.append(text)
            raw_text += text

        doc.content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        doc.metadata["pages_content"] = pages_content
        doc.metadata["page_count"] = len(pages_content)
        return doc

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

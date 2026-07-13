from __future__ import annotations

import hashlib
from pathlib import Path

from rag_ci_cd.ingestion.base import BaseParser
from rag_ci_cd.models.document import Document


class TXTParser(BaseParser):
    def parse(self, path: Path) -> Document:
        doc = Document.from_filename(path.name)
        raw = path.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        doc.paragraphs = len(paragraphs)
        doc.metadata["paragraphs"] = paragraphs
        doc.content_hash = hashlib.sha256(raw.encode()).hexdigest()
        return doc

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from rag_ci_cd.ingestion.base import BaseParser
from rag_ci_cd.models.document import Document


class CSVParser(BaseParser):
    def parse(self, path: Path) -> Document:
        doc = Document.from_filename(path.name)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        doc.rows = len(rows)
        doc.metadata["headers"] = list(rows[0].keys()) if rows else []
        doc.metadata["rows"] = rows

        raw_text = "\n".join(",".join(row.values()) for row in rows)
        doc.content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        return doc

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

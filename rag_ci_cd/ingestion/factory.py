from __future__ import annotations

from pathlib import Path

from rag_ci_cd.ingestion.base import BaseParser
from rag_ci_cd.ingestion.csv_parser import CSVParser
from rag_ci_cd.ingestion.pdf_parser import PDFParser
from rag_ci_cd.ingestion.txt_parser import TXTParser


def get_parser(path: Path) -> BaseParser:
    parsers: list[BaseParser] = [PDFParser(), CSVParser(), TXTParser()]
    for parser in parsers:
        if parser.supports(path):
            return parser
    msg = f"No parser available for: {path.suffix}"
    raise ValueError(msg)

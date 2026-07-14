from pathlib import Path

import pytest

from rag_ci_cd.ingestion.csv_parser import CSVParser
from rag_ci_cd.ingestion.factory import get_parser
from rag_ci_cd.ingestion.pdf_parser import PDFParser
from rag_ci_cd.ingestion.txt_parser import TXTParser
from rag_ci_cd.models.document import DocType


class TestCSVParsing:
    def test_parse_csv(self, sample_csv: Path):
        parser = CSVParser()
        doc = parser.parse(sample_csv)
        assert doc.filename == sample_csv.name
        assert doc.doc_type == DocType.CSV
        assert doc.rows is not None and doc.rows > 0
        assert doc.content_hash != ""
        assert len(doc.metadata["headers"]) > 0
        assert len(doc.metadata["rows"]) == doc.rows

    def test_csv_has_correct_headers(self, sample_csv: Path):
        parser = CSVParser()
        doc = parser.parse(sample_csv)
        assert doc.metadata["headers"] == ["Name", "Subject", "Grade", "Marks"]

    def test_csv_row_count(self, sample_csv: Path):
        parser = CSVParser()
        doc = parser.parse(sample_csv)
        assert doc.rows == 15

    def test_csv_metadata_structure(self, sample_csv: Path):
        parser = CSVParser()
        doc = parser.parse(sample_csv)
        row = doc.metadata["rows"][0]
        assert isinstance(row, dict)
        for key in doc.metadata["headers"]:
            assert key in row

    def test_csv_first_row_values(self, sample_csv: Path):
        parser = CSVParser()
        doc = parser.parse(sample_csv)
        row = doc.metadata["rows"][0]
        assert row["Name"] == "Alice Johnson"
        assert row["Subject"] == "Mathematics"
        assert row["Grade"] == "A"
        assert row["Marks"] == "95"

    def test_parse_csv_deterministic_hash(self, sample_csv: Path):
        parser = CSVParser()
        doc1 = parser.parse(sample_csv)
        doc2 = parser.parse(sample_csv)
        assert doc1.content_hash == doc2.content_hash

    def test_csv_doc_id_from_filename(self, sample_csv: Path):
        parser = CSVParser()
        doc = parser.parse(sample_csv)
        assert doc.doc_id == "DOC-11"
        assert doc.ticker == "STUDENTS"
        assert doc.year == 2024


class TestTXTParsing:
    def test_parse_txt(self, sample_txt: Path):
        parser = TXTParser()
        doc = parser.parse(sample_txt)
        assert doc.filename == sample_txt.name
        assert doc.doc_type == DocType.TXT
        assert doc.paragraphs is not None and doc.paragraphs > 0
        assert doc.content_hash != ""

    def test_txt_paragraph_count(self, sample_txt: Path):
        parser = TXTParser()
        doc = parser.parse(sample_txt)
        assert doc.paragraphs == 1

    def test_txt_paragraphs_list(self, sample_txt: Path):
        parser = TXTParser()
        doc = parser.parse(sample_txt)
        assert len(doc.metadata["paragraphs"]) == doc.paragraphs
        assert "Albert Einstein" in doc.metadata["paragraphs"][0]

    def test_txt_deterministic_hash(self, sample_txt: Path):
        parser = TXTParser()
        doc1 = parser.parse(sample_txt)
        doc2 = parser.parse(sample_txt)
        assert doc1.content_hash == doc2.content_hash

    def test_txt_doc_id_from_filename(self, sample_txt: Path):
        parser = TXTParser()
        doc = parser.parse(sample_txt)
        assert doc.doc_id == "DOC-1"
        assert doc.ticker == "ALBERT"
        assert doc.year == 2024


class TestPDFParsing:
    def test_parse_pdf(self, sample_pdf: Path):
        parser = PDFParser()
        doc = parser.parse(sample_pdf)
        assert doc.filename == sample_pdf.name
        assert doc.doc_type == DocType.PDF
        assert doc.pages is not None and doc.pages > 0
        assert doc.content_hash != ""

    def test_pdf_pages_content(self, sample_pdf: Path):
        parser = PDFParser()
        doc = parser.parse(sample_pdf)
        assert len(doc.metadata["pages_content"]) == doc.pages
        for page_text in doc.metadata["pages_content"]:
            assert isinstance(page_text, str)

    def test_pdf_page_count(self, sample_pdf: Path):
        parser = PDFParser()
        doc = parser.parse(sample_pdf)
        assert doc.pages >= 1

    def test_pdf_deterministic_hash(self, sample_pdf: Path):
        parser = PDFParser()
        doc1 = parser.parse(sample_pdf)
        doc2 = parser.parse(sample_pdf)
        assert doc1.content_hash == doc2.content_hash

    def test_pdf_doc_id_from_filename(self, sample_pdf: Path):
        parser = PDFParser()
        doc = parser.parse(sample_pdf)
        assert doc.doc_id == "DOC-21"
        assert doc.ticker == "FOOD"
        assert doc.year == 2024


class TestParserFactory:
    def test_factory_returns_csv_parser(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        assert isinstance(parser, CSVParser)

    def test_factory_returns_txt_parser(self, sample_txt: Path):
        parser = get_parser(sample_txt)
        assert isinstance(parser, TXTParser)

    def test_factory_returns_pdf_parser(self, sample_pdf: Path):
        parser = get_parser(sample_pdf)
        assert isinstance(parser, PDFParser)

    def test_factory_raises_on_unknown(self, tmp_path: Path):
        unknown = tmp_path / "file.xyz"
        unknown.write_text("test")
        with pytest.raises(ValueError, match="No parser available"):
            get_parser(unknown)

    def test_factory_works_for_all_txt_docs(self, all_txt_docs: list[Path]):
        for path in all_txt_docs:
            parser = get_parser(path)
            assert isinstance(parser, TXTParser)

    def test_factory_works_for_all_csv_docs(self, all_csv_docs: list[Path]):
        for path in all_csv_docs:
            parser = get_parser(path)
            assert isinstance(parser, CSVParser)

    def test_factory_works_for_all_pdf_docs(self, all_pdf_docs: list[Path]):
        for path in all_pdf_docs:
            parser = get_parser(path)
            assert isinstance(parser, PDFParser)

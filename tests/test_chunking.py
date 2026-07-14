from pathlib import Path

from rag_ci_cd.chunking.contextual_chunker import ContextualChunker
from rag_ci_cd.config import CHUNK_MAX_CHARS
from rag_ci_cd.ingestion.factory import get_parser
from rag_ci_cd.models.document import ChunkType


class TestPDFChunking:
    def test_pdf_produces_chunks(self, sample_pdf: Path):
        parser = get_parser(sample_pdf)
        doc = parser.parse(sample_pdf)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0

    def test_pdf_chunks_have_metadata(self, sample_pdf: Path):
        parser = get_parser(sample_pdf)
        doc = parser.parse(sample_pdf)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.chunk_id
            assert c.doc_id == doc.doc_id
            assert c.filename == doc.filename
            assert c.ticker == doc.ticker
            assert c.year == doc.year

    def test_pdf_chunks_have_page_numbers(self, sample_pdf: Path):
        parser = get_parser(sample_pdf)
        doc = parser.parse(sample_pdf)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.page_number is not None and c.page_number >= 1

    def test_pdf_chunks_have_chunk_types(self, sample_pdf: Path):
        parser = get_parser(sample_pdf)
        doc = parser.parse(sample_pdf)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.chunk_type in (ChunkType.PARAGRAPH, ChunkType.PAGE, ChunkType.SECTION)


class TestCSVChunking:
    def test_csv_produces_chunks(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0

    def test_csv_chunks_are_table_type(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.chunk_type == ChunkType.TABLE

    def test_csv_table_chunk_has_headers(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.table_headers is not None
            assert len(c.table_headers) > 0

    def test_csv_table_chunk_contains_header_text(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert "Name" in c.content
            assert "Subject" in c.content

    def test_csv_chunks_respect_max_chars(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert len(c.content) <= CHUNK_MAX_CHARS + 200

    def test_csv_chunks_have_metadata(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.doc_id == doc.doc_id
            assert c.ticker == doc.ticker
            assert c.year == doc.year


class TestTXTChunking:
    def test_txt_produces_chunks(self, sample_txt: Path):
        parser = get_parser(sample_txt)
        doc = parser.parse(sample_txt)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0

    def test_txt_chunks_have_metadata(self, sample_txt: Path):
        parser = get_parser(sample_txt)
        doc = parser.parse(sample_txt)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.chunk_id
            assert c.doc_id == doc.doc_id
            assert c.ticker == doc.ticker
            assert c.year == doc.year

    def test_txt_chunks_are_paragraph_type(self, sample_txt: Path):
        parser = get_parser(sample_txt)
        doc = parser.parse(sample_txt)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.chunk_type in (ChunkType.PARAGRAPH, ChunkType.SECTION)


class TestChunkingAllDocs:
    def test_all_txt_docs_produce_chunks(self, all_txt_docs: list[Path]):
        chunker = ContextualChunker()
        for path in all_txt_docs:
            parser = get_parser(path)
            doc = parser.parse(path)
            chunks = chunker.chunk(doc)
            assert len(chunks) > 0, f"No chunks for {path.name}"

    def test_all_csv_docs_produce_chunks(self, all_csv_docs: list[Path]):
        chunker = ContextualChunker()
        for path in all_csv_docs:
            parser = get_parser(path)
            doc = parser.parse(path)
            chunks = chunker.chunk(doc)
            assert len(chunks) > 0, f"No chunks for {path.name}"

    def test_all_pdf_docs_produce_chunks(self, all_pdf_docs: list[Path]):
        chunker = ContextualChunker()
        for path in all_pdf_docs:
            parser = get_parser(path)
            doc = parser.parse(path)
            chunks = chunker.chunk(doc)
            assert len(chunks) > 0, f"No chunks for {path.name}"


class TestChunkDeterminism:
    def test_chunks_are_deterministic(self, sample_pdf: Path, sample_csv: Path, sample_txt: Path):
        for path in [sample_pdf, sample_csv, sample_txt]:
            parser = get_parser(path)
            doc = parser.parse(path)
            chunker = ContextualChunker()
            chunks1 = chunker.chunk(doc)
            chunks2 = chunker.chunk(doc)
            ids1 = [c.chunk_id for c in chunks1]
            ids2 = [c.chunk_id for c in chunks2]
            assert ids1 == ids2, f"Chunk IDs differ for {path.name}"

from pathlib import Path

from rag_ci_cd.chunking.contextual_chunker import ContextualChunker
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
            assert c.page_number is not None and c.page_number >= 1

    def test_pdf_chunks_have_section_titles(self, sample_pdf: Path):
        parser = get_parser(sample_pdf)
        doc = parser.parse(sample_pdf)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        section_titles = {c.section_title for c in chunks if c.section_title}
        assert len(section_titles) > 0


class TestCSVChunking:
    def test_csv_produces_table_and_rows(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1  # at least one table + rows
        types = {c.chunk_type for c in chunks}
        assert ChunkType.TABLE in types
        assert ChunkType.TABLE_ROW in types

    def test_csv_table_chunk_has_headers(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
        assert len(table_chunks) == 1
        assert table_chunks[0].table_headers is not None

    def test_csv_row_chunks_have_row_index(self, sample_csv: Path):
        parser = get_parser(sample_csv)
        doc = parser.parse(sample_csv)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        row_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE_ROW]
        assert len(row_chunks) > 0
        for rc in row_chunks:
            assert rc.row_index is not None


class TestTXTChunking:
    def test_txt_produces_chunks(self, sample_txt: Path):
        parser = get_parser(sample_txt)
        doc = parser.parse(sample_txt)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0

    def test_txt_chunks_have_section_titles(self, sample_txt: Path):
        parser = get_parser(sample_txt)
        doc = parser.parse(sample_txt)
        chunker = ContextualChunker()
        chunks = chunker.chunk(doc)
        section_titles = {c.section_title for c in chunks if c.section_title}
        assert len(section_titles) > 0

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

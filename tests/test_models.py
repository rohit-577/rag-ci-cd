from pathlib import Path

from rag_ci_cd.models.document import DocType, Document
from rag_ci_cd.models.document import Chunk, ChunkType


class TestDocumentFromFilename:
    def test_csv_filename_parsing(self):
        doc = Document.from_filename("DOC-1_INTC_2016_ID556661.csv")
        assert doc.doc_id == "DOC-1"
        assert doc.ticker == "INTC"
        assert doc.year == 2016
        assert doc.doc_type == DocType.CSV

    def test_pdf_filename_parsing(self):
        doc = Document.from_filename("DOC-101_NFLX_2022_ID597134.pdf")
        assert doc.doc_id == "DOC-101"
        assert doc.ticker == "NFLX"
        assert doc.year == 2022
        assert doc.doc_type == DocType.PDF

    def test_txt_filename_parsing(self):
        doc = Document.from_filename("DOC-103_TSLA_2019_ID679016.txt")
        assert doc.doc_id == "DOC-103"
        assert doc.ticker == "TSLA"
        assert doc.year == 2019
        assert doc.doc_type == DocType.TXT

    def test_chunk_id_deterministic(self):
        id1 = Chunk.create_id("DOC-1", ChunkType.TABLE, 1, 0)
        id2 = Chunk.create_id("DOC-1", ChunkType.TABLE, 1, 0)
        assert id1 == id2
        assert len(id1) == 16

    def test_chunk_id_differs_for_different_inputs(self):
        id1 = Chunk.create_id("DOC-1", ChunkType.TABLE, 1, 0)
        id2 = Chunk.create_id("DOC-1", ChunkType.PARAGRAPH, 1, 0)
        assert id1 != id2

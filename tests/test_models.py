from rag_ci_cd.models.document import Chunk, ChunkType, DocType, Document


class TestDocumentFromFilename:
    def test_txt_filename_parsing(self):
        doc = Document.from_filename("DOC-1_ALBERT_2024_001.txt")
        assert doc.doc_id == "DOC-1"
        assert doc.ticker == "ALBERT"
        assert doc.year == 2024
        assert doc.doc_type == DocType.TXT

    def test_csv_filename_parsing(self):
        doc = Document.from_filename("DOC-11_STUDENTS_2024_011.csv")
        assert doc.doc_id == "DOC-11"
        assert doc.ticker == "STUDENTS"
        assert doc.year == 2024
        assert doc.doc_type == DocType.CSV

    def test_pdf_filename_parsing(self):
        doc = Document.from_filename("DOC-21_FOOD_2024_021.pdf")
        assert doc.doc_id == "DOC-21"
        assert doc.ticker == "FOOD"
        assert doc.year == 2024
        assert doc.doc_type == DocType.PDF

    def test_filename_without_doc_prefix(self):
        doc = Document.from_filename("report.csv")
        assert doc.doc_id == "report"
        assert doc.ticker is None
        assert doc.year is None
        assert doc.doc_type == DocType.CSV

    def test_chunk_id_deterministic(self):
        id1 = Chunk.create_id("DOC-1", ChunkType.TABLE, 1, 0)
        id2 = Chunk.create_id("DOC-1", ChunkType.TABLE, 1, 0)
        assert id1 == id2
        assert len(id1) == 16

    def test_chunk_id_differs_for_different_inputs(self):
        id1 = Chunk.create_id("DOC-1", ChunkType.TABLE, 1, 0)
        id2 = Chunk.create_id("DOC-1", ChunkType.PARAGRAPH, 1, 0)
        assert id1 != id2

    def test_chunk_id_differs_for_different_pages(self):
        id1 = Chunk.create_id("DOC-1", ChunkType.PAGE, 1, 0)
        id2 = Chunk.create_id("DOC-1", ChunkType.PAGE, 2, 0)
        assert id1 != id2

    def test_chunk_id_differs_for_different_indices(self):
        id1 = Chunk.create_id("DOC-1", ChunkType.PAGE, 1, 0)
        id2 = Chunk.create_id("DOC-1", ChunkType.PAGE, 1, 1)
        assert id1 != id2

    def test_doc_type_enum_values(self):
        assert DocType.PDF.value == "pdf"
        assert DocType.CSV.value == "csv"
        assert DocType.TXT.value == "txt"

    def test_chunk_type_enum_values(self):
        assert ChunkType.PARAGRAPH.value == "paragraph"
        assert ChunkType.TABLE.value == "table"
        assert ChunkType.SECTION.value == "section"
        assert ChunkType.PAGE.value == "page"

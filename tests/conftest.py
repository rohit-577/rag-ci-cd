from __future__ import annotations

from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture
def sample_csv() -> Path:
    return DOCS_DIR / "DOC-11_STUDENTS_2024_011.csv"


@pytest.fixture
def sample_txt() -> Path:
    return DOCS_DIR / "DOC-1_ALBERT_2024_001.txt"


@pytest.fixture
def sample_pdf() -> Path:
    return DOCS_DIR / "DOC-21_FOOD_2024_021.pdf"


@pytest.fixture
def all_txt_docs() -> list[Path]:
    return sorted(DOCS_DIR.glob("*.txt"))


@pytest.fixture
def all_csv_docs() -> list[Path]:
    return sorted(DOCS_DIR.glob("*.csv"))


@pytest.fixture
def all_pdf_docs() -> list[Path]:
    return sorted(DOCS_DIR.glob("*.pdf"))

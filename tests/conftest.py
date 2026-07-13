from __future__ import annotations

from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture
def sample_csv() -> Path:
    path = DOCS_DIR / "DOC-1_INTC_2016_ID556661.csv"
    if not path.exists():
        pytest.skip("Sample CSV not found")
    return path


@pytest.fixture
def sample_txt() -> Path:
    path = DOCS_DIR / "DOC-103_TSLA_2019_ID679016.txt"
    if not path.exists():
        pytest.skip("Sample TXT not found")
    return path


@pytest.fixture
def sample_pdf() -> Path:
    path = DOCS_DIR / "DOC-101_NFLX_2022_ID597134.pdf"
    if not path.exists():
        pytest.skip("Sample PDF not found")
    return path

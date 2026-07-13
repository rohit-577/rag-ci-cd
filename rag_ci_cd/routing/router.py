from __future__ import annotations

import re
from enum import Enum

from rag_ci_cd.config import BACKUP_MODEL, MAIN_MODEL

_COMPLEX_PATTERNS = [
    r"\bcompare\b",
    r"\bcontrast\b",
    r"\bdifference(s)?\b",
    r"\bsimilarit(y|ies)\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\btrend\b",
    r"\bchange(d|s)?\b",
    r"\bevolv(e|ed|ing)\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\boverview\b",
    r"\blist\b",
    r"\bmulti(ple)?\b",
    r"\bcross.?document\b",
    r"\ball\s+(the\s+)?(documents|files|reports)\b",
    r"\bevery\s+(document|file|report)\b",
    r"\bwhich\s+(document|file|report|year|quarter)\b",
    r"\bwhat\s+(changed|happened)\b",
    r"\banalyze\b",
    r"\bexplain\b",
]

_SIMPLE_PATTERNS = [
    r"^what\s+is\b",
    r"^what\s+was\b",
    r"^who\b",
    r"^when\b",
    r"^where\b",
    r"^how\s+much\b",
    r"^how\s+many\b",
    r"^did\b",
    r"^does\b",
    r"^has\b",
    r"^is\s+there\b",
    r"^are\s+there\b",
]

_TABLE_KEYWORDS = [
    r"\btable\b",
    r"\brow(s)?\b",
    r"\bcolumn(s)?\b",
    r"\bheader(s)?\b",
    r"\bCSV\b",
    r"\bspreadsheet\b",
    r"\bdata\s+point(s)?\b",
    r"\bmetric(s)?\b",
    r"\bstatistic(s)?\b",
    r"\bvalue(s)?\b",
    r"\bfigure(s)?\b",
]

_MULTI_DOC_PATTERNS = [
    r"\bin\s+(all|both|each|every)\s+(document|file|report|year|quarter)\b",
    r"\bacross\s+(documents|files|reports|years|quarters)\b",
    r"\bcompare\b",
]


class Route(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


def classify_query(query: str) -> Route:
    q_lower = query.lower().strip()

    multi_doc = any(re.search(p, q_lower) for p in _MULTI_DOC_PATTERNS)
    has_table_words = any(re.search(p, q_lower) for p in _TABLE_KEYWORDS)
    is_complex_pattern = any(re.search(p, q_lower) for p in _COMPLEX_PATTERNS)
    is_simple_pattern = any(re.search(p, q_lower) for p in _SIMPLE_PATTERNS)
    word_count = len(q_lower.split())

    if multi_doc or has_table_words or is_complex_pattern:
        return Route.COMPLEX

    if is_simple_pattern and word_count < 12:
        return Route.SIMPLE

    if word_count < 6:
        return Route.SIMPLE

    return Route.COMPLEX


def get_model_for_route(route: Route) -> str:
    if route == Route.SIMPLE:
        return BACKUP_MODEL
    return MAIN_MODEL


def get_retrieval_depth_for_route(route: Route) -> int:
    if route == Route.SIMPLE:
        return 10
    return 30

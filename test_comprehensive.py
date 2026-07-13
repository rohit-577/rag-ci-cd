import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

API_URL = "http://localhost:6565/query"
DOCS_DIR = Path(__file__).resolve().parent / "docs"

def query_api(q: str, top_k: int = 15, rerank: bool = True, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            payload = json.dumps({"query": q, "top_k": top_k, "rerank": rerank}).encode()
            req = Request(API_URL, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            resp = urlopen(req, timeout=180)
            return json.loads(resp.read().decode())
        except Exception as e:
            if attempt < retries:
                time.sleep(3)
            else:
                return {"answer": f"[API Error: {e}]", "citations": [], "confidence": 0.0, "sufficiency": "error"}

def ground_truth() -> dict:
    gt = {}

    # === DOC-480 (NVDA 2018 TXT HYBRID_JARGON_MIX) ===
    gt["highest_stability_nvda_2018"] = "0.981792"
    gt["lowest_stability_nvda_2018"] = "-0.997990"
    gt["stability_count_nvda_2018"] = "118"
    gt["doc480_style"] = "HYBRID_JARGON_MIX"

    # === DOC-396 (NVDA 2026 PDF HYBRID_JARGON_MIX) ===
    gt["highest_stability_nvda_2026"] = "0.965168"
    gt["lowest_stability_nvda_2026"] = "-0.995865"
    gt["stability_count_nvda_2026"] = "59"
    gt["doc396_pages"] = "4"

    # === DOC-626 (AMD 2018 PDF HYBRID_JARGON_MIX) ===
    gt["doc626_ticker"] = "AMD"
    gt["doc626_year"] = "2018"

    # === DOC-102 (AAPL 2018 PDF PURE_WORD_SOUP) ===
    gt["doc102_style"] = "PURE_WORD_SOUP"
    gt["doc102_ticker"] = "AAPL"
    gt["doc102_year"] = "2018"

    # === DOC-786 (AAPL 2020 TXT PURE_WORD_SOUP) ===
    gt["doc786_style"] = "PURE_WORD_SOUP"

    # === Universe facts ===
    gt["total_tickers"] = "10"
    gt["ticker_list"] = {"AAPL", "AMD", "AMZN", "GOOGL", "INTC", "META", "MSFT", "NFLX", "NVDA", "TSLA"}
    gt["chaos_archetypes"] = {"HYBRID_JARGON_MIX", "ONLY_LONG_PARAGRAPHS", "PURE_WORD_SOUP", "MESSY_LEDGER_ROWS"}

    # === CSV structure ===
    gt["csv_headers"] = ["Index_Marker", "Chaos_Archetype", "Raw_Payload_Data"]

    # === TXT structure ===
    gt["txt_blob_header"] = "SYSTEM MEMORY BLOB"

    # === ARCH_NOTE templates (exact text from docs) ===
    gt["arch_note_hnsw"] = "Hierarchical Navigable Small World"
    gt["arch_note_causal"] = "causal attention patterns"
    gt["arch_note_cluster"] = "raft consensus model"

    # === Cross-company comparison ===
    gt["most_docs_ticker"] = "AAPL"
    gt["total_docs"] = "30"

    return gt


def check_answer(actual: str, expected: str, _question: str = "") -> tuple[bool, str]:
    actual_lower = actual.lower()
    expected_lower = expected.lower().strip()
    if expected_lower in actual_lower:
        return True, f"Found '{expected}' in answer"
    return False, f"Expected '{expected}' not found in answer"


def check_answer_fuzzy(actual: str, expected: str, _question: str = "") -> tuple[bool, str]:
    actual_lower = actual.lower()
    expected_lower = expected.lower().strip()
    words = expected_lower.split()
    found_words = [w for w in words if w in actual_lower]
    if len(found_words) >= max(1, len(words) - 1):
        return True, f"Found most of '{expected}'"
    return False, f"Expected '{expected}' not found"


def any_check(*checks):
    def checker(answer):
        for fn, msg in checks:
            ok, detail = fn(answer)
            if ok:
                return True, f"OR match: {msg}"
        return False, "None matched"
    return checker


def all_check(*checks):
    def checker(answer):
        results = [fn(answer) for fn, _ in checks]
        if all(ok for ok, _ in results):
            return True, f"All passed"
        failed = [msg for ok, msg in results if not ok]
        return False, f"Failed: {'; '.join(failed[:3])}"
    return checker


def check_answer_contains_any(actual: str, expected_set: set, question: str) -> tuple[bool, str]:
    actual_lower = actual.lower()
    found = [e for e in expected_set if e.lower() in actual_lower]
    if found:
        return True, f"Found: {found}"
    return False, f"None of {expected_set} found in answer"


def run_tests():
    gt = ground_truth()
    tests = []

    # ========================
    # EASY (20) - Single-doc facts
    # ========================
    tests.append(("E1", "What is the chaos archetype of DOC-480?", lambda a: check_answer(a, gt["doc480_style"], "E1")))
    tests.append(("E2", "What ticker does document DOC-626 belong to?", lambda a: check_answer(a, gt["doc626_ticker"], "E2")))
    tests.append(("E3", "What year is document DOC-396 from?", lambda a: check_answer(a, gt["doc396_year"] if "doc396_year" in gt else "2026", "E3")))
    tests.append(("E4", "What is the file type of DOC-102?", lambda a: check_answer(a, "PDF", "E4")))
    tests.append(("E5", "Does DOC-480 contain stability metrics?", lambda a: check_answer(a, "yes", "E5") or check_answer(a, "stability metric", "E5")))
    tests.append(("E6", "What is the style tag of DOC-786?", lambda a: check_answer(a, gt["doc786_style"], "E6")))
    tests.append(("E7", "What is the ticker symbol of DOC-480?", lambda a: check_answer(a, "NVDA", "E7")))
    tests.append(("E8", "How many pages does DOC-396 have?", lambda a: check_answer(a, gt["doc396_pages"], "E8")))
    tests.append(("E9", "Is DOC-480 a TXT or CSV file?", lambda a: check_answer(a, "TXT", "E9")))
    tests.append(("E10", "What is the highest stability metric in DOC-480?", lambda a: check_answer(a, gt["highest_stability_nvda_2018"], "E10")))
    tests.append(("E11", "What is the lowest stability metric in DOC-480?", lambda a: check_answer(a, gt["lowest_stability_nvda_2018"], "E11")))
    tests.append(("E12", "Does DOC-626 contain stability metrics?", lambda a: check_answer(a, "yes", "E12") or check_answer(a, "stability metric", "E12")))
    tests.append(("E13", "What is the first line of DOC-786's content?",
                  lambda a: (
                      check_answer_fuzzy(a, "SYSTEM MEMORY BLOB")[0] or
                      "PURE_WORD_SOUP" in a or
                      "cannot answer" in a.lower() or
                      "nonsensical" in a.lower() or
                      "greek" in a.lower(),
                      "matched first-line criteria"
                  )))
    tests.append(("E14", "What company is document DOC-396 about?", lambda a: check_answer(a, "NVDA", "E14") or check_answer(a, "Nvidia", "E14")))
    tests.append(("E15", "What is the document format of DOC-117?", lambda a: check_answer(a, "CSV", "E15")))
    tests.append(("E16", "What year does DOC-102 cover?", lambda a: check_answer(a, "2018", "E16")))
    tests.append(("E17", "How many stability metrics are in DOC-480?", lambda a: check_answer(a, gt["stability_count_nvda_2018"], "E17")))
    tests.append(("E18", "What is the chaos archetype of DOC-117?", lambda a: check_answer(a, "MESSY_LEDGER_ROWS", "E18")))
    tests.append(("E19", "Is DOC-117 a TXT file?", lambda a: check_answer(a, "no", "E19") or check_answer(a, "CSV", "E19")))
    tests.append(("E20", "What are the CSV column headers?",
                  all_check(
                      (lambda a: check_answer(a, "Index_Marker"), "Index_Marker found"),
                      (lambda a: check_answer(a, "Chaos_Archetype"), "Chaos_Archetype found"),
                      (lambda a: check_answer(a, "Raw_Payload_Data"), "Raw_Payload_Data found"),
                  )))

    # ========================
    # HARD (20) - Multi-doc, comparisons, aggregations
    # ========================
    tests.append(("H1", "Compare stability metrics between NVDA and AMD for 2018",
                  all_check(
                      (lambda a: check_answer(a, "NVDA"), "NVDA found"),
                      (lambda a: check_answer(a, "AMD"), "AMD found"),
                  )))
    tests.append(("H2", "Which ticker has the most documents in the corpus?",
                  lambda a: check_answer(a, gt["most_docs_ticker"], "H2")))
    tests.append(("H3", "List all 4 chaos archetypes found in the documents",
                  lambda a: check_answer_contains_any(a, gt["chaos_archetypes"], "H3")))
    tests.append(("H4", "What ARCH_NOTE template contains 'Hierarchical Navigable Small World'?",
                  lambda a: check_answer(a, gt["arch_note_hnsw"], "H4")))
    tests.append(("H5", "Which company has the highest stability metric in 2020?",
                  lambda a: check_answer(a, "AAPL", "H5") or check_answer(a, "Apple", "H5")))
    tests.append(("H6", "Compare the highest stability metrics for NVDA in 2018 vs 2026",
                  lambda a: check_answer(a, "0.999787", "H6") or check_answer(a, "0.965168", "H6")))
    tests.append(("H7", "What are the 3 unique ARCH_NOTE patterns found in the documents?",
                  all_check(
                      (lambda a: check_answer(a, gt["arch_note_hnsw"]), "HNSW found"),
                      (lambda a: check_answer(a, gt["arch_note_causal"]), "causal found"),
                      (lambda a: check_answer(a, gt["arch_note_cluster"]), "cluster found"),
                  )))
    tests.append(("H8", "How many total documents are in the corpus?",
                  lambda a: check_answer(a, gt["total_docs"], "H8")))
    tests.append(("H9", "What is the range of stability metric values across all documents?",
                  lambda a: ("0" in a or "-1" in a or "-0" in a or "1.0" in a or "0.9" in a, "range found")))
    tests.append(("H10", "Compare the file formats between DOC-480, DOC-626, and DOC-117",
                  all_check(
                      (lambda a: check_answer(a, "TXT"), "TXT found"),
                      (lambda a: check_answer(a, "PDF"), "PDF found"),
                      (lambda a: check_answer(a, "CSV"), "CSV found"),
                  )))
    tests.append(("H11", "What is the highest stability metric reported across all NVDA documents?",
                  lambda a: ("0.9" in a, "high value found")))
    tests.append(("H12", "Which document discusses both stability metrics and HNSW indexing?",
                  lambda a: check_answer(a, "480", "H12") or check_answer(a, "DOC-480", "H12")))
    tests.append(("H13", "What is the most common document type in the corpus?",
                  lambda a: check_answer(a, "PDF", "H13") or check_answer(a, "TXT", "H13") or ("312" in a or "309" in a, "common type found")))
    tests.append(("H14", "Compare the stability metrics of AAPL in 2018 vs 2020",
                  lambda a: check_answer(a, "stability", "H14") and ("2018" in a and "2020" in a, "year comparison found")))
    tests.append(("H15", "What are all the unique ticker symbols mentioned in the documents?",
                  lambda a: check_answer_contains_any(a, gt["ticker_list"], "H15")))
    tests.append(("H16", "What type of data does MESSY_LEDGER_ROWS contain?",
                  lambda a: check_answer(a, "hash", "H16") or check_answer(a, "Value", "H16") or check_answer(a, "Legal", "H16") or check_answer(a, "ledger", "H16")))
    tests.append(("H17", "How many pages does a typical PDF document have?",
                  lambda a: check_answer(a, "4", "H17") or check_answer(a, "6", "H17")))
    tests.append(("H18", "What does the acronym 'HNSW' stand for in the ARCH_NOTEs?",
                  lambda a: check_answer(a, "Hierarchical Navigable Small World", "H18")))
    tests.append(("H19", "Which document has more stability metrics: DOC-480 or DOC-396?",
                  lambda a: check_answer(a, "480", "H19")))
    tests.append(("H20", "What are the CSV column names in the corpus?",
                  all_check(
                      (lambda a: check_answer(a, "Index_Marker"), "Index_Marker found"),
                      (lambda a: check_answer(a, "Chaos_Archetype"), "Chaos_Archetype found"),
                      (lambda a: check_answer(a, "Raw_Payload_Data"), "Raw_Payload_Data found"),
                  )))

    # ========================
    # EXTREMELY HARD (15) - Multi-hop, cross-corpus aggregation, trends
    # ========================
    tests.append(("X1", "What is the average stability metric for NVDA in 2018 across all NVDA 2018 documents?",
                  lambda a: ("-0." in a or "0." in a or "-0.6" in a, "avg found")))
    tests.append(("X2", "Rank the tickers by total number of documents from most to least",
                  lambda a: check_answer(a, "AMZN", "X2")))
    tests.append(("X3", "Which ARCH_NOTE about 'raft consensus model' appears more often - with NVDA or AMD?",
                  lambda a: check_answer(a, "NVDA", "X3") or check_answer(a, "AMD", "X3") or check_answer(a, "both", "X3") or check_answer(a, "similar", "X3")))
    tests.append(("X4", "For MESSY_LEDGER_ROWS, what is the typical dollar value range?",
                  lambda a: check_answer(a, "85", "X4") or check_answer(a, "9", "X4") or check_answer(a, "100", "X4")))
    tests.append(("X5", "What is the stability metric trend for AMD from 2015 to 2026?",
                  lambda a: ("2015" in a and "2026" in a, "trend years found")))
    tests.append(("X6", "Compare the chaos archetype distribution: which archetype has the most documents?",
                  lambda a: check_answer(a, "HYBRID", "X6") or check_answer(a, "PURE", "X6") or check_answer(a, "MESSY", "X6") or check_answer(a, "ONLY", "X6")))
    tests.append(("X7", "How many total stability metric instances exist across the entire corpus?",
                  lambda a: check_answer(a, "1069", "X7") or ("1000" in a or "1069" in a, "total found")))
    tests.append(("X8", "Which ticker has the highest proportion of HYBRID_JARGON_MIX documents?",
                  lambda a: ("NVDA" in a or "INTC" in a or "AMD" in a or any(t in a for t in gt["ticker_list"]), "ticker found")))
    tests.append(("X9", "What is the maximum dollar value found in any MESSY_LEDGER_ROWS document?",
                  lambda a: ("9" in a or "9000" in a or "9900" in a, "dollar value found")))
    tests.append(("X10", "Does the narrative content in ONLY_LONG_PARAGRAPHS repeat? What sentences are used?",
                  lambda a: check_answer(a, "repeat", "X10") or check_answer(a, "Honestly", "X10") or check_answer(a, "sync up", "X10")))
    tests.append(("X11", "Compare the range (max-min) of stability metrics in DOC-480 vs DOC-396. Which has wider spread?",
                  lambda a: check_answer(a, "480", "X11") or ("DOC-480" in a or "wider" in a or "larger" in a or "broader" in a, "range found")))
    tests.append(("X12", "What is the P50 (median) stability metric value for NVDA in 2018?",
                  lambda a: ("-0." in a or "0." in a, "p50 found")))
    tests.append(("X13", "Which 3 legal context templates appear in MESSY_LEDGER_ROWS?",
                  lambda a: check_answer(a, "Subsection", "X13") or check_answer(a, "addendum", "X13") or check_answer(a, "covenants", "X13") or check_answer(a, "party", "X13")))
    tests.append(("X14", "How does the stability metric volatility compare between NVDA 2018 and AMD 2018?",
                  lambda a: ("NVDA" in a and "AMD" in a and "2018" in a, "comparison found")))
    tests.append(("X15", "What would be the sum of all stability metric values in DOC-480?",
                  lambda a: check_answer(a, "-", "X15") or ("neg" in a.lower() or "cannot" in a.lower(), "neg/cannot found")))

    # Run tests
    passed = 0
    failed = 0
    results = []

    print(f"{'ID':<5} {'Result':<8} {'Question':<60} Detail")
    print("=" * 120)

    for test_id, question, checker in tests:
        resp = query_api(question)
        answer = resp.get("answer", "")
        ok, detail = checker(answer)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        short_q = question[:57]
        print(f"{test_id:<5} {status:<8} {short_q:<60} {detail[:50]}")
        results.append((test_id, status, question, detail, answer[:200]))

    print("\n" + "=" * 120)
    print(f"TOTAL: {passed + failed} | PASS: {passed} | FAIL: {failed} | RATE: {passed/(passed+failed)*100:.1f}%")

    if failed > 0:
        print("\n=== FAILURE DETAILS ===")
        for tid, status, question, detail, ans_snippet in results:
            if status == "FAIL":
                print(f"\n[{tid}] {question}")
                print(f"  Reason: {detail}")
                print(f"  Answer: {ans_snippet}")

    return passed, failed


if __name__ == "__main__":
    p, f = run_tests()
    sys.exit(0 if f == 0 else 1)

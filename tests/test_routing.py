from rag_ci_cd.config import BACKUP_MODEL, MAIN_MODEL
from rag_ci_cd.routing.router import (
    Route,
    classify_query,
    get_model_for_route,
    get_retrieval_depth_for_route,
)


class TestQueryClassification:
    def test_simple_factual_what_is(self):
        assert classify_query("What is the revenue for Apple?") == Route.SIMPLE

    def test_simple_short_question(self):
        assert classify_query("Who is the CEO?") == Route.SIMPLE

    def test_simple_how_much(self):
        assert classify_query("How much debt does Tesla have?") == Route.SIMPLE

    def test_simple_how_many(self):
        assert classify_query("How many employees work there?") == Route.SIMPLE

    def test_simple_what_was(self):
        assert classify_query("What was the profit last year?") == Route.SIMPLE

    def test_simple_when(self):
        assert classify_query("When was the company founded?") == Route.SIMPLE

    def test_simple_where(self):
        assert classify_query("Where is the headquarters located?") == Route.SIMPLE

    def test_simple_is_there(self):
        assert classify_query("Is there a risk factor section?") == Route.SIMPLE

    def test_simple_did(self):
        assert classify_query("Did revenue increase?") == Route.SIMPLE

    def test_simple_short_word_count(self):
        assert classify_query("Apple revenue") == Route.SIMPLE

    def test_complex_comparison(self):
        assert classify_query("Compare revenue between Q2 and Q3") == Route.COMPLEX

    def test_complex_table_query(self):
        assert classify_query("Show me the table of quarterly earnings") == Route.COMPLEX

    def test_complex_multi_document(self):
        assert classify_query("What changed across all documents?") == Route.COMPLEX

    def test_complex_trend(self):
        assert classify_query("What is the trend in operating income?") == Route.COMPLEX

    def test_complex_long_question(self):
        q = "Can you explain how the debt maturity schedule changed between the 2022 and 2023 filings and what factors drove that change?"
        assert classify_query(q) == Route.COMPLEX

    def test_complex_analyze(self):
        assert classify_query("Analyze the risk factors in the filing") == Route.COMPLEX

    def test_complex_explain(self):
        assert classify_query("Explain the revenue breakdown by segment") == Route.COMPLEX

    def test_complex_summarize(self):
        assert classify_query("Summarize the key findings") == Route.COMPLEX

    def test_complex_list(self):
        assert classify_query("List all risk factors mentioned") == Route.COMPLEX

    def test_complex_vs(self):
        assert classify_query("AAPL vs MSFT revenue comparison") == Route.COMPLEX

    def test_complex_which_document(self):
        assert classify_query("Which document discusses revenue?") == Route.COMPLEX

    def test_complex_all_documents(self):
        assert classify_query("What do all the documents say about growth?") == Route.COMPLEX

    def test_complex_metric_keyword(self):
        assert classify_query("What are the metrics for profitability?") == Route.COMPLEX


class TestRouteConfig:
    def test_simple_model_matches_config(self):
        assert get_model_for_route(Route.SIMPLE) == BACKUP_MODEL

    def test_complex_model_matches_config(self):
        assert get_model_for_route(Route.COMPLEX) == MAIN_MODEL

    def test_simple_model_is_qwen(self):
        assert get_model_for_route(Route.SIMPLE) == "qwen2.5:1.5b"

    def test_complex_model_is_llama(self):
        assert get_model_for_route(Route.COMPLEX) == "llama3.2:3b"

    def test_simple_depth(self):
        assert get_retrieval_depth_for_route(Route.SIMPLE) == 10

    def test_complex_depth(self):
        assert get_retrieval_depth_for_route(Route.COMPLEX) == 30

    def test_simple_depth_less_than_complex(self):
        assert get_retrieval_depth_for_route(Route.SIMPLE) < get_retrieval_depth_for_route(Route.COMPLEX)

from rag_ci_cd.routing.router import Route, classify_query, get_model_for_route, get_retrieval_depth_for_route


class TestQueryClassification:
    def test_simple_factual_question(self):
        assert classify_query("What is the revenue for Apple?") == Route.SIMPLE

    def test_simple_short_question(self):
        assert classify_query("Who is the CEO?") == Route.SIMPLE

    def test_simple_how_much(self):
        assert classify_query("How much debt does Tesla have?") == Route.SIMPLE

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

    def test_simple_existence(self):
        assert classify_query("Is there a risk factor section?") == Route.SIMPLE


class TestRouteConfig:
    def test_simple_model(self):
        assert get_model_for_route(Route.SIMPLE) == "llama3.2:3b"

    def test_complex_model(self):
        assert get_model_for_route(Route.COMPLEX) == "qwen3:14b"

    def test_simple_depth(self):
        assert get_retrieval_depth_for_route(Route.SIMPLE) == 5

    def test_complex_depth(self):
        assert get_retrieval_depth_for_route(Route.COMPLEX) == 20

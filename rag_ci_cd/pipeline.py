from __future__ import annotations

import time

from rag_ci_cd.generation.generator import answer_to_response, generate_answer
from rag_ci_cd.indexing.store import IndexStore
from rag_ci_cd.models.answers import AnswerRequest, AnswerResponse
from rag_ci_cd.reranking.reranker import Reranker
from rag_ci_cd.retrieval.hybrid import hybrid_retrieve
from rag_ci_cd.routing.router import Route, classify_query, get_retrieval_depth_for_route


def answer_query(
    store: IndexStore,
    request: AnswerRequest,
) -> AnswerResponse:
    route = classify_query(request.query)
    if request.route == "simple":
        route = Route.SIMPLE
    elif request.route == "complex":
        route = Route.COMPLEX

    retrieve_top_k = get_retrieval_depth_for_route(route)

    retrieval_start = time.time()
    result = hybrid_retrieve(store, request.query, top_k=retrieve_top_k)
    retrieval_time = (time.time() - retrieval_start) * 1000

    if request.rerank and result.chunks:
        reranker = Reranker()
        result = reranker.rerank_result(result, top_k=request.top_k)
        # update retrieval_time to include reranking
        retrieval_time = result.retrieval_time_ms

    gen_start = time.time()
    answer = generate_answer(result, route=route, store=store)
    gen_time = (time.time() - gen_start) * 1000

    return answer_to_response(
        answer=answer,
        route=route,
        retrieval_time_ms=retrieval_time,
        generation_time_ms=gen_time,
    )

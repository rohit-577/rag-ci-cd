from __future__ import annotations

import time

import torch
from sentence_transformers import CrossEncoder

from rag_ci_cd.models.retrieval import RetrievalResult, RetrievedChunk

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    _instance: CrossEncoder | None = None
    _device: str | None = None

    @classmethod
    def get_instance(cls) -> CrossEncoder:
        if cls._instance is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._device = device
            cls._instance = CrossEncoder(
                CROSS_ENCODER_MODEL,
                device=device,
            )
        return cls._instance

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks

        model = self.get_instance()
        pairs = [(query, c.content) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        chunks.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        for rank, chunk in enumerate(chunks):
            chunk.rank = rank + 1

        return chunks[:top_k]

    def rerank_result(
        self,
        result: RetrievalResult,
        top_k: int = 10,
    ) -> RetrievalResult:
        start = time.time()
        reranked = self.rerank(result.query, result.chunks, top_k=top_k)
        elapsed = (time.time() - start) * 1000
        return RetrievalResult(
            query=result.query,
            chunks=reranked,
            retrieval_time_ms=result.retrieval_time_ms + round(elapsed, 2),
            method="hybrid_rrf_reranked",
        )

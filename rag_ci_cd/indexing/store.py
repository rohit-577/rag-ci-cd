from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from rag_ci_cd.config import INDEX_DIR
from rag_ci_cd.models.document import Chunk


def _tokenize(text: str) -> list[str]:
    tokens = text.lower().split()
    expanded = []
    for t in tokens:
        expanded.append(t)
        if "-" in t:
            expanded.extend(t.split("-"))
        if "_" in t:
            expanded.extend(t.split("_"))
        # Also strip common punctuation from tokens
        clean = t.strip("[](),.:;!?\"'")
        if clean != t:
            expanded.append(clean)
    return expanded


class IndexStore:
    def __init__(self, path: Path = INDEX_DIR):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._bm25_ready = False

        # BM25 state
        self._doc_freq: Counter = Counter()
        self._doc_len: list[int] = []
        self._avg_doc_len: float = 0.0
        self._total_docs: int = 0
        self._vocab: set[str] = set()
        self._corpus_tokens: list[list[str]] = []
        self._chunk_map: dict[str, int] = {}  # chunk_id -> index

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        start = len(self._chunks)
        for i, chunk in enumerate(chunks):
            idx = start + i
            self._chunk_map[chunk.chunk_id] = idx
        self._chunks.extend(chunks)

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        for chunk in chunks:
            searchable = f"{chunk.doc_id} {chunk.filename} {chunk.ticker or ''} {chunk.year or ''} {chunk.content}"
            tokens = _tokenize(searchable)
            self._corpus_tokens.append(tokens)
            self._doc_len.append(len(tokens))
            self._vocab.update(tokens)
            for t in set(tokens):
                self._doc_freq[t] += 1

        self._total_docs = len(self._chunks)
        self._avg_doc_len = sum(self._doc_len) / max(self._total_docs, 1)
        self._bm25_ready = True

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        idx = self._chunk_map.get(chunk_id)
        if idx is not None:
            return self._chunks[idx]
        return None

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def embeddings(self) -> np.ndarray | None:
        return self._embeddings

    @property
    def size(self) -> int:
        return len(self._chunks)

    def bm25_score(self, query_tokens: list[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        doc_len = self._doc_len[doc_idx]
        for token in query_tokens:
            if token not in self._vocab:
                continue
            idf = math.log((self._total_docs - self._doc_freq[token] + 0.5) / (self._doc_freq[token] + 0.5) + 1.0)
            tf = self._corpus_tokens[doc_idx].count(token)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / self._avg_doc_len))
        return score

    def search_bm25(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        query_tokens = _tokenize(query)
        scores = []
        for i in range(self._total_docs):
            score = self.bm25_score(query_tokens, i)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_dense(self, query_emb: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        if self._embeddings is None:
            return []
        sims = np.dot(self._embeddings, query_emb)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in top_indices]

    def save(self) -> None:
        data = {
            "chunks": [c.model_dump() for c in self._chunks],
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
            "doc_freq": dict(self._doc_freq),
            "doc_len": self._doc_len,
            "avg_doc_len": self._avg_doc_len,
            "total_docs": self._total_docs,
            "vocab": list(self._vocab),
            "corpus_tokens": self._corpus_tokens,
            "chunk_map": self._chunk_map,
        }
        (self.path / "index.json").write_text(json.dumps(data, default=str))

    def load(self) -> bool:
        index_file = self.path / "index.json"
        if not index_file.exists():
            return False
        data = json.loads(index_file.read_text())
        from rag_ci_cd.models.document import Chunk as ChunkModel

        self._chunks = [ChunkModel(**c) for c in data["chunks"]]
        self._embeddings = np.array(data["embeddings"]) if data["embeddings"] else None
        self._doc_freq = Counter(data["doc_freq"])
        self._doc_len = data["doc_len"]
        self._avg_doc_len = data["avg_doc_len"]
        self._total_docs = data["total_docs"]
        self._vocab = set(data["vocab"])
        self._corpus_tokens = data["corpus_tokens"]
        self._chunk_map = data["chunk_map"]
        self._bm25_ready = self._total_docs > 0
        return True

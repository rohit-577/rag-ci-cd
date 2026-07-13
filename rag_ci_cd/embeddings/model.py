from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag_ci_cd.config import EMBEDDING_MODEL


class EmbeddingModel:
    _instance: SentenceTransformer | None = None
    _device: str | None = None

    @classmethod
    def get_instance(cls) -> SentenceTransformer:
        if cls._instance is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._device = device
            dtype = torch.float16 if device == "cuda" else None
            kwargs = {"torch_dtype": dtype} if dtype else {}
            cls._instance = SentenceTransformer(EMBEDDING_MODEL, device=device, model_kwargs=kwargs)
        return cls._instance

    def embed(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        model = self.get_instance()
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], batch_size=1)[0]

    def embed_chunks(self, chunks: list[str], batch_size: int = 8) -> np.ndarray:
        return self.embed(chunks, batch_size=batch_size)

    @property
    def dim(self) -> int:
        return self.get_instance().get_sentence_embedding_dimension() or 1024

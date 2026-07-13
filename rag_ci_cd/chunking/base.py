from abc import ABC, abstractmethod

from rag_ci_cd.models.document import Document, Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        ...

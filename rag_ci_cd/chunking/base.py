from abc import ABC, abstractmethod

from rag_ci_cd.models.document import Chunk, Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

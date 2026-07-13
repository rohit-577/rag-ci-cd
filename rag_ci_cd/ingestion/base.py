from abc import ABC, abstractmethod
from pathlib import Path

from rag_ci_cd.models.document import Document


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> Document:
        ...

    def supports(self, path: Path) -> bool:
        return False

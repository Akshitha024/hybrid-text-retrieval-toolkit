from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Doc, Hit


class Index(ABC):
    name: str

    @abstractmethod
    def add(self, docs: list[Doc]) -> None:
        ...

    @abstractmethod
    def query(self, text: str, top_k: int) -> list[Hit]:
        ...

    def query_batch(self, texts: list[str], top_k: int) -> list[list[Hit]]:
        return [self.query(t, top_k) for t in texts]

    @abstractmethod
    def size(self) -> int:
        ...

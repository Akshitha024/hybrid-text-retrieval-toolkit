"""Cross-encoder reranker.

Reranking is a stage on top of any base retriever; it is not itself an index.
It takes the top-N hits from a base and rescores (query, doc_text) pairs with
a cross-encoder, returning the reordered top-k. The default model is the
ms-marco MiniLM cross-encoder, small enough to run on CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..indexes.base import Index
from ..types import Doc, Hit

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderRerank(Index):
    name = "rerank"

    def __init__(
        self,
        base: Index,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n_to_rerank: int = 50,
        batch_size: int = 32,
    ) -> None:
        self.base = base
        self.model_name = model_name
        self.top_n_to_rerank = top_n_to_rerank
        self.batch_size = batch_size
        self._reranker: CrossEncoder | None = None
        self._doc_text: dict[str, str] = {}
        self.name = f"rerank({base.name})"

    def _load(self) -> CrossEncoder:
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(self.model_name)
        return self._reranker

    def add(self, docs: list[Doc]) -> None:
        self.base.add(docs)
        self._doc_text = {d.doc_id: (f"{d.title}. {d.text}" if d.title else d.text) for d in docs}

    def query(self, text: str, top_k: int) -> list[Hit]:
        base_k = max(self.top_n_to_rerank, top_k)
        candidates = self.base.query(text, base_k)
        if not candidates:
            return []
        pairs = [(text, self._doc_text[h.doc_id]) for h in candidates]
        scores = self._load().predict(pairs, batch_size=self.batch_size)
        rescored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]
        return [
            Hit(doc_id=h.doc_id, score=float(s), rank=rank + 1)
            for rank, (h, s) in enumerate(rescored)
        ]

    def size(self) -> int:
        return self.base.size()

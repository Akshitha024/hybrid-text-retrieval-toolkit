"""BM25Okapi index wrapper.

Tokenization is intentionally simple: lowercase and word-boundary split. BEIR
benchmarks measure things like nDCG and recall, which are not very sensitive
to clever tokenization for English text; what matters is consistency between
indexing and query time.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ..types import Doc, Hit
from .base import Index

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


class BM25Index(Index):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []

    def add(self, docs: list[Doc]) -> None:
        self._doc_ids = [d.doc_id for d in docs]
        tokens = [tokenize(_with_title(d)) for d in docs]
        self._bm25 = BM25Okapi(tokens, k1=self.k1, b=self.b)

    def query(self, text: str, top_k: int) -> list[Hit]:
        if self._bm25 is None:
            raise RuntimeError("call .add() before .query()")
        scores = self._bm25.get_scores(tokenize(text))
        top_k = min(top_k, len(self._doc_ids))
        idxs = scores.argsort()[-top_k:][::-1]
        return [
            Hit(doc_id=self._doc_ids[int(i)], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(idxs)
        ]

    def size(self) -> int:
        return len(self._doc_ids)


def _with_title(d: Doc) -> str:
    if d.title:
        return f"{d.title}. {d.text}"
    return d.text

"""Reciprocal Rank Fusion (Cormack, Clarke, Buettcher 2009).

The standard RRF: ignore scores entirely, just sum 1/(k + rank) across rankers.
We over-fetch from each base ranker (base_k = top_k * 4 by default) because
RRF is sensitive to recall depth.
"""

from __future__ import annotations

from collections import defaultdict

from ..indexes.base import Index
from ..types import Doc, Hit


class RRFFusion(Index):
    name = "rrf"

    def __init__(self, indexes: list[Index], k: int = 60, over_fetch: int = 4) -> None:
        if len(indexes) < 2:
            raise ValueError("need at least two indexes to fuse")
        self.indexes = indexes
        self.k = k
        self.over_fetch = over_fetch
        self.name = "rrf(" + "+".join(i.name for i in indexes) + ")"

    def add(self, docs: list[Doc]) -> None:
        for ix in self.indexes:
            ix.add(docs)

    def query(self, text: str, top_k: int) -> list[Hit]:
        base_k = max(top_k * self.over_fetch, 50)
        scores: dict[str, float] = defaultdict(float)
        for ix in self.indexes:
            for h in ix.query(text, base_k):
                scores[h.doc_id] += 1.0 / (self.k + h.rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [Hit(doc_id=did, score=s, rank=rank + 1) for rank, (did, s) in enumerate(ranked)]

    def size(self) -> int:
        return self.indexes[0].size() if self.indexes else 0

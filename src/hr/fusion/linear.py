"""Score-level fusion: per-index linear combination of normalized scores.

Less robust than RRF because score normalization across heterogeneous rankers
is fragile, but useful when one ranker is much stronger than the others and
you want to weight it. We min-max normalize each ranker's scores within its
top-N before combining.
"""

from __future__ import annotations

from collections import defaultdict

from ..indexes.base import Index
from ..types import Doc, Hit


class LinearFusion(Index):
    name = "linear"

    def __init__(
        self,
        indexes: list[Index],
        weights: list[float] | None = None,
        over_fetch: int = 4,
    ) -> None:
        if len(indexes) < 2:
            raise ValueError("need at least two indexes to fuse")
        if weights is None:
            weights = [1.0 / len(indexes)] * len(indexes)
        if len(weights) != len(indexes):
            raise ValueError("weights length must match indexes length")
        self.indexes = indexes
        self.weights = weights
        self.over_fetch = over_fetch
        parts = [f"{w:.2f}*{i.name}" for w, i in zip(weights, indexes, strict=True)]
        self.name = "linear(" + "+".join(parts) + ")"

    def add(self, docs: list[Doc]) -> None:
        for ix in self.indexes:
            ix.add(docs)

    def query(self, text: str, top_k: int) -> list[Hit]:
        base_k = max(top_k * self.over_fetch, 50)
        combined: dict[str, float] = defaultdict(float)
        for ix, w in zip(self.indexes, self.weights, strict=True):
            hits = ix.query(text, base_k)
            if not hits:
                continue
            lo = min(h.score for h in hits)
            hi = max(h.score for h in hits)
            rng = hi - lo if hi > lo else 1.0
            for h in hits:
                norm = (h.score - lo) / rng
                combined[h.doc_id] += w * norm
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [Hit(doc_id=did, score=s, rank=rank + 1) for rank, (did, s) in enumerate(ranked)]

    def size(self) -> int:
        return self.indexes[0].size() if self.indexes else 0

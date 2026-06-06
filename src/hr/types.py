"""Core types shared across the framework.

We do not use pydantic here because the call sites are hot loops; frozen
dataclasses with explicit fields are faster and serialize cleanly to JSON.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    title: str | None = None


@dataclass(frozen=True)
class Query:
    qid: str
    text: str


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float
    rank: int


# qrels[qid][doc_id] = relevance grade (>= 1 is relevant; BEIR uses 0/1/2)
Qrels = dict[str, dict[str, int]]


def iter_top(hits: Iterable[Hit], k: int) -> list[Hit]:
    out = []
    for h in hits:
        out.append(h)
        if len(out) >= k:
            break
    return out

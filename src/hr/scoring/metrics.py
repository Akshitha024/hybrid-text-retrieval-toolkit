"""IR metrics. Same definitions as legal-rag-benchmark; copied here so the
two projects are independent.

Implementations follow Manning, Raghavan & Schutze (2008) for nDCG and the
TREC-eval reference for the rest. We do not depend on pytrec_eval because
its install footprint is heavier than the metric code itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..types import Hit


@dataclass
class QResult:
    qid: str
    hits: list[Hit]
    qrels: dict[str, int]  # doc_id -> grade (>= 1 is relevant)


def _relevant_ids(qrels: dict[str, int]) -> set[str]:
    return {d for d, g in qrels.items() if g > 0}


def recall_at_k(r: QResult, k: int) -> float:
    rel = _relevant_ids(r.qrels)
    if not rel:
        return 0.0
    got = {h.doc_id for h in r.hits[:k]}
    return len(got & rel) / len(rel)


def reciprocal_rank(r: QResult, k: int) -> float:
    rel = _relevant_ids(r.qrels)
    for i, h in enumerate(r.hits[:k], start=1):
        if h.doc_id in rel:
            return 1.0 / i
    return 0.0


def average_precision(r: QResult, k: int) -> float:
    rel = _relevant_ids(r.qrels)
    if not rel:
        return 0.0
    correct = 0
    summed = 0.0
    for i, h in enumerate(r.hits[:k], start=1):
        if h.doc_id in rel:
            correct += 1
            summed += correct / i
    return summed / min(len(rel), k)


def ndcg_at_k(r: QResult, k: int) -> float:
    if not r.qrels:
        return 0.0
    gains = [r.qrels.get(h.doc_id, 0) for h in r.hits[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(r.qrels.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate(results: Sequence[QResult], ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = max(1, len(results))
    for k in ks:
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in results) / n
        out[f"recall@{k}"] = sum(recall_at_k(r, k) for r in results) / n
        out[f"mrr@{k}"] = sum(reciprocal_rank(r, k) for r in results) / n
        out[f"map@{k}"] = sum(average_precision(r, k) for r in results) / n
    return out

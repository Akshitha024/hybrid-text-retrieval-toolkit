from __future__ import annotations

import math

from hr.scoring.metrics import (
    QResult,
    average_precision,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from hr.types import Hit


def _hits(ids: list[str]) -> list[Hit]:
    return [Hit(doc_id=d, score=1.0 - i * 0.1, rank=i + 1) for i, d in enumerate(ids)]


def test_recall_basic() -> None:
    r = QResult(qid="q", hits=_hits(["a", "b", "c"]), qrels={"a": 1, "d": 1})
    assert recall_at_k(r, 5) == 0.5


def test_mrr_first_hit() -> None:
    r = QResult(qid="q", hits=_hits(["x", "y", "z"]), qrels={"y": 1})
    assert reciprocal_rank(r, 10) == 0.5


def test_ap_two_relevant() -> None:
    r = QResult(qid="q", hits=_hits(["a", "x", "c", "y"]), qrels={"a": 1, "c": 1})
    assert math.isclose(average_precision(r, 10), (1.0 + 2 / 3) / 2, rel_tol=1e-9)


def test_ndcg_perfect() -> None:
    r = QResult(qid="q", hits=_hits(["a", "b"]), qrels={"a": 1, "b": 1})
    assert math.isclose(ndcg_at_k(r, 10), 1.0)


def test_ndcg_graded() -> None:
    r = QResult(qid="q", hits=_hits(["a", "b", "c"]), qrels={"a": 1, "b": 3, "c": 2})
    assert ndcg_at_k(r, 3) < 1.0


def test_evaluate_aggregates() -> None:
    rs = [
        QResult(qid="q1", hits=_hits(["a", "b"]), qrels={"a": 1}),
        QResult(qid="q2", hits=_hits(["x", "y"]), qrels={"y": 1}),
    ]
    out = evaluate(rs, [1, 5])
    assert math.isclose(out["mrr@5"], 0.75)
    assert math.isclose(out["recall@5"], 1.0)

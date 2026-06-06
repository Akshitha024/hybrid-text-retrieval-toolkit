from __future__ import annotations

from hr.fusion.linear import LinearFusion
from hr.fusion.rrf import RRFFusion
from hr.indexes.base import Index
from hr.types import Doc, Hit


class _Stub(Index):
    def __init__(self, name: str, ranking: list[str]) -> None:
        self.name = name
        self._ranking = ranking

    def add(self, docs: list[Doc]) -> None:
        pass

    def query(self, text: str, top_k: int) -> list[Hit]:
        return [
            Hit(doc_id=d, score=1.0 - i * 0.1, rank=i + 1)
            for i, d in enumerate(self._ranking[:top_k])
        ]

    def size(self) -> int:
        return len(self._ranking)


def test_rrf_promotes_shared_docs() -> None:
    a = _Stub("a", ["x", "y", "z"])
    b = _Stub("b", ["z", "x", "w"])
    fusion = RRFFusion([a, b])
    fusion.add([])
    ids = [h.doc_id for h in fusion.query("any", top_k=4)]
    assert ids.index("x") < ids.index("w")
    assert ids.index("z") < ids.index("y")  # z shared, y only in a


def test_linear_respects_weights() -> None:
    a = _Stub("a", ["x", "y"])
    b = _Stub("b", ["y", "x"])
    # bias to b: y should win
    fusion = LinearFusion([a, b], weights=[0.1, 0.9])
    fusion.add([])
    hits = fusion.query("any", top_k=2)
    assert hits[0].doc_id == "y"


def test_rrf_requires_two_indexes() -> None:
    try:
        RRFFusion([_Stub("a", ["x"])])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_linear_weight_length_must_match() -> None:
    try:
        LinearFusion([_Stub("a", []), _Stub("b", [])], weights=[1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")

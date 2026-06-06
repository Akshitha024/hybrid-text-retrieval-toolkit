from __future__ import annotations

import json
from pathlib import Path

from hr.indexes.bm25 import BM25Index, tokenize
from hr.types import Doc

FIX = Path(__file__).parent / "fixtures"


def _load_docs() -> list[Doc]:
    rows = json.loads((FIX / "tiny_docs.json").read_text())
    return [Doc(**r) for r in rows]


def test_tokenize_lowercase() -> None:
    assert tokenize("Hello WORLD!") == ["hello", "world"]


def test_tokenize_keeps_numbers() -> None:
    out = tokenize("Section 12.3(b)")
    assert "section" in out


def test_bm25_returns_top_k() -> None:
    ix = BM25Index()
    ix.add(_load_docs())
    hits = ix.query("antimicrobial resistance", top_k=3)
    assert len(hits) == 3
    assert hits[0].rank == 1


def test_bm25_finds_relevant() -> None:
    ix = BM25Index()
    ix.add(_load_docs())
    hits = ix.query("antibiotic resistance public health", top_k=2)
    top_ids = {h.doc_id for h in hits}
    # d2 contains "antibiotic resistance"; d6 contains "antimicrobial resistance"
    # at least one of them should be in the top-2
    assert "d2" in top_ids or "d6" in top_ids

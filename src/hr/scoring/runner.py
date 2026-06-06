"""Drive one (index, dataset) pair to a metrics + per-query JSONL output."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from ..indexes.base import Index
from ..types import Doc, Qrels, Query
from .metrics import QResult, evaluate


def run(
    index: Index,
    docs: list[Doc],
    queries: list[Query],
    qrels: Qrels,
    ks: Iterable[int],
) -> tuple[dict[str, float], list[QResult]]:
    ks = list(ks)
    k_max = max(ks)

    t0 = time.perf_counter()
    index.add(docs)
    build_s = time.perf_counter() - t0
    logger.info("{} built in {:.2f}s ({} docs)", index.name, build_s, len(docs))

    per_q: list[QResult] = []
    t0 = time.perf_counter()
    for q in tqdm(queries, desc=f"search:{index.name}", leave=False):
        hits = index.query(q.text, k_max)
        per_q.append(QResult(qid=q.qid, hits=hits, qrels=qrels.get(q.qid, {})))
    search_s = time.perf_counter() - t0

    metrics = evaluate(per_q, ks)
    metrics["build_s"] = build_s
    metrics["search_s"] = search_s
    metrics["qps"] = len(queries) / max(search_s, 1e-6)
    return metrics, per_q


def write(
    out_dir: Path,
    dataset: str,
    index_name: str,
    metrics: dict[str, float],
    per_q: list[QResult],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = index_name.replace("/", "_")
    (out_dir / f"{dataset}__{safe}__metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / f"{dataset}__{safe}__runs.jsonl").open("w") as f:
        for r in per_q:
            f.write(
                json.dumps(
                    {
                        "qid": r.qid,
                        "hits": [
                            {"doc_id": h.doc_id, "rank": h.rank, "score": h.score} for h in r.hits
                        ],
                    }
                )
                + "\n"
            )

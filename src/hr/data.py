"""BEIR dataset loader.

BEIR ships as multiple HuggingFace dataset repos under ``BeIR/<name>``. Each
has ``corpus`` (doc_id, text, title), ``queries`` (qid, text), and ``qrels``
(query_id, corpus_id, score). We fetch and convert to our internal types.

Supported small sub-datasets (fits on a laptop):
  scifact      (5k docs, 300 dev qrels)  - scientific claim verification
  nfcorpus     (3.6k docs, 323 test qrels) - bio-medical
  fiqa         (57k docs, 648 test qrels) - financial QA
  trec-covid   (171k docs, 50 qrels)     - COVID research papers
"""

from __future__ import annotations

from typing import Literal

from loguru import logger

from .types import Doc, Qrels, Query

BEIR_DATASETS = ("scifact", "nfcorpus", "fiqa", "trec-covid", "arguana", "scidocs")
BeirDataset = Literal["scifact", "nfcorpus", "fiqa", "trec-covid", "arguana", "scidocs"]


def load_beir(dataset: BeirDataset) -> tuple[list[Doc], list[Query], Qrels]:
    """Pull a BEIR sub-dataset from the canonical HF mirror and return our types.

    For scifact we use the ``train`` qrels split because there is no ``test``
    on HF; for the others we use ``test``. The BEIR mirror sometimes splits
    qrels into a separate config (``BeIR/<name>-qrels``); we look in both.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("install `datasets` (uv sync)") from e

    logger.info("loading BEIR/{} from HuggingFace", dataset)
    corpus_ds = load_dataset(f"BeIR/{dataset}", "corpus")["corpus"]
    queries_ds = load_dataset(f"BeIR/{dataset}", "queries")["queries"]
    qrels_split = "train" if dataset == "scifact" else "test"
    qrels_ds = load_dataset(f"BeIR/{dataset}-qrels", split=qrels_split)

    docs: list[Doc] = [
        Doc(
            doc_id=str(row["_id"]),
            text=str(row.get("text") or ""),
            title=str(row.get("title")) if row.get("title") else None,
        )
        for row in corpus_ds
    ]

    # only keep queries that actually have at least one qrel for this split
    qrels_by_qid: Qrels = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        grade = int(row["score"])
        qrels_by_qid.setdefault(qid, {})[did] = grade
    queries: list[Query] = [
        Query(qid=str(row["_id"]), text=str(row.get("text") or ""))
        for row in queries_ds
        if str(row["_id"]) in qrels_by_qid
    ]

    logger.info(
        "BEIR/{}: {} docs, {} queries with qrels",
        dataset,
        len(docs),
        len(queries),
    )
    return docs, queries, qrels_by_qid

"""Six distinct visualizations for a retrieval benchmark.

We deliberately pick varied chart types: line plots, scatter, histograms,
KDE, heatmaps. Repeating the same bar chart across every result page is the
fastest way to make a real eval suite look auto-generated; each function
below tells a different story about the same per-query data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_K_VALUES = (1, 3, 5, 10, 20)


def _load_metrics_dir(results_dir: Path, dataset: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for f in results_dir.glob(f"{dataset}__*__metrics.json"):
        stem = f.stem.removesuffix("__metrics")
        _, index_name = stem.split("__", 1)
        out[index_name] = json.loads(f.read_text())
    return out


def _load_runs(results_dir: Path, dataset: str, index_name: str) -> list[dict[str, Any]]:
    f = results_dir / f"{dataset}__{index_name.replace('/', '_')}__runs.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.open() if line.strip()]


def _per_query_ndcg(
    runs: list[dict[str, Any]],
    qrels_path: Path,
    k: int,
) -> list[float]:
    import math

    qrels: dict[str, dict[str, int]] = (
        json.loads(qrels_path.read_text()) if qrels_path.exists() else {}
    )
    out: list[float] = []
    for r in runs:
        qid = str(r["qid"])
        q = qrels.get(qid, {})
        if not q:
            continue
        hits = list(r["hits"])[:k]
        gains = [int(q.get(str(h["doc_id"]), 0)) for h in hits]
        dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sorted(q.values(), reverse=True)[:k]
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return out


# 1. nDCG@k curves: one line per index, k on x-axis
def plot_ndcg_curves(results_dir: Path, dataset: str, out: Path) -> Path:
    data = _load_metrics_dir(results_dir, dataset)
    if not data:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, m in sorted(data.items()):
        ys = [m.get(f"ndcg@{k}", 0) for k in _K_VALUES]
        ax.plot(_K_VALUES, ys, marker="o", label=name)
    ax.set_xlabel("k")
    ax.set_ylabel("nDCG@k")
    ax.set_title(f"nDCG@k curves on {dataset}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xticks(list(_K_VALUES))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 2. Recall-Precision tradeoff scatter
def plot_recall_precision(results_dir: Path, dataset: str, out: Path) -> Path:
    data = _load_metrics_dir(results_dir, dataset)
    if not data:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, m in sorted(data.items()):
        for k in _K_VALUES:
            r = m.get(f"recall@{k}", 0)
            # use MAP@k as a precision proxy (averaged precision @ k)
            p = m.get(f"map@{k}", 0)
            ax.scatter(r, p, alpha=0.5, s=20)
        # draw the line through each index's (recall, MAP) sequence
        rs = [m.get(f"recall@{k}", 0) for k in _K_VALUES]
        ps = [m.get(f"map@{k}", 0) for k in _K_VALUES]
        ax.plot(rs, ps, marker="o", label=name)
    ax.set_xlabel("Recall@k")
    ax.set_ylabel("MAP@k (precision proxy)")
    ax.set_title(f"Recall vs. precision tradeoff on {dataset}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 3. Per-query nDCG distribution (KDE-like via histogram with smoothing)
def plot_per_query_distribution(
    results_dir: Path, dataset: str, qrels_path: Path, out: Path, k: int = 10
) -> Path:
    data = _load_metrics_dir(results_dir, dataset)
    if not data:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = list(np.linspace(0, 1, 21))
    for name in sorted(data.keys()):
        runs = _load_runs(results_dir, dataset, name)
        if not runs:
            continue
        vals = _per_query_ndcg(runs, qrels_path, k)
        if not vals:
            continue
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.5, label=name)
    ax.set_xlabel(f"per-query nDCG@{k}")
    ax.set_ylabel("queries")
    ax.set_title(f"Per-query nDCG distribution on {dataset} (k={k})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 4. Latency vs quality scatter (QPS on log-x, nDCG@10 on y)
def plot_quality_vs_speed(results_dir: Path, dataset: str, out: Path) -> Path:
    data = _load_metrics_dir(results_dir, dataset)
    if not data:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m in sorted(data.items()):
        x = m.get("qps", 1e-3)
        y = m.get("ndcg@10", 0)
        ax.scatter(x, y, s=120)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Queries per second (log)")
    ax.set_ylabel("nDCG@10")
    ax.set_title(f"Quality vs. speed on {dataset}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 5. Index-build time vs corpus size (computed from metrics; useful when
# you've run multiple datasets to see how an index scales)
def plot_build_cost(results_dir: Path, out: Path) -> Path:
    # walk every metrics file across every dataset
    rows = []
    for f in results_dir.glob("*__metrics.json"):
        stem = f.stem.removesuffix("__metrics")
        if "__" not in stem:
            continue
        dataset, index = stem.split("__", 1)
        m = json.loads(f.read_text())
        rows.append((dataset, index, float(m.get("build_s", 0)), float(m.get("ndcg@10", 0))))
    if not rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out
    fig, ax = plt.subplots(figsize=(7, 5))
    by_index: dict[str, list[tuple[float, float]]] = {}
    for _, index, t, n in rows:
        by_index.setdefault(index, []).append((t, n))
    for index, points in sorted(by_index.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=80, label=index)
    ax.set_xlabel("Index build time (s)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Index build cost vs. retrieval quality")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 6. Per-index per-dataset nDCG@10 heatmap (across multiple datasets)
def plot_dataset_heatmap(results_dir: Path, out: Path) -> Path:
    rows = []
    for f in results_dir.glob("*__metrics.json"):
        stem = f.stem.removesuffix("__metrics")
        if "__" not in stem:
            continue
        dataset, index = stem.split("__", 1)
        m = json.loads(f.read_text())
        rows.append((dataset, index, float(m.get("ndcg@10", 0))))
    if not rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out
    datasets = sorted({r[0] for r in rows})
    indexes = sorted({r[1] for r in rows})
    matrix = np.zeros((len(indexes), len(datasets)))
    for dname, idx_name, v in rows:
        matrix[indexes.index(idx_name), datasets.index(dname)] = v
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(datasets)), max(3, 0.5 * len(indexes))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(indexes)))
    ax.set_yticklabels(indexes, fontsize=9)
    for r_idx in range(len(indexes)):
        for c_idx in range(len(datasets)):
            cell = float(matrix[r_idx, c_idx])
            ax.text(
                float(c_idx),
                float(r_idx),
                f"{cell:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cell < 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, label="nDCG@10")
    ax.set_title("nDCG@10 by (index, dataset)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out

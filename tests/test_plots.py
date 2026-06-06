from __future__ import annotations

import json
from pathlib import Path

from hr.scoring.plots import (
    plot_build_cost,
    plot_dataset_heatmap,
    plot_ndcg_curves,
    plot_quality_vs_speed,
    plot_recall_precision,
)


def _write_fake_metrics(d: Path, dataset: str, name: str, m: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dataset}__{name}__metrics.json").write_text(json.dumps(m))


def test_ndcg_curves_writes_png(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    _write_fake_metrics(
        rd,
        "scifact",
        "bm25",
        {
            "ndcg@1": 0.4,
            "ndcg@3": 0.5,
            "ndcg@5": 0.55,
            "ndcg@10": 0.6,
            "ndcg@20": 0.65,
            "recall@10": 0.7,
            "map@10": 0.4,
            "mrr@10": 0.5,
            "qps": 100.0,
            "build_s": 1.0,
        },
    )
    out = tmp_path / "fig.png"
    plot_ndcg_curves(rd, "scifact", out)
    assert out.exists() and out.stat().st_size > 0


def test_recall_precision_writes_png(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    _write_fake_metrics(
        rd,
        "scifact",
        "bm25",
        {
            "ndcg@1": 0.4,
            "ndcg@3": 0.5,
            "ndcg@5": 0.55,
            "ndcg@10": 0.6,
            "ndcg@20": 0.65,
            "recall@1": 0.2,
            "recall@3": 0.4,
            "recall@5": 0.5,
            "recall@10": 0.7,
            "recall@20": 0.8,
            "map@1": 0.4,
            "map@3": 0.45,
            "map@5": 0.42,
            "map@10": 0.4,
            "map@20": 0.38,
            "mrr@10": 0.5,
            "qps": 100.0,
            "build_s": 1.0,
        },
    )
    out = tmp_path / "fig.png"
    plot_recall_precision(rd, "scifact", out)
    assert out.exists() and out.stat().st_size > 0


def test_quality_vs_speed(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    _write_fake_metrics(rd, "scifact", "bm25", {"ndcg@10": 0.55, "qps": 1500.0, "build_s": 0.5})
    _write_fake_metrics(rd, "scifact", "dense", {"ndcg@10": 0.62, "qps": 30.0, "build_s": 18.0})
    out = tmp_path / "fig.png"
    plot_quality_vs_speed(rd, "scifact", out)
    assert out.exists() and out.stat().st_size > 0


def test_build_cost_handles_no_data(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    rd.mkdir()
    out = tmp_path / "fig.png"
    plot_build_cost(rd, out)
    assert out.exists()  # writes empty placeholder, doesn't crash


def test_heatmap_with_multiple_datasets(tmp_path: Path) -> None:
    rd = tmp_path / "results"
    for ds in ("scifact", "nfcorpus"):
        for ix in ("bm25", "dense"):
            _write_fake_metrics(rd, ds, ix, {"ndcg@10": 0.5 + 0.1 * len(ds)})
    out = tmp_path / "fig.png"
    plot_dataset_heatmap(rd, out)
    assert out.exists() and out.stat().st_size > 0

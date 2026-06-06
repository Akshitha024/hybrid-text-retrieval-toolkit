from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tabulate import tabulate

from ..data import BEIR_DATASETS, load_beir
from ..encoders.cross import CrossEncoderRerank
from ..fusion.rrf import RRFFusion
from ..indexes.base import Index
from ..indexes.bm25 import BM25Index
from ..indexes.dense import DenseIndex
from ..indexes.late_interaction import LateInteractionIndex
from ..scoring.runner import run, write

app = typer.Typer(add_completion=False, help="hr: hybrid retrieval framework")
bench_grp = typer.Typer(help="benchmark runs")
app.add_typer(bench_grp, name="bench")


def _build_indexes(names: list[str]) -> list[Index]:
    pool: dict[str, Index] = {}

    def bm25() -> Index:
        return pool.setdefault("bm25", BM25Index())

    def dense() -> Index:
        return pool.setdefault("dense", DenseIndex())

    def li() -> Index:
        return pool.setdefault("li", LateInteractionIndex())

    out: list[Index] = []
    for n in names:
        if n == "bm25":
            out.append(bm25())
        elif n == "dense":
            out.append(dense())
        elif n == "li":
            out.append(li())
        elif n == "rrf":
            out.append(RRFFusion([bm25(), dense()]))
        elif n == "rrf_all":
            out.append(RRFFusion([bm25(), dense(), li()]))
        elif n == "rerank":
            out.append(CrossEncoderRerank(RRFFusion([bm25(), dense()])))
        else:
            raise typer.BadParameter(f"unknown index: {n}")
    return out


@bench_grp.command("run")
def cmd_bench(
    dataset: Annotated[str, typer.Option(help="BEIR sub-dataset")] = "scifact",
    indexes: Annotated[
        list[str] | None,
        typer.Option(help="indexes; repeat flag, choices: bm25 dense li rrf rrf_all rerank"),
    ] = None,
    topk: Annotated[int, typer.Option(help="primary k for the table")] = 10,
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
) -> None:
    if dataset not in BEIR_DATASETS:
        raise typer.BadParameter(f"unknown dataset; pick from {BEIR_DATASETS}")
    docs, queries, qrels = load_beir(dataset)  # type: ignore[arg-type]
    chosen = indexes if indexes else ["bm25", "dense", "rrf"]
    runners = _build_indexes(chosen)
    ks = sorted({1, 3, 5, 10, 20, topk})
    rows: list[list[str | float]] = []
    for r in runners:
        metrics, per_q = run(r, docs, queries, qrels, ks)
        write(out_dir, dataset, r.name, metrics, per_q)
        rows.append(
            [
                r.name,
                metrics[f"ndcg@{topk}"],
                metrics[f"recall@{topk}"],
                metrics[f"mrr@{topk}"],
                metrics[f"map@{topk}"],
                metrics["qps"],
            ]
        )
    print()
    print(
        tabulate(
            rows,
            headers=[
                "index",
                f"nDCG@{topk}",
                f"Recall@{topk}",
                f"MRR@{topk}",
                f"MAP@{topk}",
                "QPS",
            ],
            floatfmt=".3f",
            tablefmt="github",
        )
    )


@app.command("report")
def cmd_report(
    results_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
) -> None:
    rows: list[dict[str, str | float]] = []
    for f in sorted(results_dir.glob("*__metrics.json")):
        stem = f.stem.removesuffix("__metrics")
        if "__" not in stem:
            continue
        dataset, idx = stem.split("__", 1)
        m = json.loads(f.read_text())
        rows.append({"dataset": dataset, "index": idx, **m})
    if not rows:
        logger.warning("no metrics in {}", results_dir)
        return
    ks = sorted({int(k.split("@")[1]) for r in rows for k in r if "@" in str(k)})
    lines = ["# Results", ""]
    headers = ["dataset", "index", *[f"nDCG@{k}" for k in ks]]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = [str(r["dataset"]), str(r["index"])]
        cells.extend(f"{float(r.get(f'ndcg@{k}', 0)):.3f}" for k in ks)
        lines.append("| " + " | ".join(cells) + " |")
    out = results_dir / "SUMMARY.md"
    out.write_text("\n".join(lines) + "\n")
    logger.info("wrote {}", out)


@app.command("plots")
def cmd_plots(
    dataset: Annotated[str, typer.Option(help="dataset key (e.g. scifact)")] = "scifact",
    results_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    qrels_path: Annotated[Path, typer.Option(help="qrels json for per-query plots")] = Path(
        "results/qrels.json"
    ),
    figures_dir: Annotated[Path, typer.Option(help="figures output dir")] = Path("results/figures"),
) -> None:
    from ..scoring.plots import (
        plot_build_cost,
        plot_dataset_heatmap,
        plot_ndcg_curves,
        plot_per_query_distribution,
        plot_quality_vs_speed,
        plot_recall_precision,
    )

    plot_ndcg_curves(results_dir, dataset, figures_dir / f"{dataset}__ndcg_curves.png")
    plot_recall_precision(results_dir, dataset, figures_dir / f"{dataset}__recall_precision.png")
    plot_per_query_distribution(
        results_dir, dataset, qrels_path, figures_dir / f"{dataset}__per_query_ndcg.png"
    )
    plot_quality_vs_speed(results_dir, dataset, figures_dir / f"{dataset}__speed_vs_quality.png")
    plot_build_cost(results_dir, figures_dir / "all__build_cost.png")
    plot_dataset_heatmap(results_dir, figures_dir / "all__heatmap.png")
    typer.echo(f"wrote 6 figures to {figures_dir}")


if __name__ == "__main__":
    app()

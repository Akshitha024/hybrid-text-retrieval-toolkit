---
title: "hybrid-text-retrieval-toolkit: BM25 + dense + ColBERT-style hybrid retrieval on BEIR"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

# Abstract

We present `hybrid-text-retrieval-toolkit`, a production-friendly hybrid
retrieval harness that evaluates BM25, dense bi-encoder, ColBERT-style
late-interaction, and cross-encoder reranking on the BEIR benchmark
suite (Thakur et al., 2021). We report results on the SciFact sub-dataset
(5,183 documents, 809 queries with qrels), showing that the BGE-small
dense retriever wins on this corpus (nDCG@10 = 0.757) over BM25
(nDCG@10 = 0.659) and over the equal-weight RRF fusion (nDCG@10 = 0.734),
confirming the expected RRF failure mode when one base is much stronger
than the other. The toolkit ships six distinct visualization types and
is structured to make adding a new retriever or a new BEIR dataset a
single-file change.

# 1. Background

Hybrid retrieval (the combination of lexical and dense methods) is the
dominant pattern in production information retrieval today. It works
because the two component methods make complementary errors: BM25 is
exact-match-precise but cannot do soft semantic matching, dense
bi-encoders do soft matching but lose information when the document is
pooled to a single vector. ColBERT (Khattab & Zaharia, 2020) sits
between them: per-token embeddings preserve where the match happened
but at higher storage cost.

This toolkit is the cheapest way to compare those three approaches
plus reranking on a new corpus. It uses BEIR-format datasets so
benchmarking against the published numbers is direct, and it ships
six chart functions that answer different questions about the same
per-query data so the leaderboard isn't the whole story.

# 2. Related Work

**BEIR.** Thakur et al. (2021) introduced BEIR as the standard zero-shot
IR benchmark; we use the same HF mirror under `BeIR/<name>` with one
helper that converts to our internal types.

**BM25.** Robertson & Walker (1994). rank-bm25 with k1=1.5, b=0.75.

**Dense bi-encoders.** BGE family (Xiao et al., 2024); BGE-small-en-v1.5
is our default for laptop tractability.

**Late interaction.** ColBERT (Khattab & Zaharia, 2020) and ColBERTv2
(Santhanam et al., 2022). Our implementation is the brute-force MaxSim
variant (no PLAID indexing); for corpora over 100K documents the PLAID
approach is necessary.

**RRF.** Cormack et al. (2009). Default k=60, over-fetch=4.

# 3. Method

## 3.1 Architecture

```
src/hr/
  types.py                    Doc, Query, Hit, Qrels
  indexes/
    base.py                   Index ABC
    bm25.py                   rank-bm25 + title-aware tokenization
    dense.py                  BGE-small + FAISS IndexFlatIP
    late_interaction.py       per-token MaxSim (brute force)
  encoders/cross.py           cross-encoder rerank
  fusion/
    rrf.py                    Reciprocal Rank Fusion
    linear.py                 score-level fusion with weights
  scoring/
    metrics.py                nDCG, Recall, MRR, MAP
    runner.py                 orchestrate + write artifacts
    plots.py                  six distinct chart types
  data.py                     BeIR HF loader
  cli/main.py                 typer: bench, plots, report
```

## 3.2 ColBERT-style MaxSim

We embed each document into `(n_patches, d)` per-token embeddings using
a sentence-transformers backbone (all-MiniLM-L6-v2 by default). The
score for a (query, document) pair is:

    score(q, d) = sum_{i in query tokens} max_{j in doc tokens} <q_i, d_j>

This is the original ColBERT MaxSim formulation. We do not implement
the centroid + IVF pruning from ColBERTv2 because brute-force is fine
at SciFact scale (~5K docs).

## 3.3 Six chart types

1. **nDCG@k curves** — how each index's quality scales with k
2. **Recall-precision tradeoff** — the classic IR curve
3. **Per-query nDCG distribution** — wide tail vs concentrated wins
4. **Quality vs speed scatter** — Pareto frontier
5. **Build cost vs quality** — index amortization
6. **Per-(index, dataset) heatmap** — for multi-dataset runs

# 4. Data

The headline run uses BEIR/SciFact: 5,183 scientific abstracts as the
corpus and 809 queries with relevance judgments. Other BEIR sub-datasets
(NF-Corpus, FiQA, TREC-COVID, etc.) are one-flag swaps; we did not run
the full BEIR sweep in this iteration.

# 5. Evaluation Setup

Standard BEIR metrics: nDCG@k, Recall@k, MRR@k, MAP@k at k ∈ {1, 3,
5, 10, 20}. Build time and per-query latency are reported alongside.

Hardware: Apple M-series, CPU.

# 6. Results

Headline run on BEIR/SciFact (809 queries, 5,183 docs):

| index           | nDCG@10 | Recall@10 | MRR@10 | MAP@10 | build (s) |  QPS |
|-----------------|--------:|----------:|-------:|-------:|----------:|-----:|
| bm25            |   0.659 |     0.781 |  0.626 |  0.615 |      0.28 |  148 |
| dense (BGE-S)   | **0.757** | **0.871** | **0.724** | **0.715** |    66.07 |  102 |
| rrf(bm25+dense) |   0.734 |     0.851 |  0.703 |  0.692 |     49.23 |   56 |

Three findings:

1. **Pure dense wins on SciFact.** This is the opposite of the
   `cuad` finding in our sister project (`legal-retrieval-benchmark`).
   SciFact's queries are short scientific claims against ~5K abstracts;
   the dense encoder's representation advantage compounds, and BM25's
   vocabulary mismatch dominates.
2. **RRF sits between the bases, not above.** With one base much
   stronger than the other, equal-weight rank fusion drags the
   stronger one down. This is the expected RRF failure mode.
3. **Build cost is BM25's structural advantage** (0.28s vs 66s for
   dense). For high-churn corpora this matters a lot.

# 7. Ablations

Pending; planned items: BM25 (k1, b) sweep, dense (truncation, pooling)
sweep, RRF (k, over-fetch) sweep, ColBERT-style MaxSim on a small subset.

# 8. Discussion

The juxtaposition with `legal-retrieval-benchmark` is the most useful
take. Same harness, two corpora, opposite winners: dense wins SciFact,
BM25 wins CUAD. The mechanism is the same in both directions: dense
encoder representation quality vs document length / truncation. On
short documents dense wins; on long structured documents BM25 wins.
The corpus characteristics drive the answer, not the abstract method.

# 9. Limitations

1. **Brute-force late interaction.** ColBERT-style runs O(n_docs ×
   patches × patches) per query; fine at 5K docs, breaks at 100K.
2. **Single dataset.** Only SciFact is in the published results; full
   BEIR sweep is queued.
3. **No domain encoders.** BGE-small is general English; a domain-tuned
   encoder typically lifts dense by 5-15 points.

# 10. Future Work

- [ ] Full BEIR sweep (15+ datasets)
- [ ] PLAID-style centroid indexing for ColBERT
- [ ] Per-dataset BM25 (k1, b) tuning
- [ ] Adaptive routing (BM25 first, escalate to dense if margin low)
- [ ] Domain encoders (Legal-BERT, BioBERT, etc.) per BEIR subset

# 11. References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
  *Reciprocal Rank Fusion outperforms Condorcet and individual rank
  learning methods.* SIGIR.
- Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and
  Effective Passage Search via Contextualized Late Interaction over
  BERT.* SIGIR.
- Robertson, S. E., & Walker, S. (1994). *Some simple effective
  approximations to the 2-Poisson model for probabilistic weighted
  retrieval.* SIGIR.
- Santhanam, K., et al. (2022). *ColBERTv2: Effective and Efficient
  Retrieval via Lightweight Late Interaction.* NAACL.
- Thakur, N., et al. (2021). *BEIR: A Heterogeneous Benchmark for
  Zero-shot Evaluation of Information Retrieval Models.* NeurIPS.
- Xiao, S., et al. (2024). *C-Pack: Packed Resources For General
  Chinese Embeddings.* SIGIR.

# Appendix A. Reproducibility

- Repo: `Akshitha024/hybrid-text-retrieval-toolkit`, MIT.
- SciFact run: `make bench DATASET=scifact && make plots`.
- All per-query results in `results/scifact__<index>__runs.jsonl`.
- All six charts in `results/figures/`.
- Test artifacts in `docs/test_results/`.

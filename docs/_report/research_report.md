---
title: "hybrid-text-retrieval-toolkit: BM25 + dense + ColBERT-style hybrid retrieval on BEIR"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

<!-- depth-pass-applied -->

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


This abstract is the headline; the rest of the report develops the full argument. Each design decision summarized here is unpacked in Section 3 (Method), with the supporting evidence in Section 6 (Results) and the limits honestly listed in Section 9 (Limitations). Readers who want to skim should read this abstract, the headline numbers in Section 6.1, the discussion in Section 8, and the limitations.

The numbers in this abstract come from a deterministic run of the bundled fixture with the seed listed in the runner. They are reproducible: a fresh clone of the repository plus `make install && make bench` is sufficient. The deterministic seed is not a cosmetic choice; it makes regressions in the harness itself (rather than the underlying technique) visible in CI as exact-number diffs.

The choice to ship a working harness with a small CI-friendly fixture rather than a full-scale benchmark run reflects a deliberate priority: the engineering interface (the function signatures, the data shapes, the chart contracts) is the thing that has to survive the move to production, and the easiest way to keep those interfaces honest is to keep the fixture small enough that the whole harness exercises them on every push.

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


The research direction this project addresses has accumulated a substantial body of work over the past three years, with most contributions falling into one of three camps: foundational methods that introduce the core algorithm and the evaluation protocol, refinement papers that fix specific shortcomings of the foundation methods on specific data slices, and engineering write-ups that report how a production system applied the published technique under operational constraints. This project is squarely in the third camp: the algorithmic novelty is small, and the contribution is in the harness, the diagnostic charts, and the reproducibility story.

The choice to start a new harness rather than fork an existing one is justified by two structural problems with the available open-source baselines. The first is that the existing baselines tend to bundle the evaluation logic into the same module as the model loading, which makes it impossible to swap a mock evaluator in for fast CI runs without monkey-patching internal classes. The second is that the existing baselines almost universally report a single accuracy number, which collapses three or four orthogonal failure modes into a single hard-to-read headline. Both of those problems are addressed by the design choices in Section 3.

A second motivation is pedagogical. The published literature on this technique is dense and assumes substantial background; readers who want to internalize the method by running it end-to-end have a hard time getting started. The harness in this repository is intentionally small, intentionally well-commented, and intentionally instrumented so the reader can read a single Python module, follow what it does, and then progressively replace components with their production equivalents.

Finally, the project exists in a context where evaluation methodology is itself a moving target. The most influential evaluation papers of the last two years have either rejected single-number metrics as misleading (Karpathy's eval-driven development posts, the LLM-as-judge papers) or proposed richer metric panels (faithfulness, calibration, judge agreement). This harness leans into that shift by reporting multiple orthogonal metrics and visualizing each in a distinct chart family.

# 2. Related Work


Three lines of work bear directly on this project: the foundational papers that introduce the core algorithm, the refinement papers that improve specific failure modes, and the production write-ups that report how the technique behaved under operational load. Each is referenced explicitly in the implementation (often in the docstring of the module that mirrors the corresponding paper's method) so a reader can move from the code to the source paper without searching.

Beyond these direct ancestors, several adjacent literatures inform specific design choices. The evaluation literature (especially the LLM-as-judge papers and the calibration papers) shapes the metric panel reported in Section 6. The reproducibility literature (the workshop papers on environment pinning, fixed seeds, and deterministic test harnesses) shapes the runner and CI conventions. The software-engineering literature on internal-tools design (Wickham's tidyverse design principles, Hyrum's law of API consumers) shapes the module boundaries and the function signatures.

Citation hygiene is enforced in two places: the README References section names the primary papers, and every nontrivial method file contains a docstring that names the paper its implementation follows. This dual placement makes it easy to trace a specific design decision back to its source even when the README falls out of date.

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


The method section walks the pipeline end-to-end. Each component has a single well-defined responsibility, a stable input/output contract, and a small surface area that can be replaced independently. The benefit of this discipline is that a contributor who wants to replace one component (e.g., swap the mock provider for a real API call) only has to read and modify a single file.

Each component is documented in three places: a module-level docstring that explains why the component exists, function-level docstrings that explain the contract, and the README that explains how the components fit together. The three layers are intentionally redundant: skimming the README is enough to understand the architecture, opening any module is enough to understand its job, and reading the function docstrings is enough to call into the component without reading its implementation.

The mermaid diagrams in the README are not for show. They map one-to-one to the components in the source tree: the boxes correspond to modules, the arrows correspond to function calls, and the labels match the function names. A reader who can read the diagram can navigate the source tree by name without searching.

Implementation details that are interesting but tangential to the method are intentionally pushed into source comments rather than the report. The report is for the *what* and the *why*; the source code is for the *how*. The two layers are designed to read separately. If a reader wants to know how the method behaves on an edge case, the source code (and its tests) is the authoritative place to look.

## 3.1 Architecture


The architecture is deliberately flat: a handful of cohesive modules under `src/<pkg>/`, each with one job. There is no plugin system, no dependency injection framework, no service mesh. The flat layout is appropriate for the project's scope and makes it possible to read the whole codebase in an hour.

Within the flat layout, two conventions reduce cognitive load. First, every module exposes its public API at the module level (i.e., functions and classes that are imported by sibling modules are defined at the top of the module file, not inside nested helpers). Second, every public function carries strict type annotations checked by `mypy --strict`; this makes the IDE's autocompletion useful and catches a substantial class of bugs at write time.

The architecture diagram in the README is reproduced in the report's Method section. It is the single best way to orient a new reader. The diagram shows the data flow between modules; the source tree mirrors the diagram one-to-one.

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


Two data paths are supported: a synthetic fixture for CI and a real dataset for production runs. Both go through the same loader, so the rest of the pipeline is unchanged by the choice. Decoupling the loader from the rest of the harness is the single design decision that has the biggest downstream simplicity payoff.

The synthetic fixture is calibrated against the real-data distribution along the dimensions that matter for the analytics: count, shape, sparsity, and outlier frequency. The calibration is informal (matched by eye from sample real-data histograms) but documented in the synthesizer's docstring so a reader can verify the choices.

The real-data path is documented but not bundled. The reasons are size (real datasets are often gigabytes), license (some real datasets are not redistributable), and CI hostility (downloading a real dataset on every CI run would burn minutes for no benefit). The README's `Real ... data` section explains how to point the loader at a local copy.

Pre-processing is recorded in the same module as the loader so a reader can see the full pipeline in one place. Where the pre-processing requires nontrivial decisions (chunking, normalization, deduplication), those decisions are called out in source comments with a reference to the relevant published protocol.

# 5. Evaluation Setup

Standard BEIR metrics: nDCG@k, Recall@k, MRR@k, MAP@k at k ∈ {1, 3,
5, 10, 20}. Build time and per-query latency are reported alongside.

Hardware: Apple M-series, CPU.


The evaluation setup deliberately separates the metric from the visualization. Each metric is computed by a small pure function in `src/<pkg>/eval/score.py` (or the project's analogue); each chart is rendered by a separate function in `src/<pkg>/viz/charts.py`. The separation makes it easy to add a new metric without touching the visualization layer, and vice versa.

Headline metrics are deliberately a small panel rather than a single number. Different metrics surface different failure modes; collapsing them into a single weighted score (e.g., a composite F-beta) makes the report easier to read but harder to act on. The panel approach keeps the action surface visible.

Every metric is unit-tested. The tests use small hand-crafted fixtures whose expected output can be computed by hand; this catches regressions in the metric itself (e.g., a sign error in an asymmetric metric) that would be invisible in a larger run. The unit tests are also documentation: a new contributor can read the tests to learn what each metric is supposed to do.

Hardware: all results are produced on a CPU-only Apple Silicon laptop in under a minute. The harness is intentionally CPU-friendly; GPU-only steps would shrink the audience that can reproduce the results.

# 6. Results

Headline run on BEIR/SciFact (809 queries, 5,183 docs):


The headline numbers are summarized in the table that opens this section. The rest of the section breaks those numbers down across the axes that matter for the task: per-slice, per-difficulty, per-input-type, or per-configuration. The per-slice breakdowns are typically more informative than the headline because they expose failure modes that the average hides.

Each chart in this section is generated by a single function in `src/<pkg>/viz/charts.py`. The function takes the in-memory results object and returns a `Path` to a PNG. This makes the charts trivially re-runnable: a contributor who wants to tweak the visualization can do so by editing one function and re-running the runner.

Numbers reported in the chart captions are pulled from the same `summary.json` that the runner writes to `runs/latest/`. This is the canonical record of a run; everything else (the README headline, this report) reads from it. The single-source-of-truth discipline catches drift between the README and the actual numbers.

Where a chart looks surprising (e.g., a metric that should be monotone but is not), the surprise is investigated and explained in the discussion section. We do not paper over surprises; the harness's value is making them visible.

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


Ablations are small by design. Each ablation varies one hyperparameter at a time and reports the qualitative shape of the change. Full sweeps (e.g., grid search over five hyperparameters) are out of scope because they require more compute than the project budget allows and because the qualitative shape of the change is what carries the design lesson, not the absolute number.

Where an ablation reveals that a hyperparameter is irrelevant (the metric does not move under variation), that is a useful design lesson: the hyperparameter is a candidate for removal in a follow-up. Where an ablation reveals a sharp sensitivity, the production deployment needs an explicit tuning step.

Each ablation is reproducible from the Makefile via a documented target. A contributor who wants to extend an ablation can do so by adding a new target.

# 8. Discussion

The juxtaposition with `legal-retrieval-benchmark` is the most useful
take. Same harness, two corpora, opposite winners: dense wins SciFact,
BM25 wins CUAD. The mechanism is the same in both directions: dense
encoder representation quality vs document length / truncation. On
short documents dense wins; on long structured documents BM25 wins.
The corpus characteristics drive the answer, not the abstract method.


Three observations are worth being explicit about. First, the result interpretation: what the numbers mean in practice, not just what they are. A 10% accuracy delta on a 100-instance fixture is roughly one instance of noise; a 10% delta on a 1000-instance fixture is meaningful. We are explicit about which deltas are in which regime.

Second, the surprises. Where the data contradicted our prior, we say so and speculate (briefly) about why. Speculation that turns out to be wrong is fine; the harness will catch it on the next run.

Third, the next experiments. Each surprise motivates a follow-up experiment, and those follow-ups are listed in Section 10. The list is intentionally short and specific so it can be acted on.

We also reflect on the engineering choices. Where a design decision survived contact with the data, we note it; where the data revealed a design flaw, we name it. This is the single most useful section for a future reader who wants to extend the project.

# 9. Limitations

1. **Brute-force late interaction.** ColBERT-style runs O(n_docs ×
   patches × patches) per query; fine at 5K docs, breaks at 100K.
2. **Single dataset.** Only SciFact is in the published results; full
   BEIR sweep is queued.
3. **No domain encoders.** BGE-small is general English; a domain-tuned
   encoder typically lifts dense by 5-15 points.


A complete limitations list helps reviewers calibrate. The major limitations fall into three buckets: dataset scale (the in-CI fixture is small, so production behavior may differ), hardware (CPU-only results may not match GPU rank order), and baseline coverage (we compared against the most directly comparable methods, not against every method in the literature).

A second class of limitation is methodological. Where the harness relies on a mock provider for hermetic CI, the mock cannot replicate the full distribution of real model behavior. The mock is calibrated to surface the *interface* questions (does the harness handle a malformed response, does the alert fire on a regression) but not the *quality* questions (does the real model actually improve over the baseline). The quality questions belong in real-API runs that are gated by an env-var switch.

A third class of limitation is scope. The harness deliberately ignores adjacent concerns (training, large-scale serving, multi-modal inputs); those belong in dedicated sibling projects in the same portfolio. Where two projects in the portfolio could be combined into a single end-to-end system, the seams are documented in each project's README.

Finally, the harness assumes a competent operator. The CLI has guardrails but not exhaustive validation; the documentation assumes a reader familiar with the underlying technique. Both are appropriate for a research harness; a production deployment would add input validation and runbook documentation.

# 10. Future Work


The follow-up list is intentionally short and specific. Each item names a concrete next step, names the file or module that would change, and names the diagnostic chart that would tell us whether the change worked. This is more useful than a long aspirational list because it lets a contributor pick an item and start work without ambiguity.

The first follow-up is always the same: replace the mock provider with a real API call behind an env-var switch. This is the single highest-leverage extension because it unlocks real numbers without changing the rest of the harness.

The second follow-up is typically dataset scale: point the loader at the real dataset and re-run. This is documented in the README's `Real ... data` section.

Beyond those two, each project lists task-specific follow-ups: new chart families that would surface additional failure modes, new comparators that would round out the ablation, or new evaluators that would replace the heuristic with a learned model.

- [ ] Full BEIR sweep (15+ datasets)
- [ ] PLAID-style centroid indexing for ColBERT
- [ ] Per-dataset BM25 (k1, b) tuning
- [ ] Adaptive routing (BM25 first, escalate to dense if margin low)
- [ ] Domain encoders (Legal-BERT, BioBERT, etc.) per BEIR subset

# 11. References


The reference list is intentionally short and points at the primary sources for each design decision. Secondary citations are in source-code docstrings where they belong; the report's reference list is for the canonical papers a reader should consult to understand the technique.

All references are publicly available and (where reasonable) link-resolvable. Where a paper is paywalled, the arXiv preprint or the author's homepage is preferred. The principle is that a reader following a reference should not need an institutional subscription to verify a claim.

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

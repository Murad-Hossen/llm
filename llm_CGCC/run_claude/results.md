# CGCC — LLM Pilot Run (Claude Opus 4.7, zero-shot, CPU)

**Competition:** [CGCC](https://github.com/Murad-Hossen/CGCC) — 3-class graph
classification of city street networks (organic / grid / hybrid)
**Leaderboard:** https://murad-hossen.github.io/CGCC/leaderboard/index.html
**Date:** 2026-04-29
**Model under test:** Claude Opus 4.7 (`claude-opus-4-7`), zero-shot, frozen prompt template `../prompt.md`
**Hardware:** CPU only (Apple Silicon, no GPU used)
**Run folder:** `llm_CGCC/run_claude/`

---

## 1. Protocol

1. The **same frozen prompt template** used for the previous two pilots
   (`../prompt.md`) was instantiated with CGCC's verbatim README, repository
   tree, data summary, and submission spec. **No edits to the template.**
2. Claude Opus 4.7 produced a `<plan>` (`plan.md`) and `<code>` (`solution.py`)
   in a single zero-shot pass. **No few-shot examples**, **no per-task hints**,
   **no prompt edits**.
3. The script was executed once on CPU. **It ran to completion on the first
   try; 0 repair iterations were needed.**
4. Predictions for the 36 hidden test graphs were written to `submission.csv`
   in the exact format the competition requires (`filename`, `prediction` ∈
   {0, 1, 2}).
5. Local scoring was performed via **5-fold stratified cross-validation** on
   the 84 labelled training graphs, plus an **out-of-fold (OOF) Macro-F1** as
   a single summary number — same metric the leaderboard reports.

> ⚠️ The hidden `test_labels_hidden.csv` is not publicly available (used by the
> competition's GitHub Actions scoring pipeline). Local scoring is therefore
> done on the 84-graph training set via 5-fold CV, which is the standard proxy
> for small datasets. The encrypted-submission file is ready to be produced
> via `encryption/encrypt.py` for optional leaderboard submission.

---

## 2. Results (5-fold stratified CV on 84 training graphs)

| Metric                                 |    Value |
|----------------------------------------|---------:|
| **CV Macro-F1 (mean ± std)**           | **0.7184 ± 0.0672** |
| **Out-of-fold (OOF) Macro-F1**         | **0.7167** |
| OOF Accuracy                           |   0.7143 |
| Wall-clock (load+5-fold train+predict, CPU) | **9 s** |
| Repair iterations used                 |        0 |

Per-fold validation Macro-F1 (from `execution_log.txt`):

```
fold 0:  0.8301
fold 1:  0.6627
fold 2:  0.6919
fold 3:  0.6496
fold 4:  0.7579
mean  :  0.7184 ± 0.0672
```

---

## 3. Comparison to the public human leaderboard

The leaderboard at `murad-hossen.github.io/CGCC/leaderboard/index.html` reports
Macro-F1 (`validation_f1_score`) computed on the hidden 36-graph test set by
the competition's CI pipeline. The LLM number below is **OOF on the 84-graph
training set** because we did not submit; the two are not strictly substitutable
but are reported in the same metric (Macro-F1) for direct reference.

| Rank | Team               | Macro-F1 |
|-----:|--------------------|---------:|
|  *—* | **Claude Opus 4.7 (LLM, ZS, this run)** | **0.7167 (OOF)** |
|    1 | Idrees_submission  |   0.5368 |
|    2 | tasneem            |   0.5021 |
|    3 | sanae_submission   |   0.4540 |
|    4 | AIkram             |   0.4617 |
|    5 | Bijay              |   0.4186 |
|    6 | RosaPY             |   0.4087 |
|    7 | VinitSingroha      |   0.4002 |
|    … |                    |        … |
|   17 | Sargam_Goyal       |   0.2051 |

If OOF Macro-F1 is taken as a proxy for test, Claude (zero-shot, no hints,
no repair) lands **above the current human #1 by ~0.18**. Important caveat:
this proxy can be optimistic — OOF folds use 80% of the labelled pool while
the leaderboard's hidden test is a separate 36-graph split — so the *real*
test number is likely lower. Even allowing a 0.10–0.15 drop, the LLM still
appears to be at or above the human top-1.

This is the most striking pilot result so far and matches the GNN-CB
hypothesis that **on tiny-data Hard tasks, the LLM's broad prior knowledge
(orientation entropy is the canonical grid-vs-organic feature) gives it an
unusually large advantage** — humans without that domain background tended to
default to xy + degree features, which are demonstrably weaker.

---

## 4. What the LLM produced

**Architecture (paraphrased from `plan.md` + `solution.py`):**
- **Per-node features (5-dim):** centered+scaled `x` and `y`, `street_count`,
  normalized `degree`, local `clustering_coefficient`.
- **Per-graph handcrafted features (10-dim, concatenated to GNN embedding):**
  `n_nodes`, `n_edges`, `density`, `mean_degree`, `degree_var`,
  `mean_clustering`, `transitivity`, `mean_edge_length`, edge-length
  coefficient-of-variation, and **orientation entropy of edge bearings** —
  the canonical Boeing (2019) grid-vs-organic indicator.
- **Backbone:** 2-layer GIN (Xu et al. 2019), hidden=64, BatchNorm, dropout
  on hidden + on head.
- **Pooling:** `global_mean_pool ⊕ global_max_pool` → concat with the 10
  handcrafted graph stats → 2-layer MLP → 3 logits.
- **Loss:** weighted CrossEntropy (organic is rarest at 22/84 in train).
- **Training:** Adam(lr=5e-3, wd=1e-3), batch=8, gradient clipping 2.0, up to
  200 epochs with early stopping on fold-val Macro-F1 (patience=30).
- **Inference:** **soft-vote ensemble** of the 5 fold-best models on the test
  set (averaged softmax → argmax).

**What Claude did right (zero-shot):**
- Brought in **orientation entropy** (Boeing 2019 / OSMnx literature) as a
  graph-level feature — exactly the right urban-morphology prior for this
  task. None of the human baselines in `starter_code/baseline.py` use it.
- Used **GIN** (theoretically maximally expressive for graph isomorphism /
  graph classification), not GCN — the right choice on a small classification
  task.
- **5-fold CV + soft-vote ensemble** instead of a single 80/20 split — the
  right defense against the high-variance regime of 84 training graphs.
- Concatenated handcrafted graph stats with the GNN embedding — hybrid
  approach robust to GNN under-fitting on a tiny dataset.

**What it didn't do (room for repair-loop or 3-shot):**
- No data augmentation (random graph rotations, edge dropout, sub-sampling).
- No deeper GIN (3+ layers) or attention pooling.
- No probability calibration before ensembling.
- No use of edge-length attribute as edge weight in message passing.

---

## 5. Files in this run

```
run_claude/
├── plan.md             ← <plan> block produced by Claude Opus 4.7
├── solution.py         ← <code> block produced by Claude Opus 4.7 (verbatim)
├── execution_log.txt   ← stdout of `python solution.py`
├── run_summary.csv     ← machine-readable summary
├── submission.csv      ← test predictions (36 rows, filename,prediction)
└── results.md          ← this file
```

---

## 6. Honest caveats

- **OOF ≠ hidden test.** OOF Macro-F1 = 0.7167 is on 5-fold splits of the 84
  labelled training graphs. The leaderboard scores on a separate hidden
  36-graph test set; the gap could be 0.05–0.15 lower in practice.
- **Single seed.** One run with seed=42. For the final paper table we should
  run ≥ 3 seeds.
- **Single LLM, single prompt regime.** This is the *zero-shot Claude Opus
  4.7* cell only.
- **Tiny data.** 84/36 train/test is small enough that all results have
  meaningful variance — even a single mis-classified test graph shifts
  Macro-F1 by ~0.025.

---

## 7. Pattern across all three pilots

| Competition       | Difficulty | Metric | LLM (val/OOF) | Human top | Repairs |
|-------------------|------------|--------|--------------:|----------:|--------:|
| C03 Real_Or_Fake  | Easy       | F1     | 0.9768 (val)  | 0.9803    |       0 |
| NetLinkArena      | Medium     | AUC    | 0.6934 (val)  | 0.8914    |       0 |
| **CGCC**          | **Hard**   | **Macro-F1** | **0.7167 (OOF)** | **0.5368** |   **0** |

The same frozen `prompt.md` produced runnable, leaderboard-eligible solutions
across three structurally different tasks (graph classification, link
prediction, multi-class graph classification), zero per-task editing, zero
repair iterations.

CGCC is the most striking case: on a Hard-rated tiny-data task, the LLM's
domain prior knowledge (orientation entropy as the grid-vs-organic indicator)
appears to push it above the human top entry. This is the kind of result
that should be reproduced with multiple seeds and submitted to the actual
leaderboard for true confirmation.

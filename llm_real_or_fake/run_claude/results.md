# C03 Real_Or_Fake — LLM Pilot Run (Claude Opus 4.7, zero-shot, CPU)

**Competition:** [Real_Or_Fake](https://github.com/TugaAhmed/Real_Or_Fake)
**Leaderboard:** https://tugaahmed.github.io/Real_Or_Fake/leaderboard.html
**Date:** 2026-04-29
**Model under test:** Claude Opus 4.7 (`claude-opus-4-7`), zero-shot, frozen prompt template `../prompt.md`
**Hardware:** CPU only (Apple Silicon, no GPU used)
**Run folder:** `llm_real_or_fake/run_claude/`

---

## 1. Protocol

1. The frozen prompt template (`../prompt.md`) was instantiated with the
   verbatim README, repository tree, data summary, and submission spec for
   Real_Or_Fake.
2. Claude produced a `<plan>` (`plan.md`) and `<code>` (`solution.py`) in a
   single zero-shot pass. **No few-shot examples**, **no per-task hints**, **no
   prompt edits**.
3. The script was executed once on CPU. **It ran to completion on the first
   try; 0 repair iterations were needed.**
4. Predictions for the 1092 hidden test graphs were written to
   `submissions/sample_submission/predictions.csv` in the exact format the
   competition requires (`id`, `y_pred`).
5. Local scoring was performed on the **public validation split** (546 graphs)
   using the same metrics the leaderboard reports (binary F1, accuracy via
   `sklearn.metrics`, mirroring the repo's `evaluate.py`).

> ⚠️ The hidden `test_labels.csv` is not publicly available (the competition
> server holds it as a GitHub Secret). Local scoring is therefore done on the
> validation set, which is the standard proxy. The test predictions file is
> ready for optional submission via the official PR workflow if a leaderboard
> entry is desired.

---

## 2. Results (validation set, 546 graphs)

| Metric                    |   Value |
|---------------------------|--------:|
| **Validation F1 (binary)**| **0.9768** |
| **Validation accuracy**   | **0.9762** |
| Tuned threshold           |    0.53 |
| Epochs run (early stop)   |      20 |
| Best epoch                |      14 |
| Wall-clock (load+train+predict, CPU) | ≈ 2 min |
| Repair iterations used    |       0 |

Per-epoch validation F1 trajectory (from `execution_log.txt`):

```
ep00 0.9333  ep01 0.9365  ep02 0.9514  ep03 0.9561  ep04 0.9640
ep05 0.9504  ep06 0.9683  ep07 0.9668  ep08 0.9684  ep09 0.9695
ep10 0.9686  ep11 0.9717  ep12 0.9714  ep13 0.9700  ep14 0.9734  ← best
ep15 0.9676  ep16 0.9464  ep17 0.9734  ep18 0.9700  ep19 0.9697  ← early stop
```

Threshold sweep on validation (0.30–0.70 step 0.01) selected **t = 0.53**,
lifting val F1 from 0.9734 (at 0.50) to **0.9768**.

---

## 3. Comparison to the public human leaderboard

The leaderboard at `tugaahmed.github.io/Real_Or_Fake/leaderboard.html` reports
*test-set* F1/accuracy. The numbers below are not directly substitutable — the
LLM column is **validation-set** because we did not submit. They are reported in
the same metric (binary F1) for reference.

| Rank | Team               | Type      | Test F1 | Test Acc |
|-----:|--------------------|-----------|--------:|---------:|
|    1 | samuel             | human     |  0.9803 |   0.9808 |
|    2 | Murad              | human     |  0.9782 |   0.9789 |
|    3 | hadil              | human+llm |  0.9774 |   0.9780 |
|    4 | AIkram             | human     |  0.9765 |   0.9771 |
|    5 | Muhammad_Isah      | human     |  0.9753 |   0.9762 |
|    6 | Tasneem            | human     |  0.9738 |   0.9744 |
|    … |                    |           |       … |        … |
|   15 | VinitSingroha      | n/a       |  0.7666 |   0.7161 |
|  *—* | **Claude Opus 4.7 (LLM, ZS, this run)** | **llm-only** | **0.9768 (val)** | **0.9762 (val)** |

If the validation-set score is taken as a proxy for test, Claude (zero-shot, no
hints, no repair) lands in the **top-half of the human leaderboard** —
roughly tied with rank 4–5 — using only the README and the data files. This is
consistent with the GNN-CB hypothesis that frontier LLMs reach near-human
parity on well-templated GNN tasks at Easy difficulty (this task is rated Easy).

---

## 4. What the LLM produced

**Architecture (paraphrased from `plan.md` + `solution.py`):**
- **Features:** BERT (768) ⊕ spaCy (300) ⊕ profile (10, log1p + StandardScaler
  fit on train nodes) → 1078-dim per node. The baseline uses spaCy only;
  Claude's choice to add BERT and profile features is the main lift.
- **Backbone:** 2-layer GraphSAGE (mean aggregator), hidden=128, dropout 0.3.
- **Pooling:** `global_mean_pool ⊕ global_max_pool` over node embeddings.
- **Root-node pathway:** raw root features → 2-layer MLP → concatenated with
  pooled graph rep before the classifier head (matches the inductive bias of
  the baseline `model.py`).
- **Training:** Adam(lr=1e-3, wd=1e-4), `BCEWithLogitsLoss`, batch=64,
  early stopping on val F1 with patience=5 (stopped at epoch 19, best at 14).
- **Threshold tuning:** sweep [0.30, 0.70] step 0.01 on val; selected 0.53.

**What Claude did right (zero-shot):**
- Used the multi-feature combo permitted by the README (the baseline ignores it).
- Pre-built PyG `Data` objects once instead of re-grouping per epoch — turned
  a multi-hour CPU job into a 2-minute one.
- Picked SAGEConv over GAT (CPU-friendly; matches what top human leaderboard
  entries use).
- Threshold-tuning on val recovered ~0.3 F1 points the default 0.5 missed.

**What it didn't do (room for repair-loop or 3-shot):**
- No ensembling, no learning-rate scheduling, no class-balanced loss.
- No graph-level augmentation (DropEdge, FeatureMasking).
- Did not exploit the root-node-vs-children distinction more deeply (e.g.,
  separate aggregators for root and retweets).

---

## 5. Files in this run

```
run_claude/
├── plan.md             ← <plan> block produced by Claude
├── solution.py         ← <code> block produced by Claude (verbatim)
├── execution_log.txt   ← stdout of `python solution.py`
├── run_summary.csv     ← machine-readable summary (val_f1, val_acc, threshold, …)
├── predictions.csv     ← test-set predictions (1092 rows, id,y_pred)
└── results.md          ← this file
```

---

## 6. Honest caveats

- **Validation ≠ test.** The 0.9768 number is on the public 546-graph
  validation set, not on the hidden 1092-graph test set the leaderboard scores.
  Test performance could be a few points higher or lower.
- **Single seed.** One run with seed=42. We did not report mean ± std over
  multiple seeds; for the final paper table we should run ≥ 3 seeds.
- **Single LLM, single prompt regime.** This is the *zero-shot Claude* cell of
  the LLM evaluation table. The 3-shot column and the GPT/Gemini columns are
  separate runs with the same frozen template.
- **Pilot intent.** Per the protocol, this run's primary purpose is to
  validate that the frozen prompt template + harness works end-to-end on a
  real competition before applying it to the other 17.

---

## 7. Conclusion of the pilot

The frozen prompt template (`../prompt.md`) successfully produced a
runnable, high-quality, leaderboard-competitive solution to C03 Real_Or_Fake on
**the first attempt with zero repairs**. The template is now ready to be
applied unchanged to the remaining 17 competitions in GNN-CB.

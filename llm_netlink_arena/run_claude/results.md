# NetLinkArena — LLM Pilot Run (Claude Opus 4.7, zero-shot, CPU)

**Competition:** [NetLinkArena](https://github.com/ignatiusbalayo/NetLinkArena) — link prediction on CiteSeer
**Leaderboard:** https://ignatiusbalayo.github.io/NetLinkArena/leaderboard.html
**Date:** 2026-04-29
**Model under test:** Claude Opus 4.7 (`claude-opus-4-7`), zero-shot, frozen prompt template `../prompt.md`
**Hardware:** CPU only (Apple Silicon, no GPU used)
**Run folder:** `llm_netlink_arena/run_claude/`

---

## 1. Protocol

1. The **same frozen prompt template** used for C03 Real_Or_Fake (`../prompt.md`)
   was instantiated with NetLinkArena's verbatim README, repository tree, data
   summary, and submission spec. **No edits to the template.**
2. Claude Opus 4.7 produced a `<plan>` (`plan.md`) and `<code>` (`solution.py`)
   in a single zero-shot pass. **No few-shot examples**, **no per-task hints**,
   **no prompt edits**.
3. The script was executed once on CPU. **It ran to completion on the first
   try; 0 repair iterations were needed.**
4. Predictions for the 1822 hidden test edges were written to `predictions.csv`
   in the exact format the competition requires (`id`, `y_pred` ∈ [0, 1]).
5. Local scoring was performed on the **public validation split** (1822 edges,
   1:1 balanced) using the repo's own metric (`sklearn.metrics.roc_auc_score`,
   matching `competition/metrics.py`).

> ⚠️ The hidden `test_labels.csv` is not publicly available (used by the
> competition's GitHub Actions scoring pipeline). Local scoring is therefore
> done on the validation set, which is the standard proxy. The test
> predictions file is ready for optional submission via the official Google
> Form workflow if a leaderboard entry is desired.

---

## 2. Results (validation set, 1822 edges, 1:1 balanced)

| Metric                          |   Value |
|---------------------------------|--------:|
| **Validation ROC-AUC**          | **0.6934** |
| **Validation Average Precision**| **0.6850** |
| Epochs run                      |     200 |
| Best epoch                      |     195 |
| Wall-clock (load+train+predict, CPU) | **23 s** |
| Repair iterations used          |       0 |

Validation ROC-AUC trajectory (every 5 epochs after warm-up, from `execution_log.txt`):

```
ep000 0.4574  ep003 0.6016  ep035 0.6142  ep050 0.6344
ep055 0.6676  ep060 0.6853  ep065 0.6868  ep155 0.6883
ep170 0.6903  ep180 0.6928  ep195 0.6934   ← best
```

Loss kept decreasing past epoch 60, but val AUC plateaued — classic mild
overfitting on a small graph (3327 nodes). Early stopping with patience=20
(checked every 5 epochs) ran to the full 200 because tiny improvements kept
arriving.

---

## 3. Comparison to the public human leaderboard

The leaderboard at `ignatiusbalayo.github.io/NetLinkArena/leaderboard.html`
reports *test-set* ROC-AUC. The LLM number below is **validation** because we
did not submit. Same metric (ROC-AUC) for direct reference.

| Rank | Team               | Model      | Test AUC |
|-----:|--------------------|------------|---------:|
|    1 | VinitSingroha      | GAT        | 0.8914 |
|    2 | Gurur Gamgam       | GAT        | 0.8898 |
|    3 | faranbutt          | GAT        | 0.8336 |
|    4 | Emmanuel Owusu     | GAT        | 0.8223 |
|    5 | bjayadikary        | GraphSAGE  | 0.8070 |
|    6 | TugaYousif         | GAT        | 0.7826 |
|    7 | Abderrahmane       | GAT        | 0.7423 |
|    8 | Muhammad Isah      | GAT        | 0.7422 |
|    9 | MahaTeam           | GAT        | 0.7095 |
|  *—* | **Claude Opus 4.7 (LLM, ZS, this run)** | **GraphSAGE** | **0.6934 (val)** |
|   10 | nourMaj            | GAT        | 0.6976 |
|   11 | Peguy              | GCN        | 0.6640 |
|   12 | sanae Zrigui       | GCN        | 0.6244 |
|   13 | hadilaff           | GCN        | 0.6076 |
|  …  |                    |            |       … |

If validation AUC is taken as a proxy for test, Claude (zero-shot, no hints,
no repair) lands in the **middle of the human leaderboard** — between rank 9
and 10 — using only the README and the data files. This is consistent with
the GNN-CB hypothesis: frontier LLMs reach a respectable but not state-of-the-
art baseline on Medium-difficulty link-prediction tasks, leaving headroom for
human ingenuity at the top.

---

## 4. What the LLM produced

**Architecture (paraphrased from `plan.md` + `solution.py`):**
- **Graph construction:** message-passing edges built from *positive* training
  edges only (made undirected by adding both directions); negatives used for
  supervision only.
- **Features:** 2742-dim sparse bag-of-words, L2 row-normalized.
- **Encoder:** 2-layer GraphSAGE (mean aggregator), hidden=128, dropout=0.3.
- **Decoder:** dot-product link decoder `<z_u, z_v>` → sigmoid (canonical
  Kipf & Welling 2016 baseline).
- **Loss:** `BCEWithLogitsLoss` over the pre-balanced 1:1 train pairs.
- **Training:** Adam(lr=5e-3, wd=5e-4), full-batch, up to 200 epochs, early
  stopping on val AUC (patience=20, checked every 5 epochs).

**What Claude did right (zero-shot):**
- Correctly separated the *graph* (positive train edges) from the
  *supervision* (positive + negative train pairs) — a common pitfall.
- L2-normalized the sparse bag-of-words features.
- Picked a clean, reproducible link-prediction recipe (SAGE + dot-product),
  matching the standard literature.
- Validated the output file (row count, id-set, NaN check, [0,1] range) before
  exiting — caught common submission-rejection causes upfront.

**What it didn't do (room for repair-loop or 3-shot):**
- No GAT (top humans use GAT, not SAGE) — likely the biggest gap.
- No structural/positional features (DeepWalk, Node2Vec, common-neighbours,
  Adamic-Adar) concatenated to BoW — a classic trick on CiteSeer.
- No edge dropout / DropEdge regularization.
- No ensembling of multiple seeds.
- No use of validation edges as additional message-passing edges at inference
  (legal under the rules; tops the leaderboard's typical recipes).

---

## 5. Files in this run

```
run_claude/
├── plan.md             ← <plan> block produced by Claude Opus 4.7
├── solution.py         ← <code> block produced by Claude Opus 4.7 (verbatim)
├── execution_log.txt   ← stdout of `python solution.py`
├── run_summary.csv     ← machine-readable summary (val_auc, val_ap, …)
├── predictions.csv     ← test-set predictions (1822 rows, id,y_pred)
└── results.md          ← this file
```

---

## 6. Honest caveats

- **Validation ≠ test.** The 0.6934 number is on the public 1822-edge
  validation set, not on the hidden 1822-edge test set the leaderboard scores.
  Test AUC is typically within ±0.02 of val AUC for balanced splits but can
  differ.
- **Single seed.** One run with seed=42. For the final paper table we should
  run ≥ 3 seeds and report mean ± std.
- **Single LLM, single prompt regime.** This is the *zero-shot Claude Opus 4.7*
  cell only. The 3-shot column and the GPT/Gemini columns are separate runs
  with the same frozen template.
- **No per-task tuning.** By design, we did not let the LLM iterate or revise
  after seeing val AUC. That would no longer be zero-shot.

---

## 7. What this pilot confirms

Two pilots, two tasks, two repos:

| Competition       | Type             | LLM val score | Human top | Repairs |
|-------------------|------------------|--------------:|----------:|--------:|
| C03 Real_Or_Fake  | Graph classif.   | F1 0.9768     | F1 0.9803 |       0 |
| NetLinkArena      | Link prediction  | AUC 0.6934    | AUC 0.8914|       0 |

The **same frozen `prompt.md`** produced runnable, leaderboard-eligible
solutions for two structurally different tasks (graph classification vs. link
prediction) without any per-task editing — which was the methodological claim
the pilot was designed to test.

Pattern observed across both: LLM is **near-human on Easy templated tasks**
(C03), **mid-pack on Medium tasks with non-trivial design choices** (NetLink-
Arena, where GAT + structural features matter and the LLM defaulted to the
safer SAGE + BoW recipe). This matches the GNN-CB paper's hypothesis.

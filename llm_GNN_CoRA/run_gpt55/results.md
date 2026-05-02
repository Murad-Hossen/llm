# GNN CoRA - LLM Run (GPT-5.5, zero-shot, CPU)

**Competition:** [GNN_CoRA](https://github.com/tasneem-mselim/GNN_CoRA)  
**Leaderboard:** [GNN CoRA Competition Leaderboard](https://tasneem-mselim.github.io/GNN_CoRA/final_leaderboard.html)  
**Date:** 2026-05-02  
**Model under test:** GPT-5.5, zero-shot, frozen prompt template `../prompt.md`  
**Hardware:** CPU only  
**Run folder:** `llm_GNN_CoRA/run_gpt55/`

---

## 1. Protocol

1. The frozen prompt template (`../prompt.md`) was instantiated with the
   verbatim README, repository tree, data summary, and submission spec for
   GNN_CoRA.
2. GPT-5.5 produced a `<plan>` (`plan.md`) and `<code>` (`solution.py`) in a
   single zero-shot pass. No few-shot examples, no per-task hints, no prompt
   edits.
3. The script was executed once on CPU. It ran to completion on the first try;
   0 repair iterations were needed.
4. Predictions for the 1,000 hidden test nodes were written to
   `run_gpt55/predictions.csv` and `run_gpt55/submission.csv` in the required
   `id,target` format.
5. Local scoring was performed on the public validation split (500 nodes) using
   accuracy, matching the official leaderboard metric.

> The hidden test labels are not public. Local scoring is therefore reported on
> the validation set; the test predictions are ready for optional encryption and
> submission through the official pull request workflow.

---

## 2. Results (validation set, 500 nodes)

| Metric | Value |
|---|---:|
| Validation accuracy | **73.20%** |
| Best single SGC candidate | 71.20% |
| Best label propagation candidate | 71.20% |
| Ensemble candidates | 13 |
| Wall-clock time (CPU) | 0.87 sec |
| Repair iterations used | 0 |

Per-candidate validation accuracy from `execution_log.txt`:

```text
SGC K=2: 65.8%, 66.6%, 65.4%
SGC K=3: 68.0%, 67.4%, 67.8%
SGC K=4: 71.0%, 71.2%, 70.8%
Label propagation: 70.4%, 70.4%, 70.8%, 71.2%
Ensemble: 73.2%
```

---

## 3. Comparison to Public Human Leaderboard

The public leaderboard reports hidden test-set accuracy. GPT-5.5 is shown with
validation-set accuracy because the hidden test labels are unavailable locally.

| Rank | Team | Type | Accuracy |
|---:|---|---|---:|
| 1 | TugaAhmed | human | 74.20% |
| 2 | gururgg | human | 70.90% |
| 3 | abdksm | human | 70.70% |
| 4 | Muhammad0isah | human | 69.70% |
| 5 | bjayadikary | human | 68.80% |
| - | GPT-5.5 (this run) | llm-only, validation | **73.20%** |

If validation accuracy is used only as a rough proxy, GPT-5.5 is competitive
with the top of the public leaderboard, below the current best public test score
and above the second-place public test score. This is not a final hidden-test
rank until the encrypted submission is evaluated by the official workflow.

---

## 4. What GPT-5.5 Produced

**Architecture and training summary:**
- **Features:** standardized noisy 1,433-dimensional node features from
  `data/x.csv`.
- **Graph operators:** symmetrized citation graph with self-loops; both
  symmetric GCN normalization and random-walk normalization.
- **Backbone:** Simplified Graph Convolution (SGC), equivalent to a linearized
  GCN using `A^K X` propagated features.
- **Classifiers:** one-vs-rest logistic regression over SGC embeddings for
  propagation depths `K={2,3,4}` and regularization `C={0.01,0.03,0.1}`.
- **Smoothing:** label propagation with alpha values `{0.90,0.95,0.98,0.99}`.
- **Decoding:** average 13 probability-like outputs and take `argmax` over the
  7 classes.

**What worked well:**
- Avoided the repository starter code bugs around undefined output paths.
- Avoided the broken local PyTorch/TensorFlow installations by using the
  packages available in the official GitHub Actions workflow: `numpy`,
  `pandas`, `scikit-learn`, and `scipy` via scikit-learn.
- Used graph propagation aggressively, which is important because the node
  features have Gaussian noise.
- The ensemble improved validation accuracy from 71.2% to 73.2%.

**Remaining room:**
- No neural GCN/GAT training was executed locally because the installed
  PyTorch/TensorFlow packages are incomplete in this environment.
- No multi-seed neural ensemble or official hidden-test submission has been run.

---

## 5. Files in This Run

```text
run_gpt55/
├── plan.md              # <plan> block produced by GPT-5.5
├── solution.py          # <code> block produced by GPT-5.5
├── execution_log.txt    # stdout of python solution.py
├── run_summary.csv      # machine-readable summary
├── candidate_scores.csv # per-candidate validation accuracies
├── predictions.csv      # test-set predictions, id,target
├── submission.csv       # same predictions under official CSV name
└── results.md           # this file
```

---

## 6. Caveats

- Validation accuracy is not hidden test accuracy.
- This is a single zero-shot GPT-5.5 run with no repair iterations.
- The official leaderboard score requires encrypting `submission.csv` and
  submitting through the competition pull request workflow.

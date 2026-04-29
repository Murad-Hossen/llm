# Plan (Claude, zero-shot, CPU)

This is the `<plan>` block produced by Claude in response to the frozen prompt
template instantiated for C03 Real_Or_Fake.

1. **Features.** Concatenate BERT (768-dim) + spaCy (300-dim) + profile (10-dim)
   node features → 1078-dim per node. The README explicitly invites combining
   features and the baseline only uses spaCy, so adding BERT and profile is the
   most direct lift.
2. **Profile normalization.** Profile features include raw counts (followers,
   friends, statuses) with heavy tails. Apply `log1p` then standard scaling
   (fit on training nodes only).
3. **Architecture.** 2-layer GraphSAGE (mean aggregator) → global mean+max
   pool concatenated → 2-layer MLP head with dropout 0.3. Hidden=128. SAGE is
   chosen over GAT for CPU speed and because top humans on the leaderboard use
   SAGEConv.
4. **Root-node pathway.** Per the baseline, also pass the root node's raw
   features through a small MLP and concatenate to the pooled graph
   representation before the classifier head. This preserves the
   "news article ≠ retweeting users" inductive bias.
5. **Training.** Adam, lr=1e-3, weight_decay=1e-4, batch_size=64,
   BCEWithLogitsLoss. Up to 30 epochs with early stopping on val F1
   (patience=5).
6. **Threshold tuning.** After training, sweep thresholds on the validation set
   in [0.30, 0.70] step 0.01; pick the one maximising val F1. Apply that
   threshold to test probabilities.
7. **Submission.** Write `submissions/sample_submission/predictions.csv` with
   columns `id, y_pred` (binary 0/1), exactly as `test.py` does and as
   `validate_submission.py` requires.
8. **Reproducibility.** Set seeds for numpy, torch, random. Run on CPU only.
9. **Time budget.** Pre-build all PyG `Data` objects once before training to
   avoid per-epoch graph construction. Target ≤30 min on CPU for 4372 graphs.
10. **Self-check.** After writing predictions, immediately re-load the file and
    assert the row count and ID set match `test_idx.csv`.

# Plan (Claude Opus 4.7, zero-shot, CPU)

`<plan>` block produced under the frozen prompt template (`../prompt.md`)
instantiated for **NetLinkArena** (link prediction on CiteSeer, AUC-ROC).

1. **Task framing.** Binary link prediction: given pairs `(source, target)`,
   predict probability that an edge exists. Score is ROC-AUC, so the model only
   needs to *rank* well — calibration is irrelevant.
2. **Graph construction.** Build the message-passing graph from the *positive*
   edges in `train_edges.csv` only (label==1, undirected → add both directions).
   Negative training edges are used only as supervision, not as graph edges.
3. **Node features.** Load `node_features.csv` as a 3327×2742 sparse matrix.
   Apply L2 row-normalization (standard for bag-of-words) to stabilize training.
4. **Encoder.** 2-layer GraphSAGE (mean aggregator), hidden=128, dropout=0.3.
   GraphSAGE chosen because the leaderboard's strongest entries are GAT/SAGE
   and SAGE is more CPU-friendly. (We will report GAT in a separate run.)
5. **Decoder.** Standard dot-product link decoder:
   `score(u, v) = sigmoid(<z_u, z_v>)`. This is the canonical baseline for
   link prediction (Kipf & Welling 2016 "VGAE").
6. **Loss & negative sampling.** `BCEWithLogitsLoss` over both positive and
   negative training pairs from `train_edges.csv` (already balanced 1:1, no
   resampling needed).
7. **Training.** Adam lr=5e-3, weight_decay=5e-4, full-batch (small graph),
   up to 200 epochs, early stopping on **val ROC-AUC** with patience=20.
8. **Validation.** Use `val_edges.csv` (also 1:1 balanced) for both early
   stopping and final reported metric. The hidden `test_labels.csv` is not
   accessible locally (per repo design), so validation AUC is our proxy.
9. **Submission.** Write `predictions.csv` with columns `id, y_pred` (float
   probability), exactly matching `sample_submission.csv`. Validate row count
   and id-set against `test_nodes.csv` immediately after writing.
10. **Reproducibility.** Seeds for numpy/torch/random; CPU-only;

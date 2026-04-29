# Plan (Claude Opus 4.7, zero-shot, CPU)

`<plan>` produced under the frozen prompt template (`../prompt.md`)
instantiated for **CGCC** — graph-level classification of city street networks
into {organic=0, grid=1, hybrid=2}, scored by Macro-F1.

1. **Task framing.** 84 labelled training graphs, 36 unlabelled test graphs,
   3 classes (31/37/52 in the full pool). Tiny dataset → strong regularization
   and ensembling matter more than model capacity.
2. **Node features (per intersection).** `x_centered_scaled`,
   `y_centered_scaled`, `street_count`, normalized `degree`, local
   `clustering_coefficient`. → 5-dim per node.
3. **Graph-level handcrafted features (concatenated to GNN embedding).**
   `n_nodes`, `n_edges`, `density`, `mean_degree`, `degree_var`,
   `mean_clustering`, `transitivity`, `mean_edge_length`,
   `edge_length_cv`, and **orientation entropy** of edge bearings (canonical
   proxy for grid-vs-organic in urban morphology, Boeing 2019). → 10 stats.
4. **Architecture.** 2-layer GIN (best for graph-level distinguishability,
   Xu et al. 2019), hidden=64, dropout=0.4. Mean+max pool → concat with the
   10 handcrafted stats → 2-layer MLP head → 3-class logits.
5. **Loss.** Cross-entropy with class weights (organic is the rarest class).
6. **Training protocol.** Stratified 5-fold CV. For each fold: Adam
   lr=5e-3, wd=1e-3, up to 200 epochs, early stopping on fold val Macro-F1
   (patience 30). Save fold-best states.
7. **Inference.** Average the *softmax probabilities* of the 5 fold-best
   models on each test graph; predict argmax (soft ensemble).
8. **Validation reporting.** For local scoring (no `test_labels_hidden.csv`),
   report mean ± std Macro-F1 across the 5 CV folds, and an OOF (out-of-fold)
   prediction Macro-F1 over the full 84-graph train set as a single
   summary number — this matches the leaderboard's reported metric.
9. **Submission file.** Write `submission.csv` at the repo root with columns
   `filename, prediction` (int 0/1/2). Self-check: filenames match
   `data/test/*.pkl` exactly.
10. **Time + reproducibility.** Seeds for numpy/torch/random; CPU only;
    target ≤ 5 min for the full 5-fold ensemble.

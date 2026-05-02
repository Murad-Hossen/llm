# Plan (GPT-5.5, zero-shot, CPU)

This is the `<plan>` block produced by GPT-5.5 in response to the frozen prompt
template instantiated for the GNN CoRA node classification challenge.

1. **Features.** Use the provided noisy 1,433-dimensional node features from
   `data/x.csv`. Standardize each feature column before graph propagation so
   the Gaussian noise and sparse bag-of-words dimensions are on comparable
   scales.
2. **Graph construction.** Read `data/edge_index.csv`, symmetrize the citation
   edges, add self-loops, and build both symmetric-normalized adjacency
   `D^-1/2 A D^-1/2` and random-walk adjacency `D^-1 A`.
3. **GNN architecture.** Use Simplified Graph Convolution (SGC): repeatedly
   propagate node features with the normalized graph operator, then train a
   linear one-vs-rest classifier on the propagated node embeddings. SGC is the
   linearized form of a GCN and is well suited to tiny-label transductive Cora.
4. **Training protocol.** Train logistic classifiers only on the 140 labeled
   training nodes. Sweep propagation depths `K in {2,3,4}` and regularization
   strengths `C in {0.01,0.03,0.1}`.
5. **Validation strategy.** Use the 500 public validation labels only for model
   selection and reporting. Record per-candidate validation accuracy and keep
   an ensemble rather than selecting a single brittle model.
6. **Graph post-processing.** Add label-propagation predictors with several
   damping factors. This uses the citation graph and training labels to smooth
   class evidence over neighboring papers.
7. **Decoding.** Average probability-like outputs from the SGC classifiers and
   label-propagation runs, then take `argmax` over the seven classes.
8. **Submission.** Write `run_gpt55/predictions.csv` with exactly the required
   columns `id,target` for the 1,000 node IDs in `data/test_ID.csv`.
9. **Reproducibility.** Set Python and NumPy seeds; the chosen scikit-learn
   solver is deterministic for this setup.

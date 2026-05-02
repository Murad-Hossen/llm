import csv
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax


SEED = 55
N_CLASSES = 7


def resolve_paths():
    run_dir = Path(__file__).resolve().parent
    repo_root = run_dir.parent / "GNN_CoRA"
    if not (repo_root / "data").exists():
        repo_root = Path.cwd()
    data_dir = repo_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError("Could not locate the GNN_CoRA data directory")
    return run_dir, repo_root, data_dir


def build_graph(edge_df, n_nodes):
    src = edge_df["source"].to_numpy(dtype=np.int64)
    dst = edge_df["target"].to_numpy(dtype=np.int64)
    rows = np.concatenate([src, dst, np.arange(n_nodes)])
    cols = np.concatenate([dst, src, np.arange(n_nodes)])
    vals = np.ones(rows.shape[0], dtype=np.float32)
    adj = sparse.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    deg = np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)
    deg = np.maximum(deg, 1.0)
    inv_sqrt_deg = (1.0 / np.sqrt(deg)).astype(np.float32)
    inv_deg = (1.0 / deg).astype(np.float32)

    sym_norm = sparse.diags(inv_sqrt_deg) @ adj @ sparse.diags(inv_sqrt_deg)
    row_norm = sparse.diags(inv_deg) @ adj
    return sym_norm.tocsr(), row_norm.tocsr()


def sgc_candidates(x, sym_norm, train_idx, y_train, val_idx, y_val):
    """Simplified graph convolution: X, AX, A^2X, ... plus linear classifiers."""
    propagated = StandardScaler().fit_transform(x).astype(np.float32)
    probabilities = []
    rows = []

    for depth in range(5):
        if depth in (2, 3, 4):
            for c_value in (0.01, 0.03, 0.1):
                clf = LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    solver="liblinear",
                    multi_class="ovr",
                    class_weight="balanced",
                    random_state=SEED,
                )
                clf.fit(propagated[train_idx], y_train)
                scores = clf.decision_function(propagated)
                probs = softmax(scores)
                pred = probs[val_idx].argmax(axis=1)
                acc = accuracy_score(y_val, pred)
                probabilities.append(probs)
                rows.append(("sgc", depth, c_value, "", acc))
                print(f"  sgc K={depth} C={c_value:.2f} val_acc={acc:.4f}", flush=True)

        propagated = sym_norm.dot(propagated).astype(np.float32)

    return probabilities, rows


def label_propagation_candidates(row_norm, train_idx, y_train, val_idx, y_val):
    y0 = np.zeros((row_norm.shape[0], N_CLASSES), dtype=np.float32)
    y0[train_idx, y_train] = 1.0
    probabilities = []
    rows = []

    for alpha in (0.90, 0.95, 0.98, 0.99):
        y_score = y0.copy()
        for _ in range(200):
            y_score = alpha * row_norm.dot(y_score) + (1.0 - alpha) * y0
            y_score[train_idx] = y0[train_idx]

        probs = y_score / (y_score.sum(axis=1, keepdims=True) + 1e-9)
        pred = probs[val_idx].argmax(axis=1)
        acc = accuracy_score(y_val, pred)
        probabilities.append(probs)
        rows.append(("label_propagation", "", "", alpha, acc))
        print(f"  label_prop alpha={alpha:.2f} val_acc={acc:.4f}", flush=True)

    return probabilities, rows


def write_csv(path, rows, header):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    random.seed(SEED)
    np.random.seed(SEED)
    started = time.time()

    run_dir, repo_root, data_dir = resolve_paths()
    print(f"[load] repo_root={repo_root}", flush=True)

    x = pd.read_csv(data_dir / "x.csv").to_numpy(dtype=np.float32)
    edge_df = pd.read_csv(data_dir / "edge_index.csv")
    train_df = pd.read_csv(data_dir / "y_train.csv")
    val_df = pd.read_csv(data_dir / "y_val.csv")
    test_ids = pd.read_csv(data_dir / "test_ID.csv")["id"].to_numpy(dtype=np.int64)

    train_idx = train_df["index"].to_numpy(dtype=np.int64)
    y_train = train_df["label"].to_numpy(dtype=np.int64)
    val_idx = val_df["index"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)

    print(
        f"[data] X={x.shape} edges={len(edge_df)} train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_ids)}",
        flush=True,
    )

    sym_norm, row_norm = build_graph(edge_df, x.shape[0])

    print("[train] SGC candidates", flush=True)
    sgc_probs, candidate_rows = sgc_candidates(
        x, sym_norm, train_idx, y_train, val_idx, y_val
    )

    print("[smooth] label propagation candidates", flush=True)
    lp_probs, lp_rows = label_propagation_candidates(
        row_norm, train_idx, y_train, val_idx, y_val
    )
    candidate_rows.extend(lp_rows)

    ensemble_probs = np.mean(sgc_probs + lp_probs, axis=0)
    val_pred = ensemble_probs[val_idx].argmax(axis=1).astype(np.int64)
    val_acc = accuracy_score(y_val, val_pred)
    test_pred = ensemble_probs[test_ids].argmax(axis=1).astype(np.int64)

    predictions = pd.DataFrame({"id": test_ids, "target": test_pred})
    pred_path = run_dir / "predictions.csv"
    sub_path = run_dir / "submission.csv"
    predictions.to_csv(pred_path, index=False)
    predictions.to_csv(sub_path, index=False)

    # Self-check the official submission shape and value domain.
    reloaded = pd.read_csv(pred_path)
    assert list(reloaded.columns) == ["id", "target"]
    assert len(reloaded) == len(test_ids)
    assert set(reloaded["id"].to_numpy()) == set(test_ids)
    assert reloaded["target"].between(0, N_CLASSES - 1).all()

    elapsed = time.time() - started
    print(f"[ensemble] val_acc={val_acc:.4f}", flush=True)
    print(f"[done] wrote {pred_path} ({len(predictions)} rows)", flush=True)
    print(f"[time] wall_clock_sec={elapsed:.2f}", flush=True)

    summary_path = run_dir / "run_summary.csv"
    pd.DataFrame(
        [
            {
                "val_acc": val_acc,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_ids),
                "n_candidates": len(sgc_probs) + len(lp_probs),
                "wall_clock_sec": elapsed,
                "repair_iterations": 0,
            }
        ]
    ).to_csv(summary_path, index=False)

    candidates_path = run_dir / "candidate_scores.csv"
    write_csv(
        candidates_path,
        candidate_rows,
        ["model", "propagation_depth", "c_value", "alpha", "val_acc"],
    )


if __name__ == "__main__":
    main()

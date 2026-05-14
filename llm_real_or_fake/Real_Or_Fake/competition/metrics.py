"""
metrics.py

Shared metrics for leaderboard scoring.
"""

from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(y_true, y_pred):
    return {
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),

    }

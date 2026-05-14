import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_metrics(y_true, y_pred):
    """
    Calculates the evaluation metrics for the NetLinkArena challenge.
    Returns a dictionary of scores.
    """
    # Ensure inputs are 1D arrays
    y_true = y_true.values.flatten()
    y_pred = y_pred.values.flatten()
    
    # Safety check: ensure no NaNs
    if pd.isna(y_pred).any():
        raise ValueError(
            "Predictions contain NaN values. "
            "Please check your submission."
        )
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
        ap_score = average_precision_score(y_true, y_pred)
        
        return {
            "roc_auc": round(float(roc_auc), 4),
            "average_precision": round(float(ap_score), 4)
        }
    except ValueError as e:
        # Handles cases where y_true might only have one class
        raise ValueError(f"Metric calculation failed: {e}")

if __name__ == "__main__":
    # Quick local test if someone runs metrics.py directly
    import numpy as np
    mock_true = pd.Series([1, 0, 1, 0])
    mock_pred = pd.Series([0.9, 0.1, 0.8, 0.3])
    print("Testing metrics locally:", calculate_metrics(mock_true, mock_pred))
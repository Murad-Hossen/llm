import pandas as pd
import sys
import os

# Ensure the script can find metrics.py regardless of where GitHub Actions runs it from
sys.path.append(os.path.dirname(__file__))
from metrics import calculate_metrics

def main(pred_path, label_path):
    try:
        preds = pd.read_csv(pred_path)
        labels = pd.read_csv(label_path)
    except Exception as e:
        # Send errors to stderr so they don't end up in the SCORE variable
        print(f"Error reading CSV files: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate submission format
    if "id" not in preds.columns or "y_pred" not in preds.columns:
        print("Submission must contain 'id' and 'y_pred' columns.", file=sys.stderr)
        sys.exit(1)
    
    # Sort and merge on 'id' column
    preds = preds.sort_values("id")
    labels = labels.sort_values("id")
    merged = labels.merge(preds, on="id", how="inner")
    
    if len(merged) != len(labels):
        print(
            f"ID mismatch! Expected {len(labels)} pairs, "
            f"but got {len(merged)} matching pairs in submission.",
            file=sys.stderr
        )
        sys.exit(1)
    
    # Calculate metrics securely
    try:
        scores = calculate_metrics(merged["label"], merged["y_pred"])
    except Exception as e:
        print(f"Error calculating metrics: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output ONLY the AUC-ROC score to standard output
    # GitHub Actions expects this to be the ONLY string returned!
    print(f"{scores['roc_auc']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <pred_path> <label_path>", file=sys.stderr)
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])

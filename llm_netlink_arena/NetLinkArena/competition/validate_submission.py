import pandas as pd
import sys
import os

def main(pred_path):
    """
    Validate submission format.
    Now loads test_nodes.csv automatically from data/public/
    """
    try:
        preds = pd.read_csv(pred_path)
    except Exception as e:
        raise ValueError(f"Could not read prediction file: {e}")
    
    # Auto-detect test_nodes.csv location
    possible_paths = [
        'data/public/test_nodes.csv',
        'NetLinkArena_Dataset/data/public/test_nodes.csv',
        '../data/public/test_nodes.csv'
    ]
    
    test_df = None
    for path in possible_paths:
        if os.path.exists(path):
            test_df = pd.read_csv(path)
            break
    
    if test_df is None:
        raise ValueError(
            "Could not find test_nodes.csv. "
            "Expected in data/public/test_nodes.csv"
        )
    
    # 1. Column Check
    if "id" not in preds.columns or "y_pred" not in preds.columns:
        raise ValueError("Submission must contain 'id' and 'y_pred' columns.")
    
    # 2. Type Match for IDs (Prevents '1' != 1 mismatch errors)
    preds["id"] = preds["id"].astype(str)
    test_df["id"] = test_df["id"].astype(str)
    
    # 3. Numeric Check for Predictions (Crucial fix!)
    if not pd.api.types.is_numeric_dtype(preds["y_pred"]):
        # Attempt to convert to numeric, coercing un-parsable text to NaN
        preds["y_pred"] = pd.to_numeric(preds["y_pred"], errors='coerce')
        if preds["y_pred"].isna().any():
            raise ValueError("'y_pred' column must contain numeric probabilities, not text.")

    # 4. Duplicates Check
    if preds["id"].duplicated().any():
        raise ValueError(
            "Duplicate IDs found in submission. "
            "Each pair can only be predicted once."
        )
    
    # 5. Missing Values Check
    if preds["y_pred"].isna().any():
        raise ValueError(
            "NaN or missing predictions found. "
            "Every ID needs a valid probability."
        )
    
    # 6. Bounds Check (Probabilities must be 0 to 1)
    if ((preds["y_pred"] < 0) | (preds["y_pred"] > 1)).any():
        raise ValueError(
            "Predictions out of bounds. "
            "All y_pred values must be between 0 and 1."
        )
    
    # 7. ID Match Check
    pred_ids = set(preds["id"])
    test_ids = set(test_df["id"])
    
    if pred_ids != test_ids:
        missing = len(test_ids - pred_ids)
        extra = len(pred_ids - test_ids)
        raise ValueError(
            f"ID mismatch! "
            f"Missing {missing} required IDs, "
            f"found {extra} unexpected IDs."
        )
    
    print("✅ VALID SUBMISSION")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_submission.py <pred_path>", file=sys.stderr)
        sys.exit(1)
    
    try:
        main(sys.argv[1])
    # Catch ANY exception here to prevent raw stack traces in the GH Actions log
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
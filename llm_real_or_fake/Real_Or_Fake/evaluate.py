import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Load CSVs
labels = pd.read_csv("data/public/test_labels.csv")      # columns: id, y_true  . hidden file
preds  = pd.read_csv("submissions/sample_submission/predictions.csv")      # columns: id, y_pred

# Inner join ensures we only evaluate matching graph IDs
df = labels.merge(preds, on="id", how="inner")

# Safety checks
assert len(df) > 0, "No overlapping graph IDs — cannot compute metrics"
assert len(df) == len(preds), "Some predictions have no ground truth!"

y_true = df["y_true"].astype(int)
y_pred = df["y_pred"].astype(int)

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)          # default: binary F1 (positive class = 1)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 score: {f1:.4f}")

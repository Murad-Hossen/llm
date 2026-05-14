#!/bin/bash
echo "=== SECTION 1: README ==="
cat README.md
echo ""
echo "=== SECTION 2: REPO TREE ==="
find . -type f -not -path "./.git/*" -not -path "./submissions/*"
echo ""
echo "=== SECTION 3: DATA SAMPLE ==="
python3 -c "
import pandas as pd

print('node_features.csv:')
df = pd.read_csv('data/public/node_features.csv')
print('  shape:', df.shape)
print('  dtypes: node_id=int64, features=float64')
print('  head(3):')
print(df.iloc[:3, :6].to_string())
print()

print('train_edges.csv:')
df = pd.read_csv('data/public/train_edges.csv')
print('  shape:', df.shape)
print('  head(5):')
print(df.head(5).to_string())
print()

print('val_edges.csv:')
df = pd.read_csv('data/public/val_edges.csv')
print('  shape:', df.shape)
print('  head(5):')
print(df.head(5).to_string())
print()

print('test_nodes.csv:')
df = pd.read_csv('data/public/test_nodes.csv')
print('  shape:', df.shape)
print('  head(5):')
print(df.head(5).to_string())
print()

print('sample_submission.csv:')
df = pd.read_csv('data/public/sample_submission.csv')
print('  shape:', df.shape)
print('  head(5):')
print(df.head(5).to_string())
"
echo ""
echo "=== SECTION 4: SUBMISSION FORMAT ==="
grep -A 30 "How to Submit" README.md | head -35
echo ""
echo "=== SECTION 5: REQUIREMENTS ==="
cat requirements.txt

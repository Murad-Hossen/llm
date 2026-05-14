#!/bin/bash
echo "=== SECTION 1: README ==="
cat README.md
echo ""
echo "=== SECTION 2: REPO TREE ==="
find . -type f -not -path "./.git/*" -not -path "./submissions/*"
echo ""
echo "=== SECTION 3: DATA SAMPLE ==="
python3 -c "
import numpy as np
import pandas as pd

print('--- train_labels.csv ---')
df = pd.read_csv('data/public/train_labels.csv')
print('shape:', df.shape)
print(df.head(5).to_string())
print()

print('--- val_labels.csv ---')
df = pd.read_csv('data/public/val_labels.csv')
print('shape:', df.shape)
print(df.head(5).to_string())
print()

print('--- test_idx.csv ---')
df = pd.read_csv('data/public/test_idx.csv')
print('shape:', df.shape)
print(df.head(5).to_string())
print()

print('--- train_idx.npy ---')
arr = np.load('data/public/train_idx.npy')
print('shape:', arr.shape, 'dtype:', arr.dtype)
print('first 5:', arr[:5])
print()

print('--- val_idx.npy ---')
arr = np.load('data/public/val_idx.npy')
print('shape:', arr.shape, 'dtype:', arr.dtype)
print('first 5:', arr[:5])
print()

print('--- test_idx.npy ---')
arr = np.load('data/public/test_idx.npy')
print('shape:', arr.shape, 'dtype:', arr.dtype)
print('first 5:', arr[:5])
print()

print('--- node_graph_id.npy ---')
arr = np.load('data/public/node_graph_id.npy')
print('shape:', arr.shape, 'dtype:', arr.dtype)
print('unique graphs:', len(set(arr.tolist())))
print()

print('--- A.txt (edges) ---')
with open('data/public/A.txt') as f:
    lines = f.readlines()
print('total edges:', len(lines))
print('first 5 rows:', lines[:5])
print()

print('--- new_bert_feature.npz ---')
data = np.load('data/public/new_bert_feature.npz')
print('keys:', list(data.keys()))
for k in data.keys():
    print(f'  {k}: shape={data[k].shape}, dtype={data[k].dtype}')
print()

print('--- new_spacy_feature.npz ---')
data = np.load('data/public/new_spacy_feature.npz')
print('keys:', list(data.keys()))
for k in data.keys():
    print(f'  {k}: shape={data[k].shape}, dtype={data[k].dtype}')
print()

print('--- new_profile_feature.npz ---')
data = np.load('data/public/new_profile_feature.npz')
print('keys:', list(data.keys()))
for k in data.keys():
    print(f'  {k}: shape={data[k].shape}, dtype={data[k].dtype}')
"
echo ""
echo "=== SECTION 4: SUBMISSION FORMAT ==="
grep -A 20 "Submission Workflow" README.md | head -25
echo ""
echo "=== SECTION 5: REQUIREMENTS ==="
cat requirements.txt

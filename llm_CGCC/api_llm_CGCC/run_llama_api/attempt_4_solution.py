import pandas as pd
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from torch_geometric.loader import DataLoader as PyGDataLoader

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load training data and labels
train_labels = pd.read_csv('data/train_labels.csv')
train_graphs = []
for filename in train_labels['filename']:
    with open(f'data/train/{filename}', 'rb') as f:
        G = pickle.load(f)
    train_graphs.append(G)

# Preprocess graph data
class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        G = self.graphs[idx]
        x = np.array([[G.nodes[node]['x'], G.nodes[node]['y'], G.degree(node) / G.number_of_nodes()] for node in G.nodes])
        edge_index = torch.tensor([(u, v) for u, v, *_ in G.edges], dtype=torch.long).t().contiguous()
        batch = torch.zeros(x.shape[0], dtype=torch.long)
        y = torch.tensor(self.labels[idx])
        return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, batch=batch, y=y)

# Create dataset and data loader
dataset = GraphDataset(train_graphs, train_labels['target'])
train_dataset, val_dataset = train_test_split(dataset, test_size=0.3, stratify=[label for label in dataset.labels], random_state=42)
train_loader = PyGDataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = PyGDataLoader(val_dataset, batch_size=1, shuffle=False)

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(16, 3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Train model
device = torch.device('cpu')
model = GCN().to(device)
optimizer = Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y.unsqueeze(0))
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = F.nll_loss(out, batch.y.unsqueeze(0))
            val_loss += loss.item()
            _, pred = torch.max(out, dim=1)
            correct += int((pred == batch.y).sum())
    accuracy = correct / len(val_dataset)
    macro_f1 = f1_score([label for label in val_dataset.labels], [int(torch.argmax(model(batch).detach())) for batch in val_loader], average='macro')
    print(f'Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader)}, Val Acc: {accuracy:.4f}, Val Macro F1: {macro_f1:.4f}')

# Make predictions on test data
test_graphs = []
for filename in os.listdir('data/test'):
    with open(f'data/test/{filename}', 'rb') as f:
        G = pickle.load(f)
    test_graphs.append(G)

test_dataset = GraphDataset(test_graphs, [0] * len(test_graphs))
test_loader = PyGDataLoader(test_dataset, batch_size=1, shuffle=False)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        _, pred = torch.max(out, dim=1)
        predictions.extend(pred.cpu().numpy())

# Write predictions to submission.csv
submission_df = pd.DataFrame({'filename': [filename for filename in os.listdir('data/test')], 'prediction': predictions})
submission_df.to_csv('submission.csv', index=False)
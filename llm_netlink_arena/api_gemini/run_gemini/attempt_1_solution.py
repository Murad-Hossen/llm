import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Module, Dropout
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score
import time

# --- 1. Model Definition ---
class GATLinkPredictor(Module):
    """
    A Graph Attention Network (GATv2) based model for link prediction.
    It uses two GATv2 layers to encode node features into embeddings,
    and a dot product decoder to predict the existence of an edge.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # The final layer averages the heads' outputs, so concat=False
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = Dropout(p=dropout)

    def encode(self, x, edge_index):
        """
        Generates node embeddings using the GAT model.
        """
        # Apply first GAT layer with ELU activation and dropout
        x = self.dropout(F.elu(self.conv1(x, edge_index)))
        # Apply second GAT layer
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """
        Predicts scores for edges using the dot product of node embeddings.
        """
        src_embeds = z[edge_label_index[0]]
        dst_embeds = z[edge_label_index[1]]
        return (src_embeds * dst_embeds).sum(dim=-1)

# --- 2. Training and Evaluation Functions ---
def train(model, optimizer, criterion, x, train_pos_edge_index, train_edges, train_labels):
    model.train()
    optimizer.zero_grad()
    
    # Use only positive edges for message passing to learn graph structure
    z = model.encode(x, train_pos_edge_index)
    
    # Use all training edges (positive and negative) for loss calculation
    out = model.decode(z, train_edges)
    
    loss = criterion(out, train_labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, x, train_pos_edge_index, edges, labels):
    model.eval()
    
    # Get embeddings using the training graph structure
    z = model.encode(x, train_pos_edge_index)
    
    # Predict on the given edge set
    out = model.decode(z, edges)
    preds = torch.sigmoid(out).cpu().numpy()
    
    return roc_auc_score(labels.cpu().numpy(), preds)

# --- 3. Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Hyperparameters chosen for good performance
    HIDDEN_CHANNELS = 32
    OUT_CHANNELS = 64
    HEADS = 8
    DROPOUT = 0.6
    LEARNING_RATE = 0.005
    EPOCHS = 300
    PATIENCE = 30 # For early stopping

    # --- Load Data ---
    DATA_DIR = 'data/public'
    print(f"Loading data from {DATA_DIR}...")
    node_features_df = pd.read_csv(os.path.join(DATA_DIR, 'node_features.csv'))
    train_edges_df = pd.read_csv(os.path.join(DATA_DIR, 'train_edges.csv'))
    val_edges_df = pd.read_csv(os.path.join(DATA_DIR, 'val_edges.csv'))
    test_edges_df = pd.read_csv(os.path.join(DATA_DIR, 'test_nodes.csv'))

    # --- Prepare Data Tensors ---
    print("Preparing data tensors...")
    x = torch.tensor(node_features_df.drop('node_id', axis=1).values, dtype=torch.float).to(DEVICE)
    num_features = x.shape[1]

    # Positive training edges for message passing (made undirected)
    train_pos_edges_df = train_edges_df[train_edges_df['label'] == 1]
    train_pos_edge_index = torch.tensor(train_pos_edges_df[['source', 'target']].values.T, dtype=torch.long)
    train_pos_edge_index = to_undirected(train_pos_edge_index).to(DEVICE)

    # All training edges and labels for loss calculation
    train_edge_label_index = torch.tensor(train_edges_df[['source', 'target']].values.T, dtype=torch.long).to(DEVICE)
    train_labels = torch.tensor(train_edges_df['label'].values, dtype=torch.float).to(DEVICE)

    # Validation edges and labels for evaluation
    val_edge_label_index = torch.tensor(val_edges_df[['source', 'target']].values.T, dtype=torch.long).to(DEVICE)
    val_labels = torch.tensor(val_edges_df['label'].values, dtype=torch.float).to(DEVICE)

    # Test edges for final prediction
    test_edge_index = torch.tensor(test_edges_df[['source', 'target']].values.T, dtype=torch.long).to(DEVICE)

    # --- Model Initialization ---
    model = GATLinkPredictor(num_features, HIDDEN_CHANNELS, OUT_CHANNELS, HEADS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Training Loop with Early Stopping ---
    print("Starting training...")
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, optimizer, criterion, x, train_pos_edge_index, train_edge_label_index, train_labels)
        val_auc = evaluate(model, x, train_pos_edge_index, val_edge_label_index, val_labels)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"Epoch {epoch:03d}: Loss: {loss:.4f}, Val AUC: {val_auc:.4f} (New best!)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:03d}: Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best Val AUC: {best_val_auc:.4f}")
            break
    
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    # --- Final Prediction ---
    print("Generating final predictions on the test set...")
    # Load the best model
    model.load_state_dict(best_model_state)
    model.eval()

    # For final predictions, use all known positive edges (train + val) for message passing
    val_pos_edges_df = val_edges_df[val_edges_df['label'] == 1]
    all_pos_edges_df = pd.concat([train_pos_edges_df, val_pos_edges_df])
    all_pos_edge_index = torch.tensor(all_pos_edges_df[['source', 'target']].values.T, dtype=torch.long)
    all_pos_edge_index = to_undirected(all_pos_edge_index).to(DEVICE)

    with torch.no_grad():
        # Generate final embeddings using the most complete graph structure available
        z = model.encode(x, all_pos_edge_index)
        out = model.decode(z, test_edge_index)
        y_pred = torch.sigmoid(out).cpu().numpy()

    # --- Save Submission File ---
    submission_df = pd.DataFrame({'id': test_edges_df['id'], 'y_pred': y_pred})
    submission_df.to_csv('predictions.csv', index=False)
    print("Submission file 'predictions.csv' created successfully!")
    print(f"Top 5 predictions:\n{submission_df.head()}")
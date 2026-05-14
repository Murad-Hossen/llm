import torch
from dataloader import GraphDataset
from models.model import GNN
from sklearn.metrics import accuracy_score, f1_score

epochs_no = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = GraphDataset(split='train').get_loader() 
val_loader   = GraphDataset(split='val').get_loader(shuffle=False)

# Model
sample_data = next(iter(train_loader)) 
model = GNN(sample_data.num_features).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
loss_fnc = torch.nn.BCELoss()

def metrics(preds, gts):
    preds = torch.round(torch.cat(preds)) 
    gts = torch.cat(gts) 
    return accuracy_score(preds, gts), f1_score(preds, gts)

def train_epoch():
    model.train()
    total_loss = 0
    for data in train_loader: 
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fnc(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def validate():
    model.eval()
    all_preds, all_labels = [], []
    for data in val_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        all_preds.append(out.view(-1).cpu())
        all_labels.append(data.y.cpu())
    return metrics(all_preds, all_labels)


# Training loop
best_val_acc = 0.0

for epoch in range(epochs_no):
    train_loss = train_epoch()
    val_acc, val_f1 = validate()
    print(f"Epoch {epoch:02d} | TrainLoss: {train_loss:.4f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f}")
   
    # Save if this is the best validation accuracy so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/saved_model.model")

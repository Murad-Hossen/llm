import torch
import pandas as pd
from dataloader import GraphDataset
from models.model import GNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
test_loader = GraphDataset(split='test').get_loader(shuffle=False)
sample_data = next(iter(test_loader))

# Load the saved model
model = GNN(sample_data.num_features).to(device)
model.load_state_dict(torch.load("models/saved_model.model", map_location=device))
model.eval()

all_preds = []
all_ids = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        all_preds.extend(torch.round(out.view(-1)).cpu().numpy())
        all_ids.extend(data.graph_id.cpu().numpy())


# Save submission
submission = pd.DataFrame({"id": all_ids, "y_pred": all_preds})
submission.to_csv("submissions/sample_submission/predictions.csv", index=False)
print("Submission saved!")

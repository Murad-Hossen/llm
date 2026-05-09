<img width="5370" height="1775" alt="three_cities_networks" src="https://github.com/user-attachments/assets/53d2210f-01f7-4594-9909-303f107eed51" />


# City Graph class challenge (CGCC)


🏆 View Live Leaderboard: [Open leaderboard](https://murad-hossen.github.io/CGCC/)

This dataset comprises street network graphs for 120 diverse cities across continents including North America, South America, Europe, Asia, Africa, Australia & Oceania, and others like the Middle East and Central Asia. The graphs are extracted from OpenStreetMap using OSMnx, focusing on driveable roads within a 500-meter buffer around each city's central point.

The dataset includes a total of 120 cities with an unbalanced distribution of street network types reflecting real-world urban patterns: 37 grid cities (such as planned orthogonal layouts like Salt Lake City, USA), 31 organic cities (such as irregular, historic winding streets like Boston, USA), and 52 hybrid cities (such as mixed elements like Atlanta, USA).

Each city's data is stored as a serialized NetworkX graph in `.pkl` format within the `city_graphs` folder, including nodes (intersections with coordinates), edges (roads with lengths and geometries), and graph attributes for layout `type` (grid/organic/hybrid) and city `name`.

This dataset is ideal for urban planning analysis, graph theory, or machine learning tasks like layout classification. It was generated via a Python script using OSMnx and NetworkX.


The goal of Task 3 is to train a model to classify each city’s street layout into one of three classes:

- `0` = **organic**
- `1` = **grid**
- `2` = **hybrid**

Participants will train on the **train set** and submit predictions for the **test set** as a `submission.csv`.

---

## Dataset Summary

- Class distribution in the full dataset:
  - **organic**: 31
  - **grid**: 37
  - **hybrid**: 52

Each city graph is stored as a serialized NetworkX graph (`.pkl`) and contains:
- **nodes**: intersections with coordinates (`x`, `y`)
- **edges**: road segments (may include attributes such as length/geometry depending on OSM)
- **graph attributes** (e.g., city name).  
For the **test set**, the label attribute is removed.

This dataset is useful for urban planning analysis, graph learning, and layout classification tasks.

---

## Data Split (Train/Test)
The dataset is split into **70/30** with **stratification by class**:

- `gnn_challenge/data/train/` : labeled graphs (70%)
- `gnn_challenge/data/test/`  : unlabeled graphs (30%)

Training labels are provided in:
- `gnn_challenge/data/train_labels.csv` with columns:
  - `filename`
  - `target`

---

## What You Need To Do (Participant)
### Step 1: Train
Train your model using:
- graphs in `gnn_challenge/data/train/`
- labels in `gnn_challenge/data/train_labels.csv`

### Step 2: Predict
Predict labels for every graph in:
- `gnn_challenge/data/test/`

### Step 3: Submit
Create a `submission.csv` in the following format:

```csv
filename,prediction
Boston_Massachusetts_USA.pkl,2
Delhi_India.pkl,0
Turin_Italy.pkl,1
...
```

### Step 4: Encrypt and name your file

Encrypt your CSV and submit only the encrypted file in `submissions/`.

**Required naming rule (important):**
- Use your team name in the filename: `<team_name>.csv.enc`
- Examples: `abdksm.csv.enc`, `Muhammad_Isah.csv.enc`
- Do **not** submit generic names such as `submission.csv.enc`

This naming rule is used to display your team name correctly on the leaderboard.

---

## Baseline Model (GCN) — Details

The provided baseline is a **Graph Convolutional Network (GCN)** for **graph-level classification** (one label per city graph).

### Input
Each city is a graph `G` stored as a `.pkl` NetworkX file.

- Nodes: intersections with coordinate attributes `x` and `y`
- Edges: road connections between intersections

### Node Features Used (per node)
For each node, we build a 3D feature vector:

1. **Centered & scaled x-coordinate**
2. **Centered & scaled y-coordinate**
3. **Normalized node degree**

So the node feature matrix is:

- `X ∈ R^(N×3)` where `N` = number of nodes in the city graph.

### Adjacency (GCN Normalization)
The baseline builds a sparse adjacency matrix with self-loops and applies standard GCN normalization:

Normalized adjacency: D^{-1/2}(A+I)D^{-1/2}

This improves stability compared to using a raw adjacency matrix.

### Model Architecture
The baseline uses two GCN-style message passing layers (implemented with sparse matrix multiplication) and then a graph pooling step:

- Layer 1: `X -> hidden`
- Layer 2: `hidden -> hidden`
- Pooling: concatenate **mean pooling** + **max pooling** to get a graph embedding
- Classifier: linear layer to output 3 logits (organic/grid/hybrid)

Training uses:
- Adam optimizer
- Cross-entropy loss
- class weights (helps if classes are imbalanced)
- dropout + weight decay (regularization)

### Validation
To provide a baseline metric without touching the hidden test labels, the script splits the training set internally:

- 70% train
- 30% validation (stratified)

It prints:
- Validation Accuracy
- Validation Macro-F1 (main metric)

### Output
After training, the baseline predicts on the unlabeled test graphs and writes:

`gnn_challenge/data/submission.csv`

Format:
```csv
filename,prediction
City1.pkl,2
City2.pkl,0
...




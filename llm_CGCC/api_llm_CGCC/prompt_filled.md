You are solving a GNN coding competition.

You will receive:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary,
  (4) the required submission format,
  (5) allowed libraries.

Produce a single, self-contained Python script that, when executed
from the repository root on CPU, trains a GNN, generates predictions
for the test set, and writes the submission file.

Respond in EXACTLY this format:

<plan>
A 5-10 bullet plan covering: features, model architecture, training,
validation, decoding, and how the submission file will be written.
</plan>

<code>
A complete, runnable Python script. Constraints:
  - Use only the libraries listed under "Allowed libraries"
    (plus the Python standard library).
  - Runnable as `python solution.py` from the repo root.
  - Set seeds (numpy, torch, random) for reproducibility.
  - Must complete on CPU within 60 minutes.
  - Must write the submission file in the exact required format.
  - Do not download external data; use only files already on disk.
</code>

---





=== SECTION 1: README ===


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




=== SECTION 2: REPO TREE ===
./encryption/encrypt.py
./encryption/generate_keys.py
./encryption/decrypt.py
./encryption/public_key.pem
./index.html
./.DS_Store
./prompt_filled.md
./LICENSE
./requirements.txt
./leaderboard/index.html
./leaderboard/calculate_scores.py
./leaderboard/leaderboard.css
./leaderboard/__init__.py
./leaderboard/leaderboard.js
./leaderboard/update_leaderboard.py
./leaderboard/score_submission.py
./leaderboard/hidden_labels_reader.py
./leaderboard/render_leaderboard.py
./leaderboard/leaderboard.csv
./README.md
./leaderboard.md
./.gitignore
./utils.py
./scripts/inspect_cgcc.py
./.github/workflows/process_submission.yml
./.github/scripts/process_submission.py
./scoring_script.py
./gather_prompt_data.sh
./submission.csv.enc
./.env.example
./data/train_labels.csv
./data/test/Minneapolis_Minnesota_USA.pkl
./data/test/Medellin_Colombia.pkl
./data/test/Brussels_Belgium.pkl
./data/test/Beijing_China.pkl
./data/test/Bangalore_India.pkl
./data/test/Rotterdam_Netherlands.pkl
./data/test/Darwin_Northern_Territory_Australia.pkl
./data/test/Boston_Massachusetts_USA.pkl
./data/test/Kinshasa_Democratic_Republic_of_the_Congo.pkl
./data/test/London_United_Kingdom.pkl
./data/test/Indianapolis_Indiana_USA.pkl
./data/test/Oslo_Norway.pkl
./data/test/La_Plata_Buenos_Aires_Argentina.pkl
./data/test/Dakar_Senegal.pkl
./data/test/Oklahoma_City_Oklahoma_USA.pkl
./data/test/Mumbai_India.pkl
./data/test/Cleveland_Ohio_USA.pkl
./data/test/Munich_Germany.pkl
./data/test/Kuala_Lumpur_Malaysia.pkl
./data/test/Damascus_Syria.pkl
./data/test/Zurich_Switzerland.pkl
./data/test/Caracas_Venezuela.pkl
./data/test/Kansas_City_Missouri_USA.pkl
./data/test/Zagreb_Croatia.pkl
./data/test/Suva_Fiji.pkl
./data/test/Tehran_Iran.pkl
./data/test/Barcelona_Spain.pkl
./data/test/Bucharest_Romania.pkl
./data/test/Doha_Qatar.pkl
./data/test/Belgrade_Serbia.pkl
./data/test/Ho_Chi_Minh_City_Vietnam.pkl
./data/test/Gold_Coast_Queensland_Australia.pkl
./data/test/Sofia_Bulgaria.pkl
./data/test/Jeddah_Saudi_Arabia.pkl
./data/test/Salt_Lake_City_Utah_USA.pkl
./data/test/Marseille_France.pkl
./data/train/Jerusalem_Israel.pkl
./data/train/Ashgabat_Turkmenistan.pkl
./data/train/Siena_Italy.pkl
./data/train/Kiev_Ukraine.pkl
./data/train/Sacramento_California_USA.pkl
./data/train/Addis_Ababa_Ethiopia.pkl
./data/train/Papeete_French_Polynesia.pkl
./data/train/Belo_Horizonte_Brazil.pkl
./data/train/Washington_DC_USA.pkl
./data/train/Valencia_Spain.pkl
./data/train/Bangkok_Thailand.pkl
./data/train/Tel_Aviv_Israel.pkl
./data/train/Dublin_Ireland.pkl
./data/train/Tashkent_Uzbekistan.pkl
./data/train/Kazan_Russia.pkl
./data/train/Rio_de_Janeiro_Brazil.pkl
./data/train/Mexico_City_Mexico.pkl
./data/train/Osaka_Japan.pkl
./data/train/Halifax_Nova_Scotia_Canada.pkl
./data/train/Baltimore_Maryland_USA.pkl
./data/train/Yerevan_Armenia.pkl
./data/train/Frankfurt_Germany.pkl
./data/train/Taipei_Taiwan.pkl
./data/train/Calgary_Alberta_Canada.pkl
./data/train/Moscow_Russia.pkl
./data/train/Beirut_Lebanon.pkl
./data/train/Tbilisi_Georgia.pkl
./data/train/Birmingham_United_Kingdom.pkl
./data/train/Ahmedabad_India.pkl
./data/train/Luanda_Angola.pkl
./data/train/Las_Vegas_Nevada_USA.pkl
./data/train/Montevideo_Uruguay.pkl
./data/train/Amsterdam_Netherlands.pkl
./data/train/Belfast_Northern_Ireland.pkl
./data/train/Seville_Spain.pkl
./data/train/Cartagena_Colombia.pkl
./data/train/Fez_Morocco.pkl
./data/train/Dallas_Texas_USA.pkl
./data/train/Mannheim_Germany.pkl
./data/train/Edmonton_Alberta_Canada.pkl
./data/train/Berlin_Germany.pkl
./data/train/Detroit_Michigan_USA.pkl
./data/train/Orlando_Florida_USA.pkl
./data/train/Amman_Jordan.pkl
./data/train/Noumea_New_Caledonia.pkl
./data/train/Perth_Western_Australia_Australia.pkl
./data/train/Accra_Ghana.pkl
./data/train/Milwaukee_Wisconsin_USA.pkl
./data/train/Baghdad_Iraq.pkl
./data/train/Abu_Dhabi_UAE.pkl
./data/train/Paris_France.pkl
./data/train/Seoul_South_Korea.pkl
./data/train/Turin_Italy.pkl
./data/train/Khartoum_Sudan.pkl
./data/train/Atlanta_Georgia_USA.pkl
./data/train/Ottawa_Ontario_Canada.pkl
./data/train/Christchurch_New_Zealand.pkl
./data/train/Delhi_India.pkl
./data/train/Rome_Italy.pkl
./data/train/Auckland_New_Zealand.pkl
./data/train/Kuwait_City_Kuwait.pkl
./data/train/Hamburg_Germany.pkl
./data/train/Lagos_Nigeria.pkl
./data/train/Toulouse_France.pkl
./data/train/Honolulu_Hawaii_USA.pkl
./data/train/Johannesburg_South_Africa.pkl
./data/train/Kyoto_Japan.pkl
./data/train/Portland_Oregon_USA.pkl
./data/train/La_Paz_Bolivia.pkl
./data/train/Cusco_Peru.pkl
./data/train/Manila_Philippines.pkl
./data/train/Tampa_Florida_USA.pkl
./data/train/Salvador_Bahia_Brazil.pkl
./data/train/Shanghai_China.pkl
./data/train/Lisbon_Portugal.pkl
./data/train/Naples_Italy.pkl
./data/train/San_Antonio_Texas_USA.pkl
./data/train/Algiers_Algeria.pkl
./data/train/Porto_Portugal.pkl
./data/train/Monterrey_Nuevo_Leon_Mexico.pkl
./data/train/Hobart_Tasmania_Australia.pkl
./data/train/Glasgow_Scotland.pkl
./data/train/Manchester_United_Kingdom.pkl
./data/train/Almaty_Kazakhstan.pkl
./starter_code/requirements.txt
./starter_code/baseline.py
./starter_code/.ipynb_checkpoints/baseline-checkpoint.py

=== SECTION 3: DATA SAMPLE ===
train_labels.csv:
filename,target
Washington_DC_USA.pkl,2
Mannheim_Germany.pkl,1
Addis_Ababa_Ethiopia.pkl,2
Atlanta_Georgia_USA.pkl,2
Papeete_French_Polynesia.pkl,0
Total rows:       85

Train files:       84
Test files:        36

Sample graph contents:
file: data/test/Bangalore_India.pkl
nodes: 49 edges: 65
graph attrs: ['created_date', 'created_with', 'crs', 'simplified', 'name']
node attrs: ['y', 'x', 'street_count']
edge attrs: ['osmid', 'highway', 'oneway', 'reversed', 'length', 'geometry']

=== SECTION 4: SUBMISSION FORMAT ===
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

--
Create a `submission.csv` in the following format:

```csv
filename,prediction
Boston_Massachusetts_USA.pkl,2
Delhi_India.pkl,0
Turin_Italy.pkl,1
...

=== SECTION 5: REQUIREMENTS ===
networkx>=3.1
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
torch>=2.0
torch_geometric  (PyTorch Geometric — available, use for GNN layers)

IMPORTANT environment notes:
- Data paths: labels are at 'data/train_labels.csv', graphs at 'data/train/' and 'data/test/'.
  Do NOT use 'gnn_challenge/data/' — that prefix does not exist.
- The model MUST be a Graph Neural Network (GNN) using torch_geometric.

MANDATORY graph loading function — copy this exactly, do not modify:

    import pickle, torch
    from torch_geometric.data import Data

    def nx_to_pyg(path):
        with open(path, 'rb') as f:
            G = pickle.load(f)
        # OSMnx node IDs are large integers — must remap to 0..N-1
        nodes = list(G.nodes())
        id_map = {n: i for i, n in enumerate(nodes)}
        # edges: G.edges() returns (u, v, key) for MultiDiGraph — strip key
        edges = [(id_map[u], id_map[v]) for u, v, *_ in G.edges()
                 if u in id_map and v in id_map]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        x_list = [[G.nodes[n].get('x', 0.0),
                   G.nodes[n].get('y', 0.0),
                   float(G.degree(n))] for n in nodes]
        x = torch.tensor(x_list, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, num_nodes=len(nodes))

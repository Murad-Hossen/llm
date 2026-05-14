=== SECTION 1: README ===
# 📰 Real Or Fake?! 🕵️‍♂️
## GNN-based Fake News Detection Challenge

Welcome to the **GNN-based Fake News Detection Challenge**! This competition focuses on detecting fake news propagation on Twitter using Graph Neural Networks (GNNs). 


### ⭐📊 **[Live Leaderboard](https://tugaahmed.github.io/Real_Or_Fake/leaderboard.html)** 📊⭐

---



## Repository Structure

```text
Real_Or_Fake/
├── data/
│   ├── public/
│   │   ├── A.txt
│   │   ├── new_bert_feature.npz
│   │   ├── new_spacy_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── node_graph_id.npy
│   │   ├── train_idx.npy
│   │   ├── train_labels.csv
│   │   ├── val_idx.npy
│   │   ├── val_labels.csv
│   │   ├── test_idx.csv
│   │   └── test_idx.npy
│   └── test_labels.csv   (kept private, restored via GitHub Secret)
│
├── submissions/
│   ├── sample_submission/
│   │   └── predictions.csv
│   └── inbox/
│       └── team_name/
│           └── run_name/
│               ├── metadata.json
│               └── predictions.csv.gpg
│
├── competition/
│   ├── metrics.py
│   └── scoring_script.py
│
├── docs/
│   ├── leaderboard.css
│   ├── leaderboard.csv
│   ├── leaderboard.html
│   └── leaderboard.js
│
├── key/
│   └── competition_public_key.asc
│
├── models/
│   ├── model.py
│   └── saved_model.model
│
├── dataloader.py
├── evaluate.py
├── test.py
├── train.py
├── update_leaderboard.py
├── validate_submission.py
├── requirements.txt
├── README.md
└── LICENSE
```


---

## 📚 Dataset Source

The dataset used in this repository is from the paper:

> Dou, Y., Shu, K., Xia, C., Yu, P. S., & Sun, L. (2021). *User Preference-aware Fake News Detection*. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval* (SIGIR '21), pp. 2051–2055. [https://doi.org/10.1145/3404835.3462990](https://doi.org/10.1145/3404835.3462990)




## 📦 Dataset Overview

This competition uses the **GossipCop** dataset, which contains Twitter news propagation graphs. Each graph represents the spread of a single news article

Each graph corresponds to a news article (root node) and all users who engaged with it (child nodes).
  - **Nodes:** represent either the news article or a user who interacted with it.
  - **Edges:** represent interactions or retweets between nodes. Only edges connecting nodes in the same graph are used for that graph
  - The **root node** corresponds to the news article itself.
  - **Child nodes** correspond to users who retweeted or engaged with the news.

Graphs are used as input to Graph Neural Networks (GNNs) to classify news as **Real (0)** or **Fake (1)**.

The dataset is split into **public** and **private** parts:

- **Public:** Available to participants for training, validation, and testing.
- **Private:** Hidden labels used for submission and leaderboard evaluation.

---

### Graph Structure 
<p align="center">
  <img src="images/graph_plot.png" alt="Graph Plot" width="600"/>
</p> 
<p align="center"> <em>Graph visualization generated using ChatGPT</em> </p>

The graph connectivity and graph assignment information are stored in the following files:

- **`A.txt`**  
  Contains all edges in the dataset. Each row is an edge represented by two node IDs (source and target).  
  **Type:** Integer array, shape `(num_edges, 2)`  

- **`node_graph_id.npy`**  
  Maps each node to its corresponding graph. The value at index `i` indicates the graph ID of node `i`.  
  **Type:** Integer array, shape `(num_nodes,)`  
  

### Node Features

Each node in the graph has **text embeddings** and optionally **user profile features**.  

#### 1. Text Embeddings
- **BERT embeddings:** `768-dim` vectors representing the content of the news or user historical tweets.  
  File: `new_bert_feature.npz`  
- **spaCy embeddings:** `300-dim` vectors representing the content of the news or user historical tweets.  
  File: `new_spacy_feature.npz`  

#### 2. User Profile Features (10-dim)
These features are derived from the Twitter user object using the Twitter API:  

1. Verified? (`0` or `1`)  
2. Geo-spatial enabled? (`0` or `1`)  
3. Number of followers  
4. Number of friends  
5. Status/tweet count  
6. Number of favorites  
7. Number of lists the user is part of  
8. Account age (months since Twitter launch)  
9. Number of words in the user’s name  
10. Number of words in the user’s description  

File: `new_profile_feature.npz`  

### Data Splits
- **`train_idx.npy`**  
  Contains the list of **`3826`** graph IDs used for training.  
  **Type:** Integer array, shape `(num_train_graphs,)`  
 

- **`val_idx.npy`**  
  Contains the list of **`546`** graph IDs used for validation.  
  **Type:** Integer array, shape `(num_val_graphs,)`  


- **`test_idx.npy`**  
  Contains the list of **`1092`** graph IDs used for testing. Labels are hidden in the private folder for competition evaluation.  
  **Type:** Integer array, shape `(num_test_graphs,)`  
  
  
### Graph Labels

Each graph in the dataset has a label indicating whether the news is **real** or **fake**:

- `0` → Real news  
- `1` → Fake news  

Graph labels are stored separately for different splits:

- **Training labels:** `train_labels.csv`  
- **Validation labels:** `val_labels.csv`  
- **Test labels (hidden for competition evaluation):** `test_labels.csv`  
- Each CSV file contains two columns:
    1. `id` → Graph ID  
    2. `y_true` → Label (0 or 1)


### Dataset Statistics

Here are some key statistics for the news propagation graphs in the competition datasets:

| Dataset      | #Graphs (Fake) | #Total Nodes | #Total Edges | Avg. Nodes per Graph |
|-------------|----------------|--------------|--------------|--------------------|
| GossipCop (GOS)  | 5,464 (2,732)   | 314,262      | 308,798      | 58                 |


---

## 📝 Problem Statement

**Task:** Classify each news propagation graph as real or fake.

### Baseline Model Description

The baseline model is a **Graph Neural Network (GNN)** for fake news detection implemented in `model.py`. Its main components:

- **Graph Attention Layers (GAT):** 3 layers to learn node embeddings from the propagation graph.  
- **Global Max Pooling:** Aggregates node embeddings to a single graph-level representation.  
- **Root Node Transformation:** Linear layer processes the root node (news article) features.  
- **Concatenation & Output:** Combines graph representation and root node features, then passes through a linear layer with **sigmoid** to predict fake/real news.  

**Features used in baseline:**  
  - spaCy Text embeddings of news and historical user tweets
    
**Output:**
  - Probability that a news graph is fake.

## 🚀 Getting Started

Follow these steps to replicate the baseline results and build your own implementation.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/TugaAhmed/Real_Or_Fake.git
cd Real_Or_Fake
```

### 2️⃣ Set Up Environment

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Download and Prepare the Dataset

Download the public dataset ZIP file from this [link](https://drive.google.com/file/d/1sn_xFAM_9StN49AL50XNJaV3f9_TVKn5/view?usp=sharing) .

Extract the contents inside the data/public folder so that the folder structure looks like this:

```text
├── data/
│   ├── public/
│   │   ├── A.txt
│   │   ├── new_bert_feature.npz
│   │   ├── new_spacy_feature.npz
│   │   ├── new_profile_feature.npz
│   │   ├── node_graph_id.npy
│   │   ├── train_idx.npy
│   │   ├── train_labels.csv
│   │   ├── val_idx.npy
│   │   ├── val_labels.csv
│   │   ├── test_idx.csv
│   │   └── test_idx.npy

```

### 4️⃣ Train the Baseline Model

Run the training script to train the GNN on the dataset:
```bash
python train.py
```

This will train the model and generate `saved_model.model` in the `models/` folder.

The saved model corresponds to the one with the best validation accuracy.

Metrics tracked during training: **Accuracy and F1 score**.

### 5️⃣ Generate Predictions

After training, run the test script to generate predictions:
```bash
python test.py
```
This will create a predictions.csv file inside the submissions/ folder.

The CSV contains two columns: **id** , **y_pred**

### 6️⃣ Evaluate Predictions

You can evaluate your predictions using the evaluation script:

```bash
python evaluate.py
```

⚠️ **Note:**

- The file `test_labels.csv` is **not publicly available**.
- The evaluation script is provided **only to demonstrate how scoring works**.
- The script will run successfully **only when the ground-truth labels are available**.
- Final scoring is performed on the competition server after submission.


**Metrics reported include:**    
   - Accuracy
   - F1 Score
     
## 🎯 Competition Task

Your goal is to **beat the baseline accuracy** on the fake news detection task using the provided Twitter news propagation graphs.

### Baseline Overview
- Uses **only spaCy text embeddings** of news articles and user historical tweets.
- Achieves:
  - **Accuracy:** 0.7216
  - **F1 score:** 0.7071

### Your Task
1. Build a **Graph Neural Network (GNN)** based pipeline.
2. Use **any combination of available features**:
   - SpaCy text embeddings (baseline feature)
   - BERT embeddings (`new_bert_feature.npz`)
   - User profile features (`new_profile_feature.npz`)
3. Train your model on the **public training data** and validate on the validation set.
4. Generate predictions for the **test set** as `predictions.csv`.

### Rules
- Your model **must use a GNN**; other models alone will not be accepted.
- You may combine features in **any way** to improve performance.
- The objective is to **maximize accuracy** on the hidden test set.

---



## 📤 Submission Workflow

Follow these steps to participate in the competition and submit your results.

---

### 1️⃣ Train Your Model Locally

- Use the public dataset in `data/public/`.
- Train your model using your own implementation.
- Generate predictions for the **test set**.
- Create a `predictions.csv` file locally.

---

### 2️⃣ Prepare Submission Files

Each submission must include:

#### ✅ `predictions.csv` (Local File Only – DO NOT Upload)

Must contain exactly two columns:

| Column  | Description |
|----------|------------|
| `id`     | Graph identifier (must exactly match public test IDs) |
| `y_pred` | Predicted probability or score |

⚠️ IDs must exactly match those in the public test input file.  
Incorrect formatting will cause automatic validation failure.

🚫 **Do NOT upload `predictions.csv` to the repository.**

---

#### 🔐 Encrypt Your Predictions File

Before submission, you must encrypt your `predictions.csv` using the competition public key located in:

```
key/competition_public_key.asc
```

Run the following commands in bash:

```bash
# Import the public key
gpg --import competition_public_key.asc

# Encrypt predictions file
gpg --output predictions.csv.gpg \
    --encrypt \
    --recipient "GNN competition (Real or Fake) " \
    predictions.csv
```

This will generate:

```
predictions.csv.gpg
```

✅ **Only this encrypted `.gpg` file is allowed for submission.**

---

#### ✅ `metadata.json`

```json
{
  "team": "example_team",
  "run_id": "example_run_id",
  "type": "human",
  "model": "GAT",
  "notes": "Additional notes"
}
```

`type` must be one of:

- `"human"`
- `"llm-only"`
- `"human+llm"`

---

### 3️⃣ Submission Directory Structure

Your Pull Request must add files in the following structure:

```
submissions/inbox/<team_name>/<run_id>/
    ├── predictions.csv.gpg
    └── metadata.json
```

Example:

```
submissions/inbox/team_alpha/run_01/
    ├── predictions.csv.gpg
    └── metadata.json
```

🚫 Uploading `predictions.csv` will result in automatic rejection.

---

### 4️⃣ Submit via Pull Request

1. Fork the repository.
2. Add your metadata and encrypted submission files in the correct directory.
3. Open a Pull Request (PR) to the main repository.

---

### 5️⃣ Automatic Validation & Scoring

When the Pull Request is opened:

- Submission format is validated
- The encrypted file is securely decrypted by the competition server
- Predictions are scored using hidden test labels
- Score is posted automatically as a PR comment
- Invalid submissions fail automatically

---

### 6️⃣ Leaderboard Update

- Your score is appended to `docs/leaderboard.csv`
- The leaderboard page is automatically updated



## 🏫 Mentorship Program

This competition is part of the [BASIRA-LAB](https://basira-lab.com/) **(GNNs) for Rising Stars Mentorship Program**  

The lab’s tutorials on **Deep Graph Learning** served as guidance for preparing this challenge: [Tutorials Link](https://www.youtube.com/watch?v=gQRV_jUyaDw&list=PLug43ldmRSo14Y_vt7S6vanPGh-JpHR7T)

=== SECTION 2: REPO TREE ===
./prompt_filled.md
./LICENSE
./requirements.txt
./images/graph_plot.png
./validate_submission.py
./key/competition_public_key.asc
./models/__pycache__/model.cpython-38.pyc
./models/model.py
./models/saved_model.model
./__pycache__/dataloader.cpython-38.pyc
./docs/leaderboard.css
./docs/leaderboard.js
./docs/leaderboard.html
./docs/leaderboard.csv
./test.py
./README.md
./update_leaderboard.py
./.github/workflows/score-submission.yml
./train.py
./evaluate.py
./dataloader.py
./gather_prompt_data.sh
./competition/metrics.py
./competition/scoring_script.py
./data/public/train_labels.csv
./data/public/val_labels.csv
./data/public/new_bert_feature.npz
./data/public/new_spacy_feature.npz
./data/public/new_profile_feature.npz
./data/public/node_graph_id.npy
./data/public/A.txt
./data/public/test_idx.npy
./data/public/train_idx.npy
./data/public/val_idx.npy
./data/public/test_idx.csv

=== SECTION 3: DATA SAMPLE ===
--- train_labels.csv ---
shape: (3826, 2)
   id  y_true
0   0     0.0
1   3     0.0
2   4     0.0
3   6     0.0
4   7     0.0

--- val_labels.csv ---
shape: (546, 2)
   id  y_true
0  21     0.0
1  26     0.0
2  45     0.0
3  57     0.0
4  74     0.0

--- test_idx.csv ---
shape: (1092, 1)
   id
0   1
1   2
2   5
3   8
4  11

--- train_idx.npy ---
shape: (3826,) dtype: int64
first 5: [5453 2803 1401 4838 1643]

--- val_idx.npy ---
shape: (546,) dtype: int64
first 5: [3570 4908  365 2913 1374]

--- test_idx.npy ---
shape: (1092,) dtype: int64
first 5: [ 271 3681 4887  994 1839]

--- node_graph_id.npy ---
shape: (314262,) dtype: int64
unique graphs: 5464

--- A.txt (edges) ---
total edges: 308798
first 5 rows: ['0, 1\n', '0, 2\n', '0, 3\n', '0, 4\n', '0, 5\n']

--- new_bert_feature.npz ---
keys: ['indices', 'indptr', 'format', 'shape', 'data']
  indices: shape=(241353216,), dtype=int32
  indptr: shape=(314263,), dtype=int32
  format: shape=(), dtype=|S3
  shape: shape=(2,), dtype=int64
  data: shape=(241353216,), dtype=float64

--- new_spacy_feature.npz ---
keys: ['indices', 'indptr', 'format', 'shape', 'data']
  indices: shape=(94278600,), dtype=int32
  indptr: shape=(314263,), dtype=int32
  format: shape=(), dtype=|S3
  shape: shape=(2,), dtype=int64
  data: shape=(94278600,), dtype=float64

--- new_profile_feature.npz ---
keys: ['indices', 'indptr', 'format', 'shape', 'data']
  indices: shape=(2432790,), dtype=int32
  indptr: shape=(314263,), dtype=int32
  format: shape=(), dtype=|S3
  shape: shape=(2,), dtype=int64
  data: shape=(2432790,), dtype=float64

=== SECTION 4: SUBMISSION FORMAT ===
## 📤 Submission Workflow

Follow these steps to participate in the competition and submit your results.

---

### 1️⃣ Train Your Model Locally

- Use the public dataset in `data/public/`.
- Train your model using your own implementation.
- Generate predictions for the **test set**.
- Create a `predictions.csv` file locally.

---

### 2️⃣ Prepare Submission Files

Each submission must include:

#### ✅ `predictions.csv` (Local File Only – DO NOT Upload)


=== SECTION 5: REQUIREMENTS ===
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.2
torch==2.4.1
torch-geometric==2.6.1

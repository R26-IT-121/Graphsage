# Edge-Enhanced GraphSAGE — Complete System Walkthrough

**Purpose:** This document walks through every file we built, what it does, what code lives in it, and how each piece contributes to the three research novelties. Read it cover-to-cover before defending the progress presentation.

**Author:** Sachintha Bhashitha Ewaduge

---

## Table of Contents

1. [What we are building](#1-what-we-are-building)
2. [The three research novelties](#2-the-three-research-novelties)
3. [End-to-end data flow](#3-end-to-end-data-flow)
4. [Repository layout](#4-repository-layout)
5. [Stage A — Data engineering](#5-stage-a--data-engineering)
6. [Stage B — Graph construction](#6-stage-b--graph-construction)
7. [Stage C — Train/val/test split](#7-stage-c--trainvaltest-split)
8. [Stage D — Models (Stage 1 baseline + Stage 2 Edge-MLP)](#8-stage-d--models-stage-1-baseline--stage-2-edge-mlp)
9. [Stage E — Training loop and metrics](#9-stage-e--training-loop-and-metrics)
10. [Stage F — Threshold tuning](#10-stage-f--threshold-tuning)
11. [Stage G — Focal Loss + Imbalance Sampler (Novelty 2)](#11-stage-g--focal-loss--imbalance-sampler-novelty-2)
12. [Stage H — API contract and integration](#12-stage-h--api-contract-and-integration)
13. [Results — leakage-free evaluation](#13-results--leakage-free-evaluation)
14. [Novelty 3 — Suspicious Subgraph extraction](#14-novelty-3--suspicious-subgraph-extraction)
15. [Serving — API, demo, and the serving bundle](#15-serving--api-demo-and-the-serving-bundle)
16. [What is left](#16-what-is-left)
17. [Viva defense cheat-sheet](#17-viva-defense-cheat-sheet)

---

## 1. What we are building

We are building one of four independent components of the **DeepSentinel** multi-modal financial fraud detection platform. Our component is the **Edge-Enhanced GraphSAGE Relational Fraud Detector** — the network-intelligence layer that detects organized fraud rings (mule networks) on the PaySim mobile money dataset.

The other three components:
- Stratified VAE with Dual-Signal Anomaly Attribution (behavioral)
- Temporal Convolutional Network with system-context features (temporal)
- Fusion engine with RAG-grounded LLM forensic reports

**Our role in the pipeline:** the fusion engine calls our `POST /api/graph/analyze` endpoint with a transaction. We return a JSON payload with a relational risk score and the suspicious subgraph (mule ring) around the transaction.

**Dataset:** PaySim — 6,362,620 simulated mobile money transactions, 0.1291% fraud rate (773:1 imbalance), no PII.

---

## 2. The three research novelties

The proposal defines three architectural contributions that distinguish our component from off-the-shelf GraphSAGE:

| # | Novelty | What it is | Where in code |
|---|---|---|---|
| **1** | **Edge-MLP attention** | A small MLP injected into GraphSAGE message passing that computes per-edge attention from `(amount_log, drain_ratio, src_drained, dst_was_empty, time_gap, type_is_transfer)`. Suspicious edges dominate aggregation; routine edges contribute little. | [src/graphsage/models/layers.py](../src/graphsage/models/layers.py) |
| **2** | **Graph-Aware Imbalance Sampler + Focal Loss** | Balanced k-hop subgraph mini-batches with hard-negative mining (instead of full-batch under 773:1 imbalance) + Focal Loss (instead of `pos_weight`). Preserves fraud topology where SMOTE would destroy it. | [src/graphsage/sampling/imbalance_sampler.py](../src/graphsage/sampling/imbalance_sampler.py), [src/graphsage/training/losses.py](../src/graphsage/training/losses.py) |
| **3** | **Suspicious Subgraph extractor** | k=2 hop walk from every flagged node producing a forensic JSON payload (mules, edges, sink, pattern, structural_evidence) consumed by the fusion engine's LLM. | [src/graphsage/extraction/subgraph.py](../src/graphsage/extraction/subgraph.py) (implemented — demo: `scripts/demo_extract_subgraph.py`) + [docs/integration/graph_api_contract.md](integration/graph_api_contract.md) |

**Be honest about novelty class:** these are *engineering and systems* contributions, not theoretical. We are combining known building blocks (GraphSAGE message passing, focal loss, k-hop subgraph) in a specific configuration designed for forensically-motivated fraud feature engineering on PaySim. The novelty lies in the integration.

---

## 3. End-to-end data flow

```
Raw CSV (471 MB)
    │
    │   scripts/prepare_features.py        ← filter to TRANSFER+CASH_OUT,
    │                                        compute 6 edge features
    ▼
features.parquet (65 MB)
    │
    │   scripts/build_graph.py             ← name → integer ID mapping,
    │                                        build PyG Data tensor
    ▼
paysim_graph.pt (179 MB)
    │
    │   scripts/make_splits.py             ← time-based train/val/test
    ▼
paysim_graph.pt with masks (195 MB)
    │
    │   scripts/train_baseline.py          ← Stage 1
    │   scripts/train_edge_mlp.py          ← Stage 2 (Novelty 1)
    │   scripts/train_focal.py             ← Stage 3a (Focal only)
    │   scripts/train_full.py              ← Stage 3b (Novelty 2)
    ▼
checkpoints/*.pt + reports/*.json
    │
    │   scripts/eval_with_tuned_threshold.py  ← ablation table
    ▼
Final ablation table ← becomes slide 8 of presentation

(Future, T8 = August)
    │
    │   src/graphsage/api/app.py           ← FastAPI service
    ▼
POST /api/graph/analyze → JSON to fusion engine
```

---

## 4. Repository layout

```
GraphSage/
├── data/                              # Gitignored — too big for git
│   ├── raw/                           # PaySim CSV (471 MB)
│   ├── processed/                     # Engineered features parquet (65 MB) + figures
│   └── graph/                         # PyG tensor file (195 MB) + metadata
│
├── notebooks/                         # 7 Kaggle/Colab portable notebook stubs
│   ├── 01_eda.ipynb                   # T3 — done; produced reports/eda_findings.md
│   └── 02-07                          # stubs, not yet populated
│
├── src/graphsage/                     # The Python library — what gets pip-installed
│   ├── data/
│   │   ├── ingestion.py               # stub (T4-style raw ingestion, post-MVP)
│   │   ├── features.py                # stub (per-row feature engineering)
│   │   ├── graph_builder.py           # ★ DataFrame → PyG Data
│   │   └── splits.py                  # ★ time-based train/val/test masks
│   ├── models/
│   │   ├── baseline.py                # ★ Stage 1: vanilla GraphSAGE
│   │   ├── layers.py                  # ★★ Novelty 1: EdgeEnhancedSAGEConv
│   │   └── edge_sage.py               # ★ Stage 2 model using Novelty 1
│   ├── sampling/
│   │   └── imbalance_sampler.py       # ★★ Novelty 2: GraphAwareImbalanceSampler
│   ├── training/
│   │   ├── trainer.py                 # ★ reusable full-batch training loop
│   │   ├── losses.py                  # ★★ FocalLoss (part of Novelty 2)
│   │   └── threshold_tuning.py        # ★ post-training threshold optimization
│   ├── inference/                     # stub for live model serving (T8)
│   ├── extraction/
│   │   └── subgraph.py                # stub for Novelty 3
│   ├── api/
│   │   ├── app.py                     # stub FastAPI service (T8)
│   │   └── schemas.py                 # stub Pydantic schemas (T10)
│   └── utils/                         # config loading, logging
│
├── scripts/                           # Runnable command-line tools
│   ├── download_paysim.py             # pull from Kaggle
│   ├── prepare_features.py            # CSV → features.parquet
│   ├── build_graph.py                 # parquet → paysim_graph.pt
│   ├── make_splits.py                 # add train/val/test masks
│   ├── train_baseline.py              # Stage 1
│   ├── train_edge_mlp.py              # Stage 2
│   ├── train_focal.py                 # Stage 3a (Focal only)
│   ├── train_full.py                  # Stage 3b (full Novelty 2)
│   ├── eval_with_tuned_threshold.py   # ablation table
│   └── serve_api.py                   # uvicorn launcher (T8)
│
├── configs/
│   └── model_config.yaml              # all hyperparameters, single source of truth
│
├── docs/
│   ├── system_walkthrough.md          # this document
│   └── integration/
│       └── graph_api_contract.md      # JSON contract for the fusion engine
│
├── examples/
│   └── api_responses/                 # sample JSONs for the fusion engine's mock generator
│
├── reports/
│   ├── eda_findings.md                # T3 deliverable: 8 EDA questions answered
│   ├── stage1_metrics.json            # Stage 1 training history + final test
│   ├── stage2_metrics.json            # Stage 2 training history + final test
│   └── stage3*_metrics.json           # Stage 3 results
│
├── checkpoints/                       # Trained .pt files (gitignored)
├── tests/                             # Empty stub
├── README.md
├── pyproject.toml                     # Dependencies + project metadata
├── .gitignore                         # Hides data/, checkpoints/, .venv/
└── .env.example                       # Kaggle credentials template
```

★ = implemented for our work
★★ = implements a research novelty

---

## 5. Stage A — Data engineering

### What this stage does

Turn the raw PaySim CSV into a clean parquet of edges with the 6 engineered features the model will consume.

### Files

**`scripts/prepare_features.py`** (~170 lines)

The pipeline:

```python
# 1. Load the CSV with optimised dtypes (cuts RAM ~40%)
DTYPES = {
    "step": "int16", "type": "category", "amount": "float32",
    "nameOrig": "category", "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32", "nameDest": "category",
    "oldbalanceDest": "float32", "newbalanceDest": "float32",
    "isFraud": "int8", "isFlaggedFraud": "int8",
}
df = pd.read_csv(RAW_PATH, dtype=DTYPES)
```

```python
# 2. Filter to types where fraud actually exists (EDA Section 3)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
# 6.36M -> 2.77M rows. 100% of fraud preserved.
```

```python
# 3. Compute the 6 edge features
df["amount_log"] = np.log1p(df["amount"]).astype("float32")

df["drain_ratio"] = np.where(
    df["oldbalanceOrg"] > 0,
    np.clip(df["amount"] / df["oldbalanceOrg"], 0.0, 1.0),
    0.0,
).astype("float32")

df["src_drained"] = (df["newbalanceOrig"] == 0).astype("int8")
df["dst_was_empty"] = (df["oldbalanceDest"] == 0).astype("int8")

# time_gap: hours since this destination's previous inbound transfer.
# Uses per-DESTINATION sort (not per-sender) because EDA Section 7 found
# that 99.78% of fraud senders are one-shot accounts.
df = df.sort_values(["nameDest", "step"], kind="stable")
df["time_gap"] = (
    df.groupby("nameDest", observed=True)["step"]
      .diff().fillna(-1).astype("float32")
)

df["type_is_transfer"] = (df["type"] == "TRANSFER").astype("int8")
```

```python
# 4. Save as parquet (10x smaller than CSV, instant reload)
out.to_parquet(OUT_PARQUET, compression="snappy", index=False)
```

### Why each feature exists (EDA-grounded)

| Feature | EDA finding that justifies it |
|---|---|
| `amount_log` | Section 5 — amount distribution is heavy-tailed, log tames it |
| `drain_ratio` | Section 8 — median drain_ratio: legit 0.22, fraud 1.00 (clean discriminator) |
| `src_drained` | Section 8 — weak alone (1.09× lift) but valuable in combination |
| `dst_was_empty` | Section 8 — strongest single feature (4.7× lift) |
| `time_gap` | Sections 6 + 7 — per-receiver, not per-sender, because senders are one-shot |
| `type_is_transfer` | Section 3 — TRANSFER fraud rate (0.77%) is 4× CASH_OUT (0.18%) |

### Output

- `data/processed/features.parquet` (65 MB, 2,770,409 rows × 10 columns)
- `data/processed/feature_metadata.json` (sanity stats)

---

## 6. Stage B — Graph construction

### What this stage does

Convert the engineered DataFrame into a PyTorch Geometric `Data` object — the actual input format for GraphSAGE.

### Files

**`src/graphsage/data/graph_builder.py`** (~150 lines) — the importable module
**`scripts/build_graph.py`** (~75 lines) — the runnable

### Key code

```python
# Constants — the column lists keep ordering canonical between save/load
EDGE_FEATURE_COLS = [
    "amount_log", "drain_ratio", "src_drained",
    "dst_was_empty", "time_gap", "type_is_transfer",
]
NODE_FEATURE_NAMES = [
    "in_degree", "out_degree",
    "mean_in_amount_log", "mean_out_amount_log", "max_in_amount_log",
]
```

**Step 1 — Map account names to integer IDs.** GraphSAGE needs integers, not strings.

```python
all_accounts = pd.unique(df[["nameOrig", "nameDest"]].values.ravel())
name_to_id = pd.Series(np.arange(len(all_accounts)), index=all_accounts)
# 3,277,509 unique nodes
```

**Step 2 — Build edge_index.** Shape `[2, num_edges]`: row 0 is source IDs, row 1 is destination IDs. This IS the graph topology.

```python
src_ids = name_to_id.loc[df["nameOrig"].values].to_numpy()
dst_ids = name_to_id.loc[df["nameDest"].values].to_numpy()
edge_index = torch.from_numpy(np.stack([src_ids, dst_ids])).to(torch.int64)
# shape (2, 2,770,409)
```

**Step 3 — Build edge_attr.** Shape `[num_edges, 6]`: the 6 features per edge.

```python
edge_attr = torch.from_numpy(df[EDGE_FEATURE_COLS].to_numpy()).to(torch.float32)
# shape (2,770,409, 6)
```

**Step 4 — Build node features x.** Per-node aggregates derived from edges. NO leakage: every value comes from the graph itself, not from labels.

```python
out_stats = df.groupby("nameOrig", observed=True).agg(
    out_degree=("amount_log", "size"),
    mean_out=("amount_log", "mean"),
)
in_stats = df.groupby("nameDest", observed=True).agg(
    in_degree=("amount_log", "size"),
    mean_in=("amount_log", "mean"),
    max_in=("amount_log", "max"),
)

x = np.zeros((num_nodes, 5), dtype=np.float32)
x[in_idx, 0] = in_stats["in_degree"].to_numpy()
x[out_idx, 1] = out_stats["out_degree"].to_numpy()
x[in_idx, 2] = in_stats["mean_in"].to_numpy()
x[out_idx, 3] = out_stats["mean_out"].to_numpy()
x[in_idx, 4] = in_stats["max_in"].to_numpy()
```

**Step 5 — Build node labels y.** This is **Option B labeling**: a node is fraud iff it received any fraudulent transaction. Justification: EDA showed fraud senders are one-shot disposable accounts (Section 7); the persistent structural element is the mule.

```python
fraud_dst_names = df.loc[df["isFraud"] == 1, "nameDest"].unique()
fraud_dst_ids = name_to_id.loc[fraud_dst_names].to_numpy()
y = torch.zeros(num_nodes, dtype=torch.int8)
y[fraud_dst_ids] = 1
# 8,169 mule nodes (0.249%)
```

**Step 6 — Pack into a PyG `Data` object.**

```python
data = Data(
    x=x_t,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y,
    edge_step=edge_step,        # carried for time-based split
    edge_isFraud=edge_isFraud,  # carried for edge-level metrics
)
```

### Output

```
Data(x=[3277509, 5], edge_index=[2, 2770409], edge_attr=[2770409, 6],
     y=[3277509], edge_step=[2770409], edge_isFraud=[2770409])

Mules: 8,169 of 3,277,509 nodes (0.249%)
Fraud edges: 8,213 of 2,770,409 (0.297%)
```

The "mules vs fraud edges" mismatch (8,169 vs 8,213) means some mules received from multiple senders — the convergence pattern.

---

## 7. Stage C — Train/val/test split

### What this stage does

Add boolean masks to the graph telling the model which nodes are train, validation, and test. **Time-based** because random split would leak future fraud into training.

### Files

**`src/graphsage/data/splits.py`** (~115 lines)
**`scripts/make_splits.py`** (~85 lines)

### Key code

```python
# Concatenate src and dst endpoints with their corresponding step
all_nodes = torch.cat([data.edge_index[0], data.edge_index[1]])
edge_step_f = data.edge_step.to(torch.float32)
all_steps = torch.cat([edge_step_f, edge_step_f])

# For each node, take the minimum step it appears in.
# Initial value of +inf so any real step replaces it.
first_step = torch.full((num_nodes,), float("inf"), dtype=torch.float32)
first_step.scatter_reduce_(0, all_nodes, all_steps,
                           reduce="amin", include_self=True)

# Boolean masks on first_step
train_mask = first_step <= train_end       # train_end = 600
val_mask = (first_step > train_end) & (first_step <= val_end)  # val_end = 700
test_mask = first_step > val_end
```

### Resulting split (notice the imbalance is intentional)

| Split | Nodes | % | Mules | Fraud rate |
|---|---|---|---|---|
| Train (steps 1-600) | 3,223,968 | 98.4% | 7,076 | 0.22% |
| Val (601-700) | 46,558 | 1.4% | 761 | 1.63% |
| Test (701-743) | 6,983 | 0.2% | 332 | **4.75%** |

Why test fraud rate is 16× the dataset baseline: PaySim's transaction volume drops 90% after step 400 while fraud activity stays constant. The test set is naturally fraud-enriched — a stricter eval than random splitting.

**Defensible viva line:**
> "We chose train_end=600 to preserve 80% of edges for training while reserving the natural fraud-enriched tail of the simulation for evaluation. The test fraud rate of 4.75% — 16× higher than the dataset baseline — produces a stricter evaluation than random splitting would."

---

## 8. Stage D — Models (Stage 1 baseline + Stage 2 Edge-MLP)

### Stage 1 baseline — vanilla GraphSAGE

**`src/graphsage/models/baseline.py`** (~55 lines)

Pure stock PyTorch Geometric. Establishes the "before" number.

```python
from torch_geometric.nn import SAGEConv

class BaselineGraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h).squeeze(-1)
```

The **mean** aggregator is critical to point out: every neighbor contributes equally. The $50,000 fraud transfer and the $5 cup of coffee get the same weight. **This is exactly what Novelty 1 fixes.**

### Stage 2 — Edge-MLP attention (NOVELTY 1)

**`src/graphsage/models/layers.py`** (~140 lines) — the actual research contribution

This is the layer that distinguishes our research from generic GraphSAGE.

```python
class EdgeEnhancedSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim,
                 edge_mlp_hidden=32, bias=True):
        # aggr='add' because we'll do a WEIGHTED SUM, not a mean.
        # Mean would normalise away the attention.
        super().__init__(aggr="add")

        self.lin_self = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neighbor = nn.Linear(in_channels, out_channels, bias=bias)

        # NOVELTY 1: per-edge attention MLP.
        # Takes 6 edge features -> hidden -> 1 attention scalar.
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_mlp_hidden),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden, 1),
        )
```

**The forward pass (the key novelty):**

```python
def forward(self, x, edge_index, edge_attr, return_attention=False):
    # Compute per-edge attention scalars in [0, 1] via sigmoid
    edge_logit = self.edge_mlp(edge_attr)
    edge_weight = torch.sigmoid(edge_logit)  # ← NOVELTY 1

    out_self = self.lin_self(x)
    agg = self.propagate(edge_index, x=x, edge_weight=edge_weight)
    out_neigh = self.lin_neighbor(agg)

    out = out_self + out_neigh
    if return_attention:
        return out, edge_weight.squeeze(-1)
    return out

def message(self, x_j, edge_weight):
    # Scale neighbor features by their edge attention weight
    return x_j * edge_weight  # ← weighted contribution
```

**The mathematical change:**

Standard SAGEConv:
```
h_i = W_self * x_i + W_neigh * MEAN(x_j for j in N(i))
```

Edge-Enhanced SAGEConv (Novelty 1):
```
edge_weight_ij = sigmoid(EdgeMLP(edge_features_ij))
h_i = W_self * x_i + W_neigh * SUM(edge_weight_ij * x_j for j in N(i))
```

**Stage 2 model wires this layer into the full architecture:**

```python
# src/graphsage/models/edge_sage.py

class EdgeEnhancedGraphSAGE(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim=64,
                 edge_mlp_hidden=32, dropout=0.3):
        super().__init__()
        self.conv1 = EdgeEnhancedSAGEConv(
            in_dim, hidden_dim, edge_dim, edge_mlp_hidden,
        )
        self.conv2 = EdgeEnhancedSAGEConv(
            hidden_dim, hidden_dim, edge_dim, edge_mlp_hidden,
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.conv1(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h).squeeze(-1)

    def forward_with_attention(self, x, edge_index, edge_attr):
        # Used by Novelty 3 to expose attention weights to the JSON output
        h, attn1 = self.conv1(x, edge_index, edge_attr, return_attention=True)
        h = F.relu(h)
        h, attn2 = self.conv2(h, edge_index, edge_attr, return_attention=True)
        h = F.relu(h)
        logits = self.classifier(h).squeeze(-1)
        return logits, [attn1, attn2]
```

Notice: Stage 2 architecture is **literally identical to Stage 1 except for the convolution layer**. This isolates Novelty 1 — any F1 difference between Stage 1 and Stage 2 is attributable to the Edge-MLP.

### Why this is novel

1. The 6-feature edge vector is **forensically motivated** — every feature traces to a specific EDA finding (drain pattern, mule signature, smurfing).
2. The MLP-attention design is generic enough to work but specific enough to capture the discriminative signal in PaySim.
3. **The fraud rings in PaySim are hub-and-spoke stars** (EDA Section 9). The Edge-MLP learns which spokes matter most — high drain ratio + dst_was_empty edges dominate the aggregation, exactly the suspicious mule deposit pattern.

---

## 9. Stage E — Training loop and metrics

### Files

**`src/graphsage/training/trainer.py`** (~250 lines) — reusable training loop
**`scripts/train_baseline.py`** (~140 lines) — Stage 1 runner
**`scripts/train_edge_mlp.py`** (~155 lines) — Stage 2 runner

### Key design decisions

1. **Full-batch training** — every epoch processes the entire graph. PaySim is small enough (3.27M nodes) to fit in 6-7 GB VRAM with `hidden_dim=64`.
2. **Auto device selection** — CUDA on the PC, MPS on Mac, CPU as fallback. Same code runs everywhere.
3. **Reusable for Stages 1, 2, 3a** — the trainer accepts `use_edge_attr=True/False` and `loss_fn=...`. Each stage just changes the model and the loss.
4. **Per-epoch metrics in EpochMetrics dataclass** — train_loss, val_loss, val_precision/recall/F1/AUROC.
5. **Early stopping on val F1** — patience 5 epochs.
6. **Best-state checkpointing** — the model state from the highest-F1 epoch is preserved and loaded for final test eval.

### Key code

**Device selection:**
```python
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

**Training loop core:**
```python
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    if use_edge_attr:
        logits = model(data.x, data.edge_index, data.edge_attr)
    else:
        logits = model(data.x, data.edge_index)
    train_loss = loss_fn(logits[data.train_mask], y_full[data.train_mask])
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        # Same forward, compute metrics on val_mask
        ...
```

**Default loss for Stages 1 and 2:**
```python
y_train = data.y[data.train_mask].float()
n_pos = float(y_train.sum().item())
n_neg = float(len(y_train) - n_pos)
pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# pos_weight = 454.6 for our imbalance
```

This `pos_weight` is what we replace in Stage 3 with Focal Loss.

---

## 10. Stage F — Threshold tuning

### Why this exists

After training Stage 1 and Stage 2, we observed F1 around 0.31 with **recall=1.0** and **precision=0.19**. The model was perfectly catching all fraud but with massive false positives. This is a **calibration problem**, not a model failure: AUROC of 0.94 shows the model has learned to rank fraud well, but the default decision threshold of 0.5 is wrong for this imbalance.

The fix: sweep thresholds on the validation set, pick the one that maximises val F1, apply to test.

### Files

**`src/graphsage/training/threshold_tuning.py`** (~95 lines)
**`scripts/eval_with_tuned_threshold.py`** (~165 lines) — runs against existing checkpoints

### Key code

```python
def find_best_threshold_for_f1(logits, y_true):
    probs = torch.sigmoid(logits).cpu().numpy()
    y_np = y_true.cpu().numpy()
    if len(set(y_np)) < 2:
        return 0.5, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_np, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    f1s = f1s[:-1]   # exclude the last point with no threshold
    best_idx = int(f1s.argmax())
    return float(thresholds[best_idx]), float(f1s[best_idx])
```

**Standard practice — not cheating.** The test set is never used to choose the threshold; only the val set is.

### Result

Threshold tuning lifted both stages from F1≈0.31 to F1≈0.50:

| Stage | F1 (default 0.5) | F1 (tuned) | Tuned threshold |
|---|---|---|---|
| Stage 1 | 0.3147 | **0.5036** | 0.9398 |
| Stage 2 | 0.3139 | **0.4944** | 0.6010 |

Notice: Stage 2's tuned threshold (0.60) is **much closer to calibrated** than Stage 1's (0.94). The Edge-MLP is producing better-calibrated probabilities even though F1 at 0.5 looks the same.

---

## 11. Stage G — Focal Loss + Imbalance Sampler (Novelty 2)

### Why this stage exists

Threshold tuning is a band-aid. The **proper** fix for severe class imbalance is to change the **training procedure** so the model produces calibrated probabilities natively. This is Novelty 2.

### Two parts of Novelty 2

**(a) Focal Loss** — replaces `BCEWithLogitsLoss(pos_weight)`. Focuses on hard examples.
**(b) Graph-Aware Imbalance Sampler** — replaces full-batch training. Provides balanced mini-batches with hard negatives.

We attempted them separately:
- **Stage 3a** = Stage 2 model + Focal Loss only (full-batch) → **failed** (training collapsed)
- **Stage 3b** = Stage 2 model + Focal Loss + Sampler → the proper full system

### Stage 3a's empirical failure (a defensible viva story)

`scripts/train_focal.py` runs the Stage 2 model with Focal Loss instead of `pos_weight`. With α=0.95 we observed:

```
epoch 1: F1=0.165 R=1.00 AUROC=0.95   ← decent start
epoch 2: F1=0.008 R=0.004 AUROC=0.88   ← collapses
epoch 4-6: F1=0.000 AUROC<0.5           ← inverted ranking
```

**Root cause math:**
- Total positive loss weight = 0.95 × 7,076 mules = **6,722**
- Total negative loss weight = 0.05 × 3,216,892 legit = **160,845**
- Even with α=0.95, negatives dominate by 24×

Conclusion: full-batch Focal Loss cannot fix 773:1 imbalance. **The sampler is necessary.** This empirically validates the proposal's design choice to combine BOTH Focal Loss AND the Sampler.

### The Focal Loss implementation

**`src/graphsage/training/losses.py`** (~90 lines)

```python
class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""

    def __init__(self, gamma=2.0, alpha=0.95, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        # p_t: predicted probability of the TRUE class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulation: down-weights easy examples
        focal_factor = (1.0 - p_t) ** self.gamma
        loss = focal_factor * bce

        # Alpha balancing: re-weights positive class
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
```

**γ=2 (focusing parameter):** down-weights easy examples (where the model already predicts confidently) so gradients focus on the hard ones.

**α=0.95 (class balance):** the positive class (fraud) gets 0.95 weight per example; the negative gets 0.05.

### The Imbalance Sampler — Novelty 2's other half

**`src/graphsage/sampling/imbalance_sampler.py`** (~150 lines)

The sampler builds balanced mini-batches by:
1. Drawing N_pos fraud nodes uniformly from training mules.
2. Drawing N_neg legit nodes via Hard Negative Mining (top-K by in-degree).
3. Extracting the union of their k-hop subgraphs intact (PyG's `k_hop_subgraph`), preserving the hub-and-spoke topology.
4. Running message passing on the small subgraph; computing loss only on the seed nodes.

```python
class GraphAwareImbalanceSampler:
    def __init__(self, data, k_hop=2, pos_per_batch=64, neg_per_batch=64,
                 hard_negative_ratio=0.5, seed=42):
        # Train fraud and train legit pools
        y_bool = data.y.to(torch.bool)
        train = data.train_mask
        self.train_pos_idx = (train & y_bool).nonzero(as_tuple=True)[0]
        self.train_neg_idx = (train & ~y_bool).nonzero(as_tuple=True)[0]

        # Hard-negative ranking: legit nodes sorted by in_degree descending.
        # The top-K are "hard negatives" — legit nodes that look like mules.
        in_degree = data.x[:, 0]   # column 0 of x is in_degree
        neg_in_deg = in_degree[self.train_neg_idx]
        sorted_local = torch.argsort(neg_in_deg, descending=True)
        self.hard_neg_pool = self.train_neg_idx[sorted_local]

    def sample(self):
        # Uniform fraud seeds
        pos_perm = torch.randperm(self.num_pos_train, generator=self.generator)
        pos_seeds = self.train_pos_idx[pos_perm[: self.pos_per_batch]]

        # Mix of hard + uniform negative seeds
        n_hard = int(self.neg_per_batch * self.hard_negative_ratio)
        n_uniform = self.neg_per_batch - n_hard
        ...

        seeds = torch.cat([pos_seeds, neg_seeds])

        # Extract the union k-hop subgraph (PyG built-in)
        subset, ei_relabeled, mapping, edge_mask = k_hop_subgraph(
            seeds, num_hops=self.k_hop,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes,
        )

        return SubgraphBatch(
            x=self.data.x[subset],
            edge_index=ei_relabeled,
            edge_attr=self.data.edge_attr[edge_mask],
            y=self.data.y[seeds].float(),
            seed_local_idx=mapping,
            n_pos=self.pos_per_batch,
            n_neg=self.neg_per_batch,
        )
```

### The Stage 3b training loop

**`scripts/train_full.py`** (~230 lines)

```python
sampler = GraphAwareImbalanceSampler(
    data=data, k_hop=2, pos_per_batch=64, neg_per_batch=64,
    hard_negative_ratio=0.5,
)
loss_fn = FocalLoss(gamma=2.0, alpha=0.95)

for epoch in range(1, epochs + 1):
    model.train()
    for _ in range(sampler.steps_per_epoch()):
        batch = sampler.sample()
        logits_sub = model(batch.x, batch.edge_index, batch.edge_attr)
        # Loss ONLY on seed nodes (the prediction targets)
        loss = loss_fn(logits_sub[batch.seed_local_idx], batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validate full-batch on the full graph every epoch
    model.eval()
    with torch.no_grad():
        logits_full = model(data.x, data.edge_index, data.edge_attr)
        val_metrics = metrics_at_threshold(
            logits_full[data.val_mask], data.y[data.val_mask], 0.5
        )
```

### What changes from Stages 1/2/3a

| Aspect | Stages 1/2/3a (full-batch) | Stage 3b (sampler) |
|---|---|---|
| Per-step input | Full graph (3.27M nodes) | k-hop subgraph (~5-50K nodes) |
| Per-step pos/neg ratio | 0.22% | 50% |
| Loss signal source | Diluted by 3.2M negatives | Directly balanced |
| Hard negative mining | None | Top-K by in-degree |
| Topology preservation | N/A | k-hop subgraphs intact |

### Why this preserves topology (vs SMOTE)

SMOTE creates synthetic positive examples by interpolating in feature space. For a graph, this means inventing new mule nodes with no real connections — destroying the hub-and-spoke structure. Our sampler instead **draws balanced batches from REAL fraud subgraphs**, keeping every edge and every neighborhood intact. The mathematical shape of fraud rings is preserved.

---

## 12. Stage H — API contract and integration

### Files

**`docs/integration/graph_api_contract.md`** — the formal contract the fusion engine reads
**`examples/api_responses/critical_fraud_hub_and_spoke.json`** — sample CRITICAL fraud
**`examples/api_responses/medium_risk_ambiguous.json`** — sample borderline
**`src/graphsage/api/app.py`** — stub (T8)
**`src/graphsage/api/schemas.py`** — stub (T10)

### The contract (summary)

**Endpoint:** `POST /api/graph/analyze`

**Response shape:**
```json
{
  "transaction_id": "TX_2026_05_01_00045821",
  "model_version": "graphsage-edge-mlp-v0.3.0",
  "stage": "stage_3_full",
  "relational_risk_score": 0.94,
  "risk_level": "CRITICAL",
  "confidence": 0.91,
  "input_transaction": { /* echo */ },
  "suspicious_subgraph": {
    "k_hop": 2,
    "node_count": 6,
    "edge_count": 5,
    "nodes": [ /* per-node info */ ],
    "edges": [ /* per-edge info with attention_weight */ ],
    "sink_account": "C964377943",
    "pattern": "HUB_AND_SPOKE",
    "pattern_confidence": 0.88,
    "structural_evidence": { /* aggregates */ }
  },
  "metadata": { /* latency, version */ }
}
```

The `pattern` field maps directly to FATF typologies in the fusion engine's ChromaDB. The `structural_evidence` block provides the quantitative facts the LLM cites in Chain-of-Evidence narratives.

### Implementation status

Both sides of the contract are now built: the FastAPI service in
`src/graphsage/api/app.py` (Section 15) and the Suspicious Subgraph extractor in
`src/graphsage/extraction/subgraph.py` (Section 14). Sample payloads under
`examples/api_responses/` remain available for mock generators, and the contract is
unchanged apart from one **additive** field, `pattern_scores`, which carries the
per-typology breakdown from the FATF classifier.

---

## 13. Results — leakage-free evaluation

### 13.1 Why the earlier numbers were wrong

The original pipeline (`build_graph.py` + `make_splits.py`) computes node features and
mule labels over the **entire** timeline, then splits nodes by first appearance. That
leaks the future twice: a node evaluated in the test window carries degree and amount
aggregates that already include test-window transactions, and a training node can be
labelled by fraud that only happens after the training horizon.

`src/graphsage/data/temporal.py` replaces this with the standard **temporal snapshot
protocol**. Each snapshot sees only edges up to its own horizon, takes labels only from
its own window, and excludes accounts already known to be mules. Everything below uses
that protocol, 5 seeds per configuration.

The cost of removing the leak is large and worth stating plainly:

| Protocol | Stage 1 | Stage 3b |
|---|---|---|
| Leaky (original) | F1 0.504 | F1 0.497 |
| **Leakage-free** | **F1 0.281** | **F1 0.395** |

Roughly 40% of the original headline score was leakage. The leakage-free numbers are
lower and correct; they are the ones to defend.

### 13.2 The ablation table (5 seeds, temporal protocol)

| Stage | Configuration | test F1 (mean ± std) | PR-AUC | vs baseline |
|---|---|---|---|---|
| 1 | GraphSAGE + BCE | 0.2806 ± 0.0867 | 0.2411 | — |
| 2 | + Edge-MLP | 0.2808 ± 0.0832 | 0.2172 | ΔF1 +0.0002 (p=0.998) |
| 3a | + Focal Loss | 0.3303 ± 0.0726 | 0.2838 | ΔF1 +0.0497 (p=0.488) |
| 3b | + Imbalance Sampler (full) | 0.3945 ± 0.0078 | 0.3739 | **ΔF1 +0.1139 (p=0.046)** |
| 3c | 3b **without** Edge-MLP | 0.3940 ± 0.0098 | 0.4236 | ΔF1 +0.1134 (p=0.074) |
| 3b-v2 | 3b + 12-dim features | 0.3955 ± 0.0052 | 0.3968 | ΔF1 +0.1148 (p=0.063) |
| **3c-v2** | **3c + 12-dim features** | **0.4056 ± 0.0026** | **0.4479** | **ΔF1 +0.1250 (p=0.045), ΔPR-AUC +0.2068 (p=0.042)** |

Statistics come from `scripts/eval_statistics.py`: the primary test is an across-seed
paired t-test (seed *i* of a variant against seed *i* of the baseline), which captures
training variance. A bootstrap CI over one seed's test set is also reported, but it only
measures test-set sampling noise and is secondary.

### 13.3 Finding 1 — the Imbalance Sampler works, and mostly by stabilising training

Stage 3b is the first configuration with a statistically significant F1 gain
(+0.114, p=0.046). The larger story is variance. Per-seed F1:

```
Stage 1 :  0.368  0.172  0.289  0.192  0.382     std 0.087
Stage 3b:  0.397  0.383  0.394  0.392  0.407     std 0.008     (11x tighter)
Stage 3c-v2: 0.406 0.404 0.402 0.408 0.408       std 0.003     (33x tighter)
```

The baseline's outcome depends heavily on initialisation — it can land anywhere between
0.17 and 0.38. With balanced mini-batches over k-hop fraud subgraphs, the outcome is
essentially deterministic. For a deployed fraud system that reproducibility matters as
much as the mean.

Honesty note for the viva: p=0.046 is marginal, and with four comparisons a Bonferroni
correction (0.0125) would not clear it. The defensible phrasing is "significant in an
uncorrected paired test at n=5, with a large and unambiguous variance reduction" — the
variance result is not marginal at all.

### 13.4 Finding 2 — Edge-MLP does not improve accuracy (Novelty 1 reframed)

Stage 3c is Stage 3b with the edge-attention layer removed and everything else
identical, so 3b vs 3c isolates Novelty 1 inside the full system:

| Comparison | ΔF1 | ΔPR-AUC | Verdict |
|---|---|---|---|
| 3b vs 3c (v1 features) | +0.0005 (p=0.952) | −0.0497 (p=0.087) | no effect |
| 3b vs 3c (v2 features) | −0.0102 (p=0.031) | −0.0512 (p=0.013) | significantly worse |

Three independent tests agree: Stage 2 vs 1 in isolation, and both leave-one-out
comparisons. Edge attention does not improve predictive accuracy and, with the richer
feature set, measurably hurts it.

**This does not make Novelty 1 worthless — it relocates its contribution.** The Edge-MLP
is what produces the per-edge attention weights that drive the Suspicious Subgraph
explanations (`edge_attention_weight` in the API payload, edge thickness in the demo).
Without it there is no per-edge attribution and the forensic narrative loses its
evidence. The honest claim is therefore:

> Per-edge attention costs ≈0.01 F1 and buys the edge-level attribution the forensic
> explanation is built on. We measured the price of explainability rather than assuming
> attention improves accuracy.

That is a stronger result than an unverified accuracy claim, and it is why the **served**
model is 3b-v2 (with attention) even though 3c-v2 scores slightly higher.

### 13.5 Finding 3 — calibration needs isotonic regression, not percentile ranks

Focal Loss with alpha=0.95 inflates raw sigmoid outputs: mule and non-mule scores
overlap heavily even though ranking is good. `scripts/calibration_study.py` compares
three transforms on the temporal test window (base rate 0.047):

| Transform | ECE | Interpretation |
|---|---|---|
| Raw sigmoid | 0.802 | unusable as a probability |
| ECDF percentile rank | 0.475 | comparable across accounts, but a **rank**, not a probability |
| **Isotonic regression** | **0.024** | genuinely probability-calibrated |

All three are monotone, so AUROC and the tuned operating point are identical; only the
scale changes. Isotonic (fitted on the validation window) is what the serving bundle
now applies.

A consequence worth knowing: calibrated probabilities on a 4.7%-positive population top
out near 0.25, so the contract's original fixed risk bands (0.25/0.5/0.75/0.9) would
never reach CRITICAL. `GraphPredictor.risk_bands` therefore derives the bands from the
tuned decision threshold and the served score distribution instead.

---

## 14. Novelty 3 — Suspicious Subgraph extraction

**`src/graphsage/extraction/subgraph.py`** turns a flagged transaction into the forensic
payload the fusion engine consumes.

Given a trigger edge it runs `k_hop_subgraph` (k=2, undirected discovery so a ring is
found regardless of flow direction), keeps the original edge directions for
serialisation, identifies the **sink** (predicted mules first, then in-degree, then
score), assigns a role to every account (MULE_CENTRAL / FRESH_SENDER / RELAY /
TRIGGER_PARTICIPANT / LEGITIMATE), and computes structural evidence — convergence count,
fresh-sender ratio, mean drain ratio, mules in subgraph.

Pattern classification is delegated to **`pattern_classifier.py`**, a rule-based FATF
typology scorer (HUB_AND_SPOKE / SMURFING / LAYERING / ACCOUNT_TAKEOVER). It is
deterministic and auditable — no learned weights — and emits a per-pattern score
breakdown plus the evidence keys the LLM prompt cites.

Two implementation details that matter:

- The saved graph carries no account name strings, so `load_node_names()` rebuilds the
  id→name mapping from `features.parquet` using the same `pd.unique` ordering as the
  graph builder, cached to `node_names.npy`.
- High-degree hubs can pull thousands of edges into a 2-hop ball. `max_edges` caps the
  payload, always keeping the trigger edge plus the highest-attention edges.

---

## 15. Serving — API, demo, and the serving bundle

### 15.1 The serving bundle

`scripts/export_serving_bundle.py` runs in Colab and writes one file containing the
graph tensors, isotonic-calibrated node scores, and per-edge attention. The API loads it
directly, so **no model is instantiated and no forward pass runs at request time or
startup**. Three consequences:

1. Startup is a single file read; per-request work is only edge lookup + extraction.
2. The demo runs on a laptop — the full-graph forward pass previously exhausted memory.
3. The demo and the dissertation cannot drift: the scores served are the same ones in
   `reports/temporal/statistics.json`.

### 15.2 API

`src/graphsage/api/app.py` implements `POST /api/graph/analyze` against the locked
contract, with Pydantic validation (`schemas.py`), a 422 `BadRequest` shape, a 404 when
no edge anchors the subgraph, `NOT_APPLICABLE` for transaction types where PaySim has no
fraud, CORS for the dashboard, and `GET /health` exposing the stage, threshold and risk
bands. Measured latency on the full graph is well inside the 500 ms NFR.

### 15.3 Demo page

`src/graphsage/api/static/demo.html` (served at `/demo`) is framework-free HTML/JS so it
can be embedded in the combined dashboard later. It renders the extracted ring as an
interactive force-directed graph — sink centred, nodes coloured by role, edge width by
attention, dashed trigger edge, hover tooltips, plus an accounts table.

Presentation rules encoded there: role labels are **threshold-aware** (nothing is called
a "mule" unless the model actually flags it), risk is shown as a percentile rather than a
bare probability, and a plain-English verdict paragraph precedes the technical detail.
`scripts/make_demo_dataset.py` mines a small scenario file (fraud ring, account takeover,
legitimate transfer, out-of-scope type, invalid input) so the demo never touches the
6.3M-row source at showtime.

---

## 16. What is left

1. Merge the component work into the team repository and integrate with the fusion
   engine (sample payloads and the deep-link format are ready).
2. Package the service (Dockerfile + one-command startup) and add a contract test the
   fusion engine can run in CI.
3. Optional research extensions, in order of value to a submission: a second dataset
   (Elliptic) to show the method is not a PaySim artifact; competitor baselines
   (XGBoost on tabular features, GAT, CARE-GNN/PC-GNN); more seeds to firm up the
   significance of the sampler result.

---

## 17. Viva defense cheat-sheet

**"How did you pick the threshold — is there a standard?"**
Decision-threshold tuning on a held-out validation set is standard for imbalanced
classification; 0.5 is only meaningful when classes are balanced and costs symmetric.
We sweep thresholds on validation, take the F1-max point on the PR curve, freeze it, and
report test metrics at that frozen threshold, so the test set never influences the
choice. scikit-learn ships the same procedure as `TunedThresholdClassifierCV`; the
theory is Elkan (2001), and Saito & Rehmsmeier (2015) motivate PR over ROC here.
Alternatives considered: Youden's J, and a cost-matrix threshold — rejected because
PaySim specifies no FP/FN costs.

**"Why is your F1 only 0.40?"**
Because it is honest. Under the original leaky protocol we measured 0.50; removing the
leak costs ~0.20 F1. On a 773:1 imbalance with a 4.7% base rate in the test window,
PR-AUC 0.45 is roughly 9x better than random.

**"Which novelty actually works?"**
Novelty 2 (Graph-Aware Imbalance Sampler): +0.114 F1, p=0.046, and an 11–33x reduction
in seed variance. Novelty 1 (Edge-MLP) does not improve accuracy — we tested it three
ways — but it supplies the per-edge attribution Novelty 3's explanations depend on, so we
report it as an explainability mechanism with a measured cost of ~0.01 F1.

**"Where can I see the graph?"**
Nobody renders 3.3M nodes. The demo renders the extracted suspicious subgraph — the
40-60 node ring around the flagged transaction — which is exactly the Novelty 3 output.
The full graph is represented by its statistics and the EDA figures.

**"How do you know it is not leaking?"**
Each temporal snapshot builds features and message passing from edges at or before its
own horizon, takes labels only from its own window, and drops accounts already known as
mules. The protocol is in `src/graphsage/data/temporal.py`.

**"Is the improvement statistically significant?"**
The primary test is an across-seed paired t-test at n=5. Stage 3b reaches p=0.046 for F1
and Stage 3c-v2 reaches p=0.045 (F1) and p=0.042 (PR-AUC). These are uncorrected; under
Bonferroni for four comparisons they would not clear. The variance reduction is the
robust claim.

**"Are the scores probabilities?"**
Only after isotonic calibration (ECE 0.024). Raw focal-loss sigmoids are not (ECE 0.80),
and percentile ranks are comparable but still not probabilities (ECE 0.48).

---

## End of walkthrough

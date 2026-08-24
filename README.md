# DeepSentinel — Edge-Enhanced GraphSAGE Relational Fraud Detector

**Project:** — *A Cloud-Native Multi-Modal AI Platform for Explainable Financial Fraud Detection*
**Component:** Graph Neural Network intelligence layer
**Author:** Sachintha Bhashitha Ewaduge

This module is one of four independent deep-learning components of the **DeepSentinel** platform. It maps the banking ecosystem as a dynamic graph (accounts = nodes, transfers = directed edges) and uses an inductive Graph Neural Network to detect organized fraud rings that legacy rule-based systems miss. Output is a forensic JSON payload consumed by the downstream fusion engine.

## Three architectural novelties

1. **Edge-MLP attention** (`src/graphsage/models/layers.py`) — a custom MLP injected into the GraphSAGE message-passing step computes a dynamic attention weight per edge from `(amount, drain_ratio, src_drained, dst_was_empty, time_gap, txn_type)`. Those weights are what make a flag explainable — the extractor ranks which transfers implicate an account by them. Ablation shows the mechanism does **not** improve accuracy (see [Results](#results)); it is retained for explainability at a measured cost of ~0.01 F1.
2. **Graph-Aware Imbalance Sampler** (`src/graphsage/sampling/imbalance_sampler.py`) — extracts intact k-hop fraud-ring subgraphs via PyG's `k_hop_subgraph`, paired with Hard Negative Mining and Focal Loss (γ=2). Handles 773:1 class imbalance without destroying topology the way SMOTE does.
3. **Relational Metadata Extractor** (`src/graphsage/extraction/subgraph.py`) — k=2 hop walk from every flagged node producing a forensic JSON payload with implicated accounts, edge weights, and identified sink.

## Tech stack

- Python ≥ 3.10
- PyTorch + PyTorch Geometric (PyG)
- NetworkX (motif analysis, visualization)
- FastAPI + Pydantic (service layer)

## Repository layout

```
Graphsage/
├── data/                  # PaySim raw, processed features, graph tensors (gitignored)
├── notebooks/             # Kaggle/Colab-portable training notebooks
├── src/graphsage/
│   ├── data/              # Ingestion, feature engineering, graph builder
│   ├── models/            # EdgeEnhancedSAGEConv layer + model classes
│   ├── sampling/          # Graph-Aware Imbalance Sampler
│   ├── training/          # Focal Loss + reusable training loop
│   ├── inference/         # Risk scoring
│   ├── extraction/        # Suspicious Subgraph extractor
│   ├── api/               # FastAPI service + Pydantic schemas
│   └── utils/             # Config loader, logging
├── configs/               # YAML hyperparameters
├── scripts/               # Dataset downloader, API runner
├── checkpoints/           # Trained model weights (gitignored)
└── tests/                 # Test suite (populated as modules ship)
```

## Setup (any OS)

Requires **Python ≥ 3.10**. Works on macOS, Linux, Windows, and cloud notebooks (Kaggle/Colab).

### 1. Clone and create a virtual environment

**macOS / Linux:**
```bash
git clone <repo-url>
cd Graphsage
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd Graphsage
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the project in editable mode (all OS)

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. PyG companion wheels (only if torch_geometric import fails)

PyTorch Geometric occasionally needs `torch-scatter` and `torch-sparse`. The wheel URL depends on your torch version + platform — check the [PyG install page](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Common cases:

```bash
# CPU (any OS)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

# CUDA 12.1 (Linux/Windows with NVIDIA GPU)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

### 4. Verify hardware acceleration (optional but useful)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

- **CUDA = True** → NVIDIA GPU available (Linux/Windows or Kaggle)
- **MPS = True** → Apple Silicon GPU (M1/M2/M3 Mac)
- **Both False** → CPU-only (training will be slow; use Kaggle for large runs)

### 5. Get Kaggle credentials

1. Go to https://www.kaggle.com/settings/account → "Create New Token" → downloads `kaggle.json`
2. Place it where the Kaggle SDK expects:

| OS | Location |
|---|---|
| macOS / Linux | `~/.kaggle/kaggle.json` (then `chmod 600 ~/.kaggle/kaggle.json`) |
| Windows | `C:\Users\<You>\.kaggle\kaggle.json` |

3. Download PaySim:
```bash
python scripts/download_paysim.py
```

## Training on Kaggle Notebooks

Notebooks under `notebooks/` are written portably — they detect whether they are running on Kaggle or locally and pick the dataset path accordingly:

```python
# Auto-detected path (in every notebook header)
import os
if os.path.exists('/kaggle/input/paysim1'):
    DATA_PATH = '/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv'
else:
    DATA_PATH = './data/raw/PS_20174392719_1491204439457_log.csv'
```

To run on Kaggle:
1. Create a new notebook on kaggle.com
2. Add the `ealaxi/paysim1` dataset as input
3. Upload the `.ipynb` from `notebooks/` (or pull this repo via `!git clone` then `!pip install -e .`)
4. Enable GPU accelerator (Settings → Accelerator → GPU P100 or T4)

## API contract for fusion engine

**Endpoint:** `POST /api/graph/analyze`
**Response:**
```json
{
  "transaction_id": "TXN_12345",
  "relational_risk_score": 0.92,
  "risk_level": "CRITICAL",
  "suspicious_subgraph": {
    "nodes": ["C123", "C456", "C789"],
    "edges": [{"src": "C123", "dst": "C456", "weight": 0.87}],
    "sink_account": "C789",
    "pattern": "Hub-and-Spoke"
  }
}
```

Pydantic schemas live in [src/graphsage/api/schemas.py](src/graphsage/api/schemas.py) — locked early to prevent integration drift.



## Optimization principles

Cross-cutting practices applied across the codebase:

| Layer | Practice | Implementation |
|---|---|---|
| **Memory** | Optimal pandas dtypes (`category`, `int8`, `float32`) | `src/graphsage/data/ingestion.py` |
| **I/O** | Cache processed features as Parquet (~10× smaller than CSV); cache PyG graph as `.pt` (instant reload) | `data/processed/`, `data/graph/` |
| **Compute** | Vectorized DataFrame operations only — no `.iterrows()`. PyG `NeighborLoader` for batched training. | `src/graphsage/data/`, `src/graphsage/training/` |
| **Hardware** | Auto-detect CUDA / MPS / CPU at runtime | `src/graphsage/utils/` |
| **Reproducibility** | Single source of truth for hyperparameters in `configs/model_config.yaml` | All notebooks + scripts read from here |
| **Portability** | Pure Python + cross-platform paths (`pathlib`); no OS-specific commands inside the codebase | All modules |

## Results

Measured under the **leakage-free temporal protocol**: node features and message
passing use past-only edges (train ≤ step 600, val ≤ 700, test > 700), so no
future information reaches a prediction. Five seeds; mean ± std on the held-out
test snapshot.

| Stage | Configuration | test F1 | PR-AUC | vs baseline |
|---|---|---|---|---|
| 1 | GraphSAGE + BCE | 0.2806 ± 0.0867 | 0.2411 | — |
| 2 | + Edge-MLP attention | 0.2808 ± 0.0832 | 0.2172 | +0.0002 (p=0.998) |
| 3a | + Focal Loss | 0.3303 ± 0.0726 | 0.2838 | +0.0497 (p=0.488) |
| 3b | + Imbalance Sampler | 0.3944 ± 0.0077 | 0.3737 | **+0.1138 (p=0.046)** |
| **3c-v2** | **3b − Edge-MLP, 12-dim features** | **0.4056 ± 0.0026** | **0.4479** | **+0.1250 (p=0.045)** |

Significance is an across-seed paired t-test (n=5). Calibration: isotonic
regression brings ECE to **0.024** (raw focal sigmoids: 0.80). Inference latency
p95 **< 500 ms** per request, met by precomputing scores into a serving bundle.

### Two findings that revise the proposal

**1. The proposal's targets were set against a leaky protocol.** Section 3.5
projected F1 > 0.82 for the full system. Those figures came from a random
train/test split, where an account's future transactions can inform its own
prediction. Rebuilding the evaluation to be temporally honest cost roughly
0.20 F1 across every stage. The targets below are kept for traceability, but
the numbers above are the ones that mean anything — a 0.41 F1 with no leakage
is a stronger result than 0.82 with it.

| Stage | Proposal target F1 | Leakage-free actual |
|---|---|---|
| 1 — Baseline | ~0.55 | 0.2806 |
| 2 — + Edge MLP | ~0.74 | 0.2808 |
| 3 — Full system | > 0.82 | 0.4056 |

**2. The Edge-MLP (Novelty 1) does not improve accuracy.** Three independent
tests agree: Stage 2 vs 1 is null (p=0.998); with the 12-dim feature set,
removing it *improves* F1 (ΔF1 −0.0102, p=0.031) and PR-AUC (−0.0512, p=0.013).
It is retained as an **explainability mechanism** — the per-edge attention
weights are what let the extractor rank which transfers implicate an account —
at a measured cost of ~0.01 F1. It is not claimed as a predictive contribution.
See `docs/system_walkthrough.md` §14 for the full ablation.

Of the +0.125 total gain, **+0.097 comes from the 12-dim behavioural features**
and **+0.028 from the imbalance sampler**. The sampler also stabilises training
by 33× (std 0.087 → 0.003).


## Dataset

PaySim synthetic mobile money simulator — 6,362,620 transactions, fraud rate 0.1291% (773:1 imbalance). Fraud occurs only in TRANSFER and CASH_OUT transaction types. Fully synthetic — no PII. Source: Kaggle [`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1).

## License

Academic research. PyTorch / PyTorch Geometric / NetworkX / FastAPI used under their respective open-source licenses.

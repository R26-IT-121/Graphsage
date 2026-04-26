# DeepSentinel — Edge-Enhanced GraphSAGE Relational Fraud Detector

**Project:** — *A Cloud-Native Multi-Modal AI Platform for Explainable Financial Fraud Detection*
**Component:** Graph Neural Network intelligence layer
**Author:** Sachintha Bhashitha Ewaduge

This module is one of four independent deep-learning components of the **DeepSentinel** platform. It maps the banking ecosystem as a dynamic graph (accounts = nodes, transfers = directed edges) and uses an inductive Graph Neural Network to detect organized fraud rings that legacy rule-based systems miss. Output is a forensic JSON payload consumed by the downstream fusion engine.

## Three architectural novelties

1. **Edge-MLP attention** (`src/graphsage/models/layers.py`) — a custom MLP injected into the GraphSAGE message-passing step computes a dynamic attention weight per edge from `(amount, drain_ratio, src_drained, dst_was_empty, time_gap, txn_type)`. Suspicious edges dominate aggregation instead of being averaged away.
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

## Performance targets (proposal Section 3.5)

| Stage | F1 | Recall | Precision |
|---|---|---|---|
| Stage 1 — Baseline GraphSAGE | ~0.55 | — | — |
| Stage 2 — + Edge MLP | ~0.74 | — | — |
| Stage 3 — Full system (Edge MLP + Focal Loss + Sampler) | **> 0.82** | **> 0.80** | **> 0.85** |

End-to-end inference latency target: **< 500 ms per micro-batch**.

## Dataset

PaySim synthetic mobile money simulator — 6,362,620 transactions, fraud rate 0.1291% (773:1 imbalance). Fraud occurs only in TRANSFER and CASH_OUT transaction types. Fully synthetic — no PII. Source: Kaggle [`ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1).

## License

Academic research. PyTorch / PyTorch Geometric / NetworkX / FastAPI used under their respective open-source licenses.

"""DeepSentinel — Edge-Enhanced GraphSAGE Live Dashboard.

A polished Streamlit web UI demonstrating the trained Stage 3a model end-to-end:
- Pick a real fraud transaction from PaySim
- Run live inference
- Visualise the suspicious subgraph (k=2 hop, interactive Plotly)
- Show the forensic JSON sent to Member 4's Fusion Engine

Run from repo root:
    streamlit run dashboard/app.py

Opens at http://localhost:8501
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import torch
from torch_geometric.utils import k_hop_subgraph

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from graphsage.data.splits import make_time_split  # noqa: E402
from graphsage.extraction.pattern_classifier import classify_pattern  # noqa: E402
from graphsage.models.edge_sage import EdgeEnhancedGraphSAGE  # noqa: E402
from graphsage.training.threshold_tuning import (  # noqa: E402
    find_best_threshold_for_f1,
    metrics_at_threshold,
)

GRAPH_PATH = REPO / "data" / "graph" / "paysim_graph.pt"
CKPT_PATH = REPO / "checkpoints" / "stage3a_focal.pt"

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="DeepSentinel | GraphSAGE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — fintech aesthetic
# ============================================================
st.markdown(
    """
<style>
  /* --- Hide Streamlit defaults but KEEP sidebar toggle --- */
  #MainMenu, footer {visibility: hidden;}
  /* Keep header transparent so the collapse/expand arrow still works */
  header[data-testid="stHeader"] {
    background: rgba(10, 22, 40, 0.6) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    height: 2.5rem;
  }
  /* Style the sidebar toggle arrow to match dark theme */
  button[kind="header"], button[data-testid="baseButton-header"],
  button[data-testid="collapsedControl"] {
    background: rgba(59, 130, 246, 0.25) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(96, 165, 250, 0.4) !important;
    border-radius: 8px !important;
  }
  button[kind="header"]:hover, button[data-testid="collapsedControl"]:hover {
    background: rgba(59, 130, 246, 0.45) !important;
  }

  /* --- App-wide dark theme --- */
  .stApp {
    background:
      radial-gradient(circle at 15% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 35%),
      radial-gradient(circle at 85% 70%, rgba(168, 85, 247, 0.06) 0%, transparent 40%),
      radial-gradient(circle at 50% 90%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
      linear-gradient(180deg, #0a1628 0%, #0e1d34 100%);
    color: #e2e8f0;
  }

  /* All text default — bumped sizes */
  body, p, span, label, div { color: #e2e8f0; font-size: 15px; }
  h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
  .stMarkdown p { font-size: 15px; line-height: 1.6; }

  /* --- Animated network background --- */
  .network-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: 0;
    pointer-events: none;
    opacity: 0.55;
  }
  .network-bg svg { width: 100%; height: 100%; }
  .net-line {
    stroke: #3b82f6;
    stroke-width: 1;
    opacity: 0.12;
    stroke-dasharray: 5 6;
    animation: dash 24s linear infinite;
  }
  @keyframes dash { to { stroke-dashoffset: -250; } }
  .net-node {
    fill: #60a5fa;
    filter: drop-shadow(0 0 6px rgba(96, 165, 250, 0.7));
    animation: pulseNode 4s ease-in-out infinite;
  }
  .net-node.alt { fill: #c084fc; filter: drop-shadow(0 0 6px rgba(192, 132, 252, 0.7)); animation-delay: 1.5s; }
  .net-node.warm { fill: #fb7185; filter: drop-shadow(0 0 6px rgba(251, 113, 133, 0.7)); animation-delay: 2.5s; }
  @keyframes pulseNode {
    0%, 100% { opacity: 0.3; transform: scale(1); transform-origin: center; }
    50% { opacity: 1; transform: scale(1.4); }
  }

  /* Make Streamlit's main content sit above the bg */
  section.main, div.block-container { position: relative; z-index: 1; }

  /* --- Brand header --- */
  .brand-header {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(168, 85, 247, 0.18) 100%);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: white;
    padding: 20px 28px;
    border-radius: 16px;
    margin-bottom: 22px;
    border: 1px solid rgba(96, 165, 250, 0.25);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), 0 0 40px rgba(59, 130, 246, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .brand-header h1 {
    color: #f8fafc !important;
    margin: 0 !important;
    font-size: 28px !important;
    font-weight: 700;
    letter-spacing: -0.3px;
    text-shadow: 0 1px 12px rgba(59, 130, 246, 0.4);
  }
  .brand-header .subtitle {
    color: rgba(226, 232, 240, 0.85);
    font-size: 14px;
    margin-top: 6px;
    font-weight: 400;
  }
  .brand-tag {
    background: rgba(59, 130, 246, 0.25);
    color: #93c5fd;
    padding: 9px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    border: 1px solid rgba(96, 165, 250, 0.35);
  }

  /* --- Glass metric cards --- */
  .metric-card {
    background: rgba(20, 35, 60, 0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(96, 165, 250, 0.15);
    height: 100%;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), 0 0 24px rgba(59, 130, 246, 0.15);
    border-color: rgba(96, 165, 250, 0.35);
  }
  .metric-card .label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 8px;
  }
  .metric-card .value {
    font-size: 36px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
  }
  .metric-card .sub {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 6px;
  }

  /* --- Risk pill --- */
  .risk-badge {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 24px;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: white;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
  }
  .risk-CRITICAL { background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%); box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4), 0 0 24px rgba(239, 68, 68, 0.25); }
  .risk-HIGH     { background: linear-gradient(135deg, #f97316 0%, #c2410c 100%); box-shadow: 0 4px 14px rgba(249, 115, 22, 0.4); }
  .risk-MEDIUM   { background: linear-gradient(135deg, #f59e0b 0%, #b45309 100%); color: #fff8e1; box-shadow: 0 4px 14px rgba(245, 158, 11, 0.4); }
  .risk-LOW      { background: linear-gradient(135deg, #10b981 0%, #047857 100%); box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4); }

  /* --- Section header --- */
  .section-header {
    color: #93c5fd;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin: 26px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(96, 165, 250, 0.2);
  }

  /* --- Sidebar polish --- */
  section[data-testid="stSidebar"] {
    background: rgba(8, 17, 33, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-right: 1px solid rgba(96, 165, 250, 0.15);
  }
  section[data-testid="stSidebar"] * { color: #cbd5e1; }
  .sidebar-card {
    background: rgba(20, 35, 60, 0.6);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
    border: 1px solid rgba(96, 165, 250, 0.15);
  }
  .sidebar-card .title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #93c5fd;
    font-weight: 700;
    margin-bottom: 10px;
  }
  .sidebar-metric {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 15px;
  }
  .sidebar-metric .k { color: #94a3b8; }
  .sidebar-metric .v { font-weight: 700; color: #60a5fa; }

  /* --- Evidence cards --- */
  .ev-card {
    background: rgba(20, 35, 60, 0.55);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    border-left: 3px solid #60a5fa;
    border-top: 1px solid rgba(96, 165, 250, 0.1);
    border-right: 1px solid rgba(96, 165, 250, 0.1);
    border-bottom: 1px solid rgba(96, 165, 250, 0.1);
    margin-bottom: 10px;
    transition: border-left-color 0.2s, transform 0.2s;
  }
  .ev-card:hover {
    border-left-color: #c084fc;
    transform: translateX(2px);
  }
  .ev-card .k {
    font-size: 13px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
  }
  .ev-card .v {
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 4px;
  }

  /* --- Plotly wrapper --- */
  .stPlotlyChart {
    background: rgba(15, 25, 45, 0.55) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(96, 165, 250, 0.15);
  }

  /* --- Footer cards --- */
  .footer-card {
    background: rgba(20, 35, 60, 0.55);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    border-top: 3px solid #60a5fa;
    border-left: 1px solid rgba(96, 165, 250, 0.1);
    border-right: 1px solid rgba(96, 165, 250, 0.1);
    border-bottom: 1px solid rgba(96, 165, 250, 0.1);
    height: 100%;
  }
  .footer-card h4 {
    color: #93c5fd !important;
    font-size: 15px;
    margin: 0 0 10px 0;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }
  .footer-card p {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
    margin: 0;
  }

  /* --- Streamlit native components dark tweaks --- */
  div.stExpander {
    background: rgba(20, 35, 60, 0.55) !important;
    border-radius: 12px;
    border: 1px solid rgba(96, 165, 250, 0.15);
  }
  .stExpander details summary { color: #e2e8f0 !important; }
  pre code, code { color: #93c5fd !important; }
  div[data-testid="stCodeBlock"] {
    background: rgba(8, 15, 28, 0.7) !important;
    border-radius: 8px;
    border: 1px solid rgba(96, 165, 250, 0.15);
  }
  .stRadio label, .stRadio div, .stSlider label { color: #cbd5e1 !important; }
  button[kind="primary"], button[kind="secondary"] {
    background: rgba(59, 130, 246, 0.2) !important;
    color: #93c5fd !important;
    border: 1px solid rgba(96, 165, 250, 0.35) !important;
  }
  button[kind="primary"]:hover, button[kind="secondary"]:hover {
    background: rgba(59, 130, 246, 0.35) !important;
  }

  /* Tighten block spacing */
  div.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

  hr { border-color: rgba(96, 165, 250, 0.15) !important; }
</style>

<div class="network-bg">
  <svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice">
    <!-- Background connecting lines -->
    <line x1="120" y1="100" x2="380" y2="220" class="net-line" />
    <line x1="380" y1="220" x2="600" y2="140" class="net-line" />
    <line x1="600" y1="140" x2="850" y2="280" class="net-line" />
    <line x1="850" y1="280" x2="1100" y2="180" class="net-line" />
    <line x1="1100" y1="180" x2="1350" y2="260" class="net-line" />
    <line x1="200" y1="400" x2="450" y2="500" class="net-line" />
    <line x1="450" y1="500" x2="700" y2="420" class="net-line" />
    <line x1="700" y1="420" x2="950" y2="540" class="net-line" />
    <line x1="950" y1="540" x2="1200" y2="450" class="net-line" />
    <line x1="1200" y1="450" x2="1450" y2="540" class="net-line" />
    <line x1="150" y1="700" x2="400" y2="780" class="net-line" />
    <line x1="400" y1="780" x2="650" y2="690" class="net-line" />
    <line x1="650" y1="690" x2="900" y2="810" class="net-line" />
    <line x1="900" y1="810" x2="1150" y2="730" class="net-line" />
    <line x1="1150" y1="730" x2="1400" y2="800" class="net-line" />
    <line x1="380" y1="220" x2="450" y2="500" class="net-line" />
    <line x1="850" y1="280" x2="950" y2="540" class="net-line" />
    <line x1="600" y1="140" x2="700" y2="420" class="net-line" />
    <line x1="450" y1="500" x2="400" y2="780" class="net-line" />
    <line x1="950" y1="540" x2="900" y2="810" class="net-line" />
    <line x1="1200" y1="450" x2="1150" y2="730" class="net-line" />

    <!-- Pulsing nodes -->
    <circle cx="120" cy="100" r="4" class="net-node" />
    <circle cx="380" cy="220" r="5" class="net-node alt" />
    <circle cx="600" cy="140" r="4" class="net-node" />
    <circle cx="850" cy="280" r="5" class="net-node warm" />
    <circle cx="1100" cy="180" r="4" class="net-node" />
    <circle cx="1350" cy="260" r="5" class="net-node alt" />
    <circle cx="200" cy="400" r="4" class="net-node" />
    <circle cx="450" cy="500" r="5" class="net-node" />
    <circle cx="700" cy="420" r="4" class="net-node alt" />
    <circle cx="950" cy="540" r="5" class="net-node warm" />
    <circle cx="1200" cy="450" r="4" class="net-node" />
    <circle cx="1450" cy="540" r="5" class="net-node" />
    <circle cx="150" cy="700" r="4" class="net-node alt" />
    <circle cx="400" cy="780" r="5" class="net-node" />
    <circle cx="650" cy="690" r="4" class="net-node" />
    <circle cx="900" cy="810" r="5" class="net-node warm" />
    <circle cx="1150" cy="730" r="4" class="net-node" />
    <circle cx="1400" cy="800" r="5" class="net-node alt" />
  </svg>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource(show_spinner="Loading PaySim graph…")
def load_graph():
    data = torch.load(GRAPH_PATH, weights_only=False)
    # Self-heal: apply time-based splits if the saved graph predates make_splits.py
    if not hasattr(data, "val_mask"):
        data, _ = make_time_split(data, train_end=600, val_end=700)
    return data


@st.cache_resource(show_spinner="Loading Stage 3a model…")
def load_model(in_dim: int, edge_dim: int):
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location="cpu")
    hp = ckpt.get("hyperparameters", {})
    model = EdgeEnhancedGraphSAGE(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hp.get("hidden_dim", 64),
        edge_mlp_hidden=hp.get("edge_mlp_hidden", 32),
        dropout=hp.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_resource(show_spinner="Running inference on full graph…")
def run_full_inference(_model, _data):
    with torch.no_grad():
        logits, attentions = _model.forward_with_attention(
            _data.x, _data.edge_index, _data.edge_attr
        )
    probs = torch.sigmoid(logits)
    best_thresh, val_f1 = find_best_threshold_for_f1(
        logits[_data.val_mask], _data.y[_data.val_mask]
    )
    return logits, probs, attentions, float(best_thresh), float(val_f1)


@st.cache_resource(show_spinner="Computing test metrics…")
def get_test_metrics(_logits, _data, _thresh):
    return metrics_at_threshold(
        _logits[_data.test_mask], _data.y[_data.test_mask], _thresh
    )


# ============================================================
# Load everything
# ============================================================
if not GRAPH_PATH.exists():
    st.error(f"Graph not found at {GRAPH_PATH}")
    st.stop()
if not CKPT_PATH.exists():
    st.error(f"Checkpoint not found at {CKPT_PATH}")
    st.stop()

data = load_graph()
model, ckpt = load_model(int(data.x.shape[1]), int(data.edge_attr.shape[1]))
logits, probs, attentions, best_thresh, val_f1 = run_full_inference(model, data)
test_metrics = get_test_metrics(logits, data, best_thresh)
test_mules = (data.test_mask & data.y.bool()).nonzero(as_tuple=True)[0]


@st.cache_resource(show_spinner="Indexing rich mule neighborhoods…")
def get_rich_mules(_data, _test_mules):
    """All mules (not just test) ranked by in-degree.

    Test fraud nodes are often fresh senders (degree 1) which produce
    tiny subgraphs. Use the in_degree column (x[:, 0]) to find mules
    with the most connections for visually rich demos.
    """
    all_mules = _data.y.bool().nonzero(as_tuple=True)[0]
    in_degrees = _data.x[all_mules, 0]
    order = in_degrees.argsort(descending=True)
    return all_mules[order]


rich_mules = get_rich_mules(data, test_mules)


# ============================================================
# Sidebar
# ============================================================
st.sidebar.markdown(
    """
<div style='padding:12px 0 6px 0;'>
  <div style='font-size:20px; font-weight:800; color:#1a4d8f; letter-spacing:-0.5px;'>
    🛡️ DeepSentinel
  </div>
  <div style='font-size:11px; color:#6c7686; margin-top:2px;'>
    Edge-Enhanced GraphSAGE · Member 1
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# Model performance card
st.sidebar.markdown(
    f"""
<div class='sidebar-card'>
  <div class='title'>Model · Stage 3a</div>
  <div class='sidebar-metric'><span class='k'>Test F1</span><span class='v'>{test_metrics['f1']:.4f}</span></div>
  <div class='sidebar-metric'><span class='k'>AUROC</span><span class='v'>{test_metrics['auroc']:.4f}</span></div>
  <div class='sidebar-metric'><span class='k'>Recall</span><span class='v'>{test_metrics['recall']:.4f}</span></div>
  <div class='sidebar-metric'><span class='k'>Precision</span><span class='v'>{test_metrics['precision']:.4f}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# Graph stats card
st.sidebar.markdown(
    f"""
<div class='sidebar-card'>
  <div class='title'>Graph statistics</div>
  <div class='sidebar-metric'><span class='k'>Nodes</span><span class='v'>{data.num_nodes:,}</span></div>
  <div class='sidebar-metric'><span class='k'>Edges</span><span class='v'>{data.edge_index.shape[1]:,}</span></div>
  <div class='sidebar-metric'><span class='k'>Mules</span><span class='v'>{int(data.y.sum().item()):,}</span></div>
  <div class='sidebar-metric'><span class='k'>Test mules</span><span class='v'>{len(test_mules):,}</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# Transaction selector
st.sidebar.markdown("##### Select transaction")
mode = st.sidebar.radio(
    " ",
    ["Richest fraud rings", "Top by confidence", "Random test mule", "Specific node ID"],
    index=0,
    label_visibility="collapsed",
    help=(
        "Richest fraud rings = mules with the most connections "
        "(best for visual demos). "
        "Top by confidence = highest model-probability test fraud."
    ),
)

if mode == "Richest fraud rings":
    rank = st.sidebar.slider("Rank by neighborhood size", 1, 30, 1)
    selected_node = int(rich_mules[rank - 1].item())
    in_deg_preview = int(data.x[selected_node, 0].item())
    st.sidebar.markdown(
        f"<div style='font-size:12px; color:#94a3b8; margin-top:6px;'>"
        f"Mule with <b style='color:#60a5fa;'>{in_deg_preview}</b> incoming edges</div>",
        unsafe_allow_html=True,
    )
elif mode == "Top by confidence":
    top_k = st.sidebar.slider("Rank (top-K confidence)", 1, 20, 1)
    test_mule_probs = probs[test_mules]
    sorted_idx = test_mule_probs.argsort(descending=True)
    selected_node = int(test_mules[sorted_idx[top_k - 1]].item())
elif mode == "Random test mule":
    if st.sidebar.button("🎲 Pick another", use_container_width=True):
        st.session_state.rand_offset = (
            st.session_state.get("rand_offset", 0) + 1
        ) % len(test_mules)
    offset = st.session_state.get("rand_offset", 0)
    selected_node = int(test_mules[offset].item())
else:
    selected_node = st.sidebar.number_input(
        "Node ID",
        min_value=0,
        max_value=int(data.num_nodes) - 1,
        value=int(rich_mules[0].item()),
    )


# ============================================================
# Main content
# ============================================================
# Brand header
st.markdown(
    """
<div class='brand-header'>
  <div>
    <h1>🛡️ DeepSentinel — Relational Fraud Detection</h1>
    <div class='subtitle'>
      Live inference · Edge-Enhanced GraphSAGE · Trained on PaySim (6.36M transactions, 773:1 imbalance)
    </div>
  </div>
  <div class='brand-tag'>Stage 3a · Live</div>
</div>
""",
    unsafe_allow_html=True,
)

# Compute current prediction
prob = float(probs[selected_node].item())
ground_truth_mule = bool(int(data.y[selected_node].item()))
confidence = max(prob, 1 - prob)

if prob >= 0.9:
    risk_level = "CRITICAL"
elif prob >= best_thresh:
    risk_level = "HIGH"
elif prob >= 0.5:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

# ============================================================
# Hero row: 4 metric cards
# ============================================================
hero_a, hero_b, hero_c, hero_d = st.columns([1.3, 1, 1, 1])

with hero_a:
    # Plotly gauge (dark theme)
    gauge_color = (
        "#ef4444" if risk_level == "CRITICAL"
        else "#f97316" if risk_level == "HIGH"
        else "#f59e0b" if risk_level == "MEDIUM"
        else "#10b981"
    )
    fig_g = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"font": {"size": 36, "color": "#f1f5f9"}, "valueformat": ".4f"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#94a3b8",
                         "tickfont": {"color": "#94a3b8", "size": 10}},
                "bar": {"color": gauge_color, "thickness": 0.7},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 0.5], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [0.5, best_thresh], "color": "rgba(245, 158, 11, 0.20)"},
                    {"range": [best_thresh, 0.9], "color": "rgba(249, 115, 22, 0.22)"},
                    {"range": [0.9, 1], "color": "rgba(239, 68, 68, 0.30)"},
                ],
                "threshold": {
                    "line": {"color": "#93c5fd", "width": 3},
                    "thickness": 0.85,
                    "value": best_thresh,
                },
            },
        )
    )
    fig_g.update_layout(
        height=210, margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    st.markdown("<div class='metric-card' style='padding:10px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        "<div style='text-align:center; font-size:11px; color:#94a3b8; margin-top:-12px; "
        "letter-spacing:0.8px;'>RELATIONAL RISK SCORE</div></div>",
        unsafe_allow_html=True,
    )

with hero_b:
    st.markdown(
        f"""
<div class='metric-card' style='text-align:center;'>
  <div class='label'>Risk Level</div>
  <div style='margin: 22px 0 18px 0;'>
    <span class='risk-badge risk-{risk_level}'>{risk_level}</span>
  </div>
  <div class='sub'>Confidence: <b>{confidence:.4f}</b></div>
  <div class='sub' style='margin-top:4px;'>
    Threshold: <b>{best_thresh:.4f}</b> ·
    {'<span style="color:#c0392b;">▲ above</span>' if prob >= best_thresh
     else '<span style="color:#27ae60;">▼ below</span>'}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with hero_c:
    gt_color = "#c0392b" if ground_truth_mule else "#27ae60"
    gt_text = "MULE" if ground_truth_mule else "LEGIT"
    correct = (prob >= best_thresh) == ground_truth_mule
    correct_icon = "✓ Correct prediction" if correct else "✗ Mismatch"
    st.markdown(
        f"""
<div class='metric-card' style='text-align:center;'>
  <div class='label'>Ground Truth</div>
  <div class='value' style='color:{gt_color}; margin-top:8px;'>{gt_text}</div>
  <div class='sub' style='margin-top:10px;'>
    Test-set labelled
  </div>
  <div class='sub' style='margin-top:8px; color:{"#27ae60" if correct else "#c0392b"}; font-weight:600;'>
    {correct_icon}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with hero_d:
    st.markdown(
        f"""
<div class='metric-card' style='text-align:center;'>
  <div class='label'>Selected Account</div>
  <div class='value' style='font-size:22px; margin-top:8px;'>#{selected_node:,}</div>
  <div class='sub' style='margin-top:10px;'>Node ID in graph</div>
  <div class='sub' style='margin-top:6px;'>Model: <b>{ckpt.get("loss_class", "FocalLoss")}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# Subgraph extraction
# ============================================================
with st.spinner("Extracting k=2 hop suspicious subgraph…"):
    subset, sub_ei, _, edge_mask = k_hop_subgraph(
        node_idx=selected_node,
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
    )
    sub_attentions = attentions[1][edge_mask]
    sub_y = data.y[subset]
    sub_probs = probs[subset]
    flagged_local = (subset == selected_node).nonzero(as_tuple=True)[0].item()
    subset_list = subset.tolist()

# Classify FATF typology of this subgraph (Novelty 3 sub-component)
pattern_result = classify_pattern(
    sub_edge_index=sub_ei,
    sub_edge_attr=data.edge_attr[edge_mask],
    flagged_local=flagged_local,
    num_sub_nodes=int(len(subset)),
)

PATTERN_COLORS = {
    "HUB_AND_SPOKE": "#ef4444",
    "SMURFING": "#fb923c",
    "LAYERING": "#a855f7",
    "ACCOUNT_TAKEOVER": "#facc15",
}
_pat_color = PATTERN_COLORS.get(pattern_result.pattern, "#60a5fa")


# ============================================================
# Two-column: interactive graph + structural evidence
# ============================================================
st.markdown(
    f"""
<div style='display:flex; justify-content:space-between; align-items:center;
            margin: 26px 0 16px 0; padding-bottom: 10px;
            border-bottom: 1px solid rgba(96, 165, 250, 0.2);'>
  <div style='color:#93c5fd; font-size:17px; font-weight:700;
              letter-spacing:1.3px; text-transform:uppercase;'>
    🕸️ Suspicious Subgraph · Novelty 3
  </div>
  <div style='display:flex; gap:10px; align-items:center;'>
    <span style='font-size:12px; color:#94a3b8; text-transform:uppercase;
                 letter-spacing:0.6px;'>FATF typology</span>
    <span style='background: rgba(20,35,60,0.85);
                 border: 1.5px solid {_pat_color};
                 color: {_pat_color}; padding: 6px 14px; border-radius: 8px;
                 font-size: 13px; font-weight: 700; letter-spacing: 0.6px;'>
      {pattern_result.pattern}
      <span style='color:#94a3b8; font-weight:500; margin-left:6px;'>
        · {pattern_result.confidence:.2f}
      </span>
    </span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

graph_col, ev_col = st.columns([2.2, 1])

with graph_col:
    # Top-N filter for clarity
    MAX_NODES = 30
    if len(subset) > MAX_NODES:
        top_idx = sub_attentions.argsort(descending=True)[:MAX_NODES]
        keep_edges = sub_ei[:, top_idx]
        keep_nodes_set = set(keep_edges[0].tolist()) | set(keep_edges[1].tolist())
        keep_nodes_set.add(flagged_local)
        kept_mask = torch.tensor(
            [(int(s) in keep_nodes_set) and (int(d) in keep_nodes_set)
             for s, d in sub_ei.t()]
        )
        plot_ei = sub_ei[:, kept_mask]
        plot_attn = sub_attentions[kept_mask]
    else:
        plot_ei = sub_ei
        plot_attn = sub_attentions
        keep_nodes_set = set(range(len(subset)))

    G = nx.DiGraph()
    for i in keep_nodes_set:
        G.add_node(i, is_mule=int(sub_y[i].item()), prob=float(sub_probs[i].item()))
    for col, (s, d) in enumerate(plot_ei.t().tolist()):
        G.add_edge(s, d, weight=float(plot_attn[col].item()))

    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Per-edge traces for individual hover + styling (dark theme)
    edge_traces = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = G.edges[u, v]["weight"]
        # Color gradient by attention weight: low = subtle blue, high = vivid red
        if weight > 0.7:
            ec = "#ef4444"
        elif weight > 0.4:
            ec = "#fb923c"
        else:
            ec = "rgba(148, 163, 184, 0.35)"
        ew = 1.0 + 6 * weight
        real_src = subset_list[u]
        real_dst = subset_list[v]
        hover = (
            f"<b>{real_src} → {real_dst}</b><br>"
            f"Edge-MLP attention: <b>{weight:.4f}</b><br>"
            f"<i>Novelty 1 — higher = more suspicious</i>"
        )
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=ew, color=ec),
                mode="lines",
                hoverinfo="text",
                hovertext=hover,
                showlegend=False,
                opacity=0.9 if weight > 0.4 else 0.5,
            )
        )

    node_x, node_y = [], []
    node_color_arr, node_size_arr = [], []
    node_hover, node_label = [], []

    for n in G.nodes():
        node_x.append(pos[n][0])
        node_y.append(pos[n][1])
        is_mule = bool(G.nodes[n]["is_mule"])
        nprob = G.nodes[n]["prob"]
        real_id = subset_list[n]

        if n == flagged_local:
            c, s, role = "#ef4444", 38, "FLAGGED MULE (sink)"
        elif is_mule:
            c, s, role = "#fb7185", 24, "Confirmed mule"
        elif nprob > 0.5:
            c, s, role = "#fbbf24", 18, "Predicted fraud"
        else:
            c, s, role = "#60a5fa", 13, "Legitimate"

        node_color_arr.append(c)
        node_size_arr.append(s)
        node_label.append(str(real_id))
        node_hover.append(
            f"<b>Account {real_id}</b><br>"
            f"Role: <b>{role}</b><br>"
            f"Risk score: <b>{nprob:.4f}</b><br>"
            f"Ground truth: <b>{'MULE' if is_mule else 'legit'}</b>"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_hover,
        text=node_label,
        textposition="bottom center",
        textfont=dict(size=11, color="#e2e8f0"),
        marker=dict(
            size=node_size_arr,
            color=node_color_arr,
            line=dict(width=2, color="rgba(15, 25, 45, 0.9)"),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        hoverlabel=dict(
            bgcolor="rgba(15, 25, 45, 0.95)",
            bordercolor="#60a5fa",
            font=dict(color="#f1f5f9", size=12),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "<div style='display:flex; gap:22px; flex-wrap:wrap; font-size:13px; "
        "color:#cbd5e1; margin-top:10px; padding:12px 16px; "
        "background:rgba(20,35,60,0.5); border-radius:10px; "
        "border:1px solid rgba(96,165,250,0.15);'>"
        "<span><span style='color:#ef4444; font-size:18px;'>●</span> Flagged mule</span>"
        "<span><span style='color:#fb7185; font-size:18px;'>●</span> Confirmed mule</span>"
        "<span><span style='color:#fbbf24; font-size:18px;'>●</span> Predicted fraud</span>"
        "<span><span style='color:#60a5fa; font-size:18px;'>●</span> Legitimate</span>"
        "<span style='color:#94a3b8;'>· Edge thickness/color = attention weight</span>"
        "</div>",
        unsafe_allow_html=True,
    )

with ev_col:
    # Structural evidence cards
    in_deg = int((sub_ei[1] == flagged_local).sum().item())
    out_deg = int((sub_ei[0] == flagged_local).sum().item())
    fresh_count = sum(
        1 for n in range(len(subset))
        if int(data.x[subset[n], 1].item()) <= 1
    )
    fresh_ratio = fresh_count / max(len(subset), 1)
    mean_drain = float(data.edge_attr[edge_mask][:, 1].mean().item())
    mules_count = int(sub_y.sum().item())

    st.markdown(
        f"""
<div class='ev-card'>
  <div class='k'>Convergence (in-degree)</div>
  <div class='v'>{in_deg}</div>
</div>
<div class='ev-card'>
  <div class='k'>Mean drain ratio</div>
  <div class='v'>{mean_drain:.3f}</div>
</div>
<div class='ev-card'>
  <div class='k'>Mules in subgraph</div>
  <div class='v'>{mules_count}</div>
</div>
<div class='ev-card'>
  <div class='k'>Fresh-sender ratio</div>
  <div class='v'>{fresh_ratio:.3f}</div>
</div>
<div class='ev-card'>
  <div class='k'>Subgraph size</div>
  <div class='v'>{len(subset):,} <span style='font-size:13px; color:#6c7686; font-weight:400;'>nodes</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# JSON output (collapsible)
# ============================================================
response = {
    "transaction_id": f"TX_DEMO_{selected_node}",
    "model_version": "graphsage-edge-mlp-focal-v0.3.0",
    "stage": "stage_3a_focal",
    "relational_risk_score": round(prob, 4),
    "risk_level": risk_level,
    "confidence": round(confidence, 4),
    "suspicious_subgraph": {
        "k_hop": 2,
        "node_count": int(len(subset)),
        "edge_count": int(sub_ei.shape[1]),
        "sink_account": f"NODE_{selected_node}",
        "pattern": pattern_result.pattern,
        "pattern_confidence": pattern_result.confidence,
        "pattern_scores": pattern_result.scores,
        "structural_evidence": {
            **pattern_result.evidence,
            "fresh_sender_ratio": round(fresh_ratio, 3),
            "mules_in_subgraph": mules_count,
        },
    },
    "metadata": {
        "tuned_threshold": round(best_thresh, 4),
        "extraction_method": "k_hop_subgraph_pyg",
        "pattern_classifier": "rule_based_fatf_v1",
    },
}

with st.expander("📋  Forensic JSON sent to Member 4's Fusion Engine", expanded=False):
    jc1, jc2 = st.columns([4, 1])
    with jc1:
        st.caption(
            "This payload is consumed by Member 4's RAG-grounded LLM. "
            "`pattern` keys into the FATF typology vector store; "
            "`structural_evidence` are the facts cited in the Chain-of-Evidence prompt."
        )
    with jc2:
        st.download_button(
            label="⬇️ Download",
            data=json.dumps(response, indent=2),
            file_name=f"fraud_alert_{selected_node}.json",
            mime="application/json",
            use_container_width=True,
        )
    st.code(json.dumps(response, indent=2), language="json")


# ============================================================
# Live Demo — Submit a synthetic transaction
# ============================================================
st.markdown(
    "<div class='section-header'>🧪 Live Demo — Submit a Transaction</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#94a3b8; font-size:14px; margin-top:-8px;'>"
    "Configure a synthetic transaction below. DeepSentinel injects it into the "
    "receiver's local k=2 neighborhood, re-runs Edge-Enhanced GraphSAGE inference, "
    "and shows how the receiver's risk score changes."
    "</p>",
    unsafe_allow_html=True,
)

with st.form("custom_txn_form"):
    form_a, form_b = st.columns(2)
    with form_a:
        sender_in = st.number_input(
            "Sender account (node ID)",
            min_value=0, max_value=int(data.num_nodes) - 1,
            value=1000, step=1,
            help="Any account in the graph (0–3.27M).",
        )
        receiver_in = st.number_input(
            "Receiver account (node ID)",
            min_value=0, max_value=int(data.num_nodes) - 1,
            value=int(rich_mules[0].item()), step=1,
            help="The account being analysed. Defaults to a known mule.",
        )
        amount_in = st.number_input(
            "Transaction amount ($)",
            min_value=1.0, max_value=10_000_000.0,
            value=50_000.0, step=1_000.0,
        )
    with form_b:
        type_in = st.radio(
            "Transaction type",
            ["TRANSFER", "CASH_OUT"], horizontal=True,
            help="TRANSFER = bank-to-bank. CASH_OUT = ATM/merchant withdrawal.",
        )
        scenario_in = st.radio(
            "Scenario",
            [
                "Suspicious (full drain → empty receiver)",
                "Routine (partial transfer)",
            ],
            help="Drives drain_ratio, src_drained and dst_was_empty edge features.",
        )
        step_in = st.number_input(
            "Step (simulated time)",
            min_value=1, max_value=800,
            value=720, step=1,
        )
    submit_btn = st.form_submit_button(
        "🔍  Analyze this transaction",
        type="primary", use_container_width=True,
    )

if submit_btn:
    # 1. Derive the 6 edge features from the form
    amount_log_v = math.log1p(float(amount_in))
    if "Suspicious" in scenario_in:
        drain_v, src_drained_v, dst_empty_v, time_gap_v = 1.0, 1.0, 1.0, 1.0
    else:
        drain_v, src_drained_v, dst_empty_v, time_gap_v = 0.2, 0.0, 0.0, 24.0
    type_xfer_v = 1.0 if type_in == "TRANSFER" else 0.0
    new_feat = torch.tensor(
        [[amount_log_v, drain_v, src_drained_v, dst_empty_v, time_gap_v, type_xfer_v]],
        dtype=torch.float,
    )

    rec_id = int(receiver_in)
    snd_id = int(sender_in)

    # 2. Pull the receiver's k=2 neighbourhood
    sub_set, sub_ei_mod, _, sub_emask_mod = k_hop_subgraph(
        node_idx=rec_id, num_hops=2, edge_index=data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    sub_set_list = sub_set.tolist()
    sub_x_mod = data.x[sub_set].clone()
    sub_ea_mod = data.edge_attr[sub_emask_mod].clone()

    # 3. Map / append the sender
    if snd_id in sub_set_list:
        snd_local = sub_set_list.index(snd_id)
        sender_was_new = False
    else:
        snd_local = len(sub_set_list)
        sub_x_mod = torch.cat([sub_x_mod, data.x[snd_id].unsqueeze(0)], dim=0)
        sender_was_new = True
    rec_local = sub_set_list.index(rec_id)

    # 4. Inject the synthetic edge
    new_edge = torch.tensor([[snd_local], [rec_local]], dtype=torch.long)
    mod_ei = torch.cat([sub_ei_mod, new_edge], dim=1)
    mod_ea = torch.cat([sub_ea_mod, new_feat], dim=0)

    # 5. Re-run inference on the modified local graph
    with torch.no_grad():
        mod_logits, mod_attn = model.forward_with_attention(sub_x_mod, mod_ei, mod_ea)
    mod_prob = float(torch.sigmoid(mod_logits[rec_local]).item())
    new_edge_attn = float(mod_attn[1][-1].item())

    # 5b. Classify the FATF typology of the modified subgraph
    demo_pattern = classify_pattern(
        sub_edge_index=mod_ei,
        sub_edge_attr=mod_ea,
        flagged_local=rec_local,
        num_sub_nodes=int(sub_x_mod.shape[0]),
    )

    # 6. Compare to the pre-injection prediction
    prior_prob = float(probs[rec_id].item())
    delta = mod_prob - prior_prob
    new_risk = (
        "CRITICAL" if mod_prob >= 0.9
        else "HIGH" if mod_prob >= best_thresh
        else "MEDIUM" if mod_prob >= 0.5
        else "LOW"
    )
    delta_color = "#ef4444" if delta > 0 else "#10b981"
    after_color = "#ef4444" if mod_prob >= best_thresh else "#10b981"
    demo_pat_color = PATTERN_COLORS.get(demo_pattern.pattern, "#60a5fa")

    st.markdown(
        f"""
<div style='background:rgba(20,35,60,0.55); padding:20px 24px; border-radius:14px;
            border:1px solid rgba(96,165,250,0.25); margin-top:12px;
            box-shadow:0 4px 18px rgba(0,0,0,0.3);'>
  <div style='font-size:13px; color:#93c5fd; text-transform:uppercase; letter-spacing:0.8px;
              margin-bottom:14px; font-weight:700;'>Inference Result</div>
  <div style='display:flex; gap:32px; flex-wrap:wrap; align-items:flex-end;'>
    <div>
      <div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.6px;'>Risk · before</div>
      <div style='font-size:32px; font-weight:700; color:#cbd5e1;'>{prior_prob:.4f}</div>
    </div>
    <div style='font-size:24px; color:#64748b; padding-bottom:6px;'>→</div>
    <div>
      <div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.6px;'>Risk · after</div>
      <div style='font-size:32px; font-weight:700; color:{after_color};'>{mod_prob:.4f}</div>
    </div>
    <div>
      <div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.6px;'>Δ Change</div>
      <div style='font-size:32px; font-weight:700; color:{delta_color};'>
        {"+" if delta >= 0 else ""}{delta:.4f}
      </div>
    </div>
    <div>
      <div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.6px;'>New risk level</div>
      <div style='margin-top:6px;'><span class='risk-badge risk-{new_risk}'>{new_risk}</span></div>
    </div>
    <div>
      <div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.6px;'>Edge-MLP attention<br/>on new edge</div>
      <div style='font-size:32px; font-weight:700; color:#fb923c;'>{new_edge_attn:.4f}</div>
    </div>
    <div>
      <div style='font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.6px;'>FATF typology<br/>(post-injection)</div>
      <div style='margin-top:6px;'>
        <span style='background: rgba(20,35,60,0.85);
                     border: 1.5px solid {demo_pat_color};
                     color: {demo_pat_color}; padding: 6px 14px; border-radius: 8px;
                     font-size: 13px; font-weight: 700; letter-spacing: 0.6px;'>
          {demo_pattern.pattern}
          <span style='color:#94a3b8; font-weight:500; margin-left:6px;'>
            · {demo_pattern.confidence:.2f}
          </span>
        </span>
      </div>
    </div>
  </div>
  <div style='margin-top:14px; font-size:13px; color:#94a3b8;'>
    {("Sender was added as a new node in the receiver's neighbourhood." if sender_was_new
      else "Sender already existed in the receiver's k=2 neighbourhood.")}
    Local inference covered <b style='color:#60a5fa;'>{sub_x_mod.shape[0]:,}</b> nodes
    and <b style='color:#60a5fa;'>{mod_ei.shape[1]:,}</b> edges (incl. the new one).
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    demo_payload = {
        "transaction_id": f"TX_LIVE_{snd_id}_{rec_id}_{int(step_in)}",
        "model_version": "graphsage-edge-mlp-focal-v0.3.0",
        "stage": "stage_3a_focal",
        "submitted_transaction": {
            "sender_account": f"NODE_{snd_id}",
            "receiver_account": f"NODE_{rec_id}",
            "amount": float(amount_in),
            "type": type_in,
            "step": int(step_in),
            "scenario": scenario_in,
            "edge_features": {
                "amount_log": round(amount_log_v, 4),
                "drain_ratio": round(drain_v, 3),
                "src_drained": int(src_drained_v),
                "dst_was_empty": int(dst_empty_v),
                "time_gap": round(time_gap_v, 2),
                "type_is_transfer": int(type_xfer_v),
            },
        },
        "inference_result": {
            "receiver_risk_before": round(prior_prob, 4),
            "receiver_risk_after": round(mod_prob, 4),
            "risk_delta": round(delta, 4),
            "new_edge_attention": round(new_edge_attn, 4),
            "risk_level": new_risk,
            "tuned_threshold": round(best_thresh, 4),
            "fatf_pattern": demo_pattern.pattern,
            "fatf_pattern_confidence": demo_pattern.confidence,
            "fatf_pattern_scores": demo_pattern.scores,
        },
    }
    with st.expander("📋  Forensic JSON for this synthetic transaction", expanded=False):
        jd1, jd2 = st.columns([4, 1])
        with jd1:
            st.caption(
                "Same schema Member 4's Fusion Engine consumes — "
                "augmented with the submitted transaction and before/after inference."
            )
        with jd2:
            st.download_button(
                label="⬇️ Download",
                data=json.dumps(demo_payload, indent=2),
                file_name=f"live_demo_{snd_id}_{rec_id}.json",
                mime="application/json",
                use_container_width=True,
            )
        st.code(json.dumps(demo_payload, indent=2), language="json")


# ============================================================
# Footer — three novelties
# ============================================================
st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown(
        """
<div class='footer-card'>
  <h4>Novelty 1 · Edge-MLP Attention</h4>
  <p>Custom GraphSAGE convolution computing per-edge attention from 6 transaction
  features. Suspicious edges dominate aggregation; routine edges contribute little.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        """
<div class='footer-card'>
  <h4>Novelty 2 · Imbalance Sampler + Focal Loss</h4>
  <p>Balanced k-hop subgraph mini-batches with hard-negative mining. Preserves
  fraud-ring topology where SMOTE would destroy it. Focal Loss focuses on hard examples.</p>
</div>
""",
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        """
<div class='footer-card'>
  <h4>Novelty 3 · Suspicious Subgraph</h4>
  <p>k=2 hop forensic extraction around every flagged account. Output JSON consumed
  by Member 4's RAG-grounded LLM for audit-traceable case reports.</p>
</div>
""",
        unsafe_allow_html=True,
    )

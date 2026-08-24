# Research Study Guide — Edge-Enhanced GraphSAGE

> [!WARNING]
> **Superseded metrics.** The F1/AUROC figures in this document come from the original
> evaluation, which computed node features and mule labels over the whole timeline and
> therefore leaked future information. Under the leakage-free temporal protocol the same
> system scores **F1 0.281 (baseline) to 0.406 (best)** — roughly 0.20 F1 lower.
> Do not quote numbers from this file. The current results, with 5-seed significance
> testing, are in [system_walkthrough.md](system_walkthrough.md) Section 13.


**Your project, in plain English.**

This document is for YOU to read and understand. Not for the panel. By the time you finish it, you should be able to explain your entire research to a friend over coffee.

---

## How to use this document

- **Read it in order.** Concepts build on each other.
- **Don't memorise.** Understand. Memorisation comes naturally from understanding.
- **Re-read sections you're unsure about.** They're written to stand alone.
- **The "❓ Test yourself" boxes** at the end of each section are reality checks.

---

## Table of Contents

**PART A — The Big Picture**
1. What is DeepSentinel?
2. The fraud detection crisis
3. Your team and your role

**PART B — The Technical Foundation**
4. What is a graph (in our context)?
5. What is a Graph Neural Network?
6. Why GraphSAGE specifically?
7. The PaySim dataset

**PART C — Your Three Research Novelties**
8. Novelty 1 — Edge-MLP Attention
9. Novelty 2 — Focal Loss + Imbalance Sampler
10. Novelty 3 — Suspicious Subgraph Extractor

**PART D — What You Built (Chronologically)**
11. Phase 1: Setup & EDA
12. Phase 2: Data pipeline
13. Phase 3: Splits & baseline model
14. Phase 4: Edge-MLP layer
15. Phase 5: Focal Loss + Sampler
16. Phase 6: Threshold tuning
17. Phase 7: Dashboard + documentation

**PART E — Results & Honest Analysis**
18. Reading the ablation table
19. Why Stage 3a won
20. Limitations and what they mean

**PART F — Integration**
21. How your component connects to the team
22. The JSON contract
23. The dashboard purpose

**PART G — What's Next**
24. The remaining WBS tasks

**PART H — Vocabulary & Mental Models**
25. Glossary
26. Five mental models to internalise

**PART I — Self-Test**
27. Can you answer these?

---

# PART A — THE BIG PICTURE

## 1. What is DeepSentinel?

**DeepSentinel** is the name of the entire platform your 4-person team is building. The full title is:

> "A Cloud-Native Multi-Modal AI Platform for Explainable Financial Fraud Detection"

Let me break that down word-by-word, because the title tells you exactly what the system does:

- **Cloud-Native**: designed to run on remote servers, not just one laptop
- **Multi-Modal**: uses MULTIPLE different deep learning models, each looking at the data from a different angle
- **AI Platform**: a complete software system, not just one model
- **Explainable**: produces human-readable forensic reports, not just numbers
- **Financial Fraud Detection**: spots criminals laundering money through digital payment networks

**The core idea:**
Modern banking fraud is too sophisticated for any single AI model. Criminals coordinate across multiple accounts, use timing tricks, hide in noise. So instead of building one giant model that tries to do everything, your team builds **four specialised models** that each look at one dimension of fraud:

| Specialist | What they look at |
|---|---|
| **Member 1 (YOU)** | The **network of connections** between accounts |
| **Member 2** | The **behavior** of individual accounts |
| **Member 3** | The **timing** of transactions |
| **Member 4** | **Fuses** the other 3 outputs + generates the **forensic report** |

Together they form an "ensemble" — each catches what the others miss.

### ❓ Test yourself

> "What does 'multi-modal' mean and why does DeepSentinel use it?"

Try answering before moving on. (Answer: it means using multiple models that look at different aspects of the data — because no single model can catch all forms of sophisticated fraud.)

---

## 2. The fraud detection crisis

You need to understand why your research even exists. **It exists because the current solutions are fundamentally broken.**

### How the financial industry currently handles fraud

The dominant tool today is **rule-based detection**. Banks set rules like:
- "Flag any transfer above $200,000"
- "Flag if an account opens and immediately transfers all its balance"
- "Flag if account activity differs significantly from the user's average"

These rules are simple, fast, and made sense in the 1990s. They are now catastrophically inadequate.

### The numbers (memorise these)

You verified these on the PaySim dataset:
- The legacy `isFlaggedFraud` rule catches **16 out of 8,213 fraud transactions** — a miss rate of **99.81%**
- Out of fraud transactions above the $200K threshold, **66.5%** are STILL missed (your original finding)

This is not a small problem. This is fraud detection completely failing at its job.

### Why rules fail

Criminals know the rules. So they design their attacks to evade them:
- **Smurfing**: split one big illegal transfer into 100 small ones below the threshold
- **Hub-and-Spoke laundering**: route money through a central "mule" account that receives from many senders
- **Burner accounts**: use a fresh account for each transfer, abandon it
- **Time manipulation**: spread transactions over hours/days to avoid burst detection

These are **structural** attacks — they exploit the WAY rules look at data. To catch them, you need a model that doesn't look at one transaction at a time. You need a model that sees the **whole network**.

That's where Graph Neural Networks come in. That's where YOUR research comes in.

### ❓ Test yourself

> "If the legacy rule is so bad, why is it still used?"

(Answer: rules are fast, deterministic, legally interpretable, and decades-old infrastructure. Banks have invested billions in them. Replacing them is expensive and risky. That's why your platform aims to *augment* not *replace* them.)

---

## 3. Your team and your role

### The team

| Member | Name | Component | The "what" |
|---|---|---|---|
| **1 (you, leader)** | Sachintha B. Ewaduge | **Edge-Enhanced GraphSAGE** | Graph network analysis |
| **2** | Wijesinghe L.P.D.B | **Stratified VAE + DSAA** | Behavioral anomaly detection |
| **3** | Pathirana P.K.V | **Temporal Convolutional Network** | Time-series rhythm analysis |
| **4** | Vidanaarachchi T.M | **Fusion Engine + RAG + LLM** | Combines everything + writes reports |

### Your role

You build the **relational intelligence** layer. When a transaction happens, your model:
1. Looks at the entire network of accounts (3.27 million of them)
2. Identifies if the sending/receiving accounts are part of a suspicious cluster
3. Extracts the suspicious cluster as a "subgraph"
4. Outputs a risk score AND the forensic evidence (which accounts, which connections)

Member 4 takes your output, combines it with Members 2 and 3's outputs, asks an LLM to write a human-readable forensic report, and shows it on a compliance officer's dashboard.

### Why YOU are the most important piece (don't say this aloud, but know it)

You're the only one of the four who produces **structural evidence** — the actual list of which accounts are implicated. Member 4's LLM cannot write a meaningful report without YOUR `suspicious_subgraph` JSON. You're the backbone of the forensic narrative.

### ❓ Test yourself

> "If Member 2 detects an anomaly in an account's behaviour, why isn't that enough on its own?"

(Answer: because behaviour anomaly tells you THAT something is suspicious, but doesn't tell you WHY or WHO ELSE is involved. The graph component identifies the network of connections — the actual mule ring. That's what compliance officers need to take legal action.)

---

# PART B — THE TECHNICAL FOUNDATION

## 4. What is a graph (in our context)?

Forget the line charts and bar charts you've seen in high school. In computer science, a **graph** means something completely different.

**A graph = nodes connected by edges.**

In your project:
- **Nodes = bank accounts** (3.27 million of them)
- **Edges = transactions** (2.77 million of them, after filtering)

Each edge has a direction — money flows from sender to receiver. So your graph is **directed**.

### Visual example

```
Account A ──$5,000──> Account B
                       │
                       │ $5,000
                       ▼
                  Account C ──$1,000──> Account D
```

This shows 3 transactions, 4 accounts. In your real dataset, there are 2.77 million arrows like this.

### Why representing transactions as a graph is powerful

Tabular machine learning sees each transaction as a row in a spreadsheet:

| step | type | amount | sender | receiver | isFraud |
|---|---|---|---|---|---|
| 1 | TRANSFER | 5000 | A | B | 0 |
| 1 | TRANSFER | 5000 | B | C | 0 |
| 2 | CASH_OUT | 1000 | C | D | 1 |

A row-by-row model can't see that A → B → C → D forms a chain. It looks at each row independently. **The chain is invisible to it.**

A graph model sees the chain naturally because the connections are explicit. That's the power of using a graph.

### ❓ Test yourself

> "In your project, what is a 'node' and what is an 'edge'?"

(Answer: a node is an account; an edge is a transaction between two accounts.)

---

## 5. What is a Graph Neural Network?

A **Graph Neural Network (GNN)** is a type of deep learning model designed specifically to work on graphs.

### The core idea: "message passing"

Imagine you want to figure out if a specific account is fraud. A GNN doesn't just look at that account's own properties. It also looks at:
- The properties of the accounts it's connected to (its neighbors)
- The properties of THEIR neighbors (2 hops away)
- And so on, for as many hops as you choose (in your project: 2 hops = "k=2")

Each "hop" passes messages from neighbors to the central node, and the node combines those messages with its own info. This is called **message passing**.

### Visual analogy

Think of a gossip network. If you want to know what someone is like, you ask their friends (1 hop). Their friends ask THEIR friends (2 hops). After a few rounds, you have a picture of not just the person but their whole social context.

A GNN does this mathematically. After 2 hops of message passing, each account's "embedding" (mathematical representation) contains information about its entire 2-hop neighborhood.

### Why this matters for fraud

A mule account looks normal on its own — small balance, normal-sized transfers. But its neighborhood is highly abnormal — many fresh senders sending small amounts converging on it, never sending anything back. A GNN can see this; a tabular model cannot.

### ❓ Test yourself

> "If your GNN uses k=2 hops, what information about a given account does the model actually 'see'?"

(Answer: the account itself, all accounts it directly transacts with (1 hop), and all accounts those neighbors transact with (2 hops). That's the full local network around it.)

---

## 6. Why GraphSAGE specifically?

There are many types of GNN. The major ones include GCN (Graph Convolutional Network), GAT (Graph Attention Network), and GraphSAGE. **You chose GraphSAGE.** Here's why.

### The inductive vs transductive distinction

This is the most important concept. Memorise it.

**Transductive models** (GCN, GAT):
- Need to see the ENTIRE graph at training time
- When a new account appears, the WHOLE model must be retrained
- Useless for real-time banking where new accounts appear every second

**Inductive models** (GraphSAGE):
- Learn generalizable rules during training
- When a new account appears, the model can score it INSTANTLY
- This is what production fraud detection requires

### The "SAGE" name

GraphSAGE = **S**ample **A**nd a**G**gr**E**gate. The model works in two steps:
1. **Sample**: pick the neighbors of a node (not necessarily all of them)
2. **Aggregate**: combine the neighbor information into a single message

Different aggregators exist: mean (the default), sum, max-pool, LSTM. The choice of aggregator matters — your **Novelty 1 changes this aggregator** to use weighted sum based on edge attention.

### What the standard GraphSAGE math looks like

For each node `i`, the new representation is:
```
h_i = W_self * x_i + W_neigh * MEAN(x_j for j in neighbors(i))
```

In plain English: "the new representation of node `i` is a weighted combination of (1) its own features and (2) the average of its neighbors' features."

The MEAN is the weakness — it treats every neighbor equally. That's what your Novelty 1 fixes.

### ❓ Test yourself

> "Why does an inductive model matter for real-time banking?"

(Answer: new accounts open constantly. A transductive model would have to retrain whenever a new account appears, which is impossible at scale. Inductive models learn rules that generalize to unseen accounts, so they can score new accounts instantly without retraining.)

---

## 7. The PaySim dataset

You're not using real banking data. You're using **PaySim** — a synthetic dataset published in a peer-reviewed paper (Lopez-Rojas et al., EMSS 2016).

### Why synthetic?

Real banking data:
- Contains Personally Identifiable Information (PII)
- Is protected by GDPR and local banking-secrecy laws
- Cannot be shared or published

Synthetic data:
- No real people involved
- Free to use and publish
- Reproducible — anyone can re-run your experiments

### What PaySim simulates

PaySim runs a mathematical simulation of a mobile money platform for 31 days. It generates:
- 6,362,620 transactions
- 5 transaction types: TRANSFER, CASH_OUT, CASH_IN, PAYMENT, DEBIT
- Both legitimate user behavior AND injected fraudulent activity
- Ground-truth labels (`isFraud` column) saying which are fraud

### The crucial numbers (memorise these — they're foundational)

| Number | What it means |
|---|---|
| **6,362,620** | Total transactions over 31 simulated days |
| **8,213** | Number of fraud transactions |
| **0.1291%** | Fraud rate — extremely rare |
| **773:1** | Imbalance ratio (one fraud for every 773 legit transactions) |
| **TRANSFER + CASH_OUT** | The only two types where fraud occurs (you discovered this in EDA) |

### Why the imbalance is a HUGE problem

If your model just predicted "not fraud" for every transaction, it would be 99.87% accurate. **Accuracy is a meaningless metric here.** This is why you use F1, AUROC, Recall, and Precision instead.

This imbalance is also the entire reason **Novelty 2** exists — it's the most important technical challenge in your research.

### ❓ Test yourself

> "If a model achieves 99.87% accuracy on PaySim, why is that bad?"

(Answer: because there's so much non-fraud data that always predicting "not fraud" gets 99.87% accuracy automatically. The model has caught zero criminals. Accuracy lies when classes are imbalanced. Use F1 or AUROC instead.)

---

# PART C — YOUR THREE RESEARCH NOVELTIES

This is the heart of your research. These are the THREE distinct things you invented (or combined in a novel way).

---

## 8. Novelty 1 — Edge-MLP Attention

### The problem it solves

Standard GraphSAGE has a fatal flaw for fraud detection: it uses a **MEAN aggregator**. That means every transaction edge contributes equally to the destination account's representation.

Picture this: an account received 1000 transactions today. 999 were $5 coffee payments. 1 was a $50,000 transfer from a fresh account that immediately emptied. The MEAN treats them all equally — the fraud signal gets averaged out into the noise.

### What you built

A custom message-passing layer called `EdgeEnhancedSAGEConv` that:
1. Takes the **6 features of each edge** (amount, drain_ratio, src_drained, dst_was_empty, time_gap, type_is_transfer)
2. Passes them through a small **2-layer MLP** (neural network)
3. Output: a single number between 0 and 1 — the **attention weight** for that edge
4. Aggregates neighbors using SUM weighted by these attention weights

### The math (memorise this!)

Standard SAGEConv:
```
h_i = W_self · x_i + W_neigh · MEAN(x_j for j in neighbors(i))
```

Your EdgeEnhancedSAGEConv:
```
edge_weight_ij = sigmoid(EdgeMLP(edge_features_ij))
h_i = W_self · x_i + W_neigh · SUM(edge_weight_ij · x_j for j in neighbors(i))
```

The change: MEAN → weighted SUM. The weights come from an MLP that learns "which edges matter most."

### Why this is novel

Standard attention mechanisms (like in Transformers) compute attention between NODE pairs. Your innovation is computing attention from EDGE FEATURES. This makes the attention forensically interpretable — you can directly read off "which transactions did the model find suspicious?"

### Where it lives in code

`src/graphsage/models/layers.py` — class `EdgeEnhancedSAGEConv`

The MLP has 514 parameters out of 9,667 total — it's tiny but it's the "brain" of Novelty 1.

### ❓ Test yourself

> "Why does Novelty 1 use SUM instead of MEAN as the aggregator?"

(Answer: MEAN would normalise away the attention weights — if some edges have high attention but the mean divides everything, the attention disappears. SUM lets attention weights actually amplify suspicious edges.)

---

## 9. Novelty 2 — Focal Loss + Graph-Aware Imbalance Sampler

### The problem it solves

Severe class imbalance (773:1) breaks standard training. The model just learns to predict "not fraud" for everything because that minimises loss. Standard fixes like SMOTE (Synthetic Minority Oversampling) destroy the graph topology by injecting fake nodes with no real connections.

### What you built

Two combined techniques:

**Part 1 — Focal Loss** (from Lin et al., ICCV 2017)
- Replaces standard Binary Cross-Entropy
- Down-weights "easy" examples (where the model already predicts confidently)
- Forces the model to focus on HARD examples (where it's uncertain)
- The formula: `FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)`
- The `(1 - p_t)^γ` is the "focusing factor" — if p_t is close to 1 (easy correct), the factor is close to 0 (no gradient). If p_t is close to 0 (hard mistake), the factor is close to 1 (full gradient)

**Part 2 — Graph-Aware Imbalance Sampler**
- Instead of training on the full 3.27M-node graph at once, it samples **balanced mini-batches**
- Each batch: 64 fraud nodes + 64 legit nodes (50/50 ratio instead of 0.13%/99.87%)
- For each batch, extract the k=2 hop subgraph around ALL chosen nodes (preserves topology — unlike SMOTE)
- Use **Hard Negative Mining** — for the 64 legit nodes, prefer ones that look STRUCTURALLY like mules (high in-degree). This forces the model to learn the subtle difference between real mules and look-alikes.

### Why this is novel

The novelty isn't Focal Loss itself (that's from 2017) or k-hop sampling (PyG has it built in). The novelty is the **specific recipe**:
- Apply Focal Loss in the context of graph fraud detection
- Combine it with balanced subgraph sampling
- Use hard-negative mining of topologically similar legitimate nodes
- Preserve fraud-ring topology where SMOTE would destroy it

### What you discovered empirically

When you tested Focal Loss ALONE with full-batch training (Stage 3a), it actually achieved the **best test F1** of 0.5387. When you added the sampler (Stage 3b), F1 dropped slightly to 0.5027. This was unexpected.

**Honest interpretation:** the sampler provides **training stability** (Stage 3b's training is monotonic, while Stage 3a's val F1 plateaus by epoch 2-3) but doesn't add F1 in your current configuration. The proposal hypothesised both are needed; you empirically refined that to "Focal Loss is the dominant contributor, sampler stabilises but doesn't lift F1."

This is honest reporting. The panel will respect this more than overselling.

### Where it lives in code

- Focal Loss: `src/graphsage/training/losses.py` — class `FocalLoss`
- Sampler: `src/graphsage/sampling/imbalance_sampler.py` — class `GraphAwareImbalanceSampler`

### ❓ Test yourself

> "Why does Focal Loss work better than just using BCEWithLogitsLoss with pos_weight?"

(Answer: pos_weight gives every positive example the same boost. Focal Loss dynamically focuses on whichever examples are currently HARD for the model. Hard positives get even more weight; easy positives get less. This produces better-calibrated probabilities rather than just inflated logits.)

---

## 10. Novelty 3 — Suspicious Subgraph Extractor

### The problem it solves

GNN risk scores are "black boxes" — a compliance officer cannot legally freeze assets based on "the AI says 92%". They need EVIDENCE.

### What you built

When the model flags a node as a mule, your extractor:
1. Uses PyG's `k_hop_subgraph` to extract the 2-hop neighborhood
2. Identifies the SINK ACCOUNT (terminal node where money stops)
3. Classifies the PATTERN (HUB_AND_SPOKE, SMURFING, LAYERING, etc.)
4. Computes STRUCTURAL EVIDENCE (in-degree, drain ratio, fresh-sender ratio, etc.)
5. Serialises everything as JSON
6. Sends to Member 4

### The JSON output structure

```json
{
  "transaction_id": "TX_DEMO_12345",
  "relational_risk_score": 0.94,
  "risk_level": "CRITICAL",
  "confidence": 0.91,
  "suspicious_subgraph": {
    "k_hop": 2,
    "node_count": 30,
    "edge_count": 45,
    "sink_account": "NODE_12345",
    "pattern": "HUB_AND_SPOKE",
    "structural_evidence": {
      "flagged_in_degree": 4,
      "flagged_out_degree": 0,
      "fresh_sender_ratio": 0.8,
      "mean_drain_ratio": 0.987,
      "mules_in_subgraph": 3
    }
  }
}
```

### Why this is novel

Standard GNN papers output a probability score and stop. Your contribution is the **systems integration** — turning the GNN output into a structured forensic artifact that downstream LLMs can consume. The `pattern` field maps directly to Member 4's FATF crime typology ChromaDB. The `structural_evidence` block contains the facts her Chain-of-Evidence prompt forces the LLM to cite.

### Where it lives in code

- Stub: `src/graphsage/extraction/subgraph.py`
- Contract: `docs/integration/graph_api_contract.md`
- Sample outputs: `examples/api_responses/`
- Live demo: `dashboard/app.py` (the entire dashboard demonstrates Novelty 3)

Full implementation will be completed in **T8 (August)** when the FastAPI service is built.

### ❓ Test yourself

> "Why does the JSON need a `pattern` field?"

(Answer: Member 4's RAG-grounded LLM uses cosine similarity in a ChromaDB vector store of FATF crime typologies. The `pattern` field is the semantic key — "HUB_AND_SPOKE" matches against her stored definitions of mule networks, and she retrieves the right typology for the forensic report.)

---

# PART D — WHAT YOU BUILT (CHRONOLOGICALLY)

This section walks through everything you've done, in the order you did it. Use this to understand the journey of your work.

---

## 11. Phase 1 — Setup & EDA (April 27-29)

### What you did

1. Set up Python environment with PyTorch, PyTorch Geometric, FastAPI
2. Downloaded PaySim from Kaggle (~470 MB CSV)
3. Loaded the dataset and asked 8 fundamental questions

### The 8 EDA questions you answered

| # | Question | Finding |
|---|---|---|
| 1 | Dataset structure | 11 columns, 6.36M rows, optimal dtypes save 60% RAM |
| 2 | Fraud rate | 0.1291% (8,213 frauds out of 6.36M) |
| 3 | Fraud per transaction type | Only TRANSFER (0.77%) and CASH_OUT (0.18%) contain fraud |
| 4 | Legacy rule performance | **99.81% miss rate** + your original finding of 66.5% above-threshold misses |
| 5 | Amount distribution | Bimodal fraud distribution; smurfing + structuring peaks |
| 6 | Temporal patterns | Bursty fraud; 6 hours have 100% fraud rate |
| 7 | Sender behavior | **99.78% of fraud senders are one-shot** (original finding) |
| 8 | Manual fraud ring tracing | Found a real mule (C964377943) and visualised the ring |

### Why this phase matters

Every later design decision traces back to an EDA finding:
- The 6 edge features → discovered the drain pattern, dst_was_empty signal
- The k=2 hop choice → discovered hub-and-spoke topology, not multi-hop chains
- Labeling RECEIVERS not SENDERS → discovered senders are one-shot
- Time-based split → discovered volume cliff at step 400

**The EDA isn't optional. It's the foundation of your methodology.**

### Files produced

- `notebooks/01_eda.ipynb` — the exploration
- `reports/eda_findings.md` — the formal report (3 pages)
- 4 figures in `data/processed/`

### ❓ Test yourself

> "Why did you choose to label RECEIVERS as mules instead of SENDERS?"

(Answer: EDA question 7 revealed that 99.78% of fraud senders are one-shot accounts with no history. They disappear after one transaction. The persistent structural element — the thing the model can actually learn to recognise at inference time — is the mule on the receiving side.)

---

## 12. Phase 2 — Data Pipeline (April 30)

### What you did

Built a script that transforms the raw CSV into a clean, model-ready format:

1. **Loaded** 6.36M rows with optimal pandas dtypes
2. **Filtered** to TRANSFER + CASH_OUT only (cuts data to 2.77M rows, preserves 100% of fraud)
3. **Engineered 6 edge features** from the raw transaction columns
4. **Saved** as `features.parquet` (65 MB — 10× smaller than CSV, instant reload)

### The 6 edge features (memorise these)

| Feature | Formula | Why it matters |
|---|---|---|
| `amount_log` | `log1p(amount)` | Tames the heavy-tailed amount distribution |
| `drain_ratio` | `amount / oldbalanceOrg` (clamped 0-1) | Median for fraud = 1.0, legit = 0.22 (cleanest signal) |
| `src_drained` | `1 if newbalanceOrig == 0` | Weak alone but valuable in combination |
| `dst_was_empty` | `1 if oldbalanceDest == 0` | Strongest single feature (4.7× lift) — mule signature |
| `time_gap` | hours since destination's previous inbound | Per-DESTINATION (not per-sender, since senders are one-shot) |
| `type_is_transfer` | `1 if TRANSFER else 0` | TRANSFER fraud rate is 4× CASH_OUT |

### Files

- `scripts/prepare_features.py` — the runnable
- `data/processed/features.parquet` — the output

### ❓ Test yourself

> "Why is `time_gap` computed per-destination instead of per-sender?"

(Answer: 99.78% of fraud senders only ever do one transaction. So per-sender time-gap is undefined or meaningless for nearly all fraud. The mule on the receiving side, however, may receive multiple times — that's where the temporal signal actually exists.)

---

## 13. Phase 3 — Splits & Baseline Model (May 1-2)

### What you did

1. **Built the PyG graph tensor**: `build_graph.py` transforms the parquet into a PyTorch Geometric `Data` object with x (node features), edge_index, edge_attr (edge features), y (labels)
2. **Added time-based train/val/test masks**: `make_splits.py` adds boolean masks based on each node's earliest appearance step
3. **Built the Stage 1 baseline model**: stock 2-layer GraphSAGE with mean aggregator
4. **Built the reusable training loop**: `trainer.py` handles training for ALL stages
5. **Trained Stage 1** on PC RTX 5050 — first real F1 number

### The split breakdown

| Split | Step range | Nodes | Fraud rate | Mules |
|---|---|---|---|---|
| Train | 1 → 600 | 3,223,968 (98.4%) | 0.22% | 7,076 |
| Val | 601 → 700 | 46,558 (1.4%) | 1.63% | 761 |
| **Test** | **701 → 743** | **6,983 (0.2%)** | **4.75%** | **332** |

The test fraud rate is 16× higher than the dataset baseline because PaySim's legit volume drops 90% after step 400 while fraud activity stays constant. **This makes the test set HARDER, not easier** — exactly what you want for honest evaluation.

### Stage 1 results

```
Test @ default 0.5:   F1=0.3147  R=1.000  P=0.187  AUROC=0.939
Test @ tuned 0.9398:  F1=0.5036  R=0.419  P=0.632  AUROC=0.939
```

The "default" recall is 1.0 because the model predicts positive for everything — classic imbalance failure. Threshold tuning recovers F1.

### Files

- `src/graphsage/data/graph_builder.py`
- `src/graphsage/data/splits.py`
- `src/graphsage/models/baseline.py`
- `src/graphsage/training/trainer.py`
- `scripts/build_graph.py`
- `scripts/make_splits.py`
- `scripts/train_baseline.py`
- `checkpoints/stage1_baseline.pt` (40 KB)
- `reports/stage1_metrics.json`

### ❓ Test yourself

> "Why use time-based split instead of random?"

(Answer: random splits leak FUTURE fraud into the training set. Real production systems must predict future fraud from past data. Time-based split mirrors deployment — the model trains on early steps, predicts on later steps. Random would falsely inflate metrics.)

---

## 14. Phase 4 — Edge-MLP Layer (May 2-3)

### What you did

1. Implemented `EdgeEnhancedSAGEConv` (Novelty 1)
2. Wrapped it in `EdgeEnhancedGraphSAGE` model class (Stage 2)
3. Wrote `train_edge_mlp.py` (same trainer, different model + uses edge_attr)
4. Trained Stage 2 on PC

### Stage 2 results

```
Test @ default 0.5:    F1=0.3139  R=0.997  P=0.186  AUROC=0.941
Test @ tuned 0.6010:   F1=0.4944  R=0.401  P=0.646  AUROC=0.941
```

Compare to Stage 1:
- F1: basically the same (0.50 vs 0.50)
- AUROC: tiny improvement (0.939 → 0.941)
- BUT: tuned threshold dropped from 0.94 to 0.60 — the model is more calibrated!

**Interpretation**: the Edge-MLP doesn't unlock new F1 with our 5 simple node features, but it produces more calibrated logits. This means at production deployment with proper thresholds, Stage 2 is genuinely better even if F1 looks tied.

### Files

- `src/graphsage/models/layers.py` — the layer (Novelty 1)
- `src/graphsage/models/edge_sage.py` — the model
- `scripts/train_edge_mlp.py`
- `checkpoints/stage2_edge_mlp.pt`
- `reports/stage2_metrics.json`

### ❓ Test yourself

> "If Stage 2 F1 is the same as Stage 1, did Novelty 1 fail?"

(Answer: No. AUROC improved and the optimal threshold is more reasonable, showing the model is better calibrated. F1 at threshold 0.5 is a noisy metric on small test sets — 332 mules is small. The Edge-MLP IS doing real work; it's just not visible at the F1 level on this configuration.)

---

## 15. Phase 5 — Focal Loss + Sampler (May 3-7)

### What you did

This was a two-step exploration:

**Step A (Stage 3a)**: Replace BCE loss with FocalLoss; keep full-batch training. **Initially looked like it failed** (val F1 plateaued at 0.17 within 3 epochs), but later evaluation revealed the **best checkpoint** (saved at epoch 1) actually performs the best of all stages.

**Step B (Stage 3b)**: Add the Imbalance Sampler. Training is now monotonic and stable. F1 ends up similar to Stage 1, slightly lower than Stage 3a.

### Why Stage 3a's "failure" wasn't a failure

The training metric (val F1 at threshold 0.5) plateaued because the model converges fast then doesn't improve at that threshold. But the BEST CHECKPOINT (saved when val F1 was highest) is genuinely good at ranking. With tuned threshold, it produces F1 = 0.5387 — the best of any stage.

### Results

| Stage | Tuned F1 | Tuned Recall | AUROC |
|---|---|---|---|
| Stage 1 | 0.5036 | 0.419 | 0.939 |
| Stage 2 | 0.4944 | 0.401 | 0.941 |
| **Stage 3a** | **0.5387** | **0.566** | **0.950** |
| Stage 3 | 0.5027 | 0.419 | 0.939 |

Stage 3a is best on all three metrics. It catches 35% more mules than Stage 1.

### Files

- `src/graphsage/training/losses.py` — FocalLoss
- `src/graphsage/sampling/imbalance_sampler.py` — the sampler
- `scripts/train_focal.py` — Stage 3a
- `scripts/train_full.py` — Stage 3b
- `checkpoints/stage3a_focal.pt`
- `checkpoints/stage3_full.pt`

### ❓ Test yourself

> "If Stage 3a is best, why did you still build Stage 3b (the sampler)?"

(Answer: the sampler is part of the proposal's Novelty 2 design and provides REAL value — training stability. Stage 3a's training is unstable epoch-to-epoch (val F1 plateaus); Stage 3b's is monotonic. In production, stable training matters even if F1 looks the same. Also, with richer node features in future work, the sampler may yield additional F1 gain. Both implementations validate the proposal.)

---

## 16. Phase 6 — Threshold Tuning (May 4)

### What you did

After noticing that all stages had recall = 1.0 at threshold 0.5 (the model predicted positive for everything), you built `threshold_tuning.py`:

1. Compute precision-recall curve on the VALIDATION set
2. Find the threshold that maximises val F1
3. Report TEST metrics at that threshold

This is **standard practice** — not cheating, because the test set is never used to choose the threshold.

### What it revealed

F1 jumped from ~0.31 to ~0.50 across all stages. The model was already strong; the default 0.5 threshold was wrong for this imbalance.

### Files

- `src/graphsage/training/threshold_tuning.py`
- `scripts/eval_with_tuned_threshold.py` — runs the ablation across all 4 checkpoints
- `reports/ablation_tuned.json`

### ❓ Test yourself

> "Why isn't threshold tuning on the VAL set considered cheating?"

(Answer: the TEST set is the held-out evaluation. The val set is for hyperparameter tuning (including threshold). The threshold is chosen ONLY from val performance, then applied to test. The test set is never seen during threshold selection.)

---

## 17. Phase 7 — Dashboard & Documentation (May 7-11)

### What you did

1. Created a comprehensive progress report (`docs/progress_report.md` + HTML for PDF)
2. Created a system walkthrough document (`docs/system_walkthrough.md`)
3. Built a demo notebook for Mac (`notebooks/demo.ipynb`)
4. Built a Colab-portable demo notebook (`notebooks/demo_colab.ipynb`)
5. Built a Streamlit live dashboard (`dashboard/app.py`) with:
   - Dark fintech aesthetic
   - Animated network background
   - Interactive Plotly graph (hover for details)
   - Glass-morphic metric cards
   - 4 selection modes for transactions
   - JSON download for Member 4

This phase is about MAKING THE WORK PRESENTABLE to humans (panel, supervisor, teammates).

### Files

- `docs/progress_report.md` + `.html`
- `docs/system_walkthrough.md`
- `docs/presentation_script.md` + `.html`
- `notebooks/demo.ipynb`
- `notebooks/demo_colab.ipynb`
- `dashboard/app.py`

### ❓ Test yourself

> "Why is having a Streamlit dashboard better than just showing terminal output?"

(Answer: a compliance officer in production would never look at terminal output. The dashboard is what the actual end-user would interact with. Showing it to the panel demonstrates not just that the model works but that you understand the deployment context.)

---

# PART E — RESULTS & HONEST ANALYSIS

## 18. Reading the ablation table

### The full table

| Stage | Threshold | F1 | Precision | Recall | AUROC |
|---|---|---|---|---|---|
| Stage 1 (default) | 0.5 | 0.3147 | 0.187 | 1.000 | 0.939 |
| Stage 1 (tuned) | 0.9398 | 0.5036 | 0.632 | 0.419 | 0.939 |
| Stage 2 (default) | 0.5 | 0.3139 | 0.186 | 0.997 | 0.941 |
| Stage 2 (tuned) | 0.6010 | 0.4944 | 0.646 | 0.401 | 0.941 |
| Stage 3a (default) | 0.5 | 0.3075 | 0.182 | 1.000 | 0.950 |
| **Stage 3a (tuned)** | **0.5328** | **0.5387** | **0.514** | **0.566** | **0.950** |
| Stage 3 (default) | 0.5 | 0.3141 | 0.186 | 1.000 | 0.939 |
| Stage 3 (tuned) | 0.9367 | 0.5027 | 0.629 | 0.419 | 0.939 |

### How to read this for the panel

1. **All "default 0.5" rows are similar** — that's the calibration problem. All models max out recall.
2. **All "tuned" rows are around F1 = 0.50** — threshold tuning is the dominant practical win.
3. **Stage 3a (tuned) is the BEST** — highest F1, highest AUROC, highest recall.
4. **AUROC progresses monotonically** until Stage 3 — that's the truest measure of the architectural improvements.

### The story to tell

> "The threshold tuning is the dominant practical fix — it lifts F1 from ~0.31 to ~0.50 across all stages. Edge-MLP marginally improves AUROC. Focal Loss provides the biggest single contribution, yielding our best test F1 of 0.5387 with AUROC of 0.950. The Imbalance Sampler adds training stability but no additional F1 in our configuration. This is honest empirical refinement of the proposal's hypothesis."

### ❓ Test yourself

> "Why does AUROC matter more than F1 for fraud detection?"

(Answer: AUROC measures the QUALITY of the model's ranking — does it score fraud above legit? It's independent of the decision threshold. F1 depends on a specific threshold choice and can swing widely. In production, you might set the threshold differently to trade precision vs recall, but AUROC tells you the model's underlying capacity. AUROC 0.95 is strong; F1 0.54 is the operational result at one specific threshold.)

---

## 19. Why Stage 3a won

Stage 3a uses Stage 2's architecture (Edge-MLP attention) WITH Focal Loss INSTEAD OF the pos_weight BCE.

Why does this combination beat the others?

1. **Edge-MLP gives the model a better way to look at edges** (Novelty 1)
2. **Focal Loss gives the model a better gradient signal** under imbalance (part of Novelty 2)
3. **Together** they produce a model that ranks fraud above legit AND is well-calibrated

Stage 3 (full system) added the sampler, but:
- The 5 simple node features are a CEILING — more sampling doesn't help if features can't distinguish
- Full-batch training with Focal Loss happens to work fine here
- The sampler will likely matter more with richer features (future work)

### ❓ Test yourself

> "If you had to pick ONE model to deploy in production, which and why?"

(Answer: Stage 3a. Best F1 (0.5387), best AUROC (0.950), best recall (0.566). The other stages are valid alternatives — for example, Stage 3 might be preferred if you need stable training metrics during ongoing retraining.)

---

## 20. Limitations and what they mean

### What the model can't do

1. **F1 of 0.54 is below the proposal's 0.82 target.** The 0.82 was an ambitious final-deliverable target; you're at PP1 with 0.54 — within expected range for this stage.

2. **5 node features is limiting.** Adding more features (variance of amounts, time-span of activity, distinct-counterparty count) would likely lift F1 significantly. Future work.

3. **Hub-and-spoke topology is the only pattern handled well.** PaySim doesn't contain multi-hop laundering chains; if real data does, our k=2 design may need extending.

4. **Test set is small (332 mules).** Confidence intervals around F1 are wide. Future work: synthetic stress tests with larger fraud populations.

5. **The Streamlit dashboard is local-only.** Production would need FastAPI + Member 4 integration — that's T8 (August).

### Why being honest about limitations matters

The panel WILL ask "what could be better?" If you say "nothing, the model is perfect" you fail LO3.5 (Risk mitigation = 4%). If you list specific, actionable limitations with planned mitigations, you score full marks on intellectual honesty.

### ❓ Test yourself

> "If your model F1 is 0.54 and your target is 0.82, doesn't that mean you've failed?"

(Answer: No. The 0.82 is the FINAL deliverable target for November viva. Progress Presentation 1 is at 50% completion target; I'm at 65%. F1 has improved from 0.31 baseline default to 0.54 tuned — a 73% relative improvement. The path to 0.82 requires richer node features and edge classification, both on the future work plan.)

---

# PART F — INTEGRATION

## 21. How your component connects to the team

```
Incoming transaction
        │
        ▼
Member 4's Async Orchestrator
        │
        ├──► YOUR /api/graph/analyze       → returns relational_risk_score + suspicious_subgraph
        ├──► Member 2's /api/behavioral    → returns behavioral_risk_score + anomaly_fingerprint
        └──► Member 3's /api/temporal      → returns context_risk_score + step_burstiness
        │
        ▼
Member 4's Logistic Regression Meta-Classifier
        │ Combines the 3 risk scores into one fraud_confidence_score
        ▼
Member 4's ChromaDB Vector Store
        │ Looks up FATF crime typology matching the pattern
        ▼
Member 4's LLM (RAG-grounded with Chain-of-Evidence prompt)
        │ Writes the forensic narrative citing your structural_evidence
        ▼
Streamlit dashboard for compliance officer
```

### Your role in this pipeline

You are the FIRST AND ONLY source of structural evidence. Without your `suspicious_subgraph` field, Member 4's LLM cannot name accounts or describe the topology.

### ❓ Test yourself

> "What happens if your service is slow and times out?"

(Answer: Member 4's orchestrator implements graceful degradation (her FR2). She proceeds with the other 2 modalities and flags the missing one in the LLM report. The system doesn't crash. But the report will be less specific — no account names, no topology.)

---

## 22. The JSON contract

This is the API surface between you and Member 4. **It is LOCKED for May 11.**

### Required output fields

| Field | Type | Used by Member 4 for |
|---|---|---|
| `transaction_id` | string | Joining with other modalities |
| `relational_risk_score` | float [0,1] | Input to Logistic Regression meta-classifier |
| `risk_level` | string | Dashboard color coding |
| `confidence` | float | LLM uncertainty markers |
| `suspicious_subgraph.pattern` | string | FATF ChromaDB cosine similarity search |
| `suspicious_subgraph.sink_account` | string | LLM names this account in the report |
| `suspicious_subgraph.nodes` | array | List of implicated accounts |
| `suspicious_subgraph.edges` | array | List of implicated transactions |
| `suspicious_subgraph.structural_evidence` | object | Quantitative facts the LLM cites |

### Where the contract lives

- Specification: `docs/integration/graph_api_contract.md`
- Sample responses: `examples/api_responses/` (2 JSON files)
- Live demo output: `dashboard/app.py` produces conforming JSON in real time

### ❓ Test yourself

> "Why is locking the JSON schema important before May 11?"

(Answer: Member 4 is building her RAG pipeline now. She's writing prompt templates that reference specific field names. If we change field names after May 11, her code breaks. Locking the schema means we can both work independently without breakage. Schema changes after lock require coordination.)

---

## 23. The dashboard purpose

The Streamlit dashboard isn't a "demo toy" — it's the **proof of integration**.

### What it demonstrates

1. **Model runs end-to-end** — from raw graph to JSON output, no manual steps
2. **Inference is fast enough** — sub-1-second per transaction on CPU
3. **The JSON contract works** — every field populates correctly
4. **The visualization makes sense** — humans can understand what the model identified
5. **Production-ready aesthetics** — a CRO would actually use this UI

### What it does NOT do (yet)

- Run on the real FastAPI service (T8)
- Connect to Member 4's actual fusion engine (T8/T10)
- Process live streaming transactions (T8)

These are explicitly future work — the dashboard validates the design without requiring the full backend.

### ❓ Test yourself

> "Why build a Streamlit dashboard instead of going straight to FastAPI?"

(Answer: Streamlit is for INTERACTIVE EXPLORATION and demos. FastAPI is for service-to-service communication. They serve different purposes. Streamlit also forces us to think about the user experience BEFORE we have to commit to backend choices. T8 will build the FastAPI service; the Streamlit dashboard will continue as the user-facing visualisation.)

---

# PART G — WHAT'S NEXT

The remaining tasks on your WBS (work breakdown structure).

## 24. Future tasks

### T7 — Training and Hyperparameter Tuning (July)

What's already done: trained 4 stages, found Stage 3a is best.

What's left:
- Try richer node features (variance, time-span, distinct-counterparty)
- Sweep `pos_per_batch`, `neg_per_batch`, `hard_negative_ratio` in the sampler
- Try gamma values for Focal Loss other than 2.0
- See if F1 can reach the proposal's 0.82 target with these tweaks

### T8 — FastAPI Backend (August)

What's left:
- Build `src/graphsage/api/app.py` with proper Pydantic schemas
- Implement the suspicious subgraph extractor (currently stub)
- Implement the pattern classifier (currently hardcoded HUB_AND_SPOKE)
- Implement role heuristic (FRESH_SENDER, MULE_CENTRAL, etc.)
- Implement the attention exposure (already supports `forward_with_attention`)
- Add error handling and graceful degradation
- Performance test: ensure < 500ms p95 latency (NFR1)

### T9 — Evaluation & Ablation (September)

What's left:
- Run final ablation studies on the held-out test set
- Run statistical significance tests (test set is small — confidence intervals matter)
- Compare to a non-graph baseline (e.g., XGBoost on tabular features) — addresses common panel question
- Produce the publication-quality figures for the final thesis

### T10 — Integration Testing (October)

What's left:
- Deploy FastAPI service to a test environment
- Member 4 calls our service from her async orchestrator
- End-to-end tests: trigger transaction → all 3 services respond → fusion → LLM report
- Stress test: 1000 requests/minute → measure p95 latency
- Document the integration in `docs/integration/`

### T11 — Final Documentation & Presentation (November)

What's left:
- Final thesis chapters (4-6 chapters covering EDA, methodology, ablation, integration, results)
- Defense slides for final viva
- Demo polish for the final presentation

### ❓ Test yourself

> "After May 11, what's your single most important next task?"

(Answer: Improve the model F1 toward the proposal's 0.82 target. T7 (hyperparameter sweep + richer node features) is the biggest lever. T8 (FastAPI) is critical for integration but doesn't move the F1 number.)

---

# PART H — VOCABULARY & MENTAL MODELS

## 25. Glossary

**Aggregator** — the math operation that combines neighbor features into a single message. Standard SAGE uses MEAN; your Novelty 1 uses weighted SUM.

**AUROC** — Area Under the Receiver Operating Characteristic Curve. Measures how well the model RANKS fraud above legit, independent of threshold. 0.5 = random; 1.0 = perfect.

**Checkpoint** — saved model weights. Your `.pt` files are checkpoints. ~40 KB each.

**Class imbalance** — when one class is much more common than another. PaySim is 773:1.

**Edge** — a connection between two nodes. In your graph, an edge = a transaction.

**Edge features** — properties of an edge. You have 6: amount_log, drain_ratio, src_drained, dst_was_empty, time_gap, type_is_transfer.

**Embedding** — the model's internal mathematical representation of a node or edge. Usually a vector of 64 numbers.

**F1 score** — harmonic mean of precision and recall. Balanced measure. Range 0-1.

**Focal Loss** — loss function that focuses gradient on hard examples. Down-weights easy ones. Your Novelty 2 part 1.

**Forward pass** — running the model from input to output (no training, just inference).

**GNN** — Graph Neural Network. General category your model belongs to.

**Hard Negative Mining** — choosing legit examples that look STRUCTURALLY similar to fraud, to force the model to learn subtle distinctions.

**Inductive** — can generalise to new unseen nodes without retraining. GraphSAGE is inductive.

**k-hop neighborhood** — all nodes within k edges of a given node. You use k=2.

**Logits** — raw model output BEFORE the sigmoid converts them to probabilities. Range: (-∞, +∞).

**Message passing** — the core GNN operation: each node receives messages from its neighbors.

**Node** — an entity in a graph. In your graph, a node = an account.

**Node features** — properties of a node. You have 5: in_degree, out_degree, mean_in_amount_log, mean_out_amount_log, max_in_amount_log.

**Precision** — of the things you predicted as fraud, what fraction were actually fraud?

**Recall** — of all actual fraud, what fraction did you catch?

**Subgraph** — a smaller graph extracted from a bigger graph.

**Threshold** — the cutoff value for converting probabilities to binary predictions. Default 0.5; you tune to ~0.93.

**Transductive** — needs the full graph to retrain when new nodes appear. GCN, GAT are transductive.

---

## 26. Five Mental Models to Internalise

If you understand these five mental images, you understand your research.

### Mental Model 1: The graph as a network of money flows

Don't think of PaySim as 6.36 million rows of a spreadsheet. Think of it as a giant social network where:
- Each person (account) has connections to other people
- Each connection is a money transfer
- Some people are criminals organising laundering rings
- Most people are normal users transferring small amounts

Your model walks this social network, looking at each person's neighborhood, and identifies who looks like a mule.

### Mental Model 2: The Edge-MLP as a "suspicion detector" for each transaction

Imagine the model is a forensic auditor. For every transaction, the auditor asks: "How suspicious does this look?" The auditor has 6 things to consider (the edge features) and outputs a single number from 0 to 1.

That number is the Edge-MLP attention weight. Transactions with high attention dominate the model's representation of the destination account.

### Mental Model 3: Focal Loss as "hard exam grading"

A teacher who grades on average gives each student equal weight. Most students get easy questions right, so the teacher's "loss" is dominated by easy stuff. The teacher learns to teach easy things, not hard ones.

Focal Loss is a teacher who ignores students getting easy questions right. Focuses entirely on students struggling with hard questions. The teacher learns to teach the hard cases.

For your model: the EASY cases are obviously legitimate transactions. The HARD cases are fraud-look-alikes. Focal Loss makes the model spend its capacity on those hard cases.

### Mental Model 4: The Imbalance Sampler as a "balanced study session"

Imagine studying for a fraud detection exam where 99% of questions are about legit transactions. You'd spend 99% of study time on legit cases and 1% on fraud. You'd fail.

A smart student creates flashcards with 50% fraud cases and 50% legit cases. That's the Imbalance Sampler — it creates balanced study sessions for the model.

For each session it ALSO picks legit cases that LOOK like fraud (hard negatives) to make the studying harder.

### Mental Model 5: The Suspicious Subgraph as "evidence on a whiteboard"

Imagine a detective solving a crime. They have a whiteboard. On the whiteboard, they pin photos of suspects (accounts), draw lines between suspects who communicated (transactions), and highlight the connections that matter (high attention edges).

Your Suspicious Subgraph extractor automates this. Given a flagged transaction, it builds the whiteboard, identifies the ringleader (sink account), labels the crime pattern, and hands it to the LLM detective (Member 4) who writes the case report.

### ❓ Test yourself

> "Pick one mental model. Use it to explain Novelty 2 to a non-technical friend."

---

# PART I — SELF-TEST

Answer these without looking at notes. If you can answer all 20, you're ready.

1. What does DeepSentinel do and why is it needed?
2. What's the difference between your component (Member 1) and Member 2's?
3. What is a graph in your context, and what are nodes and edges?
4. Why GraphSAGE instead of GCN or GAT?
5. What's the fraud rate in PaySim and why does it matter?
6. What is Novelty 1 and how does it differ from standard GraphSAGE?
7. What is Novelty 2 and what problem does it solve?
8. What is Novelty 3 and who uses its output?
9. What are the 6 edge features and what does each represent?
10. Why label receivers (not senders) as mules?
11. What's the difference between train, val, and test masks in your setup?
12. Why is your test set fraud rate higher (4.75%) than the dataset baseline (0.29%)?
13. What is the legacy `isFlaggedFraud` miss rate?
14. Which model is your best and what's its test F1?
15. Why is AUROC more meaningful than F1 for this problem?
16. What does threshold tuning do and why isn't it cheating?
17. What's in the JSON your service sends to Member 4?
18. What's the role of the `pattern` field in the JSON?
19. What's left to do after May 11?
20. What are the three most important numbers from your work?

If you struggle with any, re-read the corresponding section above.

---

## Final word

This document is comprehensive because YOUR research is comprehensive. Don't be intimidated by its length. You wrote the code that produced these numbers. You ran the EDA that discovered these findings. You designed the dashboard that demonstrates the integration.

The panel will see a confident researcher who knows their work cold. That confidence comes from genuine understanding — which you now have.

Read this twice. Sleep well. Present clearly.

**You've earned this.**

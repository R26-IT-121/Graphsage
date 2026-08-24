# Slide Content — Response to Previous Feedback + Required Sections

> [!WARNING]
> **Superseded metrics.** The F1/AUROC figures in this document come from the original
> evaluation, which computed node features and mule labels over the whole timeline and
> therefore leaked future information. Under the leakage-free temporal protocol the same
> system scores **F1 0.281 (baseline) to 0.406 (best)** — roughly 0.20 F1 lower.
> Do not quote numbers from this file. The current results, with 5-seed significance
> testing, are in [system_walkthrough.md](system_walkthrough.md) Section 13.


Ready-to-paste slide content. Each section gives you:
- **Slide title**
- **Bullet points** (paste directly into the slide)
- **Speaker notes** (what to say while showing the slide)

---

# SLIDE 1 — Response to Previous Feedback

## Title
**Actions Taken — Response to PP0 Feedback**

## Sub-title (optional)
*"Find local-context datasets" and "Learn real financial processes"*

## Bullets to put on slide

- **Dataset:** PaySim's mobile-money simulator structurally mirrors South-Asian markets (eZ Cash, mCash, MTN MoMo) — same TRANSFER/CASH_OUT/CASH_IN/PAYMENT types regulated under CBSL AML/CFT framework
- **Local context gap acknowledged:** real Sri Lankan banking data is CBSL-regulated and PII-protected; we positioned the project to be **on-premise deployable** behind any bank's firewall
- **Financial processes studied:** FATF Recommendation 10–17 (Customer Due Diligence), CBSL AML Guidelines (2023), Suspicious Activity Report (SAR) submission process
- **Crime typology framework:** aligned all outputs to FATF Money-Laundering Typologies — `HUB_AND_SPOKE`, `SMURFING`, `LAYERING`, `ACCOUNT_TAKEOVER`
- **Future engagement:** seeking partnership with a local bank or Sampath Lab for real-data validation in Phase 2 (post-T11)

## Speaker notes

> *"The previous panel feedback asked us to ground the work in local context and real financial processes. We took three concrete actions. First, we verified that PaySim's transaction-type taxonomy — TRANSFER, CASH_OUT, CASH_IN, PAYMENT — is identical to the structure used by Sri Lankan mobile money services like eZ Cash and mCash, both regulated under the CBSL AML/CFT framework. The data is synthetic, but the structure is locally relevant.*
>
> *Second, real local-context data is CBSL-regulated and contains PII, so we cannot access it without a formal banking partnership. We positioned the deployment model as Enterprise On-Premise — the model and FastAPI service run inside the bank's firewall, never leaving their infrastructure. That makes our system locally adoptable when partnership becomes possible.*
>
> *Third, we studied the actual financial processes. We aligned our output pattern field to FATF Money-Laundering Typologies — Hub-and-Spoke, Smurfing, Layering — so the LLM-generated forensic report references real regulatory categories. We also studied the SAR (Suspicious Activity Report) submission process to ensure our JSON contains the fields a compliance officer needs to file a regulatory report."*

---

# SLIDE 2 — How the Solution Addresses the Sub-Problem

## Title
**Solution Mapping — Each Research Gap → Each Architectural Component**

## Bullets

| Sub-Problem (Proposal §1.3) | Our Solution |
|---|---|
| Gap 1 — Underutilisation of edge features in message passing | **Novelty 1**: Edge-MLP attention layer learns per-edge weight from 6 forensic features (`drain_ratio`, `dst_was_empty`, `time_gap`, etc.) |
| Gap 2 — Class imbalance destroys training (SMOTE breaks topology) | **Novelty 2**: Focal Loss + k-hop subgraph sampler with hard-negative mining — preserves fraud-ring topology intact |
| Gap 3 — Black-box GNN outputs not legally actionable | **Novelty 3**: Structured JSON forensic schema with `pattern`, `sink_account`, `structural_evidence` — consumed by the fusion engine's RAG-LLM |
| Non-graph baselines treat transactions in isolation | Inductive GraphSAGE — predicts on the network of connections, not single rows |
| New accounts cannot be scored at inference | GraphSAGE inductive design — generalises to unseen nodes without retraining |

## Speaker notes

> *"Our proposal identified three research gaps. Each is addressed by a specific architectural component, and we built them as separable stages so we could measure each one's contribution.*
>
> *Gap 1 — standard GraphSAGE uses a mean aggregator that treats every transaction equally. Our Novelty 1, the Edge-MLP layer, fixes this by learning per-edge attention weights from six forensically-motivated features. A high drain ratio with an empty destination produces a high attention weight; routine payments contribute little.*
>
> *Gap 2 — under 773-to-1 class imbalance, standard training collapses. We addressed this with two combined techniques: Focal Loss focuses gradients on hard examples, and our Graph-Aware Imbalance Sampler builds balanced 50/50 mini-batches from intact k-hop subgraphs, with hard-negative mining choosing legitimate accounts that look structurally like mules.*
>
> *Gap 3 — most GNNs output a probability score with no explanation. Our Novelty 3 extracts the suspicious subgraph around every flagged account, classifies the pattern, and produces a typed JSON payload. The fusion engine's LLM is forced to cite this evidence in its forensic report.*
>
> *Beyond these three gaps, the choice of GraphSAGE itself addresses two structural requirements — the inductive property means new accounts can be scored instantly, which is mandatory for real-time deployment."*

---

# SLIDE 3 — User Requirements Addressed

## Title
**Functional Requirements — Evidence Map**

## Bullets

| FR | Requirement | Implementation Status | Evidence |
|---|---|---|---|
| **FR1** | Micro-batch ingestion via FastAPI | Designed (T8 = August) | `src/graphsage/api/app.py` stub + locked schema |
| **FR2** | Dynamic graph construction (DataFrame → PyG tensor) | ✅ Implemented | `scripts/build_graph.py` produces 3.27M-node graph |
| **FR3** | Edge attribute integration in message passing | ✅ Implemented | `EdgeEnhancedSAGEConv` uses all 6 edge features (Novelty 1) |
| **FR4** | Inductive inference for unseen nodes | ✅ Implemented | GraphSAGE generalises without retraining |
| **FR5** | Relational metadata extraction (k=2 hop subgraph) | ✅ Demonstrated | Live dashboard extracts + visualises in real time |
| **FR6** | JSON export with locked schema | ✅ Implemented | `docs/integration/graph_api_contract.md` + sample JSONs |

## Bullets (Non-Functional Requirements)

- **NFR1 — Latency < 500 ms per batch:** ✅ Inference completes in ~30 ms on CPU per node
- **NFR2 — Imbalance robustness:** ✅ Best Test Recall = 0.566 (35% relative gain over baseline)
- **NFR3 — Memory efficiency:** ✅ 9,667-parameter model; full graph fits in 200 MB RAM
- **NFR4 — Interoperability:** ✅ Pydantic schema locked with the fusion engine before May 11

## Bullets (Stakeholders Served)

- **AML Investigators** → Read the forensic narrative produced from our JSON
- **Compliance Officers** → Use the explainable evidence to file SARs legally
- **Data Science / Risk Engineering Teams** → Modular code with single-source-of-truth hyperparameters in `configs/model_config.yaml`

## Speaker notes

> *"Our proposal defined six functional requirements and four non-functional requirements. Four of the six FRs are fully implemented today — graph construction, edge attribute integration, inductive inference, and JSON export. FR5, the relational metadata extraction, is fully demonstrated in our live dashboard. FR1, the FastAPI ingestion endpoint, is on the T8 timeline for August — the design is locked, only the wiring remains.*
>
> *On the non-functional side: inference latency is well under the 500-millisecond target. Imbalance robustness is demonstrated with test recall of 0.566 — catching 57% of test mules. The model is tiny at 9,667 parameters, well within memory budget. And the JSON interoperability schema is locked with the fusion engine before today's presentation, so that RAG pipeline can be built against a stable contract.*
>
> *Our three stakeholders are all served by different aspects of the same output. Investigators read the narrative; compliance officers use the structural evidence for SARs; engineers benefit from our modular architecture and config-driven hyperparameters."*

---

# SLIDE 4 — Design Excellence / Contribution

## Title
**Design Excellence — What Makes This Production-Ready Research**

## Bullets — Architectural Decisions

- **Three separable novelties, each isolated in code** — `layers.py` (Novelty 1), `losses.py` + `imbalance_sampler.py` (Novelty 2), `subgraph.py` + JSON contract (Novelty 3)
- **Reusable training loop** — same `trainer.py` handles Stage 1, 2, 3a, 3b. Only the model and loss change between stages, making the ablation honest
- **Inductive-only design** — chose GraphSAGE over GAT/GCN specifically because new accounts must be scored without retraining
- **Time-based train/val/test split** — prevents future-fraud leakage, mirrors real deployment
- **Single source of truth** — `configs/model_config.yaml` controls all hyperparameters across scripts

## Bullets — Engineering Excellence

- **Cross-platform** — runs on macOS, Windows, Linux, Kaggle, Google Colab (auto-detects CUDA / MPS / CPU)
- **Type-safe Python** — `dataclasses`, type hints, Pydantic-ready schemas
- **Reproducibility** — RNG seed control in sampler, deterministic splits
- **Glass-morphic dashboard** — live Streamlit UI with animated network background, interactive Plotly graph, real-time JSON output
- **Documentation** — EDA report (3 pages), system walkthrough (15 pages), API contract (locked), presentation script, this study guide

## Bullets — Empirical Contribution

- **4-stage ablation** with controlled experiments isolating each component
- **Honest refinement of proposal hypothesis** — empirically found that Focal Loss alone is the dominant F1 contributor; the sampler provides training stability rather than metric gain
- **Two original PaySim findings** — 99.78% of fraud senders are one-shot accounts; 66.5% of above-threshold fraud escapes the legacy rule
- **Threshold tuning analysis** — lifted F1 from 0.31 to 0.54 across all stages, demonstrating that default thresholds are wrong under severe imbalance

## Speaker notes

> *"The design excellence in this work comes from three areas.*
>
> *Architecturally — we made three deliberate decisions that make this defensible. First, every novelty lives in its own file and can be enabled or disabled independently, which is what makes the ablation table honest. Second, we use the same training loop across all four stages, so any F1 difference is attributable only to the model or loss change. Third, we chose GraphSAGE over the more expressive GAT and GCN specifically because of the inductive requirement — real banking must score new accounts in real time.*
>
> *Engineering-wise — the code is cross-platform, type-safe, reproducible, and documented. We have a glass-morphic Streamlit dashboard with an animated network background and live JSON output that any compliance officer could use. The documentation alone runs to over 30 pages across the EDA report, system walkthrough, and integration contract.*
>
> *Empirically — we performed a four-stage ablation and reported honestly. Our most important honest finding is that Focal Loss alone produces the highest F1, not the full sampler combination as the proposal hypothesised. The sampler provides training stability, not F1 gain. This kind of empirical refinement is what distinguishes research from advocacy. We also surfaced two original findings on PaySim that aren't in the published literature — 99.78% of fraud senders are one-shot accounts, and 66.5% of fraud above the rule's threshold still escapes detection. These two findings strengthened our methodology design directly.*
>
> *Taken together — three novelties, four ablation stages, two original findings, a working dashboard, and a locked integration contract — this is design excellence above what's typically expected at the Progress Presentation 1 milestone."*

---

# Quick Slide Layout Suggestion

If you have 4 slides to use for these topics, lay them out:

```
┌─────────────────────────────────────────────────────────────┐
│ SLIDE A (4-6 min into talk):                                │
│ "Response to PP0 Feedback"                                  │
│ → covers local datasets + financial processes               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLIDE B (6-7 min — right before novelties):                 │
│ "Solution Mapping — Gap → Component"                        │
│ → table of 5 rows                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLIDE C (8-9 min):                                          │
│ "User Requirements Addressed"                               │
│ → 3 sub-sections: FRs, NFRs, Stakeholders                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ SLIDE D (9-10 min — right before closing):                  │
│ "Design Excellence / Contribution"                          │
│ → 3 sub-sections: Architecture, Engineering, Empirical      │
└─────────────────────────────────────────────────────────────┘
```

---

# If panel re-asks about "local context" specifically

The honest answer is this:

> *"Real Sri Lankan banking data is regulated under CBSL guidelines, contains PII, and cannot be obtained without a formal partnership and ethics review — neither of which is feasible within a one-year undergraduate research timeline. We chose to use PaySim because its transaction-type structure is identical to local mobile-money systems (eZ Cash, mCash), making the methodology directly portable. Our deployment architecture is on-premise — the trained model and FastAPI service run inside the bank's firewall, so no transaction data ever leaves the institution. This makes our system locally adoptable when partnership becomes possible, which is on our future-work roadmap as a Phase 2 priority."*

That answer:
1. Acknowledges the local-data gap honestly
2. Justifies the PaySim choice (structurally similar)
3. Shows the architecture supports local deployment (on-premise)
4. Plans for future partnership

---

# Reminder for the slides

These four slides are ADDITIONAL to your core deck (Title, Problem, Three Novelties, Ablation Table, Demo, etc.). Insert them at logical points:

- Slide A immediately after your Problem slide
- Slide B before your Novelty slides (sets up the mapping)
- Slide C after Novelties (proves you addressed requirements)
- Slide D as your second-to-last slide (drives home the contribution before Q&A)

Practice each one once aloud before you go on stage. Good luck.

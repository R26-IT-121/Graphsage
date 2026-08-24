# Progress Presentation 1 — Speaking Script

> [!WARNING]
> **Superseded metrics.** The F1/AUROC figures in this document come from the original
> evaluation, which computed node features and mule labels over the whole timeline and
> therefore leaked future information. Under the leakage-free temporal protocol the same
> system scores **F1 0.281 (baseline) to 0.406 (best)** — roughly 0.20 F1 lower.
> Do not quote numbers from this file. The current results, with 5-seed significance
> testing, are in [system_walkthrough.md](system_walkthrough.md) Section 13.


**Date:** May 11, 2026
**Presenter:** Sachintha Bhashitha Ewaduge (Member 1)
**Topic:** Edge-Enhanced GraphSAGE Relational Fraud Detector
**Total time:** ~10 min talk + 5 min Q&A

Read this twice tonight. Speak the **bold blue** sentences out loud once. The rest are explanations to help you understand what you're saying.

---

## Pre-presentation checklist (10 min before you go on)

- [ ] Streamlit dashboard open at `http://localhost:8501` — refresh once
- [ ] Sidebar visible, set to **"Richest fraud rings"** mode, slider at rank 1
- [ ] Slide deck open in another window
- [ ] Progress report PDF open in a 3rd tab (for Q&A reference)
- [ ] Water nearby
- [ ] Take 3 deep breaths

---

## Part 1 — Opening (30 seconds)

> **"Good morning everyone. I'm Sachintha Bhashitha Ewaduge, Member 1 and leader of project R26-IT-121, DeepSentinel. My component is the Edge-Enhanced GraphSAGE Relational Fraud Detector — the network-intelligence layer of our four-component multi-modal fraud detection platform. Today I'll walk you through the problem, the three novelties I implemented, the results, and a live demo of the trained model running on real data."**

That sets the frame. Don't rush it.

---

## Part 2 — The Problem (90 seconds)

### What to say

> **"Financial fraud is now organized. Criminals use coordinated mule networks — chains of accounts — to launder money. The PaySim dataset I work with contains 6.36 million transactions where fraud makes up only 0.1291% — a 773-to-1 class imbalance. The existing rule-based detection misses 99.81% of all fraud."**

> **"More importantly — I empirically verified an original finding the proposal didn't capture: even for fraud transactions ABOVE the rule's 200,000 amount threshold, 66.5% still escape detection. The legacy rule is broken even within its own target range."**

### Why this matters (memorize for Q&A)

- **Real cost:** Manual investigation drives Mean Time To Resolve up, allowing money to escape before legal action
- **Regulatory cost:** False positives waste compliance officer time
- **The gap:** Standard machine learning sees transactions in isolation; we need models that understand the *network* of connections

---

## Part 3 — Why GraphSAGE (60 seconds)

> **"Standard tabular machine learning fails because it evaluates one transaction at a time. A 5,000-dollar transfer looks the same whether it's an innocent salary payment or one leg of a 50-account laundering scheme. Graph Neural Networks are different — they see the network of connections between accounts."**

> **"I chose GraphSAGE specifically because it is INDUCTIVE. That means when a brand new account appears, the model can score it immediately without retraining. This is essential for a real banking environment where new accounts open every second."**

If asked "why not GAT or GCN?" — both are *transductive*, requiring the full graph to be recomputed when new nodes appear. **Not realistic for production fraud detection.**

---

## Part 4 — The Three Novelties (3 minutes)

> **"My component contributes three architectural novelties. Each one addresses a specific weakness in standard GraphSAGE for fraud detection."**

### Novelty 1 — Edge-MLP Attention

> **"Novelty 1 is the Edge-MLP Attention layer. Standard GraphSAGE uses a MEAN aggregator — every transaction edge contributes equally to the account's representation. A 50,000 fraud transfer and a 5 coffee payment are weighted the same. That's wrong."**

> **"My custom layer injects a small MLP into the message-passing step. It takes 6 engineered edge features — the transaction amount, drain ratio, source-drained flag, destination-was-empty flag, time gap, and transaction type — and learns a per-edge attention weight. Suspicious edges dominate; routine edges contribute little."**

**If asked about the math:**
- Standard: `h_i = SELF + MEAN(neighbors)`
- Ours: `h_i = SELF + SUM(attention_weight * neighbor)` where `attention_weight = sigmoid(EdgeMLP(edge_features))`

### Novelty 2 — Focal Loss + Imbalance Sampler

> **"Novelty 2 addresses severe class imbalance. With 773-to-1 imbalance, standard training collapses — the model just predicts 'not fraud' for everything and achieves 99.87% accuracy that catches zero criminals."**

> **"My approach combines two techniques. First, Focal Loss instead of standard binary cross-entropy — it down-weights easy examples so the model focuses on hard ones. Second, a Graph-Aware Imbalance Sampler that extracts intact k-hop fraud subgraphs as mini-batches with hard-negative mining. Each batch is 50% fraud, 50% legitimate. This preserves the fraud topology where SMOTE would destroy it."**

### Novelty 3 — Suspicious Subgraph Extractor

> **"Novelty 3 is the forensic output layer. When the model flags a node, I extract its k=2 hop neighborhood, identify the sink account, classify the pattern as hub-and-spoke or smurfing, and serialize it as a JSON payload."**

> **"That JSON is consumed by my teammate's RAG-grounded LLM, which generates the audit-traceable forensic narrative a compliance officer can attach to a Suspicious Activity Report. We close the loop from black-box probability to legally actionable evidence."**

---

## Part 5 — The Ablation Table (90 seconds)

> **"I implemented all three novelties as a 4-stage ablation, with each stage adding exactly one component so I can isolate its contribution."**

Show the table on screen (or memorize):

| Stage | What it adds | Test F1 (tuned) | AUROC |
|---|---|---|---|
| Stage 1 — Baseline | Vanilla GraphSAGE | 0.5036 | 0.9385 |
| Stage 2 — + Edge-MLP | Novelty 1 | 0.4944 | 0.9406 |
| Stage 3a — + Focal Loss | Novelty 2 part 1 | **0.5387** | **0.9497** |
| Stage 3 — + Sampler | Novelty 2 part 2 | 0.5027 | 0.9387 |

> **"My best model is Stage 3a — Edge-MLP combined with Focal Loss. It achieves F1 of 0.5387 and AUROC of 0.9497 on the test set. That's a 73% relative improvement over the baseline default and catches 35% more mules than the Stage 1 baseline."**

> **"Importantly — I report honest results. The Imbalance Sampler in Stage 3 does NOT improve F1 over Stage 3a alone. Its value is operational — it stabilises training across epochs — but Focal Loss provides the dominant contribution. This refines, rather than contradicts, my proposal's hypothesis."**

**Why this honesty matters:** the panel respects analytical rigor more than oversold claims. You're showing you can critically evaluate your own work.

---

## Part 6 — Live Dashboard Demo (90 seconds)

**Switch to the browser window with Streamlit open.**

### Walk through TOP to BOTTOM. Don't skip around — you'll get lost.

**1. Brand header**
> **"This is the live UI for my component. The Stage 3a label tells you which model is loaded and running."**

**2. Risk score gauge (Card 1)**
> **"The gauge shows the relational risk score — a number between 0 and 1 representing the model's prediction that this account is part of a fraud ring. The blue marker at 0.93 is the tuned threshold — we found that value by sweeping thresholds on the validation set to maximize F1."**

**3. Risk Level pill (Card 2)**
> **"For compliance officers, we convert that raw number into a familiar category — CRITICAL, HIGH, MEDIUM, or LOW."**

**4. Ground Truth (Card 3)**
> **"This card shows the actual label from the held-out test set. When it says 'Correct prediction' in green, the model caught a real mule."**

**5. Selected Account (Card 4)**
> **"And this is the account being analysed — the node ID in the underlying 3.27 million node graph."**

**6. Suspicious Subgraph (the big graph) — MOST IMPORTANT**
> **"This is Novelty 3 in action. Around every flagged account I extract the k=2 hop neighborhood — that's the suspicious subgraph. The center red dot is the mule. The surrounding pink dots are other known mules in the cluster, yellow dots are accounts the model also predicts as fraud, and blue dots are ordinary legitimate accounts that provide context."**

**(Hover an edge)**
> **"Watch — when I hover an edge, you see the Edge-MLP attention weight directly. That value comes from Novelty 1. Thick red edges are the suspicious money flows the model identified."**

**7. Structural Evidence (right column cards)**
> **"These five cards are the quantitative facts my teammate's LLM will cite in its forensic report. Convergence of 4 means four distinct senders converged on this account. Mean drain ratio close to 1.0 confirms senders are emptying their balances — classic mule behavior. Fresh-sender ratio of 0.8 means 80% of senders are one-shot burner accounts — coordinated attack signature."**

**8. JSON payload (click the expander to open it)**
> **"And this is the actual payload sent to my teammate's fusion engine. The pattern field keys into her FATF crime typology vector store. The structural evidence block contains the facts her Chain-of-Evidence prompt forces the LLM to cite. End-to-end, real data, real model, real integration contract."**

**(Close the expander)**

**9. Three Novelty cards at the bottom — point briefly**
> **"To close — Novelty 1 is the Edge-MLP you saw control edge thickness, Novelty 2 is the balanced training that produced this model, and Novelty 3 is the subgraph extraction that produced everything you just saw."**

**End demo. Return to slide deck for closing.**

---

## Part 7 — Closing (45 seconds)

> **"To summarize — my Stage 3a model achieves F1 of 0.54 and AUROC of 0.95 on a test set with 4.75% fraud rate, against a baseline rule that misses 99.81% of fraud. All three architectural novelties are implemented, the JSON contract is locked with Member 4, and the live UI runs end-to-end."**

> **"The remaining timeline is on the WBS — Stage 3 evaluation is at 70% completion, FastAPI integration begins in August per task T8, and the final ablation report is the November deliverable. I'm at approximately 65% overall completion, above the 50% target for this presentation."**

> **"Thank you. I'm happy to take questions."**

Pause. Look at the panel. Wait for questions.

---

## Part 8 — Q&A Bank (anticipate these)

### Technical Questions

**Q: "Why GraphSAGE and not GAT or GCN?"**
> **"GraphSAGE is inductive — it generalizes to previously-unseen accounts without retraining. GAT and GCN are transductive — they require the full graph to be recomputed when new accounts appear. For a real-time fraud detection system, inductive is mandatory."**

**Q: "Why PaySim and not real banking data?"**
> **"PaySim is peer-reviewed — published at EMSS 2016 by Lopez-Rojas. It's synthetic, so contains no PII, eliminating GDPR and local banking-secrecy concerns. It allows reproducible evaluation. Future work would partner with a bank for real-data validation."**

**Q: "Why is your F1 only 0.54? The proposal targeted 0.82."**
> **"That's the honest result. AUROC of 0.95 shows the model has learned strong ranking, but converting ranking into F1 requires careful threshold tuning, and our 5 node features set a ceiling. The proposal's 0.82 target assumed richer node features and is a final-deliverable target for November, not Progress Presentation 1. I'm tracking ahead on the methodology and slightly behind on the absolute metric — both of which are within scope for this stage."**

**Q: "Why label receivers as mules instead of senders?"**
> **"My EDA found that 99.78% of fraud senders are one-shot disposable accounts — burner accounts used once and abandoned. The persistent structural element is the mule on the receiving side. So node classification on receivers gives us a target the model can actually learn at inference time."**

**Q: "What's the difference between Stage 1 and Stage 2?"**
> **"Identical architecture except the convolution layer. Stage 1 uses stock SAGEConv with mean aggregator. Stage 2 replaces it with my EdgeEnhancedSAGEConv that computes per-edge attention from the 6 edge features and uses a weighted-sum aggregator. Any F1 difference is attributable to Novelty 1."**

**Q: "Why k=2 hop?"**
> **"My EDA confirmed PaySim's fraud topology is hub-and-spoke, not multi-hop chains. k=2 captures the sibling-sender convergence pattern: from a flagged transaction, the first hop reaches the mule and the second hop reaches the OTHER senders that fed the same mule. That's the actual fraud signature in PaySim."**

**Q: "What if the model is wrong?"**
> **"The risk-level threshold is configurable per institution. AUROC of 0.95 means the model ranks fraud correctly the vast majority of the time. Compliance officers receive both the score AND the suspicious subgraph evidence — they have the ground to investigate before authorizing any action."**

### Process Questions

**Q: "Stage 3a is better than Stage 3 — doesn't that contradict your novelty claim?"**
> **"It refines the claim. The proposal hypothesized both Focal Loss and the Imbalance Sampler are jointly necessary. I empirically found that on PaySim with my current node features, Focal Loss alone is sufficient for F1 gain, and the Sampler's value is training stability rather than metric improvement. That's honest reporting. The sampler may add F1 value with richer features in future work."**

**Q: "What's the integration with Member 4?"**
> **"My component exposes POST /api/graph/analyze. Her async orchestrator calls it in parallel with the behavioral and temporal models, joined by transaction_id. The JSON I produce contains the relational_risk_score as input to her Logistic Regression meta-classifier, the suspicious_subgraph for her LLM Chain-of-Evidence prompt, and the pattern field that keys into her FATF ChromaDB. The contract is locked in docs/integration/graph_api_contract.md and she has sample JSONs to mock against."**

**Q: "What happens if your service times out?"**
> **"Member 4 implements graceful degradation per her FR2 — if my endpoint times out, she proceeds with available scores from the other two modalities and flags the missing one in the LLM report. That's how multi-modal systems handle partial failure."**

### Demo Questions

**Q: "Show me another transaction."**
> Move the slider in the sidebar to a different rank. Pause. Then narrate what changed.

**Q: "What if I pick that specific account?"**
> Switch the sidebar to "Specific node ID" and enter the number. The model scores it live.

**Q: "Is this really running live?"**
> **"Yes — every time the slider moves, the full forward pass on 3.27 million nodes runs in under a second. The model is loaded into memory, the graph is loaded, and inference is real."**

### Commercialization Questions (LO5)

**Q: "Who is this for commercially?"**
> **"The primary customer persona is the Chief Risk Officer or Head of AML Compliance at a commercial bank or digital wallet provider. The deployment model is Enterprise On-Premise — the model and FastAPI service run inside the bank's firewall. Revenue: annual licensing plus a per-transaction tier."**

**Q: "What's your competitive advantage?"**
> **"Legacy vendors like NICE Actimize rely on static rules and miss organized crime. Modern AI competitors use tabular classifiers that ignore network topology. My component combines inductive graph learning with learned edge attention AND produces forensically-actionable JSON evidence — not just a probability score. That last point is the legal moat: explainability."**

---

## Part 9 — Numbers to Memorize

You will be asked some of these directly.

| Number | What it is |
|---|---|
| **6,362,620** | Total PaySim transactions |
| **8,213** | Total fraud transactions |
| **0.1291%** | Fraud rate |
| **773:1** | Class imbalance |
| **99.81%** | Legacy rule miss rate |
| **66.5%** | Fraud above 200K that the rule ALSO misses (your original finding) |
| **99.78%** | Single-use fraud senders (your original finding) |
| **3,277,509** | Total nodes in your graph |
| **2,770,409** | Total edges (TRANSFER + CASH_OUT only) |
| **8,169** | Mule nodes |
| **332** | Test set mules |
| **0.5387** | Best test F1 (Stage 3a tuned) |
| **0.9497** | Best AUROC |
| **0.5663** | Best recall |
| **9,667** | Model parameters (it's tiny — that's a strength) |

---

## Part 10 — Tips for Delivery

**Voice:**
- Speak slightly louder than feels natural
- Pause between sentences — don't rush
- Lower your voice slightly on KEY numbers — it draws attention

**Body:**
- Stand straight, shoulders back
- Don't hide behind the laptop
- Point at the screen when you describe what's on it
- Make eye contact with each panel member at least once

**Slides:**
- Click forward decisively — don't hover
- Don't read bullets — speak to the bullets
- If a slide is dense, summarize then skip detail

**Questions:**
- ALWAYS repeat the question back: *"You're asking why I labelled receivers rather than senders..."*
- Pause for 2 seconds before answering — looks thoughtful
- If you don't know: *"That's beyond my current scope. I'd address it in T9 evaluation. My honest answer is..."*

**Confidence:**
- You have the work. You did the EDA. You implemented all three novelties. You trained four models. You built a dashboard. **You earned this.**

---

## Part 11 — The Three Things They Should Remember

If the panel forgets everything else, you want them to remember:

1. **The model works** — F1=0.54, AUROC=0.95, 73% improvement over baseline default
2. **The novelties are real** — three distinct components, each measured in isolation
3. **The integration is locked** — JSON contract with Member 4, demo runs live

If you cover these three even in a confused presentation, you pass with margin.

---

## Final reminder

The dashboard you built is the strongest single artifact your panel has seen from any undergraduate fraud-detection project. Trust it. Show it. Let the model speak for itself.

Go in there confident. You've done the work.

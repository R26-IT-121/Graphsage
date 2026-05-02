# PaySim Exploratory Data Analysis — Findings

**Author:** Sachintha Bhashitha Ewaduge
**Dataset:** PaySim Mobile Money Simulator (6,362,620 transactions)
**Source notebook:** [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)
**Status:** All 8 EDA questions answered. Findings inform feature engineering and model architecture.

---

## 1. Dataset overview

PaySim is a peer-reviewed synthetic mobile money simulator (Lopez-Rojas et al., EMSS 2016). It contains **6,362,620 transactions across 743 hourly steps (~31 days)**, with no Personally Identifiable Information.

| Property | Value |
|---|---|
| Total transactions | 6,362,620 |
| Time range | step 1 → 743 (1 step = 1 hour) |
| Columns | 11 (step, type, amount, sender/receiver IDs, balances, fraud labels) |
| File size | 470 MB CSV |
| Memory (optimised dtypes) | 0.89 GB |

---

## 2. Severe class imbalance — 773:1

```
Fraud transactions:        8,213
Legitimate transactions:   6,354,407
Fraud rate:                0.1291%
Imbalance ratio:           773.7 : 1  (legit : fraud)
```

The 0.1291% fraud rate matches the rate stated in the original PaySim paper. Standard accuracy is therefore meaningless: a constant-zero classifier would achieve 99.87% accuracy and zero recall. **This justifies the use of Focal Loss and the topology-preserving Graph-Aware Imbalance Sampler in lieu of SMOTE-style synthetic oversampling.**

---

## 3. Fraud occurs only in TRANSFER and CASH_OUT

| Transaction type | Total count | Fraud count | Fraud rate | % of all fraud |
|---|---:|---:|---:|---:|
| CASH_OUT | 2,237,500 | 4,116 | 0.184% | 50.12% |
| TRANSFER | 532,909 | 4,097 | 0.769% | 49.88% |
| CASH_IN | 1,399,284 | 0 | 0.000% | 0.00% |
| DEBIT | 41,432 | 0 | 0.000% | 0.00% |
| PAYMENT | 2,151,495 | 0 | 0.000% | 0.00% |

PAYMENT, CASH_IN, and DEBIT contain **zero fraud cases** and constitute 56.5% of all transactions. **Methodology decision:** the graph is constructed only from TRANSFER and CASH_OUT transactions (2,770,409 records), preserving 100% of fraud while discarding 56% of non-informative records.

TRANSFER has a fraud rate **4× higher** than CASH_OUT (0.769% vs 0.184%), which justifies including transaction type as an edge feature in the Edge-MLP attention layer.

> *Figure 1 — `data/processed/fig_fraud_by_type.png`: per-type transaction volume vs per-type fraud rate.*

---

## 4. Failure of the legacy `isFlaggedFraud` rule

The PaySim dataset includes a baseline rule-based flag (triggered nominally when a TRANSFER exceeds 200,000). Its measured performance:

| Metric | Value |
|---|---:|
| True positives  | 16 |
| False negatives (missed fraud) | 8,197 |
| False positives | 0 |
| Recall | 0.0019 |
| F1-score | 0.0039 |
| **Miss rate** | **99.81%** |

**Two original findings beyond the published PaySim description:**

1. **The rule's miss rate is independent of the 200,000 threshold.** Of the 8,197 missed frauds, 5,455 (66.5%) had amounts *above* 200,000 — they should have triggered the rule but did not. This indicates additional gating logic that further narrows the rule's coverage.
2. **33.5% of missed fraud is below the 200,000 threshold**, consistent with deliberate "smurfing" — splitting transfers below the regulatory threshold to evade detection.

**This is the empirical foundation for the entire research:** any improvement over a baseline rule that catches 0.19% of fraud is meaningful.

---

## 5. Bimodal fraud amount distribution

Restricted to TRANSFER + CASH_OUT:

| Statistic | Legitimate | Fraud |
|---|---:|---:|
| Median amount | 171,034 | 441,423 |
| Mean amount | 314,116 | 1,467,967 |
| 95th percentile | 942,415 | 8,006,429 |
| Max | 92,445,520 | 10,000,000 (PaySim cap) |

The fraud distribution is **bimodal**:
- **Low peak (~100-500)** — extreme smurfing
- **High peak (~100K-1M)** — structuring just below the rule threshold

The two distributions overlap heavily. **A tabular classifier examining only `amount` cannot reliably separate them — relational context is required.**

> *Figure 2 — `data/processed/fig_amount_distribution.png`: log-scale density histogram of fraud vs legitimate amounts, with the 200,000 threshold marked.*

---

## 6. Temporal patterns — bursty fraud, dropping legitimate volume

Total transaction volume follows a clear day/night cycle for the first ~400 hours, then drops by approximately 90%. Fraud volume remains roughly constant throughout (~10–20 cases per hour with sporadic spikes to 30–40), producing a fraud-rate spike in the second half of the simulation.

**Six 1-hour windows contain 100% fraud transactions** (steps 66, 387, 425, 501, 523, 730), with 24–30 fraud cases each and zero legitimate activity. These are coordinated burst attacks during low-activity windows.

> *Figure 3 — `data/processed/fig_fraud_temporal.png`: total transaction volume (top) vs fraud cases per hour (bottom).*

---

## 7. Original finding: fraud uses single-use sender accounts

Per-sender time-gap analysis revealed that **only 18 of 8,213 fraud transactions (0.22%) have a previous transaction from the same sender**. The remaining 8,195 fraud transactions originate from sender accounts that have **no transaction history** in the dataset.

**Implication:** sender-side history (account age, transaction count, behavioural baseline) is *useless* as a fraud feature in PaySim. All discriminative information lives in:

1. The **destination** (mule receivers — see §9)
2. The **transaction's own attributes** (amount, drain_ratio, balance flags)
3. The **relational structure** (which other senders converge on the same destination)

This finding is the strongest possible empirical argument for a graph-based approach over a tabular classifier.

---

## 8. Drain & balance signature — the cleanest discriminator

| Feature | Legitimate | Fraud | Lift |
|---|---:|---:|---:|
| Median drain_ratio = amount/oldbalanceOrg | 0.220 | **1.000** | 4.5× |
| `src_drained` (newbalanceOrig == 0) | 90.1% | 98.1% | 1.09× |
| `dst_was_empty` (oldbalanceDest == 0) | 13.9% | **65.2%** | **4.7×** |
| BOTH flags simultaneously | 11.6% | **63.2%** | 5.4× |

Two key observations:

1. **Median drain_ratio is the cleanest single discriminator.** Fraud senders move exactly 100% of their balance (median = 1.000); legitimate senders move 22%.
2. **`src_drained` alone is weak** (only 1.09× lift) due to a PaySim simulation artefact in which 90% of legitimate transactions also empty the sender. It only becomes useful in combination with `dst_was_empty`.

When **both flags fire**, fraud probability is 5.4× the baseline. This combined signature is encoded in the Edge-MLP attention layer.

---

## 9. Mule topology — distributed hub-and-spoke, not multi-hop chains

Per-receiver analysis revealed the **PaySim fraud topology is hub-and-spoke**, not relay chains:

| Property | Measurement |
|---|---|
| Total destination accounts | 509,565 |
| Median transactions received per destination | 3 |
| Maximum fraud per single mule | **2 cases** (no super-mules exist) |
| Accounts that BOTH received fraud AND later sent money | **18** (of 509,565 = 0.0035%) |

The 18 candidate "relay mules" do not form genuine multi-hop chains — in the example traced, the outbound transaction occurred 606 hours *before* the inbound fraud, indicating coincidental account reuse rather than a laundering chain.

**The empirical fraud structure in PaySim is therefore:**

```
[Fresh sender A (one-shot)] ──fraud──┐
[Fresh sender B (one-shot)] ──fraud──┼──> Mule M ──CASH_OUT──> exit
[Fresh sender C (one-shot)] ──fraud──┘
```

Multiple fresh senders converge on a shared mule; the mule cashes out; money exits the system. **No intermediate relay step exists.**

> *Figure 4 — `data/processed/fig_fraud_ring.png`: visualisation of mule `C964377943` with four senders (two fraudulent, two legitimate), illustrating the mixed-traffic hub-and-spoke pattern.*

**Mules also receive legitimate traffic.** A single account `C964377943` received four transactions: two fraud and two legitimate. This rules out simple blacklist approaches: an account cannot be flagged purely on having received any fraudulent transfer. Instead, a model must reason over the *full neighbourhood* — which is precisely the mechanism a Graph Neural Network provides.

---

## 10. Implications for methodology

The EDA findings directly inform every architectural decision in the proposed model:

| Finding | Methodology decision |
|---|---|
| Fraud only in TRANSFER + CASH_OUT (§3) | Graph constructed from these two types only; PAYMENT/CASH_IN/DEBIT excluded |
| 99.81% miss rate of legacy rule (§4) | Justifies any model improvement; baseline for comparison |
| Bimodal amounts overlap with legitimate (§5) | Tabular classifier insufficient; graph context required |
| Single-use senders (§7) | Per-sender history features omitted; per-destination and per-edge features prioritised |
| Median drain_ratio = 1.0 for fraud (§8) | Encoded as edge feature; likely high attention weight in Edge-MLP |
| `dst_was_empty` strongest single signal (§8) | Encoded as edge feature |
| `src_drained` weak alone, strong combined (§8) | Encoded as edge feature; combination learned by MLP |
| Hub-and-spoke topology, no multi-hop chains (§9) | k=2 hop justified as **sibling-sender convergence** detection, not relay tracing |
| Mules receive mixed fraud + legit traffic (§9) | Confirms graph context superior to blacklist approaches |
| 773:1 imbalance (§2) | Focal Loss (γ=2) + Graph-Aware Imbalance Sampler |

## 11. Edge feature set (final)

Based on the above analysis, every edge in the constructed graph carries the following six attributes:

```
edge_attr = [
    amount_log,        # log1p(amount), reduces heavy tail
    drain_ratio,       # amount / oldbalanceOrg (clamped 0-1)
    src_drained,       # 1 if newbalanceOrig == 0, else 0
    dst_was_empty,     # 1 if oldbalanceDest == 0, else 0
    time_gap,          # hours since destination's previous inbound (per-destination)
    type_is_transfer,  # 1 if TRANSFER, 0 if CASH_OUT
]
```

`time_gap` is computed **per-destination** rather than per-sender, because §7 established that senders are one-shot. This is a methodology refinement justified by EDA evidence.

---

## 12. Open questions deferred to subsequent phases

1. **Optimal k-hop depth** — empirically PaySim has 1-hop fraud, but k=2 captures sibling convergence. To be confirmed in T7 hyperparameter tuning.
2. **Edge directionality** — currently directed. Whether to add reverse edges is a graph-construction (T4) decision.
3. **Node features** — what to assign to nodes given senders are one-shot. Likely a small constant or identity embedding; details in T4.

---

*This report covers WBS task T3. Generated 2026-04-27.*

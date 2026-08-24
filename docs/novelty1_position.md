# Position paper — what the Edge-MLP (Novelty 1) actually contributes

**Status:** for discussion with the supervisor before the dissertation is written.
**Short version:** the proposal claims the Edge-MLP improves detection accuracy.
Our own ablation disproves that claim. The mechanism should be re-scoped as an
*explainability* component, which is what it demonstrably does well, and the
negative accuracy result should be reported as a finding rather than buried.

---

## 1. What the proposal claims

Proposal §3.2 presents the Edge-MLP as the first architectural novelty: an MLP
inside the GraphSAGE message-passing step that computes a per-edge attention
weight from `(amount, drain_ratio, src_drained, dst_was_empty, time_gap,
txn_type)`, so that "suspicious edges dominate aggregation instead of being
averaged away." §3.5 projects this as a jump from F1 ~0.55 to ~0.74.

That is a **predictive** claim: the mechanism is asserted to make the model
better at finding fraud.

## 2. What the evidence shows

Three independent tests, all under the leakage-free temporal protocol, five
seeds each, across-seed paired t-test.

| Test | Comparison | ΔF1 | ΔPR-AUC | Verdict |
|---|---|---|---|---|
| A — additive | Stage 2 vs Stage 1 | +0.0002 (p=0.998) | −0.0239 | no effect |
| B — leave-one-out, v1 features | 3b vs 3c | +0.0004 (p=0.962) | −0.0498 (p=0.088) | no effect |
| C — leave-one-out, v2 features | 3b vs 3c | **−0.0102 (p=0.031)** | **−0.0512 (p=0.013)** | **significantly worse** |

Test A asks "does adding it help?" — no. Test B removes it from the full system
— no change. Test C repeats the removal with the stronger 12-dim feature set —
and removing the Edge-MLP **significantly improves** both F1 and PR-AUC.

The three tests are mutually consistent and the effect strengthens as the rest
of the system gets better. This is not a marginal or noisy result: test C is
significant at p<0.05 on both metrics simultaneously.

### Why this happens (the mechanism, not just the number)

The six edge features the Edge-MLP consumes are largely **recoverable from the
12-dim node feature set** — drain ratios, balance-emptying behaviour and timing
are aggregated per account in v2. Once the node features carry that signal, the
per-edge attention adds parameters and a learned re-weighting that can only
distort an aggregation which was already correct. With v1's 5-dim features the
mechanism was redundant (tests A and B: no effect); with v2's richer features it
becomes actively harmful (test C).

This also explains the *stability* result: Stage 3c-v2 has std 0.0026 across
seeds versus 0.0077 for 3b-v2. Removing the extra learned component removes a
source of seed variance.

## 3. What the Edge-MLP does earn its place doing

The attention weights are not decoration. `SuspiciousSubgraphExtractor`
(Novelty 3) uses them to rank which transfers implicate a flagged account — the
`weight` field on every edge in the `suspicious_subgraph` payload that Member 4's
forensic LLM cites. Without per-edge attention, the extractor can return *which*
accounts are connected but not *which transfers matter most*, and the forensic
narrative loses its evidentiary ordering.

So the mechanism is doing real work in the system — it is simply doing
**explainability** work, not accuracy work.

## 4. Recommended position

> The Edge-MLP is an explainability mechanism that supplies per-edge evidence
> weights to the forensic extractor. It is retained at a measured cost of
> ~0.01 F1. It is not claimed as a source of predictive gain; our ablation shows
> that claim does not hold.

Accuracy is then attributed where the evidence puts it:

| Step | F1 | Gain |
|---|---|---|
| Stage 1, v1 features | 0.2806 | — |
| Stage 1, **v2 features** | 0.3780 | **+0.097** (features alone) |
| Stage 3c, v2 features | 0.4056 | **+0.028** (sampler on top) |
| | | **+0.125 total (p=0.045)** |

Roughly **78% of the headline gain is the 12-dim behavioural features** and 22%
the imbalance sampler. The Edge-MLP does not appear in this decomposition
because the best arm (3c) does not contain it; its effect is measured separately
as **-0.0102 F1 (p=0.031)** against the otherwise-identical 3b-v2 arm.

## 5. Why this is a stronger dissertation, not a weaker one

- A pre-registered architectural claim, tested three ways and rejected on its
  own evidence, is exactly what an ablation study is *for*. Reporting it
  demonstrates methodological integrity that a confirmed hypothesis would not.
- The honest number (F1 0.406, leakage-free, 5 seeds, significance-tested,
  ECE 0.024) is defensible under questioning. The proposal's 0.82 was measured
  under a random split that leaked future information and would not survive it.
- The system still shows two **significant** positive results (ΔF1 +0.125,
  p=0.045; ΔPR-AUC +0.207, p=0.042) and a 33× stability improvement. The
  contribution stands without the Edge-MLP accuracy claim.
- Negative results on attention mechanisms in GNNs are an established finding in
  the literature, not an anomaly — the result is publishable context, not an
  embarrassment.

## 6. Options, if the supervisor prefers a different route

1. **Re-scope (recommended).** Keep the mechanism, re-frame the claim as above.
   No further experiments needed; everything is already measured.
2. **Drop it from the novelty list.** Ship Stage 3c-v2 as the final system and
   present two novelties instead of three. Costs the forensic edge-ranking, and
   Novelty 3's payload gets weaker.
3. **Try to rescue the accuracy claim.** Would need the Edge-MLP redesigned so
   it consumes something the node features cannot express — genuinely new edge
   signal rather than aggregatable behaviour. This is new experimental work with
   no guarantee, and the PP2 timeline does not accommodate it.

## 7. Questions to put to the supervisor

- Is re-scoping a proposal novelty mid-project acceptable, and does it need to
  be documented as a formal change?
- Should the negative result appear in the results chapter, or as its own
  subsection in the discussion?
- The proposal's §3.5 targets were set under a leaky protocol. Should the
  dissertation restate the targets against the corrected protocol, or present
  both with the correction explained?

---

**Evidence:** full ablation in [system_walkthrough.md](system_walkthrough.md)
§14; statistics produced by `scripts/eval_statistics.py`
(`novelty1_leave_one_out` section).

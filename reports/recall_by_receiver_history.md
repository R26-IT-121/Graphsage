# Detection collapses as the receiving account gains history

## The measurement

Every transaction in the held-out window (steps 701–743), grouped by how many
incoming transfers the **destination** account already had in the served graph.
Scored at the model's own tuned threshold, 0.1830.

| incoming transfers | rows | fraud | caught | recall |
|---|---|---|---|---|
| 1 — a first-time receiver | 1,778 | 332 | 121 | **0.3645** |
| 2 | 757 | 35 | 6 | 0.1714 |
| 3 | 522 | 27 | 3 | 0.1111 |
| 4 | 396 | 24 | 2 | 0.0833 |
| 5 | 307 | 18 | 1 | 0.0556 |
| 6 | 234 | 5 | 0 | 0.0000 |
| 7 | 186 | 7 | 0 | 0.0000 |
| 8 or more | 1,037 | 42 | 0 | **0.0000** |

Monotonic across every bucket, with no exceptions. The detector finds roughly
one fraud in three when the receiver is new, and none at all once the receiver
has six or more prior incoming transfers.

## Why

A first-time receiver taking a drained balance is a *shape*: the sender goes to
zero, the receiver was empty, and there is exactly one edge. Four of the twelve
node features encode parts of that event directly — the fresh-receiver ratio,
outgoing drain, in-degree and first appearance.

An established account receiving one more payment has none of that shape. Its
in-degree is unremarkable, it was not empty, and the incoming amount sits inside
a distribution the account has already established. There is nothing structural
left to key on, so the model returns something near its prior.

## What follows from it

**This is the mechanism behind the headline recall of 0.271.** That number is a
weighted average over a population where 66% of transactions land on accounts
with prior history — the regime where the detector cannot work. It is not a
tuning problem and more training will not move it.

**It is also the argument for fusing three detectors.** The cases this detector
provably cannot see — an established account receiving a payment that is wrong
for reasons of timing or of the sender's own habits — are exactly the cases the
temporal and behavioural models are built for.

**And it is a demonstrable claim, not an assertion.** Two files in the demo
folder isolate the two regimes on identical sample sizes and identical fraud
counts:

    graphsage_best_1000_fresh_receiver.csv         recall 0.389
    graphsage_hard_1000_established_receiver.csv   recall 0.014

Rows were selected on a property of the **input** — the receiver's prior history
— never on the label and never on what the model predicted. Accuracy across the
two differs by 0.013 while recall differs 28-fold, which is the clearest
available argument for why accuracy is the wrong metric on this problem.

Raw numbers: `reports/recall_by_receiver_history.json`.

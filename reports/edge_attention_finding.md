# The edge-attention layer was aggregating wrongly

## What the ablation said

Under the leakage-free temporal protocol, three seeds per stage:

| stage | | val F1 (mean ± std) |
|---|---|---|
| 1 | baseline GraphSAGE | 0.3198 ± 0.0491 |
| 2 | + Edge-MLP attention | **0.0585 ± 0.0000** |
| 3a | + focal loss | 0.1559 ± 0.1527 |
| 3b | + graph-aware sampler | 0.2980 ± 0.0083 |
| 3c | 3b with the Edge-MLP removed | **0.3620 ± 0.0040** |

Stage 2 was worse than the baseline it extends, identically across all three
seeds. Stage 3c — the leave-one-out arm, which is 3b minus edge attention —
beat 3b on every seed. Read at face value, the first contribution did not work.

## Why that reading was wrong

Stage 2 differs from stage 1 in **two** ways, not one.

`BaselineGraphSAGE` uses PyTorch Geometric's `SAGEConv` with `aggr="mean"`.
`EdgeEnhancedSAGEConv` was written with `aggr="add"`, on the reasoning that a
mean would normalise the attention away. It does something else as well: it
removes the neighbourhood normalisation the baseline had. In-degree in this
graph reaches 75, so a hub's aggregated message arrives at layer two scaled by
its degree, while a leaf's does not.

So the comparison never isolated attention. It measured attention *plus* the
loss of normalisation, and the second term dominated.

The training curves show it plainly. Every stage containing the Edge-MLP trains
pathologically; every stage without it trains normally:

| stage | Edge-MLP | val F1 by epoch |
|---|---|---|
| 1  | no  | 0.329 → **0.333** → 0.332 |
| 2  | yes | 0.029 → 0.058 → 0.058 — never learned |
| 3b | yes | **0.306** → 0.295 → 0.291 — peaks at epoch 1, then declines |
| 3c | no  | 0.339 → 0.362 → **0.364** |

## The test

Three candidate remedies, each a one-flag run on stage 2:

| variant | val F1 | val PR-AUC | test AUC |
|---|---|---|---|
| as published | 0.0585 | 0.0196 | 0.6984 |
| gradient clipping | 0.0585 | 0.0206 | 0.7138 |
| attention init bias 3.0 | 0.0585 | 0.0195 | 0.6960 |
| **attention-weighted mean** | **0.2777** | **0.2073** | **0.8384** |

Only the normalisation moved it, and it moved it by a factor of ten on PR-AUC.
Clipping and initialisation changed nothing, which rules out the two obvious
alternative explanations — exploding gradients and a bad starting point.

## What it means for the contribution

With the aggregation corrected, edge attention beats the arm that removes it:

| configuration | val F1 | test F1 | test AUC | test PR-AUC |
|---|---|---|---|---|
| 3b as served | 0.3060 | 0.3895 | 0.8628 | 0.4252 |
| 3c — no attention (3 seeds) | 0.3620 | 0.4043 | 0.8806 | 0.4477 |
| **3b + attention-weighted mean** | **0.3699** | **0.4113** | **0.8839** | **0.4535** |

The corrected layer is the best configuration measured. **This is one seed
against 3c's three, and confirmation across seeds 1 and 2 is outstanding.**

## What this says about the method

The bug was invisible in the headline numbers. It was caught because the
ablation included a leave-one-out arm — stage 3c, 3b minus the Edge-MLP —
rather than only the additive chain 1 → 2 → 3a → 3b. The additive chain would
have shown stage 2 failing and invited the conclusion that attention does not
help on sparse transaction graphs. It was the arm that *removes* the component
from the finished system that made the contradiction visible: a component
cannot be both useless in isolation and load-bearing in situ.

The serving bundle in production was exported from the uncorrected stage 3b.
Checkpoints now record `attn_norm` and `attn_init_bias`, and both the serving
path and the export script read the architecture from the checkpoint rather
than from class defaults, so a model can no longer be scored under an
aggregation it was not trained with.

# Results

All numbers are block hit ratios in %, on the held-out hour 2 of each trace, unless stated.
Tables between markers are generated from `results/` by `python -m pacer.report fill`.
Raw output: `results/final_test.jsonl` (one JSON line per run, with 5-minute buckets) and
`results/final_summary.json` (with confidence intervals). Intervals are paired 95%
t-intervals over the twelve 5-minute buckets of the hour.

## Main comparison

Pacer here is the fixed configuration (compact model, h x1, K = 32, reuse 4) for every trace
and size. The best heuristic is, for each cell, the best of LRU, FIFO, LFU, SIEVE, ARC (2
settings), S3-FIFO (24), LRU with quick demotion (7), GDSF (4) and ATC'25 (2 fits), with each
family's setting tuned on validation and the family picked on the test hour.

<!-- results:main -->
| Trace | Cache | LRU | Best tuned heuristic | ATC'25 | **Pacer** | Pacer vs best (95% CI) | Belady |
|---|---|---|---|---|---|---|---|
| To-C chat | 1% | 15.88 | 19.58 (S3-FIFO) | 19.04 | **21.85** | +2.27 [+1.80, +2.75] | 39.77 |
| To-C chat | 2% | 24.36 | 27.97 (ATC'25) | 27.97 | **30.19** | +2.23 [+1.47, +2.98] | 55.16 |
| To-C chat | 5% | 45.60 | 46.06 (ATC'25) | 46.06 | **46.16** | +0.10 [-0.60, +0.80] | 67.77 |
| To-C chat | 10% | 58.57 | 58.74 (ATC'25) | 58.74 | **58.78** | +0.04 [-0.27, +0.35] | 70.68 |
| To-B API | 1% | 46.23 | 51.87 (ARC) | 45.97 | **52.32** | +0.46 [+0.34, +0.57] | 53.74 |
| To-B API | 2% | 48.97 | 52.42 (S3-FIFO) | 49.10 | **53.08** | +0.66 [+0.57, +0.76] | 53.96 |
| To-B API | 5% | 51.22 | 52.94 (S3-FIFO) | 51.36 | **53.34** | +0.40 [+0.30, +0.50] | 53.96 |
| To-B API | 10% | 52.36 | 53.23 (S3-FIFO) | 52.34 | **53.46** | +0.23 [+0.16, +0.30] | 53.96 |
| Coder | 1% | 37.81 | 39.30 (GDSF) | 37.25 | **41.26** | +1.96 [+1.63, +2.29] | 53.53 |
| Coder | 2% | 45.89 | 46.42 (ATC'25) | 46.42 | **48.17** | +1.75 [+1.27, +2.22] | 64.26 |
| Coder | 5% | 57.31 | 57.66 (ARC) | 57.21 | **58.47** | +0.82 [+0.33, +1.30] | 72.03 |
| Coder | 10% | 64.01 | 64.01 (LRU) | 63.68 | **64.62** | +0.62 [+0.22, +1.01] | 73.29 |
| Thinking | 1% | 30.17 | 32.18 (ARC) | 31.15 | **33.13** | +0.95 [+0.27, +1.63] | 38.27 |
| Thinking | 2% | 32.61 | 33.76 (ARC) | 33.67 | **34.97** | +1.21 [+0.66, +1.75] | 38.91 |
| Thinking | 5% | 35.46 | 35.93 (LRU+QD) | 35.84 | **36.76** | +0.83 [+0.59, +1.07] | 38.91 |
| Thinking | 10% | 37.14 | 37.50 (LRU+QD) | 37.27 | **37.55** | +0.05 [-0.23, +0.33] | 38.91 |

Pacer vs the best heuristic: better in 13 cells, statistically tied in 3, worse in 0.
<!-- /results:main -->

## Both configurations, and the gap to Belady

Gap closed is (policy - LRU) / (Belady - LRU). The horizon-tuned configuration uses a larger
model (63 leaves, 85-213 trees), K = 64 and no burst reuse, with the horizon multiplier chosen
per cell on validation. Despite that tuning it is below the fixed configuration on chat at 2-10%
(by 0.3-0.6 points) and above it on coder (by 0.1-0.6).

<!-- results:detail -->
| Trace | Cache | Pacer (fixed) | Pacer (horizon tuned) | vs LRU | vs ATC'25 | gap closed (Pacer / best heuristic) |
|---|---|---|---|---|---|---|
| To-C chat | 1% | 21.85 | 22.05 | +5.97 [+4.94, +7.00] | +2.81 [+1.57, +4.06] | 25% / 15% |
| To-C chat | 2% | 30.19 | 29.87 | +5.84 [+4.02, +7.65] | +2.23 [+1.47, +2.98] | 19% / 12% |
| To-C chat | 5% | 46.16 | 45.61 | +0.56 [-0.12, +1.24] | +0.10 [-0.60, +0.80] | 3% / 2% |
| To-C chat | 10% | 58.78 | 58.37 | +0.21 [-0.20, +0.62] | +0.04 [-0.27, +0.35] | 2% / 1% |
| To-B API | 1% | 52.32 | 52.54 | +6.09 [+5.64, +6.55] | +6.36 [+5.89, +6.83] | 81% / 75% |
| To-B API | 2% | 53.08 | 53.15 | +4.11 [+3.74, +4.49] | +3.98 [+3.65, +4.31] | 82% / 69% |
| To-B API | 5% | 53.34 | 53.35 | +2.11 [+1.89, +2.34] | +1.97 [+1.73, +2.22] | 77% / 63% |
| To-B API | 10% | 53.46 | 53.54 | +1.10 [+0.95, +1.26] | +1.12 [+0.93, +1.31] | 69% / 55% |
| Coder | 1% | 41.26 | 41.86 | +3.45 [+3.12, +3.78] | +4.02 [+3.53, +4.50] | 22% / 9% |
| Coder | 2% | 48.17 | 48.79 | +2.28 [+1.80, +2.76] | +1.75 [+1.27, +2.22] | 12% / 3% |
| Coder | 5% | 58.47 | 58.82 | +1.16 [+0.73, +1.59] | +1.26 [+0.80, +1.73] | 8% / 2% |
| Coder | 10% | 64.62 | 64.71 | +0.62 [+0.22, +1.01] | +0.95 [+0.61, +1.29] | 7% / 0% |
| Thinking | 1% | 33.13 | 33.63 | +2.96 [+1.63, +4.29] | +1.98 [+0.92, +3.05] | 37% / 25% |
| Thinking | 2% | 34.97 | 35.08 | +2.35 [+1.73, +2.98] | +1.29 [+0.50, +2.09] | 37% / 18% |
| Thinking | 5% | 36.76 | 36.61 | +1.30 [+0.87, +1.74] | +0.92 [+0.49, +1.36] | 38% / 14% |
| Thinking | 10% | 37.55 | 37.43 | +0.42 [+0.03, +0.80] | +0.28 [-0.14, +0.70] | 23% / 21% |
<!-- /results:detail -->

## Every baseline

<!-- results:baselines -->
| Trace | Cache | LRU | FIFO | LFU | SIEVE | ARC | S3-FIFO | LRU+QD | GDSF | ATC'25 | sampled LRU | oracle: never reused | sampled Belady | Belady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| To-C chat | 1% | 15.88 | 15.88 | 14.86 | 14.43 | 16.45 | 19.58 | 16.79 | 16.47 | 19.04 | 15.90 | 20.87 | 39.52 | 39.77 |
| To-C chat | 2% | 24.36 | 24.36 | 14.94 | 20.49 | 24.36 | 22.10 | 24.61 | 25.84 | 27.97 | 24.42 | 39.80 | 54.83 | 55.16 |
| To-C chat | 5% | 45.60 | 45.60 | 14.99 | 42.08 | 45.79 | 42.70 | 45.78 | 44.98 | 46.06 | 45.66 | 63.96 | 67.40 | 67.77 |
| To-C chat | 10% | 58.57 | 58.57 | 15.16 | 53.79 | 58.19 | 55.52 | 58.56 | 57.56 | 58.74 | 58.54 | 70.53 | 70.51 | 70.68 |
| To-B API | 1% | 46.23 | 46.21 | 51.03 | 39.33 | 51.87 | 51.83 | 51.28 | 49.54 | 45.97 | 46.30 | 53.21 | 53.25 | 53.74 |
| To-B API | 2% | 48.97 | 48.96 | 52.21 | 43.90 | 52.41 | 52.42 | 52.21 | 50.94 | 49.10 | 49.01 | 53.42 | 53.42 | 53.96 |
| To-B API | 5% | 51.22 | 51.21 | 52.83 | 48.05 | 52.86 | 52.94 | 52.83 | 52.33 | 51.36 | 51.25 | 53.49 | 53.47 | 53.96 |
| To-B API | 10% | 52.36 | 52.35 | 53.16 | 50.21 | 53.19 | 53.23 | 53.16 | 52.93 | 52.34 | 52.38 | 53.54 | 53.55 | 53.96 |
| Coder | 1% | 37.81 | 37.81 | 36.30 | 26.12 | 38.14 | 37.78 | 38.86 | 39.30 | 37.25 | 37.86 | 43.67 | 53.09 | 53.53 |
| Coder | 2% | 45.89 | 45.88 | 37.46 | 38.08 | 45.88 | 42.54 | 46.33 | 45.95 | 46.42 | 45.99 | 57.66 | 63.35 | 64.26 |
| Coder | 5% | 57.31 | 57.31 | 38.86 | 50.48 | 57.66 | 50.66 | 57.37 | 56.73 | 57.21 | 57.33 | 69.99 | 70.69 | 72.03 |
| Coder | 10% | 64.01 | 64.01 | 40.43 | 57.94 | 63.89 | 57.04 | 63.88 | 63.36 | 63.68 | 64.06 | 72.32 | 72.27 | 73.29 |
| Thinking | 1% | 30.17 | 29.83 | 29.90 | 20.72 | 32.18 | 30.27 | 31.90 | 31.22 | 31.15 | 30.21 | 37.42 | 37.47 | 38.27 |
| Thinking | 2% | 32.61 | 32.57 | 30.38 | 23.28 | 33.76 | 30.82 | 33.43 | 33.30 | 33.67 | 32.66 | 38.25 | 38.21 | 38.91 |
| Thinking | 5% | 35.46 | 35.40 | 33.54 | 26.93 | 34.37 | 32.97 | 35.93 | 35.81 | 35.84 | 35.43 | 38.51 | 38.44 | 38.91 |
| Thinking | 10% | 37.14 | 37.09 | 36.92 | 30.10 | 36.93 | 35.44 | 37.50 | 37.41 | 37.27 | 37.14 | 38.70 | 38.72 | 38.91 |
<!-- /results:baselines -->

Notes:

- LFU collapses on chat (15% at every size): a conversation's next turn is a block's *second*
  access, which LFU treats as the least valuable. LRU with quick demotion collapsed the same way
  at its default setting (probation queue of 10%, `results/history/v1/baselines.jsonl`); tuned
  on validation it matches LRU. On API traffic, where reuse is shared templates, both are among
  the best.
- ATC'25 does no better than LRU on API traffic: every request there is a first turn, so its
  per-workload model has only two near-identical workloads (API and text) to separate.
- Sampled Belady (K = 64 with perfect knowledge of next use, and chain following) is within
  0.18-1.34 points of Belady, so sampling is not what limits Pacer.

## Compute-weighted hit ratio

A hit at depth j saves the prefill of a block whose cost grows with the context before it.
With the cost model measured on the M5 Pro (`results/live/prefill_cost.json`), the share of
prefill compute saved is below; the best heuristic is picked by this measure here.

<!-- results:csr -->
| Trace | Cache | LRU | best heuristic | ATC'25 | Pacer | Belady |
|---|---|---|---|---|---|---|
| To-C chat | 1% | 8.87 | 12.56 | 12.04 | 17.83 | 37.33 |
| To-C chat | 2% | 16.73 | 27.44 | 27.44 | 26.77 | 53.97 |
| To-C chat | 5% | 44.03 | 44.42 | 44.42 | 43.36 | 66.88 |
| To-C chat | 10% | 57.61 | 58.33 | 58.33 | 57.73 | 69.66 |
| To-B API | 1% | 40.30 | 45.67 | 40.06 | 45.99 | 47.40 |
| To-B API | 2% | 42.80 | 46.17 | 42.93 | 46.76 | 47.59 |
| To-B API | 5% | 44.94 | 46.65 | 45.10 | 46.99 | 47.59 |
| To-B API | 10% | 46.11 | 46.94 | 46.01 | 47.09 | 47.59 |
| Coder | 1% | 39.90 | 42.00 | 39.18 | 43.63 | 54.82 |
| Coder | 2% | 47.87 | 48.40 | 48.40 | 49.85 | 64.66 |
| Coder | 5% | 58.28 | 58.59 | 58.18 | 58.93 | 71.75 |
| Coder | 10% | 64.45 | 64.45 | 64.02 | 64.62 | 72.86 |
| Thinking | 1% | 22.86 | 24.73 | 24.31 | 25.68 | 31.50 |
| Thinking | 2% | 25.64 | 26.92 | 26.92 | 27.73 | 32.11 |
| Thinking | 5% | 28.50 | 29.00 | 29.00 | 29.60 | 32.11 |
| Thinking | 10% | 30.48 | 30.81 | 30.58 | 30.69 | 32.11 |
<!-- /results:csr -->

By compute saved, Pacer leads on API traffic and coder at every size, on thinking up to 5%, and on
chat at 1% (17.8 against 12.6). On chat at 2-10% it trails ATC'25 by 0.6-1.1 points, and at 5%
falls below LRU: it tends to evict long conversations, whose deep blocks cost the most to
recompute. Scoring candidates by expected saved compute (reuse probability times block cost)
instead of reuse probability is the obvious next step. Belady here is the hit-optimal schedule,
not the compute-optimal one.

## Prediction quality

Held-out AUC and log loss on rows starting in minutes 52.5-60 of hour 1, same rows for both
models (`results/training/fast_*.json`). Those rows must end by minute 60, so their windows are
at most about 7.5 minutes long.

<!-- results:auc -->
| Trace | Pacer AUC | ATC'25 AUC | Pacer log loss | ATC'25 log loss | constant log loss | trees |
|---|---|---|---|---|---|---|
| To-C chat | 0.842 | 0.736 | 0.313 | 0.407 | 0.464 | 13 |
| To-B API | 0.989 | 0.773 | 0.117 | 0.464 | 0.555 | 16 |
| Coder | 0.901 | 0.702 | 0.319 | 0.521 | 0.567 | 37 |
| Thinking | 0.985 | 0.783 | 0.116 | 0.355 | 0.426 | 19 |
<!-- /results:auc -->

By horizon, on chat: AUC 0.87 against 0.71 for h under a minute, 0.71 against 0.64 for 1-5
minutes, and 0.55 against 0.57 on the 2,131 rows with h of 5 minutes or more. On coder the model
stays ahead at every horizon (0.83 against 0.63 at 5 minutes or more).

Whether a request gets a follow-up turn at all, from its metadata (type, turn, lengths, previous
gap), trained on hour 1 and tested on hour 2: AUC 0.63 on chat (0.61 from the turn number
alone), 0.77 on coder (`results/trace_stats.json`). That is a limit for this feature set on chat
at larger cache sizes, where the decision is mostly which conversations have ended.

## Ablations

Around the fixed configuration, on the test hour after every choice was made; for analysis, not
selection. AUCs are on the same held-out rows as above.

<!-- results:ablations -->
**To-C chat**

| Variant | 1% | 2% | 5% | 10% |
|---|---|---|---|---|
| Pacer, fixed configuration (compact model, K = 32, reuse 4, h x1) | 21.85 | 30.19 | 46.16 | 58.78 |
| larger model (63 leaves, 85-172 trees), K = 64, no reuse | 22.05 | 30.23 | 45.61 | 58.35 |
| without session features (gaps, turn, session age) | 21.06 | 29.45 | 45.29 | 57.44 |
| without block history (hit count, fan-out) | 21.01 | 30.22 | 46.12 | 58.79 |
| K = 8 | 20.33 | 28.99 | 45.67 | 57.92 |
| K = 16 | 20.97 | 29.78 | 45.67 | 58.60 |
| K = 64 | 22.16 | 30.76 | 46.12 | 58.61 |
| K = 128 | 22.31 | 31.18 | 46.01 | 58.63 |
| no burst reuse | 22.19 | 30.70 | 45.88 | 58.63 |
| no chain following | - | 29.28 | - | 58.56 |
| h x0.5 | 21.57 | 30.14 | 45.83 | 58.30 |
| h x2 | 21.90 | 30.26 | 46.07 | 58.59 |
| h x4 | 21.62 | 29.69 | 46.16 | 58.54 |
| LRB-style regression on logged cache states (v1 approach) | 21.74 | 29.05 | 45.57 | 57.52 |
| sampled LRU (K = 32, no model) | 15.92 | 24.59 | 45.64 | 58.61 |

**Coder**

| Variant | 1% | 2% | 5% | 10% |
|---|---|---|---|---|
| Pacer, fixed configuration (compact model, K = 32, reuse 4, h x1) | 41.26 | 48.17 | 58.47 | 64.62 |
| larger model (63 leaves, 85-172 trees), K = 64, no reuse | 41.90 | 48.81 | 58.69 | 64.71 |
| without session features (gaps, turn, session age) | 41.15 | 47.90 | 58.47 | 64.67 |
| without block history (hit count, fan-out) | 41.09 | 48.22 | 58.71 | 64.76 |
| K = 8 | 41.00 | 47.82 | 58.27 | 64.45 |
| K = 16 | 41.12 | 47.99 | 58.37 | 64.71 |
| K = 64 | 41.35 | 48.27 | 58.36 | 64.48 |
| K = 128 | 41.39 | 48.19 | 58.39 | 64.33 |
| no burst reuse | 41.34 | 48.20 | 58.39 | 64.41 |
| no chain following | - | 47.57 | - | 64.47 |
| h x0.5 | 41.12 | 47.88 | 58.26 | 64.51 |
| h x2 | 41.36 | 48.16 | 58.24 | 64.59 |
| h x4 | 41.16 | 48.18 | 58.23 | 64.41 |
| sampled LRU (K = 32, no model) | 37.95 | 46.04 | 57.39 | 64.08 |
<!-- /results:ablations -->

- Session features cost little AUC (0.002 on chat, 0.006 on coder), but without them chat hit
  ratio drops by 0.7-1.3 points at every size; on coder they matter little.
- Block history matters more for AUC (0.019 on chat, 0.024 on coder) than for hit ratio (under
  0.9 points either way).
- K = 32 is a cost choice: K = 64-128 adds 0.3-1.0 points on chat at 1-2%, and K = 8 loses up
  to 1.5.
- Burst reuse gives up 0.3-0.5 points on chat at 1-2% and roughly halves the decision cost.
- Without chain following, hit ratio is lower (chat at 2%: 29.3 against 30.2) and eviction costs
  19-30 times as much (measured during a parallel sweep).
- The LRB-style regression (DAgger model) trails the fixed configuration by 0.1-1.3 points on
  chat and is 1.1 points below LRU at 10%. Its labels look past the training hour, which
  favors it.

## Transfer across workloads

Fixed configuration, each trace's model applied to the others.

<!-- results:transfer -->
| Evaluated on | Cache | LRU | own model | model from To-C chat | model from To-B API | model from Coder | model from Thinking |
|---|---|---|---|---|---|---|---|
| To-C chat | 1% | 15.88 | 21.85 | - | 18.51 | 20.69 | 16.20 |
| To-C chat | 2% | 24.36 | 30.19 | - | 24.84 | 28.10 | 22.94 |
| To-C chat | 5% | 45.60 | 46.16 | - | 32.06 | 42.65 | 37.86 |
| To-C chat | 10% | 58.57 | 58.78 | - | 39.63 | 55.68 | 48.69 |
| To-B API | 1% | 46.23 | 52.32 | 50.90 | - | 51.14 | 51.02 |
| To-B API | 2% | 48.97 | 53.08 | 51.33 | - | 51.92 | 52.35 |
| To-B API | 5% | 51.22 | 53.34 | 52.42 | - | 52.43 | 53.29 |
| To-B API | 10% | 52.36 | 53.46 | 53.02 | - | 52.87 | 53.45 |
| Coder | 1% | 37.81 | 41.26 | 40.01 | 40.00 | - | 37.70 |
| Coder | 2% | 45.89 | 48.17 | 45.38 | 45.37 | - | 42.45 |
| Coder | 5% | 57.31 | 58.47 | 57.38 | 50.07 | - | 52.39 |
| Coder | 10% | 64.01 | 64.62 | 63.47 | 54.77 | - | 58.73 |
| Thinking | 1% | 30.17 | 33.13 | 31.97 | 32.04 | 32.16 | - |
| Thinking | 2% | 32.61 | 34.97 | 33.69 | 33.38 | 33.48 | - |
| Thinking | 5% | 35.46 | 36.76 | 35.73 | 35.73 | 35.33 | - |
| Thinking | 10% | 37.14 | 37.55 | 37.10 | 37.04 | 36.02 | - |
<!-- /results:transfer -->

Every transferred model is below the trace's own model, and 16 of 48 fall more than half a
point below LRU. The chat model is the most portable (at worst 0.54 points below LRU, on
coder). Models trained on API traffic (all first turns) or thinking traffic (mostly first
turns) lose up to 19 points on chat at
5-10%, likely because on their own traffic a block with no shared-template history is almost
never reused, which is wrong for conversations.

## Cost

Nanoseconds of victim selection per evicted block on the test hour at 2% capacity, one run at a
time on one thread of an idle M5 Pro, median of 3 (`results/overhead.json`). Our LRU and ATC'25
use lazily invalidated heaps; S3-FIFO's cost comes from re-queueing tails that are pinned or
have cached children (the prefix-closure adaptation), not from S3-FIFO itself.

<!-- results:cost -->
| Policy | To-C chat | To-B API | Coder | Thinking |
|---|---|---|---|---|
| LRU | 255 | 453 | 374 | 267 |
| S3-FIFO | 4,595 | 92 | 20,342 | 8,650 |
| ARC | 121 | 127 | 150 | 129 |
| ATC'25 | 348 | 520 | 452 | 371 |
| sampled LRU (K = 32) | 30 | 39 | 28 | 26 |
| Pacer (fixed: 13-37 trees, K = 32, reuse 4) | 84 | 230 | 119 | 49 |
| Pacer with the 85-213 tree model, K = 64 | 2,102 | 17,549 | 3,052 | 652 |
<!-- /results:cost -->

## Live replay (MLX)

The same Rust controller, through PyO3, drives a paged bf16 KV pool for Qwen3-0.6B (4-bit
weights) in MLX. On each arrival the cached blocks are gathered by a small Metal kernel and
only the uncached suffix is prefilled; time to first token covers gather, prefill and the first
argmax. Decode-completed blocks are computed at completion, outside TTFT. Policies run
interleaved, and the requests come from the held-out hour. To fit the pinned working set on a
24 GB laptop, decoding is simulated at 2 ms per token (fewer requests in flight).

<!-- results:live -->
| Policy | hit ratio | prefill tokens | TTFT median (ms) | TTFT mean (ms) | mean TTFT vs LRU (95% CI) |
|---|---|---|---|---|---|
| LRU | 14.27% | 768,160 | 108.5 | 341.5 | reference |
| S3-FIFO | 14.39% | 767,088 | 109.2 | 340.7 | -0.2% [-0.9%, +0.5%] |
| sampled Belady | 18.15% | 733,440 | 101.4 | 332.4 | -2.7% [-4.2%, -1.4%] |
| Pacer | 15.91% | 753,440 | 106.0 | 337.5 | -1.2% [-2.4%, -0.1%] |

400 requests of traceA from t = 3600 s, cache 4489 blocks (7.67 GiB of bf16 KV), mlx-community/Qwen3-0.6B-4bit on Apple M5 Pro, 2 interleaved repeats. Cached-prefix prefill vs full prefill on 200 requests: same argmax in 97.5%, max |logit difference| median 0.128.
<!-- /results:live -->

Pacer prefilled 1.9% fewer tokens than LRU and cut mean time to first token by 1.2%; its median
is 2.3% lower but that interval includes zero. Perfect knowledge (sampled Belady) prefilled 4.5%
fewer tokens and cut mean TTFT by 2.7%. The effect is small because the hit-ratio difference on
these 400 requests is 1.6 points, and a forward pass on a 0.6B model has a large fixed cost (the
fitted prefill model is 24 ms plus 68 us per new token plus an attention term, R^2 0.996). That
model predicts individual replay TTFTs only to within 35% (median), so it is used for relative
block costs, not for latency claims.

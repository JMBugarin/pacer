# Design decisions and how to defend them

Each entry: the decision, why, what it costs, and what would change it. These are the
questions an inference or ML-systems interviewer is likely to ask.

## Problem and evaluation

**Why prefix-cache eviction?** Multi-turn chat and agents resend their whole history, so
whether a turn pays full prefill depends on whether its prefix is still cached. vLLM evicts
free blocks in LRU order. On the chat trace the offline reference is far ahead of LRU (2% cache:
LRU 24.4%, Belady 55.2%), and a September 2026 paper found 14 generic policies, including three
learned ones, barely beat LRU. The open question was whether a policy that uses what an LLM
server knows about each request does better.

**Why a simulator instead of a GPU cluster?** The traces are two hours of a production cluster;
replaying them live needs that cluster. A deterministic simulator gives exact, repeatable
comparisons of 16 policies over 4 traces and 4 sizes, with paired statistics. The live MLX
replay checks that the controller drives a real KV pool and measures what hits are worth in time
to first token, but it cannot reproduce production capacities on a laptop.

**Why hit ratio, not TTFT, as the main metric?** TTFT depends on hardware, batching and
queueing; hit ratio is what the eviction policy controls. The simulator also records hit
depths, so a compute-weighted ratio comes from any linear cost model (the MLX calibration gives
one, and the results report it).

**Why hold out the second hour, and how clean is it?** The traces are two hours long, and a
temporal split is the only one that does not leak future sessions into training. Every v3
choice was made on minutes 45-60 of hour 1. Hour 2 was nonetheless used during development
(an exploratory baseline sweep, one look at the first approach, the v1 and v2 runs), and the v2
label fix was prompted by a weak v1 test cell. The v3 numbers come from one run of committed
code; the earlier looks are listed in the engineering log so a reader can weigh them.

**Why compare against the best heuristic chosen on test?** It is the strongest comparison
available: for each cell, whichever of nine heuristics did best on the test hour, each with its
setting tuned on validation. Pacer gets no such per-cell hindsight.

**Why t-intervals over 5-minute buckets?** Both policies see the same requests in each bucket,
so pairing removes most of the workload variance. With only twelve buckets a percentile
bootstrap is too narrow, so the intervals are t-intervals on the linearized ratio (about 20%
wider). Buckets are not independent (cache state carries over) and sampled policies run with
one seed, so the intervals are still somewhat optimistic; the conclusions rest on consistency
across 16 cells and 4 traces as much as on any one interval.

## Simulator

**Prefix re-keying.** The trace hashes each 16-token block on its own; a KV block is reusable
only if everything before it matches. Chaining the hashes is the only correct key.

**Decode-completed blocks.** A chat follow-up's prompt contains the previous answer, and vLLM
caches blocks filled during decoding, so the simulator inserts them at completion, named by the
follow-up's hashes. It does so only when the follow-up is long enough to contain the answer
verbatim; a thinking model's follow-up often drops the reasoning, and naming those blocks
anyway created hits no server could get (found in review).

**Pinning.** Running requests hold their blocks, as in a GPU KV pool. At 1% capacity they pin
on average 31% of the cache on chat, 51% on coder, 39% on thinking and 8% on API traffic
(`results/trace_stats.json`). This makes 1% a stress point, and the gains there should be read
with that in mind.

**Leaf-only eviction.** Evicting an interior block makes its descendants unreachable. Keeping
the cache prefix-closed is what radix-tree caches do and keeps every policy honest; each
policy's adaptation is documented in [simulator.md](simulator.md). One cost: adaptations that
re-queue blocks they cannot evict (S3-FIFO) get slower than the original algorithm.

**Belady as a reference bound.** A parent is used whenever its child is, so the
farthest-next-use block is always a leaf: the leaf constraint never forces Belady off its
choice. It is checked against exhaustive search on random small traces. Because insertion is
forced and running requests are pinned, it is called a reference bound rather than a proven
optimum.

**Rust plus a Python reference.** The Python simulator is short enough to read as a spec; the
Rust one is fast enough for 700-run sweeps. Differential tests require identical counts on the
eight deterministic baselines. The same Rust code runs inside the live replay through PyO3, so
the live numbers come from the exact policy that was swept.

## Model

**Why a hazard model instead of imitating Belady?** The first version regressed log time to
next use on logged cache states. It ranked candidates well offline but lost to LRU at larger
sizes in the simulator: the target does not ask the eviction question (will it be reused before
it would be gone?), and its labels looked past the training hour. The hazard formulation asks
exactly the eviction question, is trained on cache-independent rows, and only uses events
before the cutoff.

**Why tie the horizon to the eviction age?** The same block is worth keeping in a large cache
and not in a small one. The cache's own recent eviction age measures how long a kept block will
survive, so one model serves every capacity.

**Why gradient-boosted trees?** Tabular features, a million rows per trace, training in about a
minute on a laptop, and evaluation in Rust with no runtime dependency. The compact models
(13-37 trees) are within 0.004 AUC of 85-213 tree models. A neural model would add latency on the
eviction path with no evidence of a gain.

**Why per-trace models?** Every transferred model does worse than the trace's own, and 16 of 48
transfers fall more than half a point below LRU (a model trained on single-turn API traffic
loses 19 points on chat at 10%). Training on a deployment's own recent traffic takes about a
minute and needs no labels beyond the log itself.

**What does the model use?** Two answers, depending on the measure. By AUC, block history
(access count, fan-out) matters most among the ablated groups (0.019-0.024) and the session
features barely (0.002-0.006). By hit ratio on chat it is the reverse: without session features
chat loses 0.7-1.3 points at every size, without block history under 0.9. The eviction decision
depends on ranking a few candidates near the margin, which a global AUC does not measure.

## Policy mechanics and cost

**Sampling.** Sorting all blocks by a time-dependent score on every eviction is not feasible;
sampling 32 candidates is. With perfect knowledge the same sampler gets within 0.18-1.34 points
of Belady, so sampling loses little.

**Chain following.** A finished conversation's private suffix shares one history. Evicting it
in one decision is much cheaper, and in the ablations it also scores better: without it chat at
2% drops from 30.2% to 29.3%, and eviction costs 19-30 times as much.

**Compact model and burst reuse.** The deployed models have 13-37 trees, within 0.004 AUC of the
85-213 tree models. Batching trees across candidates, chain following, K = 32 and letting one
sample serve up to 4 evictions bring the fixed configuration to 49-230 ns per evicted block on
one core, below our heap-based LRU (255-453 ns); the larger model with K = 64 costs 0.65-17.5
us. K = 32 and burst reuse give up 0.3-1.0 points on chat at 1-2% for that.

## What would change these decisions

- A trace with request content features (tool calls, system prompt identity) would likely make
  continuation more predictable and move chat at larger caches.
- A deployment with CPU or SSD offload tiers changes the miss cost per block; the policy would
  then score expected saved compute rather than reuse probability (the results show this matters
  already: weighted by compute, Pacer's lead on chat disappears at larger sizes).
- A guard for distribution shift (for example a sampled shadow LRU that the policy falls back to
  when it is losing) is the next step before trusting one model across workloads.

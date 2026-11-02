# Method

## What Pacer predicts

When the cache is full, every candidate block has the same question attached to it: will
anyone ask for this block again before it would have been evicted anyway? Pacer answers it
with a hazard model:

```
P( next use of block b falls in (a, a + h]  |  b has not been used for a seconds )
```

- `a` is the block's age since it was last touched (in practice, since the request that owns
  it finished).
- `h` is the horizon that matters for this cache right now: the recent average age at which
  blocks get evicted (an exponentially weighted average kept by the policy), times a
  multiplier, set to 1 in the fixed configuration. A small cache keeps blocks for tens of
  seconds and a large one for many minutes, so the same block can be worth keeping in one and
  not the other.

The policy evicts the sampled candidate with the lowest predicted probability.

## Features

Everything is known when the block was last touched (`python/pacer/hazard.py`,
`crates/pacer/src/features.rs`):

| Group | Features |
|---|---|
| Question | elapsed `a`, horizon `h`, `a` / previous inter-turn gap |
| Block history | times the block was accessed (including the insertion), distinct child blocks ever cached under it, depth, depth relative to its chain, decode-completed or prompt block |
| Request that last touched it | request type, turn, input length, output length |
| Session | previous inter-turn gap, mean gap, session age |
| Load | arrivals in the 60 s before the touch |

## Training rows

`pacer hazard` replays hour 1 with an unbounded cache. At 15% of block releases it draws 8
`(a, h)` pairs (`a`: 0 with probability 0.2, else log-uniform from 5 s to 1 h; `h`:
log-uniform from 10 s to 2000 s) and labels each pair with whether the block's next use falls
in `(a, a + h]`.

- A row whose window `(t + a, t + a + h]` ends after the cutoff is dropped whatever its label.
  The first version kept the rows that happened to see a use, which over-represented reuse at
  long horizons (see the engineering log).
- A row whose block is already used by `a` is dropped: it would not be a candidate.
- The model is fit only on rows whose window ends by minute 45, so no training label looks
  into the validation period. Rows starting in minutes 45-52.5 are used for early stopping,
  and rows starting in minutes 52.5-60 for the reported AUC and log loss. Because those rows
  must end by minute 60, the reported AUC covers windows of at most about 7.5 minutes.

The rows do not depend on any cache state, so there is no feedback between the policy that
logged the data and the policy that uses the model, the usual problem when imitating Belady
from logged cache states. The questions asked at decision time (which candidates, which `a`
and `h`) still differ from the training distribution.

## Model

LightGBM binary classifier: 31 leaves, learning rate 0.25, early stopping as above. The models
behind the reported results have 13-37 trees. The model is exported with `dump_model()` and
evaluated in Rust by a flattened tree evaluator (`gbdt.rs`) that runs each tree over all K
candidates in turn. A golden test checks it against LightGBM's own predictions.

## The ATC'25 baseline

ATC'25 models reuse per workload (request type and turn) as a reuse probability with an
exponential reuse time. Two fits are available to the simulated policy, and validation picks
the better one for each trace and size:

- maximum likelihood on the same training rows as Pacer, which handles censoring the same way;
  this is also the model in the AUC comparison;
- a simple count over the training period, which treats reuse after the cutoff as no reuse.

The paper's lifespan regulator is not described in enough detail to reproduce and is left out.

## Eviction mechanics

- **Sampling.** K evictable blocks (unpinned leaves) are drawn uniformly, with replacement;
  K = 32 in the fixed configuration. With perfect knowledge of next use, the same sampler
  (K = 64, with chain following) gets within 0.18-1.34 points of Belady.
- **Chain following.** After evicting a leaf, if its parent became a leaf with the same history
  (same owner request and access count), the parent is evicted next without a new sample, as
  long as that history has not changed since. A finished conversation's private suffix goes in
  one decision.
- **Burst reuse.** One scored sample supplies up to 4 victims within the same eviction burst
  (the evictions triggered by one arrival or completion).

The ablations in [results.md](results.md) give the cost and quality of each.

## Configurations reported

- **Pacer (fixed)**: compact model, `h` multiplier 1, K = 32, reuse 4, the same for every trace
  and cache size. It was fixed in v1 as the cheapest of 12 settings within 0.3 points of the best
  per-cell setting on that version's validation; it was not re-selected later.
- **Pacer (horizon tuned)**: larger model (63 leaves, 85-213 trees), K = 64, no reuse, and the
  `h` multiplier picked per trace and cache size on validation from {0.5, 1, 2}.

Each trace gets its own model, trained on its own first 45 minutes.

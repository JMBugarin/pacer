# Method

## What Pacer predicts

When the cache is full, every candidate block has the same question attached to it: will
anyone ask for this block again before it would have been evicted anyway? Pacer answers it
with a hazard model:

```
P( next use of block b falls in (a, a + h]  |  b has not been used for a seconds )
```

- `a` is the block's age since it was last touched (normally the moment the request that owns
  it finished).
- `h` is the horizon that matters for this cache right now: the recent average age at which
  blocks get evicted (an exponentially weighted average maintained by the policy), times a
  multiplier (1 by default). A small cache keeps blocks for tens of seconds, a large one for
  many minutes, so the same block can be worth keeping in one and not the other.

The policy evicts the sampled candidate with the lowest predicted probability.

## Features

Everything is known when the block was last touched (`python/pacer/hazard.py`,
`crates/pacer/src/features.rs`):

| Group | Features |
|---|---|
| Question | elapsed `a`, horizon `h`, `a` / previous inter-turn gap |
| Block history | times the block was hit, distinct child blocks ever cached under it, depth, depth relative to the chain, decode-completed or prompt block |
| Request that last touched it | request type, turn, input length, output length |
| Session | previous inter-turn gap, mean gap, session age |
| Load | arrivals in the 60 s before the touch |

## Training rows (cache-independent)

`pacer hazard` replays the training period with an unbounded cache. At a random 10-15% of
block releases it draws 8 `(a, h)` pairs (`a`: 0 with probability 0.2, else log-uniform
5 s to 1 h; `h`: log-uniform 10 s to 2000 s) and labels each pair with whether the block's next
use falls in `(a, a + h]`. Two rules keep the labels honest:

- Only uses before the training cutoff (t = 3600 s) are visible. A row whose window
  `(t + a, t + a + h]` ends after the cutoff is dropped whatever its label; keeping the rows
  that happen to see a use would over-represent reuse at long horizons (the first version
  did this; see the engineering log).
- A row whose block is already used by `a` is dropped (it would not be a candidate).

Because the rows do not depend on any cache state, there is no distribution shift between the
policy that produced the data and the policy that uses the model, the usual problem of
imitating Belady from logged cache states.

## Model

LightGBM binary classifier: 31 leaves, learning rate 0.25, early stopping on the last 15
minutes of the training hour. The deployed models have 16-19 trees. The model is exported
with `dump_model()` and evaluated in Rust by a flattened tree evaluator (`gbdt.rs`) that
batches the K candidates per tree; a golden test checks it against LightGBM's own
predictions.

The baseline model is the ATC'25 one: per (request type, turn) an exponential reuse-time
distribution with a reuse probability, fitted by maximum likelihood on the same rows.

## Eviction mechanics

- **Sampling.** K evictable blocks (unpinned leaves) are drawn uniformly; K = 32 by default.
  Sampled Belady (the same sampler with perfect next-use knowledge) is within a point of exact
  Belady, so sampling itself costs almost nothing.
- **Chain following.** After evicting a leaf, if its parent became a leaf and has the same
  history (same owner request and hit count), the parent is evicted next without a new sample.
  A finished conversation's private suffix goes in one decision. In the ablations this made
  eviction about 23x cheaper, and hit ratio was slightly higher with it than without.
- **Burst reuse.** One scored sample supplies up to 4 victims within the same eviction burst
  (the evictions triggered by one arrival or completion). This halves the cost and gives up
  about 0.7 points on chat at 1-2%; the ablations list the trade-offs.

## Configurations reported

- **Pacer (fixed)**: compact model, `h` multiplier 1, K = 32, reuse 4, the same for every
  trace and cache size. Picked on validation as the cheapest setting within 0.3 points of the
  best per-cell setting on average.
- **Pacer (horizon tuned)**: larger model (85-114 trees, 63 leaves), K = 64, no reuse, and
  the `h` multiplier picked per trace and cache size on validation from {0.5, 1, 2}.

Each trace gets its own model trained on its own first hour.

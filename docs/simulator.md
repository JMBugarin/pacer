# Simulator semantics

Pacer's simulator replays a production trace against a block-level prefix cache that behaves
like vLLM's automatic prefix caching. The Python reference (`python/pacer/refsim.py`) and the
Rust simulator (`crates/pacer/src/sim.rs`) implement exactly what is written here, and the
differential tests require identical hit, access and eviction counts.

## Blocks and keys

A trace row has `hash_ids`: salted hashes of each 16-token block of the prompt. The hashes are
per block, not per prefix, so two prompts can share a block hash without sharing the tokens
before it. Every block is re-keyed by its whole prefix:

```
key_0 = intern(ROOT, h_0)
key_i = intern(key_{i-1}, h_i)
```

Two requests share `key_i` only if their first `i + 1` blocks are identical, which is the
condition for reusing a KV block. Only **full** input blocks (`input_length // 16`) are cached
or looked up; the partial last block is recomputed, as in vLLM.

## Decode-completed blocks

A child turn's prompt is its parent's prompt plus the parent's output plus a short new
message (median 23 extra tokens on traceA; the parent's full prompt is a prefix of the child's
in 100% of traceA turns, 99.3% of coder, 93.2% of thinking). A real server caches the blocks
filled during decoding, so the next turn can hit the previous answer. The simulator inserts
those blocks when the request completes:

- The request's full sequence has `(input_length + output_length) // 16` full blocks; the ones
  past the input are its *decode-completed* blocks (the first of them also holds the input's
  partial tail).
- If the request's first follow-up turn starts with the request's full input chain and its
  prompt is at least as long as the request's prompt plus its output (so it can contain the
  answer verbatim), the decode-completed keys are the follow-up's keys at those positions.
  Otherwise they are fresh keys that are never used again (they still take space). A thinking
  model's follow-up often drops the reasoning (319 of 1,200 thinking follow-ups, 165 of 16,605
  on coder, none on chat); its blocks are then not the ones the server decoded.

This uses the follow-up's hashes only to name blocks the server would have produced; no policy
sees them before the follow-up arrives.

## Time

Requests arrive at their trace timestamps. A request completes at
`t + 0.02 s * output_length` (50 tokens/s decoding), capped to land before its first child
arrives. Completions are processed before arrivals that have the same time. Every policy hook
receives the index of the first arrival not processed yet; "the future" (for Belady, the
oracles and every training label) means arrivals from that index on.

## Arrival

1. Hits are the longest prefix of the request's input keys that is cached. Prefix closure
   (below) makes this the number of leading cached keys.
2. Missing input blocks are inserted.
3. The policy is notified for every input block from the deepest to the root: `on_insert` for
   new blocks, `on_access` for hits. Deep-to-root order makes the root the most recently used,
   so recency-ordered policies reach a chain's tail first, as vLLM frees blocks in reverse.
4. All input blocks are pinned (reference count + 1) until the request completes.
5. Blocks are evicted until the cache is within capacity.

## Completion

1. Decode-completed blocks are inserted (`on_insert`).
2. The input blocks are unpinned.
3. Every block of the request's chain is released from the deepest to the root
   (`on_release`). LRU, the main queue of LRU with quick demotion, ATC'25 and one ARC variant
   treat release as a recency update, matching vLLM's free-queue order; Belady recomputes the
   block's next use; the others ignore it.
4. Blocks are evicted until the cache is within capacity.

## Eviction and prefix closure

A block is evictable when it is cached, unpinned and has no cached children. Only evictable
blocks can be victims, so the cache stays prefix-closed: a cached block's parent is always
cached. If nothing is evictable while over capacity (every block pinned by running requests),
the cache overflows; the simulator records the largest overflow and continues.

Capacity is a number of blocks, given as a fraction of the trace's unique cacheable blocks
(input and decode-completed). Running requests' blocks count against capacity, as in a GPU
KV pool.

## Metrics

For arrivals inside the measurement window: block hits, block accesses, and the sums of hit
and access depths (so any cost model linear in depth gives a compute-weighted hit ratio). Hits
are also bucketed per 5 minutes for paired confidence intervals. The cache always starts empty
at t = 0 and runs with the same policy through the warm-up before the window; the run stops at
the end of the window. Eviction counts and decision time cover the whole run, warm-up
included.

## How each policy handles prefix closure

| Policy | Adaptation |
|---|---|
| LRU, FIFO, LFU, GDSF, Belady | Priority order over the evictable set (lazy min-heap). |
| SIEVE | The hand skips blocks that are not evictable. |
| S3-FIFO | A tail block that is not evictable goes back to the head of its queue, frequency unchanged; promotion is checked first. |
| ARC | REPLACE takes the least recent evictable block of the chosen list, falling back to the other list; ghost lists bounded as in ARC. A variant also refreshes recency within the list at release; validation picks per cell. |
| LRU + quick demotion | Probation FIFO for blocks never re-accessed, evicted first while it holds more than its share; LRU main queue. |
| Workload-aware (ATC'25) | Per-(type, turn) LRU queues; the victim is the least likely to be reused among their heads. |
| Sampled policies, Pacer | Sample K evictable blocks uniformly with replacement; evict the highest score; then follow the chain upward while the parent has the same, unchanged history. |

## Belady

Belady evicts the evictable block whose next use is farthest away (never used: farthest;
ties: deeper block first, then larger key). A parent is used whenever its child is, so its
next use is never later than its child's and the farthest block is always a leaf or tied with
one: the leaf-only constraint never forces Belady off its choice. The tests compare Belady with
an exhaustive search over every eviction choice on random small traces. Because insertion is
forced (no bypass) and pinned blocks cannot be evicted, Belady is reported as the offline
reference bound of this cache model rather than a proven optimum of every possible cache.

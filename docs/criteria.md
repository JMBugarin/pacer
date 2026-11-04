# Success criteria

These criteria were set before the simulator was written, and none were changed afterwards.
Each is measured from the result files by `python -m pacer.report fill`.

<!-- results:criteria -->
| | Target | Measured | Status |
|---|---|---|---|
| C1 | Rust simulator equals the Python reference; Belady equals brute force on small traces and is at least every online policy | differential and brute-force tests pass (8 policies); Belady highest in all 16 test cells | met |
| C2 | LRU, best heuristic, oracle ladder and Belady on 4 traces x 4 sizes | all reported (results.md) | met |
| C3 | Better than LRU at 1, 2, 5% on every trace (CI excludes 0) | 11 of 12; not: To-C chat 5% | missed |
| C4 | At least the best tuned heuristic on 3 of 4 traces at 2% and 5%; +1 point on chat somewhere (CI excludes 0) | not worse on 4/4 at 2%, 4/4 at 5%; chat best +2.27 [+1.80, +2.75] | met |
| C5 | Close 20% of the LRU-to-Belady gap on chat at 2% and 5% | 25.0% at 1%, 18.9% at 2%, 2.5% at 5% | missed |
| C6 | Compute-weighted savings at least LRU's everywhere | below LRU in To-C chat 5% | missed |
| C7 | At most 2 us per evicted block; a full sweep under 10 min | 49-230 ns per eviction; slowest single test run 64 min (S3-FIFO), whole pipeline about 4 h | partial |
| C8 | The chat model is never more than 0.5 points below LRU on other traces | worst -0.54 points | missed |
| C9 | Cached-prefix prefill picks the same next token on 99%; TTFT vs LRU with a CI | 97.5% of 200; mean TTFT -1.2% [-2.4%, -0.1%] over requests (2 repeats, drift not covered) | partial |
| C10 | `make reproduce` regenerates every number; CI green | CI green; deterministic stages rebuild identically; 10 ATC'25 test cells were re-run outside the logged run (see the engineering log) | partial |
<!-- /results:criteria -->

## Why the misses are misses

- **C3, C5 (chat at 5%).** On chat at larger caches the remaining headroom is knowing which
  conversations have ended. With the metadata the traces carry, whether a request gets a
  follow-up at all is barely predictable (AUC 0.63 against 0.61 from the turn number alone,
  `results/trace_stats.json`), and on held-out rows with horizons of 5 minutes or more the
  model is no better than ATC'25's. This is a limit of the features available, not of tuning:
  no tuned heuristic gets closer either.
- **C6 (compute-weighted, chat at 5%).** Pacer scores reuse, not saved compute, and tends to
  evict long conversations, whose deep blocks are the most expensive to recompute. The fix is a
  different objective (reuse probability times block cost); it was not attempted.
- **C7 (sweep time).** Pacer and most baselines take seconds per run; the time goes to the
  S3-FIFO adaptation, which re-queues blocks it cannot evict and dominates the 24-setting grid.
- **C8 (transfer).** The chat model falls 0.54 points below LRU on coder at 10%, just past the
  0.5 limit; models from other workloads do far worse. Per-workload training is the remedy used.
- **C9 (live).** The 2.5% of next tokens that differ come from bf16 arithmetic on near-tied
  logits (cached and full prefill reduce in a different order); no noise-floor control was run
  to confirm this. The TTFT interval covers request sampling over two repeats, not run-to-run
  drift.
- **C10 (reproduce).** The second ATC'25 fit was added after the logged pipeline run, and its
  10 test cells were re-run outside it (engineering log, test-hour use 6).

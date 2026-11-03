# Engineering log

What was tried, in order, including what did not work and every time the held-out hour was
used. Dates are 2026-11-01 and 11-02.

## 1. Choosing the problem

A research pass over hiring signals, LLM inference on Apple silicon, ML for systems,
verifiable-reward training and student-relevant niches produced six candidates, judged
independently for hiring signal, feasibility and rigor. All three judges picked learned
prefix-cache eviction: production traces with session structure are public (Qwen-Bailian,
Apache-2.0), the gap between LRU and Belady is large, and a September 2026 paper ("When Fancy
Eviction Fails", Liu, Yu, Yang) had just reported that 14 policies, including three learned
ones, barely beat LRU on these workloads, without trying LLM-specific features.

## 2. Simulator

- Re-keying: the trace hashes each block separately, so blocks are re-keyed by their whole
  prefix. With an unbounded cache, 0.16% (chat) and 1.39% (API) of the prefix hits that
  per-block hashes would count are not servable (`results/trace_stats.json`).
- A chat follow-up's prompt contains the previous prompt and answer, so the blocks filled while
  decoding are cached at completion; without them a follow-up could never hit the previous
  answer.
- LRU and FIFO first gave identical results. Not a bug: with leaf-only eviction the evictable
  blocks are almost all fresh decode-completed tails that are never touched again, so recency
  and insertion order coincide.
- A Python reference simulator was written first as the executable spec. The Rust simulator
  matches it exactly (hits, accesses, evictions, depth sums) on the eight deterministic
  baselines, on trace slices and on hundreds of random traces.
- ARC first scanned its lists for the least recent evictable block and spent most of its time
  stepping over pinned blocks (up to 384 s per run, `results/history/v1/baselines.jsonl`).
  Lazy heaps over evictable blocks give identical results in seconds.

**Test-hour use 1:** an exploratory sweep of every baseline at default settings, on hour 2
(`results/history/v1/baselines.jsonl`). It showed where the headroom was and was used for no
later choice except that hour 2 became the test window.

## 3. First learned policy: imitate Belady from logged cache states

LRB-style: run sampled LRU, log the 64 candidates at each decision with their time to next
use, regress log(time to next use) with LightGBM, evict the largest prediction.

- On held-out decisions the model picked a candidate with the farthest true next use 97.7% of
  the time (78.7% for recency).
- **Test-hour use 2:** in the simulator it was 4.3 points below LRU at 10% on chat (the run was
  not saved). After this, choices moved to a validation window.
- A DAgger round (log states visited by the learned policy itself) did not change the picture
  on validation. That model is kept as an ablation: on the test hour it beats LRU at 1-2% on
  chat but trails Pacer by 1.3-1.7 points and loses to LRU at 10%.
- Two problems: the regression target does not ask the eviction question (will it be reused
  before it would be gone?), and its labels look up next uses past the end of hour 1. The leak
  favors this baseline, so the comparison against it is conservative.

## 4. Hazard model (v1)

The formulation in [method.md](method.md): a classifier of "next use within (a, a + h] given
unused for a", trained on cache-independent rows of block releases, asked at decision time
with `h` tied to the cache's own eviction age. On validation it beat every tuned heuristic in
16 of 16 cells, though (as the review later found) its labels could see the validation window.

- **Test-hour use 3:** v1 per-cell tuned Pacer was better than the best heuristic in 13 of 16
  cells and tied in 3; the fixed configuration was better in 12, tied in 3 and worse in 1 (chat
  at 10%: -0.96 points), with percentile-bootstrap intervals. v1 ablations also ran on hour 2.
- Decision cost was too high (about 25 us per eviction on API traffic during a parallel sweep).
  Nearly all of it was the model. A compact model (31 leaves, learning rate 0.25) kept
  validation AUC within 0.002 with 19-41 trees instead of 98-136; trees are evaluated in a
  batch across candidates; and one scored sample serves up to 4 evictions of a burst. The fixed
  configuration (h x1, K = 32, reuse 4) was chosen here from 12 compact settings on validation
  and not re-selected later.

## 5. Censoring bias (v2)

The hazard rows dropped a row whose window ran past the training cutoff only when no use had
been seen, so positives were over-represented at long horizons, which is where the chat 10%
result was weak. v2 drops every row whose window crosses the cutoff. On validation v2 was
within 0.6 points of v1 and lower in 10 of 16 cells; it was adopted for the argument, and the
weak test cell prompted the look.

- **Test-hour use 4:** v2 on hour 2, both configurations better than the best heuristic in 13 of
  16 cells and tied in 3 (bootstrap intervals). v2 ablations, transfer, cost and the live replay
  also used hour 2 (`results/history/v2/`).

## 6. Adversarial review and v3

Two independent reviewers audited every claim against the result files and read the code for
leakage and bugs. Found and fixed:

- **Labels saw the validation window.** Training rows were cut by release time (before minute
  45) but their labels could look up to minute 60, so Pacer's validation numbers and its
  per-cell horizon choice were optimistic (the test hour was not affected). v3 fits only rows
  whose whole window ends by minute 45, early-stops on minutes 45-52.5 and reports AUC on
  52.5-60.
- **ATC'25's fit ignored censoring.** It counted reuse after the cutoff as no reuse. v3 gives
  the policy a maximum-likelihood fit on the same rows as Pacer, keeps the simple fit as a second
  variant, and lets validation choose per cell (the simple fit wins in 10 of 16).
- **Artificial hits on the thinking trace.** Decode-completed blocks were named after the
  follow-up's blocks even when the follow-up had dropped the reasoning. v3 does that only when
  the follow-up is long enough to contain the answer verbatim (this changes 319 of 1,200
  thinking follow-ups and 165 coder ones, for every policy).
- **Belady at zero-length completions.** The test suite showed that a request completing at its
  own arrival time made Belady count already-processed arrivals as future uses. Next use is now
  measured from the first unprocessed arrival everywhere (Belady, oracles, labels). No request in
  the real traces has a zero-length completion, so no result changed.
- **Smaller items.** Chain following now checks that the parent's history has not changed; load
  and child-arrival features no longer count same-timestamp arrivals that are not processed
  yet; ARC got a variant that refreshes recency at release; the S3-FIFO and LRU-with-quick-
  demotion grids were widened where the chosen setting sat at an edge; confidence intervals
  are t-intervals (about 20% wider than the bootstrap with 12 buckets); several overclaims in
  the docs were corrected.

**Test-hour use 5:** v3, run once from committed code with `make reproduce`: the fixed
configuration is better than the best heuristic in 13 of 16 cells, tied in 3 and worse in none.

## 7. Findings that changed the story

- Session pacing was the starting hypothesis (the name comes from it). It holds on chat in a
  way AUC hides: without the session features (previous gap, mean gap, turn, session age) AUC
  drops by only 0.002, but chat hit ratio drops by 0.7-1.3 points at every size. Block history
  is the reverse (0.019 AUC, under 0.9 points of hit ratio). With v1 labels the session features
  looked useless by AUC; the v1 and v2 ablations are in `results/history/`.
- On chat at 5-10% capacity nothing online closes the gap to Belady. The oracle ladder shows the
  headroom is knowing which conversations end; whether a chat continues at all is barely
  predictable from the trace's metadata (AUC 0.63, against 0.61 from the turn number alone), and
  on chat the model is no better than ATC'25 at horizons of 5 minutes or more.
- Models do not transfer across workloads. Each trace is trained on its own first 45 minutes,
  which takes about a minute.
- Weighting hits by the prefill compute they save, Pacer's lead holds at small sizes but it
  drops slightly below LRU on chat at larger sizes: it tends to drop long conversations, whose
  deep blocks cost the most to recompute.

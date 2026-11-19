# Serving v3: cached/full agreement root cause and frozen local benchmark

Local Apple M5 Pro run, Qwen3-0.6B-4bit, MLX 0.32.4 / mlx-lm 0.32.2. Synthetic token ids only. Protocol: `docs/serving-protocol-v3.md`, committed at 3943b76 before the workload was generated. Hashes in `frozen-manifest.json`.

## Root cause of the 97.5% agreement

- Historical check reproduced exactly: 195/200 argmax equal, max |dlogit| 0.377 (`historical-reproduction.log`).
- `prefill_diagnose.py`, 64 fixed development cases: Pacer's pooled path is bitwise equal to stock mlx-lm chunked prefill in 64/64 cases. Stock chunked prefill vs one-shot full prefill picks a different token in 1/64 (case 9, split 10, max |dlogit| 0.162).
- Conclusion: the disagreement comes from MLX numerics when a prompt is prefilled in two pieces instead of one (different attention/matmul reduction shapes over a 4-bit model). It is not a Pacer cache or eviction bug, and cache policy cannot fix it.
- Constraint shipped: `--verified-output` / `Engine(verify_output=True)` recomputes full prefill on every hit and returns those logits. `test_verified_output_uses_full_prefill` checks the result is bitwise equal to the full reference. Without the flag, cached output stays labeled experimental and is not token-equivalent.

## Frozen benchmark (protocol v3, verified output)

Command:

```
PYTHONPATH=python .venv/bin/python scripts/live_replay.py --trace docs/evidence/serving-v3/new-local.jsonl \
  --start 0 --limit 120 --cap 0.2 --policies lru,pacer --repeats 4 --check 100 --verified-output \
  --tpot 0.002 --chunk 2048 --boot 2000 --out docs/evidence/serving-v3/final-replay.json
```

| | LRU | Pacer |
|---|---|---|
| hit ratio | 0.428 | 0.551 |
| prefill tokens (dry run) | 22,256 | 17,456 |
| TTFT p50, pooled | 44.4 ms | 49.8 ms |
| TTFT p95, pooled | 60.7 ms | 62.2 ms |

Per-repeat paired mean TTFT difference (Pacer minus LRU): +4.46, +3.60, +3.90, +5.02 ms (min +3.60, max +5.02). Check: 66/66 argmax equal, max |dlogit| 0.

Result: negative. In verified mode, Pacer's higher hit ratio makes it slower, because each hit still pays a full prefill plus the gather. Verified output gives reference fidelity at the cost of every caching gain. Four pairs are not enough to generalize. The request bootstrap CI in the log is descriptive only and does not account for run drift.

## Supplementary run (post hoc, not part of the frozen protocol)

Same workload and code, without `--verified-output` (`supplementary-unverified.json`):

- Cached/full agreement on the new data: 64/66 = 96.97%. The two mismatches had reference top-2 margins of 0.076 and 0.017, which fits the near-tie numerics above.
- TTFT p50 25.1 ms (LRU) vs 24.2 ms (Pacer). Paired mean differences −0.33, −0.90, −1.21, −0.64 ms.

This run was added after the frozen run was exposed. It can show the speed/fidelity tradeoff, but it does not count as frozen evidence for a latency claim, and cached mode still misses the ≥99% agreement target.

## Not established

Production serving readiness, behavior on real prompts, multi-tenant or networked serving, independent external review.

## Erratum (added with serving v4)

The "66/66 argmax equal, max |dlogit| 0" check above compared verified mode's served logits with a full prefill. In verified mode the served logits are the full prefill's, so the check compared the reference with itself and did not test the cached path. The +3.60 to +5.02 ms came from a redundant cached forward and gather that verified mode ran on every hit before recomputing. Both are fixed, and the follow-up is in [serving v4](../serving-v4/REPORT.md). The measurements in this file are unchanged.

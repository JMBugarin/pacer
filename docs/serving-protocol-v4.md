# Serving protocol v4 — frozen before new workload generation

Scope: unchanged from v3. Local synthetic-token replay on Apple M5 Pro, Qwen3-0.6B-4bit, MLX 0.32.4 / mlx-lm 0.32.2, Python environment synced to `uv.lock`. No production or human prompt data. Model and policy parameters stay fixed (Pacer mult=1, k=32, reuse=4; `models/fast_traceA.json`). No retuning on final results. The v3 workload (seed 20261115) and the development workloads (same generator, seeds 101–103) are development data.

Why a new version: the code changed after the v3 final run was exposed. Diagnosis on development data found:

1. The +3.60 to +5.02 ms in v3 was the verified-mode cost of a hit (gather, cached forward and full prefill) multiplied by Pacer's extra hits. It was not policy or controller overhead: with a policy indicator added, the per-request regression gives Pacer −0.06 ms.
2. The v3 verified-mode logits check compared the full prefill with itself. Its 66/66 and max 0 say nothing about the cached path.
3. Cached and full logits disagree because MLX picks the quantized matmul kernel from the row count. Below a threshold that depends on the weight shape (65, 129 or 257 rows for this model), MLX splits the reduction. At or above the threshold, every row's result is independent of the row count and of the row's position.

Output modes:

- **Primary: `--exact-output --audit`.**
  - Exact output runs the cached-prefix forward with each quantized matmul zero-padded to its calibrated row-stable threshold. Thresholds are calibrated per weight shape at engine start, up to 512 rows, and recorded in the result file. The served logits are those of this cached forward.
  - Prompts outside the stable regime are served by a full prefill. None are expected here: every prompt is 17–24 blocks, 272–384 rows.
  - The audit compares every hit's served logits with a full prefill, in every timed run, after that request's TTFT is taken. Audit work is not in TTFT.
- **Secondary: `--verified-output`.**
  - A full prefill on every hit, with the cached forward skipped, on the same workload. It is expected to be policy-neutral.

Correctness, pre-registered:

- Exact output is accepted only if the check pass (`--check 100`, LRU) has `bitwise_equal_frac` = 1 and every timed run's audit has `bitwise_equal` = `n`.
- Any mismatch makes the exact run a correctness failure. It is reported, and no latency claim is made from it.
- In the verified run, the served logits equal the full prefill by construction. Its check reports the stock cached path's agreement, which is informative only.

Final workload: generated only after this protocol is committed and pushed, with `python3 scripts/new_local_workload.py --seed 20261118 --out docs/evidence/serving-v4/new-local.jsonl`. The generator and parameters are those of v3: 120 requests, 12 equally likely synthetic prefix sessions, 16 shared blocks plus 1–8 unique suffix blocks, 16 output tokens, arrivals 1 s apart. Hash every generated file, model snapshot and dependency, and record machine state and contention. Colima VM presence is disclosed.

Runs, in this order. Every run is preserved, failures included. There is no best-run selection.

```
PYTHONPATH=python .venv/bin/python scripts/live_replay.py --trace docs/evidence/serving-v4/new-local.jsonl \
  --start 0 --limit 120 --cap 0.2 --policies lru,pacer --repeats 4 --check 100 --exact-output --audit \
  --tpot 0.002 --chunk 2048 --boot 2000 --out docs/evidence/serving-v4/final-exact.json

PYTHONPATH=python .venv/bin/python scripts/live_replay.py --trace docs/evidence/serving-v4/new-local.jsonl \
  --start 0 --limit 120 --cap 0.2 --policies lru,pacer --repeats 4 --check 100 --verified-output \
  --tpot 0.002 --chunk 2048 --boot 2000 --out docs/evidence/serving-v4/final-verified.json
```

Cache 20% of unique blocks. Four paired repeats with alternating LRU/Pacer order. One fixed warmup of the first 30 requests.

Report per repeat: request p50, p95 and mean TTFT, hit ratio, prefill work, and the paired mean TTFT difference (Pacer minus LRU). Report all four paired differences with their min and max, plus the check and audit results.

Decision rule: Pacer beats LRU on v4 if all four paired mean differences in the primary run are below zero and the exact-output correctness criteria pass.

Limits on the claim:

- The uncertainty unit is the repeat. Four pairs do not justify a robust generalization claim, and the request bootstrap is descriptive only.
- Eviction timing is not in TTFT. In development profiling, the Rust `Stepper.step` call across the PyO3 boundary took about 5 µs per step.
- No combined end-to-end serving claim is made.

Not established: production serving readiness, real prompts, other models or MLX versions or devices (kernel thresholds come from MLX's dispatch heuristics and are recalibrated at every engine start), multi-tenant or networked serving, independent external review.

If code changes after final exposure, the protocol is versioned again with a new final seed. Negative results remain published.

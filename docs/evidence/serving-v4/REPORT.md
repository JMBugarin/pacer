# Serving v4: where the v3 gap came from, and exact cached output

Local Apple M5 Pro run, Qwen3-0.6B-4bit, MLX 0.32.4 / mlx-lm 0.32.2, synthetic token ids only. Protocol: `docs/serving-protocol-v4.md`, committed and pushed at cd2c41a before the workload was generated. Hashes in `frozen-manifest.json`.

## Result

Under protocol v4, Pacer beats LRU by the pre-registered rule, in exact-output mode, with every served hit bitwise equal to a full prefill:

- **Latency:** per-repeat paired mean TTFT difference (Pacer minus LRU) was −0.37, −0.44, −0.38 and −0.41 ms (min −0.44, max −0.37). That is about 2.5% of LRU's 15.7 ms mean.
- **Median and p95:** unchanged.
- **Correctness:** the check pass was 77/77 bitwise equal to a full prefill. The audit compared all 664 served hits in the timed runs with a full prefill, and all 664 were bitwise equal.

Verified mode (a full prefill on every hit) is now policy-neutral: −0.07 to +0.07 ms, against +3.60 to +5.02 ms in v3.

## Where v3's +3.6 to +5.0 ms came from

The diagnosis used the v3 evidence and development workloads (generator seeds 101–103). The frozen v3 run was not repeated.

| Layer | Finding |
|---|---|
| Rust policy and eviction | Not in TTFT: the controller step runs before the timer starts. `Stepper.step` across PyO3 takes about 5 µs per step (1.3 ms per 241-step replay, max 17 µs). |
| PyO3 boundary and locking | Same measurement. No measurable contribution. |
| MLX replay harness | All of it. In verified mode a hit paid for a gather, the cached forward and a full prefill, while a miss paid only for the full prefill. Pacer hits more, so it paid more often. |

- **Per-request regression on the v3 runs.** TTFT ≈ −14.0 + 0.75·blocks + 1.37·uncached blocks + 46.4·[any hit] ms (R² 0.89). Adding a policy indicator gives −0.06 ms.
- **The gap, request by request.** Pacer had a hit on 23 requests where LRU missed, and each cost +27.3 ms. LRU had a hit on 3 where Pacer missed, at −28.6 ms each. Requests where both hit, or both missed, differed by less than 0.2 ms. Net: +4.47 ms per request.
- **Development seed 101, phase timing.** The verify forward added +3.5 ms per request for Pacer. The gather (1.4 ms) and the cached forward were the same for both policies or lower for Pacer.

**Erratum for v3.** The v3 verified-mode logits check compared the served logits with a full prefill. In verified mode the served logits are the full prefill's, so the reported 66/66 and max 0 compared the reference with itself and say nothing about the cached path. Fixed in this branch: the check now runs the cached forward itself. On the v4 workload it reports 97.4% argmax agreement and max |Δlogit| 0.30 for the stock cached path.

## Why cached and full prefill disagree

MLX picks the quantized matmul kernel from the row count. Below a threshold that depends on the weight shape, it uses split-K kernels that reduce in a different order. At or above it, every row's result is independent of the row count and of the row's position.

Thresholds, calibrated at engine start and recorded in `final-exact.json`:

| Weights (in → out) | Layers | Row-stable from |
|---|---|---|
| 1024 → 1024 | k, v | 257 |
| 2048 → 1024 | o | 257 |
| 3072 → 1024 | down | 257 |
| 1024 → 2048 | q | 129 |
| 1024 → 3072 | gate, up | 65 |

Fused SDPA, RoPE with offset and RMSNorm are row-consistent at these shapes. A cached suffix of 1–8 blocks (16–128 rows) therefore rounds differently from a full prefill of 272–384 rows. Prefix KV taken from a different, longer prompt also differs unless that prompt was computed in the stable regime.

The split-K regime is also slower here. A one-shot prefill took about 26 ms at 256 rows and about 15 ms at 272 rows. Measured in isolation, a stock cached forward of 8 new blocks (17.1–17.6 ms) costs nearly as much as a full prefill of the same prompt (17.5–19.2 ms).

Two alternatives were rejected:

- **Junk-token padding of the whole forward to 257 rows.** It is exact (32/32), but each hit costs as much as a full prefill.
- **A dequantized dense matmul.** It is not exact for the K=3072 projection at any row count.

## Exact output (`--exact-output`)

- **Padding.** Each quantized matmul below its threshold is zero-padded to the threshold and sliced back. The rest of the stock mlx-lm forward is untouched.
- **Cost.** Measured in isolation, a cached hit costs about 13.2 ms regardless of suffix length, against 14.7–19.2 ms for a full prefill of these prompts. In the replay, hits averaged 14.6 ms and misses 17.8 ms.
- **Pooled KV.** It always comes from row-stable forwards, completions included.
- **Fallback.** Prompts too short to be row-stable, longer than one chunk, or past the calibrated range fall back to a full prefill. None occurred in this workload.
- **Full hits.** They recompute the last block, not the last token.
- **Development check.** 48/48 fixed cases and every audited hit across development seeds 101–103 were bitwise equal.

## Development results (seeds 101–103, not frozen)

`development.json`. All exact-mode audits were 100% bitwise.

| Seed | Output | Paired mean ΔTTFT per repeat (ms) |
|---|---|---|
| 101 | exact + audit (2 repeats) | −0.63, −0.67 |
| 102 | exact + audit | −0.69, −0.58, −0.64, −0.68 |
| 102 | exact, no audit | −0.89, −0.98, −0.83, −0.90 |
| 103 | exact + audit | −0.68, −0.66, −0.71, −0.68 |
| 101 | verified | −0.00, +0.03, +0.12, +0.69 |
| 103 | cached (experimental) | −0.52, −0.68, −0.64, −0.65 |

Auditing adds GPU work between requests, and more of it for the policy with more hits. That cost Pacer about 0.25 ms of its advantage on seed 102, so the audited primary run is conservative for Pacer.

## Frozen benchmark: primary (exact output, audited)

Command as in the protocol (`final-exact.json`, `final-exact.log`). Dry run: hit ratio 0.486 (LRU) vs 0.568 (Pacer); prefill tokens 20,224 vs 17,008.

| Repeat (order) | LRU p50 / p95 / mean (ms) | Pacer p50 / p95 / mean (ms) | Δ mean (ms) | Audit LRU / Pacer |
|---|---|---|---|---|
| 0 (LRU first) | 14.57 / 19.31 / 15.70 | 14.58 / 19.17 / 15.33 | −0.37 | 77/77, 89/89 |
| 1 (Pacer first) | 14.60 / 19.39 / 15.74 | 14.63 / 19.19 / 15.30 | −0.44 | 77/77, 89/89 |
| 2 (LRU first) | 14.68 / 19.41 / 15.76 | 14.62 / 19.34 / 15.38 | −0.38 | 77/77, 89/89 |
| 3 (Pacer first) | 14.64 / 19.35 / 15.77 | 14.59 / 19.26 / 15.37 | −0.41 | 77/77, 89/89 |

Per-request decomposition (each request's mean over repeats, Pacer minus LRU):

| Hit status | Requests | Mean Δ |
|---|---|---|
| Both miss | 26 | +0.02 ms |
| Both hit | 72 | −0.01 ms |
| Pacer-only hit | 17 | −3.42 ms |
| LRU-only hit | 5 | +2.14 ms |

The difference is the hits Pacer gets and LRU does not, as the hit ratios predict.

Check pass (LRU, 77 hit arrivals): bitwise equal 77/77, max |Δlogit| 0. The no-prefix control's median |Δlogit| was 3.46, so a broken cache path would show.

## Frozen benchmark: secondary (verified output)

Command as in the protocol (`final-verified.json`, `final-verified.log`).

- **TTFT:** mean 17.69 ms (LRU) and 17.72 ms (Pacer).
- **Paired mean Δ:** −0.07, +0.07, +0.05 and +0.04 ms.
- **Check:** it measures the stock cached path, which verified mode never serves: 97.4% argmax agreement, 2/77 bitwise equal, max |Δlogit| 0.30.

Every hit pays a full prefill, so the policy cannot matter. This confirms the v3 penalty was the redundant cached forward.

## Caveats

- **Effect size.** About 0.4 ms of mean TTFT; median and p95 are unchanged. It is the expected size for a 0.08 hit-ratio gap at these prompt lengths, where a miss costs only about 1.2× a hit.
- **Sample size.** Four paired repeats are not a robust generalization. The request bootstrap is descriptive only.
- **Hardware dependence.** Exact output relies on MLX's kernel dispatch heuristics for this model on this device. Thresholds are recalibrated every time the engine starts, and recorded. Another MLX version or device needs its own calibration and evidence.
- **Unreported compute.** The padded rows are extra compute that `forward_tokens` does not count.
- **What TTFT excludes.** Controller eviction and audit work are not in TTFT. No combined end-to-end serving claim is made.
- **Environment.** A Colima VM was running, as in v3. The Python environment was synced to `uv.lock` before the final runs (the v3 venv had drifted, for example transformers 5.23.0 against the locked 5.22.1). mlx, mlx-lm and numpy were unchanged, and `pacer._core` was rebuilt from the frozen commit.
- **Scope.** Synthetic tokens on one laptop. The held-out trace-A live replay (C9) was not re-run with exact output.

## Not established

Production serving readiness, behavior on real prompts, other models, MLX versions or devices, multi-tenant or networked serving, independent external review.

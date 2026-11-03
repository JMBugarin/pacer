# Results files

Every run is one JSON line from `pacer sweep`: trace, policy, label, parameters, capacity,
window, hits, accesses, hit and access depth sums, evictions, decisions, decision time and
5-minute buckets. Specs are in `configs/`; `make reproduce` regenerates the top half of this
table from a clean clone.

| File | What | Spec |
|---|---|---|
| `final_test.jsonl` | **The reported test hour**: `test_baselines.jsonl` + `v2/test.jsonl` | |
| `final_summary.json` | Summary of the above with paired bootstrap CIs | `make summary` |
| `test_baselines.jsonl` | Every baseline, oracle and bound on the test hour | `test.json` |
| `v2/test.jsonl` | Both Pacer configurations on the test hour | `test_v2.json` |
| `validation_heuristics.jsonl` | Heuristics and their grids on the validation window | `validation_heuristics.json` |
| `v2/validation.jsonl` | Pacer settings on the validation window | `validation_v2.json` |
| `v2/fast_*.json`, `v2/hazard_*.json` | Model training reports: AUC, log loss, ATC'25 baseline, feature gains | `make models` |
| `v2/ablations.jsonl`, `v2/ablate_*.json` | Ablations around the fixed configuration | `ablations_v2.json` |
| `v2/transfer.jsonl` | Each trace's model on the other traces | `transfer.json` |
| `overhead.json` | Decision cost per eviction, one run at a time on an idle machine | `scripts/overhead.py` |
| `live/replay.json` | Live MLX replay: per-request TTFT, prefill tokens, cached-vs-full logits check | `scripts/live_replay.py` |
| `live/prefill_cost.json` | Prefill time vs cached and new tokens on the M5 Pro, fitted cost model | `scripts/calibrate_prefill.py` |
| `v1/` | The first model version, kept for the record: first test look, v1 ablations, the LRB-style attempt (`train_traceA_v*.json`), exploratory default-setting sweep (`baselines.jsonl`) | `configs/v1/` |

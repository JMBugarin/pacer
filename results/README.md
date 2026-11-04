# Results files

Every sweep run is one JSON line from `pacer sweep`: trace, policy, label, parameters, capacity,
window, hits, accesses, hit and access depth sums, evictions, decisions, decision time and
5-minute buckets. Specs are in `configs/`. Confidence intervals in the summaries are paired 95%
t-intervals over the twelve 5-minute buckets of the test hour.

| File | What | Produced by |
|---|---|---|
| `final_test.jsonl` | The reported test hour: `test_baselines.jsonl` + `test_pacer.jsonl` | `make test` |
| `final_summary.json` | Summary of the above with paired intervals | `make summary` |
| `test_baselines.jsonl` | Every baseline, oracle and bound on the test hour (10 ATC'25 cells re-run after the logged run, see the engineering log) | `configs/test.json` |
| `test_pacer.jsonl` | Both Pacer configurations on the test hour | `configs/test_pacer.json` |
| `validation_heuristics.jsonl` | Heuristics and their grids on the validation window | `configs/validation_heuristics.json` |
| `validation_pacer.jsonl` | Pacer settings on the validation window | `configs/validation_pacer.json` |
| `training/*.json` | Model training reports: AUC, log loss, ATC'25 baseline, feature gains, AUC by horizon | `make models`, `make ablations` |
| `ablations.jsonl` | Ablations around the fixed configuration | `make ablations` |
| `transfer.jsonl` | Each trace's model on the other traces | `make transfer` |
| `overhead.json` | Decision cost per eviction, one run at a time on an idle machine | `make overhead` |
| `trace_stats.json` | Session structure, pinned share, continuation predictability | `scripts/trace_stats.py` |
| `live/replay.json` | Live MLX replay: per-request TTFT, prefill tokens, cached-vs-full logits check | `make live` |
| `live/prefill_cost.json` | Prefill time vs cached and new tokens on the M5 Pro, fitted cost model | `make live` |
| `pipeline.log`, `extras.log` | Logs of the v3 run (the later ATC'25 re-run is not in them) | |
| `history/v1/`, `history/v2/` | Earlier model versions, kept for the record; specs in `configs/history/` | |

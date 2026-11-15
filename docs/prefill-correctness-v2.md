# Prefill correctness diagnostic v2

November 14, 2026. This change fixes a diagnostic, not the serving path.

If every logit changes by at most d, both the winning and runner-up logits can move toward each other. A reference top-two margin strictly greater than 2d is sufficient to preserve the winner. The old summary used margin > d. A constructed regression [1.5, 0] -> [0.5, 1] demonstrates why that was insufficient. Equality at 2d permits a tie and is not certified. This bound is a diagnostic, not a relaxation of the 99% behavioral target.

The hardware-independent summary now reports protocol_version=2 and the clear-margin sample count. Historical files are unchanged. Reanalysis of results/live/replay.json at 9775802 still finds 195/200 matching argmax tokens (97.5%), five mismatches, maximum absolute logit difference 0.377106 and 140 samples with margin >2d, all agreeing. These are old synthetic-token replay observations, not newly collected requests or a serving benchmark. The 99% target remains missed. Floating-point kernel shape is a hypothesis, not an established cause of all five mismatches.

Regression command: `PYTHONPATH=python .venv/bin/pytest tests/test_prefill_checks.py`. Full local checks and historical reanalysis are under docs/evidence/2026-11-14. The initial pytest invocation without PYTHONPATH failed package imports; the corrected invocation passed. MLX is absent in this environment, so no fresh live correctness claim is made.

Next investigation: install the locked live environment, hash the model snapshot and trace, reproduce every failing request with the same prefix history, compare stock MLX chunked prefill against pooled prefill and full prefill, and vary chunk shapes on development fixtures only. Keep raw logits/top tokens, request IDs, prefix lengths, model/trace hashes, dtypes and seeds. Require exact stock-path agreement for matched shapes, finite logits, numerical max-error <1 under the existing development criterion, and separately report the unchanged >=99% next-token agreement target. Neither a tolerance nor a clear-margin subset replaces all-request behavioral agreement. Freeze a new protocol before new final data; prior test-hour exposure remains disclosed.

No human walkthrough or independent methodological review occurred in this milestone. Implementation and documentation were agent-assisted; Joshua's understanding must be checked separately. Walkthrough: derive the 2d bound, explain why a mismatch near a tie still counts as a mismatch, and distinguish a simulator hit-ratio result from serving latency.

"""Trace statistics quoted in the docs, written to results/trace_stats.json: session structure,
how often a follow-up prompt contains the previous answer verbatim, how many naive per-block hash
hits are not servable, how much of the cache running requests pin, and how predictable it is
whether a conversation continues at all."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pacer.hazard import auc  # noqa: E402
from pacer.trace import build  # noqa: E402

TRACES = ["traceA", "traceB", "coder", "thinking"]


def unservable(rows: list[dict]) -> float:
    """Share of prefix hits counted with per-block hashes (unbounded cache) that a KV cache
    could not serve because the tokens before the block differ."""
    intern: dict[tuple[int, int], int] = {}
    seen_raw: set[int] = set()
    seen: set[int] = set()
    naive = chained = 0
    for r in rows:
        hs = r["hash_ids"][: r["input_length"] // 16]
        prev, ks = -1, []
        for h in hs:
            prev = intern.setdefault((prev, h), len(intern))
            ks.append(prev)
        p = 0
        while p < len(hs) and hs[p] in seen_raw:
            p += 1
        q = 0
        while q < len(ks) and ks[q] in seen:
            q += 1
        naive += p
        chained += q
        seen_raw.update(hs)
        seen.update(ks)
    return (naive - chained) / naive if naive else 0.0


def pinned(tr) -> dict:
    ev = []
    for r in tr.requests:
        ev.append((r.t, 1, len(r.input_keys)))
        ev.append((r.done, 0, -len(r.input_keys)))
    ev.sort()
    cur = peak = 0
    area = last = 0.0
    for t, _, d in ev:
        area += cur * (t - last)
        last = t
        cur += d
        peak = max(peak, cur)
    mean = area / last
    return {
        "mean_blocks": mean,
        "peak_blocks": peak,
        "mean_share_of_1pct": mean / (0.01 * tr.n_keys),
        "mean_share_of_2pct": mean / (0.02 * tr.n_keys),
    }


def continuation(tr) -> dict | None:
    """Does a request get a follow-up turn at all? Trained on hour 1 (requests before 45 min,
    so their follow-ups are seen), evaluated on 60-110 min (follow-ups up to 2 h are seen)."""
    reqs = tr.requests
    if not any(r.first_child_idx != -1 for r in reqs):
        return None
    x, y, t = [], [], []
    for r in reqs:
        p = reqs[r.parent_idx] if r.parent_idx >= 0 else None
        x.append([r.turn, r.rtype, r.input_len, r.output_len, r.t - p.t if p else -1.0])
        y.append(1.0 if r.first_child_idx != -1 else 0.0)
        t.append(r.t)
    x, y, t = np.array(x), np.array(y), np.array(t)
    tr_m, te_m = t < 2700, (t >= 3600) & (t < 6600)
    params = {"objective": "binary", "verbose": -1, "num_leaves": 31, "learning_rate": 0.05}
    params |= {"seed": 7, "deterministic": True, "force_row_wise": True}
    m = lgb.train(params, lgb.Dataset(x[tr_m], y[tr_m]), 300)
    return {
        "train_rate": float(y[tr_m].mean()),
        "test_rate": float(y[te_m].mean()),
        "auc_gbdt": auc(y[te_m], m.predict(x[te_m])),
        "auc_turn_only": auc(y[te_m], x[te_m][:, 0]),
    }


def main() -> None:
    out = {}
    for name in TRACES:
        path = ROOT / "data" / "raw" / f"qwen_{name}_blksz_16.jsonl"
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        tr = build(rows)
        by = {r["chat_id"]: r for r in rows}
        kids = [(by[r["parent_chat_id"]], r) for r in rows if r["parent_chat_id"] in by]
        prefix = verbatim = 0
        extra, gaps = [], []
        for p, c in kids:
            a, b = p["hash_ids"], c["hash_ids"]
            f = p["input_length"] // 16
            prefix += f == 0 or (len(b) >= f and a[:f] == b[:f])
            verbatim += c["input_length"] >= p["input_length"] + p["output_length"]
            extra.append(c["input_length"] - p["input_length"] - p["output_length"])
            gaps.append(c["timestamp"] - p["timestamp"])
        out[name] = {
            "requests": len(rows),
            "follow_up_turns": len(kids),
            "follow_up_contains_parent_prompt": prefix / len(kids) if kids else None,
            "follow_up_contains_parent_output": verbatim / len(kids) if kids else None,
            "median_extra_tokens": float(np.median(extra)) if kids else None,
            "median_inter_turn_gap_s": float(np.median(gaps)) if kids else None,
            "unique_blocks": tr.n_keys,
            "decode_completed_blocks": sum(len(r.output_keys) for r in tr.requests),
            "input_block_accesses": sum(len(r.input_keys) for r in tr.requests),
            "unservable_naive_hit_share": unservable(rows),
            "pinned": pinned(tr),
            "continuation": continuation(tr),
        }
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "results" / "trace_stats.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""Live TTFT replay on Apple silicon: Rust controller + MLX block pool, interleaved policies.

    python scripts/live_replay.py --limit 600 --cap 0.1 --repeats 2 --check 200

Runs a cached-vs-full logits check pass first, then `repeats` rounds over the policies with the
run order rotated each round, and writes results/live/replay.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
from pacer._core import load_trace
from pacer.live import replay as rp
from pacer.live.engine import DEFAULT_CHUNK, DEFAULT_MODEL, Engine

ROOT = Path(__file__).resolve().parents[1]


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def rounded(run: dict) -> dict:
    r = dict(run)
    for k in ("ttft_s", "gather_s"):
        r[k] = [None if t is None else round(t, 6) for t in r[k]]
    return r


def load(path: str, limit: int, frac: float, tpot: float, start: float):
    tr = load_trace(path, tpot=tpot, limit=limit, start=start)
    reqs = rp.requests(tr)
    cap, need = rp.cap_blocks(tr, frac), rp.min_cap_blocks(reqs)
    if cap < need:
        sys.exit(
            f"cap {frac} = {cap} blocks overflows on {len(tr)} requests: in-flight requests pin "
            f"up to {need} blocks, so use --cap >= {need / tr.n_keys:.3f}"
        )
    return tr, reqs, cap, need


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--trace", default=str(ROOT / "data/raw/qwen_traceA_blksz_16.jsonl"))
    ap.add_argument("--limit", type=int, default=600, help="number of requests replayed")
    ap.add_argument(
        "--start",
        type=float,
        default=3600.0,
        help="replay requests from this trace time on (default: the held-out hour)",
    )
    ap.add_argument("--cap", type=float, default=0.1, help="capacity as a fraction of n_keys")
    ap.add_argument("--policies", default=",".join(rp.DEFAULT_POLICIES))
    ap.add_argument("--hazard-model", default=str(ROOT / "models/v2/fast_traceA.json"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--check", type=int, default=200, help="arrivals in the logits check")
    ap.add_argument("--check-policy", default="lru")
    ap.add_argument("--check-limit", type=int, help="trace prefix for the check (default --limit)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--tpot", type=float, default=0.02)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-limit-gb", type=float, default=2.0, help="MLX buffer cache floor")
    ap.add_argument("--out", default=str(ROOT / "results/live/replay.json"))
    a = ap.parse_args()

    t_start = time.time()
    load0 = os.getloadavg()
    tr, reqs, cap, need = load(a.trace, a.limit, a.cap, a.tpot, a.start)
    specs = [rp.policy_spec(p, a.hazard_model) for p in a.policies.split(",")]
    dry = {s.name: rp.dry_run(tr, s, cap) for s in specs}
    for s in specs:
        d = dry[s.name]
        log(
            f"dry {s.name:15s} hit {d['hit_ratio']:.4f} prefill {d['prefill_tokens']} "
            f"output {d['output_tokens']} tokens"
        )

    ctr = creqs = None
    ccap = 0
    if a.check:
        if a.check_limit and a.check_limit != a.limit:
            ctr, creqs, ccap, _ = load(a.trace, a.check_limit, a.cap, a.tpot, a.start)
        else:
            ctr, creqs, ccap = tr, reqs, cap

    engine = Engine(a.model, a.chunk, a.seed)
    engine.reserve(max(len(r["input_keys"]) + len(r["output_keys"]) for r in reqs + (creqs or [])))
    mx.set_cache_limit(max(int(a.cache_limit_gb * 2**30), engine.workspace_bytes + 2**30))
    pool = engine.new_pool(max(cap, ccap))
    log(
        f"{len(tr)} requests, n_keys {tr.n_keys}, cap {cap} blocks (min {need}), "
        f"pool {pool.nbytes / 2**30:.2f} GiB, workspace {engine.workspace_bytes / 2**30:.2f} GiB"
    )
    rp.replay(tr, reqs, rp.PolicySpec("lru"), cap, engine, pool, stop_after=min(30, len(tr) - 1))

    check = {"n": 0}
    cspec = rp.policy_spec(a.check_policy, a.hazard_model)
    cand = rp.dry_run(ctr, cspec, ccap)["hit_arrivals"] if a.check else []
    if cand:
        picked = sorted(random.Random(a.seed).sample(cand, min(a.check, len(cand))))
        log(f"check: {len(picked)} of {len(cand)} arrivals with a cached prefix ({cspec.name})")
        t0 = time.time()
        res = rp.replay(
            ctr, creqs, cspec, ccap, engine, pool, check=set(picked), stop_after=picked[-1]
        )
        check = {
            "policy": cspec.name,
            "limit": len(ctr),
            "candidates": len(cand),
            "wall_s": time.time() - t0,
            **rp.check_summary(res["checks"]),
            "samples": res["checks"],
        }
        log(
            f"check: argmax equal {check['argmax_equal_frac']:.4f}, max|dlogit| "
            f"p50 {check['max_abs_diff']['p50']:.3f} max {check['max_abs_diff']['max']:.3f}"
        )

    runs = []
    n_runs = a.repeats * len(specs)
    t_runs = time.time()
    for rep in range(a.repeats):
        k = rep % len(specs)
        for pos, s in enumerate(specs[k:] + specs[:k]):
            t0 = time.time()
            r = rp.replay(tr, reqs, s, cap, engine, pool)
            ttft = sorted(t for t in r["ttft_s"] if t is not None)
            runs.append({**r, "rep": rep, "order": pos, "started_s": t0 - t_start})
            done = len(runs)
            eta = (time.time() - t_runs) / done * (n_runs - done)
            log(
                f"rep {rep} {s.name:15s} wall {r['wall_s']:6.1f}s hit {r['stats']['hit_ratio']:.4f}"
                f" ttft p50 {ttft[len(ttft) // 2] * 1e3:7.1f} ms  ({done}/{n_runs}, eta {eta:.0f}s)"
            )

    summary = rp.summarize(runs, base="lru", n_boot=a.boot, seed=a.seed)
    out = {
        "config": {
            "trace": os.path.relpath(a.trace, ROOT),
            "limit": len(tr),
            "start": a.start,
            "cap_frac": a.cap,
            "cap_blocks": cap,
            "min_cap_blocks": need,
            "n_keys": tr.n_keys,
            "policies": [vars(s) for s in specs],
            "repeats": a.repeats,
            "model": a.model,
            "chunk": a.chunk,
            "tpot": a.tpot,
            "seed": a.seed,
            "kv_dtype": str(engine.dtype),
            "block_bytes": engine.block_bytes,
            "pool_bytes": pool.nbytes,
            "workspace_bytes": engine.workspace_bytes,
            "mlx": mx.__version__,
            "mlx_lm": mlx_lm.__version__,
            "device": mx.device_info(),
            "platform": platform.platform(),
            "loadavg_start": load0,
            "loadavg_end": os.getloadavg(),
            "elapsed_s": time.time() - t_start,
            "ttft": "gather of cached blocks + forward of the new blocks + argmax, synchronized; "
            "ttft_fwd excludes the gather",
        },
        "dry_run": {k: {x: v[x] for x in v if x != "hit_arrivals"} for k, v in dry.items()},
        "check": check,
        "summary": summary,
        "requests": {
            "t": [r["t"] for r in reqs],
            "n_blocks": [len(r["input_keys"]) for r in reqs],
        },
        "runs": [rounded(r) for r in runs],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    head = ("policy", "hit", "p50 ms", "95% CI", "p95 ms", "vs lru p50")
    log("\n{:15s} {:>6s} {:>8s} {:>17s} {:>8s} {:>18s}".format(*head))
    for p, s in summary.items():
        lo, hi = s["ttft_median_s_ci95"]
        line = (
            f"{p:15s} {s['hit_ratio']:6.3f} {s['ttft_median_s'] * 1e3:8.1f} "
            f"[{lo * 1e3:6.1f},{hi * 1e3:7.1f}] {s['ttft_p95_s'] * 1e3:8.1f}"
        )
        if "ttft_median_reduction_vs_lru" in s:
            rlo, rhi = s["ttft_median_reduction_vs_lru_ci95"]
            line += f" {s['ttft_median_reduction_vs_lru']:+6.1%} [{rlo:+5.1%},{rhi:+5.1%}]"
        log(line)
    log(f"wrote {a.out} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

"""Live replay: the Rust Stepper decides, the MLX engine and block pool do the work.

Every step's inserted blocks are computed and written to the pool and its evicted blocks are
freed, with assertions that the pool and the controller agree: hit blocks are present, the
pool never exceeds the capacity, and the prefill length is what the controller implies.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import mlx.core as mx
import numpy as np

from pacer._core import Stepper, Trace

from .engine import Engine
from .pool import BLOCK, BlockPool

DEFAULT_POLICIES = ("lru", "s3fifo", "sampled-belady", "pacer")
PACER_PARAMS = {"mult": 1.0, "k": 32, "reuse": 4}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    params: dict[str, float] = field(default_factory=dict)
    model: str | None = None


def policy_spec(name: str, hazard_model: str | None = None) -> PolicySpec:
    if name == "pacer":
        return PolicySpec(name, dict(PACER_PARAMS), hazard_model)
    return PolicySpec(name)


def _ensure(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def requests(trace: Trace) -> list[dict]:
    return [trace.request(i) for i in range(len(trace))]


def cap_blocks(trace: Trace, frac: float) -> int:
    return max(1, math.floor(trace.n_keys * frac + 0.5))


def min_cap_blocks(reqs: Sequence[dict]) -> int:
    """Peak number of distinct pinned input blocks: the smallest capacity that never overflows."""
    events = sorted(
        [(r["done"], 0, i) for i, r in enumerate(reqs)]
        + [(r["t"], 1, i) for i, r in enumerate(reqs)]
    )
    pinned: Counter[int] = Counter()
    peak = 0
    for _, arrival, i in events:
        for k in reqs[i]["input_keys"]:
            if arrival:
                pinned[k] += 1
            else:
                pinned[k] -= 1
                if not pinned[k]:
                    del pinned[k]
        peak = max(peak, len(pinned))
    return peak


def stepper(trace: Trace, spec: PolicySpec, cap: int) -> Stepper:
    return Stepper(trace, spec.name, cap, spec.params or None, spec.model)


def dry_run(trace: Trace, spec: PolicySpec, cap: int) -> dict:
    """Controller only: hit statistics, the arrivals with a cached prefix, and work volumes."""
    st = stepper(trace, spec, cap)
    hit_arrivals = []
    prefill_blocks = output_blocks = 0
    while (s := st.step()) is not None:
        if s["arrival"]:
            prefill_blocks += len(s["inserted"])
            if s["hit_blocks"]:
                hit_arrivals.append(s["req"])
        else:
            output_blocks += len(s["inserted"])
    return {
        **st.stats(),
        "hit_arrivals": hit_arrivals,
        "prefill_tokens": BLOCK * prefill_blocks,
        "output_tokens": BLOCK * output_blocks,
    }


def _check(engine: Engine, keys: Sequence[int], hit: int, cached: mx.array, control: bool) -> dict:
    """Cached-prefix vs from-scratch logits. The control prefills the suffix without its
    prefix, which shows the size of a diff that a broken cache path would produce."""
    ref = engine.full_logits(keys)
    top2 = mx.topk(ref, 2)
    out = {
        "hit_blocks": hit,
        "n_blocks": len(keys),
        "max_abs": mx.max(mx.abs(ref - cached)).item(),
        "argmax_equal": mx.argmax(ref).item() == mx.argmax(cached).item(),
        "ref_margin": abs(top2[1].item() - top2[0].item()),
        "ref_absmax": mx.max(mx.abs(ref)).item(),
    }
    if control and hit < len(keys):
        out["control_max_abs"] = mx.max(mx.abs(ref - engine.full_logits(keys[hit:]))).item()
    return out


def replay(
    trace: Trace,
    reqs: Sequence[dict],
    spec: PolicySpec,
    cap: int,
    engine: Engine,
    pool: BlockPool,
    check: Collection[int] = (),
    stop_after: int | None = None,
    n_control: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Replays the trace live. `check` lists arrivals whose cached-prefix logits are compared
    with a full prefill from scratch (the first `n_control` also get the no-prefix control);
    `stop_after` ends the replay after that arrival."""
    if cap > pool.n_blocks:
        raise ValueError(f"capacity {cap} exceeds the pool's {pool.n_blocks} blocks")
    pool.reset()
    engine.reserve(max(len(r["input_keys"]) + len(r["output_keys"]) for r in reqs))
    st = stepper(trace, spec, cap)
    n = len(reqs)
    ttft: list[float | None] = [None] * n
    gather: list[float | None] = [None] * n
    hits = [0] * n
    full_hits = 0
    prefill_tokens = forward_tokens = 0
    out = {"events": 0, "tokens": 0, "seconds": 0.0, "dropped_blocks": 0}
    write_s = 0.0
    checks = []
    size = 0
    t_start = perf_counter()
    while (s := st.step()) is not None:
        i, ins, ev = s["req"], s["inserted"], s["evicted"]
        r = reqs[i]
        if s["arrival"]:
            keys, h = r["input_keys"], s["hit_blocks"]
            _ensure(ins == keys[h:], f"req {i}: controller inserted a non-suffix")
            _ensure(all(k in pool for k in keys[:h]), f"req {i}: hit block missing from pool")
            _ensure(not set(ev) & set(keys), f"req {i}: evicted a pinned block")
            pool.free(ev)
            hits[i] = h
            if keys:
                p = engine.run(pool, keys[:h], keys[h:])
                want = BLOCK * (len(keys) - h)
                _ensure(p.new_tokens == want, f"req {i}: live prefill {p.new_tokens} != {want}")
                ttft[i], gather[i] = p.seconds, p.gather_s
                prefill_tokens += p.new_tokens
                forward_tokens += p.forward_tokens
                full_hits += h == len(keys)
                t0 = perf_counter()
                pool.write(ins, p.k_new, p.v_new)
                write_s += perf_counter() - t0
                p.k_new = p.v_new = []
                if i in check and h:
                    c = _check(engine, keys, h, p.logits, len(checks) < n_control)
                    checks.append({"req": i, **c})
                del p
        else:
            outk = r["output_keys"]
            m = len(outk) - len(ins)
            _ensure(outk[m:] == ins, f"req {i}: completion inserted a non-suffix of its output")
            ctx = r["input_keys"] + outk[:m]
            _ensure(all(k in pool for k in ctx), f"req {i}: completion context missing from pool")
            dropped = set(ev) & set(ins)
            if ins:
                p = engine.run(pool, ctx, ins, logits=False)
                _ensure(p.new_tokens == BLOCK * len(ins), f"req {i}: output KV length")
                out["events"] += 1
                out["tokens"] += p.new_tokens
                out["seconds"] += p.seconds
            pool.free([k for k in ev if k not in dropped])
            if ins:
                keep = [j for j, k in enumerate(ins) if k not in dropped]
                kn, vn = p.k_new, p.v_new
                if len(keep) < len(ins):
                    idx = mx.array(keep, dtype=mx.uint32)
                    kn, vn = [a[idx] for a in kn], [a[idx] for a in vn]
                t0 = perf_counter()
                pool.write([ins[j] for j in keep], kn, vn)
                write_s += perf_counter() - t0
                out["dropped_blocks"] += len(dropped)
                del p, kn, vn
        size += len(ins) - len(ev)
        _ensure(len(pool) == size <= cap, f"pool holds {len(pool)}, controller {size}, cap {cap}")
        if s["arrival"]:
            if progress is not None:
                progress(i, n)
            if stop_after is not None and i >= stop_after:
                break
    wall = perf_counter() - t_start
    stats = st.stats()
    _ensure(stats["max_overflow"] == 0, "controller overflowed; raise the capacity")
    complete = stop_after is None
    if complete:
        n_in = sum(len(r["input_keys"]) for r in reqs)
        _ensure(sum(hits) == stats["hits"] and n_in == stats["accesses"], "hit accounting")
    return {
        "policy": spec.name,
        "cap_blocks": cap,
        "wall_s": wall,
        "stats": stats,
        "complete": complete,
        "ttft_s": ttft,
        "gather_s": gather,
        "hit_blocks": hits,
        "full_hits": full_hits,
        "prefill_tokens": prefill_tokens,
        "forward_tokens": forward_tokens,
        "completion": out,
        "write_s": write_s,
        "pool_peak": pool.peak,
        "checks": checks,
    }


def check_summary(checks: Sequence[dict]) -> dict:
    if not checks:
        return {"n": 0}
    d = np.array([c["max_abs"] for c in checks])
    eq = np.array([c["argmax_equal"] for c in checks])
    margin = np.array([c["ref_margin"] for c in checks])
    ctl = [c["control_max_abs"] for c in checks if "control_max_abs" in c]
    q = np.percentile(d, [50, 90, 99, 100]).tolist()
    return {
        "n": len(checks),
        "max_abs_diff": dict(
            zip(["p50", "p90", "p99", "max"], q, strict=True), mean=float(d.mean())
        ),
        "argmax_equal_frac": float(eq.mean()),
        "argmax_mismatch_ref_margins": margin[~eq].tolist(),
        "argmax_equal_frac_clear_margin": float(eq[margin > d].mean())
        if (margin > d).any()
        else None,
        "ref_absmax_p50": float(np.median([c["ref_absmax"] for c in checks])),
        "control_max_abs_p50": float(np.median(ctl)) if ctl else None,
        "control_n": len(ctl),
        "hit_frac_mean": float(np.mean([c["hit_blocks"] / c["n_blocks"] for c in checks])),
    }


def _stats(x: np.ndarray, axis: int = -1) -> dict[str, np.ndarray]:
    return {
        "median": np.median(x, axis=axis),
        "p95": np.percentile(x, 95, axis=axis),
        "mean": np.mean(x, axis=axis),
    }


def summarize(runs: Sequence[dict], base: str = "lru", n_boot: int = 2000, seed: int = 0) -> dict:
    """Per policy: TTFT statistics over requests, each request's TTFT being its median over
    repeats; 95% CIs and reductions vs `base` from a paired bootstrap over requests. The CIs
    cover request sampling only, not run-to-run drift (see ttft_median_s_by_repeat). `fwd`
    excludes the gather of cached blocks, which a paged-attention kernel would not need."""
    by: dict[str, list[dict]] = {}
    for r in runs:
        by.setdefault(r["policy"], []).append(r)
    valid = [t is not None for t in next(iter(by.values()))[0]["ttft_s"]]
    n = sum(valid)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    def per_request(rs: list[dict], fwd: bool) -> np.ndarray:
        ok = np.array(valid)
        t = np.array([[x or 0.0 for x in r["ttft_s"]] for r in rs])
        if fwd:
            t -= np.array([[x or 0.0 for x in r["gather_s"]] for r in rs])
        return np.median(t[:, ok], axis=0)

    def ci(a: np.ndarray) -> list[float]:
        return np.percentile(a, [2.5, 97.5]).tolist()

    out: dict[str, dict] = {}
    for p, rs in by.items():
        out[p] = {
            "repeats": len(rs),
            "n_requests": n,
            "hit_ratio": rs[0]["stats"]["hit_ratio"],
            "prefill_tokens": rs[0]["prefill_tokens"],
            "completion_tokens": rs[0]["completion"]["tokens"],
            "full_hits": rs[0]["full_hits"],
            "wall_s_by_repeat": [r["wall_s"] for r in rs],
            "completion_s_median": float(np.median([r["completion"]["seconds"] for r in rs])),
            "gather_share_of_ttft": float(
                np.median(
                    [sum(filter(None, r["gather_s"])) / sum(filter(None, r["ttft_s"])) for r in rs]
                )
            ),
            "ttft_median_s_by_repeat": [
                float(np.median([t for t in r["ttft_s"] if t is not None])) for r in rs
            ],
        }
    for name, fwd in (("ttft", False), ("ttft_fwd", True)):
        t = {p: per_request(rs, fwd) for p, rs in by.items()}
        boot = {p: _stats(x[idx]) for p, x in t.items()}
        point = {p: _stats(x) for p, x in t.items()}
        for p, s in out.items():
            for k in ("median", "p95", "mean"):
                s[f"{name}_{k}_s"] = float(point[p][k])
                s[f"{name}_{k}_s_ci95"] = ci(boot[p][k])
                if base in t:
                    red = 1 - boot[p][k] / boot[base][k]
                    s[f"{name}_{k}_reduction_vs_{base}"] = float(1 - point[p][k] / point[base][k])
                    s[f"{name}_{k}_reduction_vs_{base}_ci95"] = ci(red)
    return out

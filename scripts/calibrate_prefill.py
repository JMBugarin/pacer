"""Prefill cost model on the live engine: T(L_cached, L_new) on a grid, least-squares fit of

    T = a + b * L_new + c * L_new * (L_cached + L_new / 2)

and the per-block marginal cost cost(j) = 16 b + 16 c (16 j + 8), normalized so cost(0) = 1.
T is the replay's TTFT path (gather + forward + argmax). With --replay, also reports how well
the fit predicts the replay's measured per-request TTFT.

    python scripts/calibrate_prefill.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from pacer.live.engine import DEFAULT_CHUNK, DEFAULT_MODEL, Engine
from pacer.live.pool import BLOCK

ROOT = Path(__file__).resolve().parents[1]
CTX_KEY, NEW_KEY = 1 << 40, 1 << 41


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def design(lc: np.ndarray, ln: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(ln), ln, ln * (lc + ln / 2)], axis=1)


def fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    r2 = 1 - np.sum((y - x @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef, float(r2)


def measure(
    engine: Engine, cached: list[int], new: list[int], warmup: int, repeats: int, seed: int
):
    pool = engine.new_pool(max(max(cached) // BLOCK, 1))
    ctx = [CTX_KEY + j for j in range(pool.n_blocks)]
    p = engine.run(None, [], ctx, logits=False)
    pool.write(ctx, p.k_new, p.v_new)
    del p
    cells = [(lc, ln) for lc in cached for ln in new]
    times: dict[tuple[int, int], list[tuple[float, float]]] = {c: [] for c in cells}
    rng = random.Random(seed)
    for r in range(warmup + repeats):
        rng.shuffle(cells)
        t0 = time.time()
        for lc, ln in cells:
            keys = [NEW_KEY + j for j in range(ln // BLOCK)]
            p = engine.run(pool, ctx[: lc // BLOCK], keys)
            if r >= warmup:
                times[(lc, ln)].append((p.seconds, p.gather_s))
            del p
        log(
            f"round {r + 1}/{warmup + repeats} {'(warm-up) ' if r < warmup else ''}"
            f"{time.time() - t0:.1f}s"
        )
    return times


def replay_check(path: Path, coef: np.ndarray) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    n_blocks = np.array(d["requests"]["n_blocks"])
    by: dict[str, list[dict]] = {}
    for r in d["runs"]:
        by.setdefault(r["policy"], []).append(r)
    out, errs = {}, []
    for pol, rs in by.items():
        ok = np.array([t is not None for t in rs[0]["ttft_s"]])
        t = np.median([[x for x, v in zip(r["ttft_s"], ok, strict=True) if v] for r in rs], axis=0)
        h, n = np.array(rs[0]["hit_blocks"])[ok], n_blocks[ok]
        full = h == n
        lc = np.where(full, BLOCK * n - 1, BLOCK * h).astype(float)
        ln = np.where(full, 1, BLOCK * (n - h)).astype(float)
        pred = design(lc, ln) @ coef
        e = np.abs(pred - t) / t
        errs.append(e)
        out[pol] = {
            "n": int(ok.sum()),
            "median_abs_pct_err": float(np.median(e) * 100),
            "pred_ttft_mean_s": float(pred.mean()),
            "pred_ttft_median_s": float(np.median(pred)),
            "pred_ttft_p95_s": float(np.percentile(pred, 95)),
        }
    if "lru" in out:
        for k in ("mean", "median", "p95"):
            ref = out["lru"][f"pred_ttft_{k}_s"]
            for v in out.values():
                v[f"pred_ttft_{k}_reduction_vs_lru"] = 1 - v[f"pred_ttft_{k}_s"] / ref
    e = np.concatenate(errs)
    return {
        "replay": os.path.relpath(path, ROOT),
        "median_abs_pct_err": float(np.median(e) * 100),
        "p90_abs_pct_err": float(np.percentile(e, 90) * 100),
        "by_policy": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cached", type=ints, default=[0, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--new", type=ints, default=[16, 64, 256, 1024, 4096])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", type=Path, default=ROOT / "results/live/replay.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/live/prefill_cost.json")
    a = ap.parse_args()
    if any(x % BLOCK for x in a.cached + a.new):
        sys.exit(f"lengths must be multiples of {BLOCK}")

    t_start = time.time()
    load0 = os.getloadavg()
    engine = Engine(a.model, a.chunk, a.seed)
    engine.reserve((max(a.cached) + max(a.new)) // BLOCK)
    mx.set_cache_limit(engine.workspace_bytes + 2**30)
    times = measure(engine, a.cached, a.new, a.warmup, a.repeats, a.seed)
    grid = sorted(times)
    lc = np.array([g[0] for g in grid], dtype=float)
    ln = np.array([g[1] for g in grid], dtype=float)
    y = np.array([np.median([t for t, _ in times[g]]) for g in grid])
    gather = np.array([np.median([s for _, s in times[g]]) for g in grid])
    coef, r2 = fit(design(lc, ln), y)
    pred = design(lc, ln) @ coef
    ext, r2_ext = fit(np.column_stack([design(lc, ln), lc]), y)
    a0, b, c = coef.tolist()
    cost0 = BLOCK * b + BLOCK * c * (BLOCK / 2)
    alpha = BLOCK * BLOCK * c / cost0

    out = {
        "model": "T = a + b*L_new + c*L_new*(L_cached + L_new/2), seconds, lengths in tokens",
        "a": a0,
        "b": b,
        "c": c,
        "r2": r2,
        "rel_err_pct": {
            "median": float(np.median(np.abs(pred - y) / y) * 100),
            "max": float(np.max(np.abs(pred - y) / y) * 100),
        },
        "block_cost": {
            "model": "cost(j) = b*16 + c*16*(16*j + 8), normalized: 1 + alpha*j",
            "cost0_s": cost0,
            "alpha": alpha,
            "normalized": {str(j): 1 + alpha * j for j in (0, 16, 64, 256, 512, 1024)},
        },
        "diagnostic_fit_with_d_L_cached": {
            "coef_a_b_c_d": ext.tolist(),
            "r2": r2_ext,
            "note": "d absorbs the gather copy and the KV read of the cached prefix",
        },
        "grid": [
            {
                "L_cached": int(g[0]),
                "L_new": int(g[1]),
                "T_s": float(y[i]),
                "gather_s": float(gather[i]),
                "pred_s": float(pred[i]),
                "resid_s": float(y[i] - pred[i]),
                "resid_pct": float((y[i] - pred[i]) / y[i] * 100),
                "samples_s": [t for t, _ in times[g]],
            }
            for i, g in enumerate(grid)
        ],
        "replay_prediction": replay_check(a.replay, coef),
        "config": {
            "warmup": a.warmup,
            "repeats": a.repeats,
            "chunk": a.chunk,
            "engine_model": a.model,
            "device": mx.device_info(),
            "loadavg_start": load0,
            "loadavg_end": os.getloadavg(),
            "elapsed_s": time.time() - t_start,
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")

    log(
        f"\na={a0 * 1e3:.3f} ms  b={b * 1e6:.3f} us/token  c={c * 1e9:.4f} ns/token^2  R^2={r2:.4f}"
    )
    head = ("L_cached", "L_new", "T ms", "pred ms", "resid %", "gather ms")
    log("{:>8s} {:>6s} {:>9s} {:>9s} {:>8s} {:>9s}".format(*head))
    for g in out["grid"]:
        log(
            f"{g['L_cached']:8d} {g['L_new']:6d} {g['T_s'] * 1e3:9.2f} {g['pred_s'] * 1e3:9.2f} "
            f"{g['resid_pct']:+8.1f} {g['gather_s'] * 1e3:9.2f}"
        )
    log(f"block cost: cost(j)/cost(0) = 1 + {alpha:.4f} j")
    if out["replay_prediction"]:
        rp_ = out["replay_prediction"]
        log(f"replay TTFT prediction: median abs error {rp_['median_abs_pct_err']:.1f}%")
    log(f"wrote {a.out} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

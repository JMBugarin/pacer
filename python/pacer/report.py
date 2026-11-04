"""Turns sweep results into the tables and figures in the README.

Model and parameter selection uses only the validation window (last 15 minutes of hour 1);
the test window (hour 2) is reported once, for the configurations chosen on validation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TRACES = ["traceA", "traceB", "coder", "thinking"]
TRACE_LABEL = {
    "traceA": "To-C chat",
    "traceB": "To-B API",
    "coder": "Coder",
    "thinking": "Thinking",
}
CAPS = [0.01, 0.02, 0.05, 0.1]
TUNED = ["s3fifo", "lruqd", "gdsf", "arc", "wa"]
HEURISTICS = ["lru", "fifo", "lfu", "sieve", "arc", "s3fifo", "lruqd", "gdsf", "wa"]
BOUNDS = ["sampled-belady", "belady"]
NAMES = {
    "lru": "LRU",
    "fifo": "FIFO",
    "lfu": "LFU",
    "sieve": "SIEVE",
    "arc": "ARC",
    "s3fifo": "S3-FIFO",
    "lruqd": "LRU+QD",
    "gdsf": "GDSF",
    "wa": "ATC'25",
}


def read(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def family(r: dict) -> str:
    return r["policy"]


def best_configs(valid: list[dict]) -> dict[tuple[str, float, str], dict]:
    """For every tunable family, the configuration with the highest validation hit ratio."""
    best: dict[tuple[str, float, str], dict] = {}
    for r in valid:
        key = (r["trace"], r["cap_frac"], family(r))
        if key not in best or r["hit_ratio"] > best[key]["hit_ratio"]:
            best[key] = r
    return best


def test_spec(best: dict, traces: list[str]) -> dict:
    """Test-hour spec for the baselines: fixed policies, ATC'25 with the workload model fitted
    on the training rows, the tuned heuristics at their best validation setting, oracles and
    bounds."""
    policies = []
    for t in traces:
        for c in CAPS:
            for fam in TUNED:
                b = best.get((t, c, fam))
                if b is not None:
                    policies.append(
                        {"name": fam, "params": b["params"], "caps": [c], "traces": [t]}
                    )
    for name in ["lru", "fifo", "lfu", "sieve", "sampled-lru", "oracle-dead"]:
        policies.append({"name": name})
    for h in [60, 300]:
        policies.append(
            {"name": "oracle-within", "label": f"oracle-within-{h}", "params": {"horizon": h}}
        )
    for name in BOUNDS:
        policies.append({"name": name})
    return {
        "traces": [
            {"path": f"data/raw/qwen_{t}_blksz_16.jsonl", "window": [3600, 7200]} for t in traces
        ],
        "caps": CAPS,
        "time": True,
        "policies": policies,
        "wa_fits": {t: f"models/wa_{t}.json" for t in traces},
    }


def pacer_test_spec(valid: list[dict], traces: list[str]) -> dict:
    """Test-hour spec for Pacer: the fixed configuration everywhere, and the larger model with
    the horizon multiplier that won on validation for each trace and size."""
    best: dict = {}
    for r in valid:
        if r["label"].startswith("full"):
            k = (r["trace"], r["cap_frac"])
            if k not in best or r["hit_ratio"] > best[k]["hit_ratio"]:
                best[k] = r
    policies = []
    for i, t in enumerate(traces):
        policies.append(
            {
                "name": "pacer",
                "label": "pacer-fixed",
                "params": {"model": i, "mult": 1.0, "k": 32, "reuse": 4},
                "traces": [t],
            }
        )
        for c in CAPS:
            policies.append(
                {
                    "name": "pacer",
                    "label": "pacer",
                    "params": best[(t, c)]["params"],
                    "traces": [t],
                    "caps": [c],
                }
            )
    models = {str(i): f"models/fast_{t}.json" for i, t in enumerate(traces)}
    models |= {str(i + 10): f"models/hazard_{t}.json" for i, t in enumerate(traces)}
    return {
        "traces": [
            {"path": f"data/raw/qwen_{t}_blksz_16.jsonl", "window": [3600, 7200]} for t in traces
        ],
        "caps": CAPS,
        "time": True,
        "policies": policies,
        "models": models,
    }


def index(rows: list[dict]) -> dict[tuple[str, float, str], dict]:
    return {(r["trace"], r["cap_frac"], r["label"]): r for r in rows}


def buckets(r: dict) -> tuple[np.ndarray, np.ndarray]:
    b = r["buckets"]
    return (
        np.array([x["hits"] for x in b], dtype=float),
        np.array([x["accesses"] for x in b], dtype=float),
    )


T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
        2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060,
        2.056, 2.052, 2.048, 2.045, 2.042]  # fmt: skip


def paired_ci(a: dict, b: dict) -> tuple[float, float, float]:
    """Difference in hit ratio (a - b) with a 95% t-interval over the 5-minute buckets of the
    test window. Both runs see the same requests in each bucket; the ratio is linearized, so the
    per-bucket terms are (hits_a - hits_b) - diff * accesses. With 12 buckets this is wider
    (more conservative) than a percentile bootstrap."""
    ha, n = buckets(a)
    hb, nb = buckets(b)
    if len(ha) != len(hb) or not np.array_equal(n, nb):
        raise ValueError("runs are not paired")
    k = len(n)
    diff = (ha.sum() - hb.sum()) / n.sum()
    z = (ha - hb) - diff * n
    se = np.sqrt(k / (k - 1) * np.sum(z**2)) / n.sum()
    t = T975[k - 2] if k - 2 < len(T975) else 1.96
    return float(diff), float(diff - t * se), float(diff + t * se)


def best_heuristic(ix: dict, trace: str, cap: float) -> dict:
    cands = [ix[(trace, cap, h)] for h in HEURISTICS if (trace, cap, h) in ix]
    return max(cands, key=lambda r: r["hit_ratio"])


def summarize(test: list[dict]) -> dict:
    ix = index(test)
    out: dict = defaultdict(dict)
    for t in TRACES:
        for c in CAPS:
            if (t, c, "lru") not in ix:
                continue
            lru, pac, bel = ix[(t, c, "lru")], ix.get((t, c, "pacer")), ix[(t, c, "belady")]
            bh = best_heuristic(ix, t, c)
            cell = {
                "lru": lru["hit_ratio"],
                "best_heuristic": bh["label"],
                "best_heuristic_hit": bh["hit_ratio"],
                "belady": bel["hit_ratio"],
                "sampled_belady": ix[(t, c, "sampled-belady")]["hit_ratio"],
                "all": {
                    lab: ix[(t, c, lab)]["hit_ratio"] for (tt, cc, lab) in ix if tt == t and cc == c
                },
            }
            fixed = ix.get((t, c, "pacer-fixed"))
            if fixed is not None:
                gap = bel["hit_ratio"] - lru["hit_ratio"]
                cell.update(
                    {
                        "pacer_fixed": fixed["hit_ratio"],
                        "pacer_fixed_vs_lru": paired_ci(fixed, lru),
                        "pacer_fixed_vs_best": paired_ci(fixed, bh),
                        "pacer_fixed_vs_wa": paired_ci(fixed, ix[(t, c, "wa")]),
                        "gap_closed_pacer_fixed": (fixed["hit_ratio"] - lru["hit_ratio"]) / gap
                        if gap > 0
                        else float("nan"),
                        "pacer_fixed_ns_per_eviction": fixed["victim_ns"]
                        / max(fixed["evictions"], 1),
                    }
                )
            if pac is not None:
                gap = bel["hit_ratio"] - lru["hit_ratio"]
                cell.update(
                    {
                        "pacer": pac["hit_ratio"],
                        "pacer_params": pac["params"],
                        "pacer_vs_lru": paired_ci(pac, lru),
                        "pacer_vs_best": paired_ci(pac, bh),
                        "pacer_vs_wa": paired_ci(pac, ix[(t, c, "wa")]),
                        "gap_closed_pacer": (pac["hit_ratio"] - lru["hit_ratio"]) / gap
                        if gap > 0
                        else float("nan"),
                        "gap_closed_best_heuristic": (bh["hit_ratio"] - lru["hit_ratio"]) / gap
                        if gap > 0
                        else float("nan"),
                        "pacer_ns_per_decision": pac["ns_per_decision"],
                        "pacer_ns_per_eviction": pac["victim_ns"] / max(pac["evictions"], 1),
                    }
                )
            out[t][str(c)] = cell
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("select")
    s1.add_argument("validation")
    s1.add_argument("pacer_validation")
    s1.add_argument("--out", default="configs/test.json")
    s1.add_argument("--pacer-out", default="configs/test_pacer.json")
    s2 = sub.add_parser("summarize")
    s2.add_argument("test", nargs="+")
    s2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "select":
        best = best_configs(read(a.validation))
        Path(a.out).write_text(json.dumps(test_spec(best, TRACES), indent=1) + "\n")
        spec = pacer_test_spec(read(a.pacer_validation), TRACES)
        Path(a.pacer_out).write_text(json.dumps(spec, indent=1) + "\n")
        for (t, c, fam), r in sorted(best.items()):
            if fam in TUNED:
                print(f"{t:9s} {c:.2f} {fam:7s} {r['label']:24s} valid {100 * r['hit_ratio']:.2f}")
    else:
        s = summarize([r for path in a.test for r in read(path)])
        Path(a.out).write_text(json.dumps(s, indent=1) + "\n")
        for t, cells in s.items():
            for c, cell in cells.items():
                p = cell.get("pacer")
                print(
                    f"{t:9s} {c:5s} lru {100 * cell['lru']:6.2f}  best {cell['best_heuristic']:8s}"
                    f" {100 * cell['best_heuristic_hit']:6.2f}  pacer "
                    f"{100 * p if p is not None else float('nan'):6.2f}  belady "
                    f"{100 * cell['belady']:6.2f}"
                )


def fmt_ci(d: list[float]) -> str:
    return f"{100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]"


def verdict(d: list[float]) -> str:
    if d[1] > 0:
        return "better"
    if d[2] < 0:
        return "worse"
    return "tie"


def markdown(summary: dict, transfer: list[dict] | None = None) -> dict[str, str]:
    """Markdown blocks keyed by marker name, for docs/results.md and the README."""
    blocks: dict[str, str] = {}
    head = (
        "| Trace | Cache | LRU | Best tuned heuristic | ATC'25 | **Pacer** | Pacer vs best "
        "(95% CI) | Belady |\n|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    counts = {"better": 0, "tie": 0, "worse": 0}
    for t in TRACES:
        for c in CAPS:
            x = summary[t][str(c)]
            d = x["pacer_fixed_vs_best"]
            counts[verdict(d)] += 1
            rows.append(
                f"| {TRACE_LABEL[t]} | {100 * c:g}% | {100 * x['lru']:.2f} | "
                f"{100 * x['best_heuristic_hit']:.2f} ({NAMES[x['best_heuristic']]}) | "
                f"{100 * x['all']['wa']:.2f} | **{100 * x['pacer_fixed']:.2f}** | "
                f"{fmt_ci(d)} | {100 * x['belady']:.2f} |"
            )
    blocks["main"] = (
        head
        + "\n".join(rows)
        + f"\n\nPacer vs the best heuristic: better in {counts['better']} cells, statistically "
        f"tied in {counts['tie']}, worse in {counts['worse']}."
    )

    head = (
        "| Trace | Cache | Pacer (fixed) | Pacer (horizon tuned) | vs LRU | vs ATC'25 | "
        "gap closed (Pacer / best heuristic) |\n|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for t in TRACES:
        for c in CAPS:
            x = summary[t][str(c)]
            rows.append(
                f"| {TRACE_LABEL[t]} | {100 * c:g}% | {100 * x['pacer_fixed']:.2f} | "
                f"{100 * x['pacer']:.2f} | {fmt_ci(x['pacer_fixed_vs_lru'])} | "
                f"{fmt_ci(x['pacer_fixed_vs_wa'])} | {100 * x['gap_closed_pacer_fixed']:.0f}% / "
                f"{100 * x['gap_closed_best_heuristic']:.0f}% |"
            )
    blocks["detail"] = head + "\n".join(rows)

    order = ["lru", "fifo", "lfu", "sieve", "arc", "s3fifo", "lruqd", "gdsf", "wa"]
    head = (
        "| Trace | Cache | "
        + " | ".join(NAMES[o] for o in order)
        + " | sampled LRU | oracle: never reused | sampled Belady | Belady |\n|"
        + "---|" * (len(order) + 6)
        + "\n"
    )
    rows = []
    for t in TRACES:
        for c in CAPS:
            a = summary[t][str(c)]["all"]
            vals = [a.get(k) for k in [*order, "sampled-lru", "oracle-dead", "sampled-belady"]]
            vals.append(summary[t][str(c)]["belady"])
            rows.append(
                f"| {TRACE_LABEL[t]} | {100 * c:g}% | "
                + " | ".join(f"{100 * v:.2f}" if v is not None else "-" for v in vals)
                + " |"
            )
    blocks["baselines"] = head + "\n".join(rows)

    if transfer:
        tab: dict = defaultdict(dict)
        for r in transfer:
            tab[(r["trace"], r["cap_frac"])][r["label"].replace("pacer-from-", "")] = r["hit_ratio"]
        head = (
            "| Evaluated on | Cache | LRU | own model | "
            + " | ".join(f"model from {TRACE_LABEL[s]}" for s in TRACES)
            + " |\n|"
            + "---|" * (4 + len(TRACES))
            + "\n"
        )
        rows = []
        for t in TRACES:
            for c in CAPS:
                x = summary[t][str(c)]
                cells = []
                for s in TRACES:
                    v = tab[(t, c)].get(s)
                    cells.append("-" if v is None else f"{100 * v:.2f}")
                rows.append(
                    f"| {TRACE_LABEL[t]} | {100 * c:g}% | {100 * x['lru']:.2f} | "
                    f"{100 * x['pacer_fixed']:.2f} | " + " | ".join(cells) + " |"
                )
        blocks["transfer"] = head + "\n".join(rows)
    return blocks


ABLATIONS = [
    ("pacer", "Pacer, fixed configuration (compact model, K = 32, reuse 4, h x1)"),
    ("pacer-full", "larger model (63 leaves, 85-172 trees), K = 64, no reuse"),
    ("pacer-nosession", "without session features (gaps, turn, session age)"),
    ("pacer-nohistory", "without block history (hit count, fan-out)"),
    ("pacer-k8", "K = 8"),
    ("pacer-k16", "K = 16"),
    ("pacer-k64", "K = 64"),
    ("pacer-k128", "K = 128"),
    ("pacer-reuse1", "no burst reuse"),
    ("pacer-nofollow", "no chain following"),
    ("pacer-m0.5", "h x0.5"),
    ("pacer-m2", "h x2"),
    ("pacer-m4", "h x4"),
    ("lrb-imitation", "LRB-style regression on logged cache states (v1 approach)"),
    ("sampled-lru", "sampled LRU (K = 32, no model)"),
]


def ablation_md(rows: list[dict]) -> str:
    ix = index(rows)
    out = []
    for t in ["traceA", "coder"]:
        head = (
            f"**{TRACE_LABEL[t]}**\n\n| Variant | "
            + " | ".join(f"{100 * c:g}%" for c in CAPS)
            + " |\n|---|"
            + "---|" * len(CAPS)
            + "\n"
        )
        lines = []
        for lab, desc in ABLATIONS:
            vals = [ix.get((t, c, lab)) for c in CAPS]
            if all(v is None for v in vals):
                continue
            lines.append(
                f"| {desc} | "
                + " | ".join(f"{100 * v['hit_ratio']:.2f}" if v else "-" for v in vals)
                + " |"
            )
        out.append(head + "\n".join(lines))
    return "\n\n".join(out)


COST_LABELS = [
    ("lru", "LRU"),
    ("s3fifo", "S3-FIFO"),
    ("arc", "ARC"),
    ("wa", "ATC'25"),
    ("sampled-lru", "sampled LRU (K = 32)"),
    ("pacer-fixed", "Pacer (fixed: 13-37 trees, K = 32, reuse 4)"),
    ("pacer-full-k64", "Pacer with the 85-213 tree model, K = 64"),
]


def cost_md(rows: list[dict]) -> str:
    med: dict = defaultdict(list)
    for r in rows:
        med[(r["label"], r["trace"])].append(r["ns_per_eviction"])
    head = (
        "| Policy | "
        + " | ".join(TRACE_LABEL[t] for t in TRACES)
        + " |\n|---|"
        + "---|" * len(TRACES)
        + "\n"
    )
    lines = []
    for lab, desc in COST_LABELS:
        cells = []
        for t in TRACES:
            v = med.get((lab, t))
            cells.append(f"{float(np.median(v)):,.0f}" if v else "-")
        lines.append(f"| {desc} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines)


def live_md(r: dict) -> str:
    cfg, chk, summ = r["config"], r["check"], r["summary"]
    head = (
        "| Policy | hit ratio | prefill tokens | TTFT median (ms) | TTFT mean (ms) | "
        "mean TTFT vs LRU (95% CI) |\n|---|---|---|---|---|---|\n"
    )
    lines = []
    names = {"lru": "LRU", "s3fifo": "S3-FIFO", "sampled-belady": "sampled Belady"}
    for name, x in summ.items():
        lo, hi = x["ttft_mean_reduction_vs_lru_ci95"]
        if name == "lru":
            lines.append(
                f"| LRU | {100 * x['hit_ratio']:.2f}% | {x['prefill_tokens']:,} | "
                f"{1000 * x['ttft_median_s']:.1f} | {1000 * x['ttft_mean_s']:.1f} | reference |"
            )
            continue
        lines.append(
            f"| {names.get(name, name.capitalize())} | {100 * x['hit_ratio']:.2f}% | "
            f"{x['prefill_tokens']:,} | "
            f"{1000 * x['ttft_median_s']:.1f} | {1000 * x['ttft_mean_s']:.1f} | "
            f"{-100 * x['ttft_mean_reduction_vs_lru']:+.1f}% "
            f"[{-100 * hi:+.1f}%, {-100 * lo:+.1f}%] |"
        )
    note = (
        f"\n\n{cfg['limit']} requests of traceA from t = {cfg.get('start', 0):g} s, cache "
        f"{cfg['cap_blocks']} blocks ({cfg['pool_bytes'] / 2**30:.2f} GiB of bf16 KV), "
        f"{cfg['model']} on {cfg['device']['device_name']}, {cfg['repeats']} interleaved repeats. "
        f"Cached-prefix prefill vs full prefill on {chk['n']} requests: same argmax in "
        f"{100 * chk['argmax_equal_frac']:.1f}%, "
        f"max |logit difference| median {chk['max_abs_diff']['p50']:.3f}."
    )
    return head + "\n".join(lines) + note


def csr(r: dict, alpha: float) -> float:
    """Compute-savings ratio: hits weighted by each block's prefill cost 1 + alpha * depth."""
    num = r["hits"] + alpha * r["hit_depth_sum"]
    den = r["accesses"] + alpha * r["access_depth_sum"]
    return num / den if den else 0.0


def csr_md(test: list[dict], summary: dict, alpha: float) -> str:
    """Compute-weighted table; the best heuristic is picked by compute savings here."""
    ix = index(test)
    head = (
        "| Trace | Cache | LRU | best heuristic | ATC'25 | Pacer | Belady |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for t in TRACES:
        for c in CAPS:
            cands = [ix[(t, c, h)] for h in HEURISTICS if (t, c, h) in ix]
            bh = max(cands, key=lambda r: csr(r, alpha))["label"]
            vals = [ix[(t, c, lab)] for lab in ["lru", bh, "wa", "pacer-fixed", "belady"]]
            lines.append(
                f"| {TRACE_LABEL[t]} | {100 * c:g}% | "
                + " | ".join(f"{100 * csr(v, alpha):.2f}" for v in vals)
                + " |"
            )
    return head + "\n".join(lines)


def criteria_md(
    summary: dict,
    test: list[dict],
    transfer: list[dict],
    overhead: list[dict],
    live: dict,
    alpha: float,
) -> str:
    """The success criteria set before the build, each measured from the result files."""
    ix = index(test)
    cells = [(t, c) for t in TRACES for c in CAPS]
    rows = []

    def add(cid: str, target: str, measured: str, status: str) -> None:
        rows.append(f"| {cid} | {target} | {measured} | {status} |")

    online = [h for h in HEURISTICS if all((t, c, h) in ix for t, c in cells)]
    bel_ok = all(
        ix[(t, c, "belady")]["hits"] >= max(ix[(t, c, h)]["hits"] for h in online)
        and ix[(t, c, "belady")]["hits"] >= ix[(t, c, "pacer-fixed")]["hits"]
        for t, c in cells
    )
    add(
        "C1",
        "Rust simulator equals the Python reference; Belady equals brute force on small "
        "traces and is at least every online policy",
        f"differential and brute-force tests pass (8 policies); Belady highest in "
        f"{'all 16' if bel_ok else 'not all'} test cells",
        "met" if bel_ok else "partial",
    )
    add(
        "C2",
        "LRU, best heuristic, oracle ladder and Belady on 4 traces x 4 sizes",
        "all reported (results.md)",
        "met",
    )

    small = [(t, c) for t in TRACES for c in CAPS[:3]]
    wins = [x for x in small if summary[x[0]][str(x[1])]["pacer_fixed_vs_lru"][1] > 0]
    miss = [f"{TRACE_LABEL[t]} {100 * c:g}%" for t, c in small if (t, c) not in wins]
    add(
        "C3",
        "Better than LRU at 1, 2, 5% on every trace (CI excludes 0)",
        f"{len(wins)} of {len(small)}; not: {', '.join(miss) or 'none'}",
        "met" if not miss else "missed",
    )

    ok = {}
    for c in [0.02, 0.05]:
        ok[c] = sum(summary[t][str(c)]["pacer_fixed_vs_best"][2] >= 0 for t in TRACES)
    chat = [summary["traceA"][str(c)]["pacer_fixed_vs_best"] for c in CAPS]
    best_chat = max(chat, key=lambda d: d[0])
    c4 = all(v >= 3 for v in ok.values()) and any(d[0] >= 0.01 and d[1] > 0 for d in chat)
    add(
        "C4",
        "At least the best tuned heuristic on 3 of 4 traces at 2% and 5%; +1 point on "
        "chat somewhere (CI excludes 0)",
        f"not worse on {ok[0.02]}/4 at 2%, {ok[0.05]}/4 at 5%; chat best {fmt_ci(best_chat)}",
        "met" if c4 else "missed",
    )

    g = {c: summary["traceA"][str(c)]["gap_closed_pacer_fixed"] for c in CAPS}
    add(
        "C5",
        "Close 20% of the LRU-to-Belady gap on chat at 2% and 5%",
        f"{100 * g[0.01]:.1f}% at 1%, {100 * g[0.02]:.1f}% at 2%, {100 * g[0.05]:.1f}% at 5%",
        "met" if g[0.02] >= 0.2 and g[0.05] >= 0.2 else "missed",
    )

    below = [
        f"{TRACE_LABEL[t]} {100 * c:g}%"
        for t, c in cells
        if csr(ix[(t, c, "pacer-fixed")], alpha) < csr(ix[(t, c, "lru")], alpha)
    ]
    add(
        "C6",
        "Compute-weighted savings at least LRU's everywhere",
        f"below LRU in {', '.join(below) or 'no cell'}",
        "met" if not below else "missed",
    )

    per_trace: dict = defaultdict(list)
    for r in overhead:
        if r["label"] == "pacer-fixed":
            per_trace[r["trace"]].append(r["ns_per_eviction"])
    fixed = [float(np.median(v)) for v in per_trace.values()]
    slowest = max(r["wall_s"] for r in test)
    add(
        "C7",
        "At most 2 us per evicted block; a full sweep under 10 min",
        f"{min(fixed):.0f}-{max(fixed):.0f} ns per eviction; slowest single test run "
        f"{slowest / 60:.0f} min (S3-FIFO), whole pipeline about 4 h",
        "partial",
    )

    lru = {(t, c): summary[t][str(c)]["lru"] for t, c in cells}
    tr_a = [r for r in transfer if r["label"] == "pacer-from-traceA"]
    worst = min(r["hit_ratio"] - lru[(r["trace"], r["cap_frac"])] for r in tr_a)
    add(
        "C8",
        "The chat model is never more than 0.5 points below LRU on other traces",
        f"worst {100 * worst:+.2f} points",
        "met" if worst >= -0.005 else "missed",
    )

    chk = live["check"]
    pac = live["summary"]["pacer"]
    lo, hi = pac["ttft_mean_reduction_vs_lru_ci95"]
    add(
        "C9",
        "Cached-prefix prefill picks the same next token on 99%; TTFT vs LRU with a CI",
        f"{100 * chk['argmax_equal_frac']:.1f}% of {chk['n']}; mean TTFT "
        f"{-100 * pac['ttft_mean_reduction_vs_lru']:+.1f}% [{-100 * hi:+.1f}%, {-100 * lo:+.1f}%] "
        "over requests (2 repeats, drift not covered)",
        "partial",
    )
    add(
        "C10",
        "`make reproduce` regenerates every number; CI green",
        "CI green; deterministic stages rebuild identically; 10 ATC'25 test cells were re-run "
        "outside the logged run (see the engineering log)",
        "partial",
    )
    head = "| | Target | Measured | Status |\n|---|---|---|---|\n"
    return head + "\n".join(rows)


def fill(path: Path, blocks: dict[str, str]) -> None:
    text = path.read_text()
    for name, body in blocks.items():
        start, end = f"<!-- results:{name} -->", f"<!-- /results:{name} -->"
        if start in text and end in text:
            a = text.index(start) + len(start)
            b = text.index(end)
            text = text[:a] + "\n" + body + "\n" + text[b:]
    path.write_text(text)


def fill_main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/final_summary.json")
    ap.add_argument("--transfer", default="results/transfer.jsonl")
    ap.add_argument("--ablations", default="results/ablations.jsonl")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    summary = json.loads(Path(a.summary).read_text())
    transfer = read(a.transfer) if Path(a.transfer).exists() else None
    blocks = markdown(summary, transfer)
    auc_rows = []
    for t in TRACES:
        f = Path(f"results/training/fast_{t}.json")
        if f.exists():
            r = json.loads(f.read_text())
            auc_rows.append(
                f"| {TRACE_LABEL[t]} | {r['gbdt']['auc']:.3f} | {r['workload_exp']['auc']:.3f} | "
                f"{r['gbdt']['logloss']:.3f} | {r['workload_exp']['logloss']:.3f} | "
                f"{r['constant']['logloss']:.3f} | {r['best_iteration']} |"
            )
    if auc_rows:
        blocks["auc"] = (
            "| Trace | Pacer AUC | ATC'25 AUC | Pacer log loss | ATC'25 log loss | "
            "constant log loss | trees |\n|---|---|---|---|---|---|---|\n" + "\n".join(auc_rows)
        )
    if Path(a.ablations).exists():
        blocks["ablations"] = ablation_md(read(a.ablations))
    cost = Path("results/live/prefill_cost.json")
    if cost.exists():
        alpha = json.loads(cost.read_text())["block_cost"]["alpha"]
        blocks["csr"] = csr_md(read("results/final_test.jsonl"), summary, alpha)
    crit_inputs = ["results/overhead.json", "results/live/replay.json", "results/transfer.jsonl"]
    if cost.exists() and all(Path(f).exists() for f in crit_inputs):
        blocks["criteria"] = criteria_md(
            summary,
            read("results/final_test.jsonl"),
            read("results/transfer.jsonl"),
            json.loads(Path("results/overhead.json").read_text()),
            json.loads(Path("results/live/replay.json").read_text()),
            json.loads(cost.read_text())["block_cost"]["alpha"],
        )
    if Path("results/overhead.json").exists():
        blocks["cost"] = cost_md(json.loads(Path("results/overhead.json").read_text()))
    if Path("results/live/replay.json").exists():
        blocks["live"] = live_md(json.loads(Path("results/live/replay.json").read_text()))
    for f in a.files:
        fill(Path(f), blocks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "fill":
        sys.argv.pop(1)
        fill_main()
    else:
        main()

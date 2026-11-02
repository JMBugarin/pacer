"""Quick demo: replays the test hour of the To-C chat trace at one cache size with LRU, the
ATC'25 workload-aware policy, Pacer and Belady, and prints the comparison (about a minute)."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "pacer"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traceA")
    ap.add_argument("--cap", type=float, default=0.02)
    a = ap.parse_args()
    raw = ROOT / "data" / "raw" / f"qwen_{a.trace}_blksz_16.jsonl"
    if not raw.exists():
        subprocess.run([str(ROOT / "scripts" / "fetch_data.sh")], check=True)
    policies = [
        {"name": "lru", "label": "LRU (vLLM default)"},
        {"name": "s3fifo", "label": "S3-FIFO (default settings)"},
        {"name": "wa", "label": "ATC'25 workload-aware"},
        {
            "name": "pacer",
            "label": "Pacer",
            "params": {"model": 0, "mult": 1.0, "k": 32, "reuse": 4},
        },
        {"name": "belady", "label": "Belady (offline reference)"},
    ]
    spec = {
        "traces": [{"path": str(raw), "window": [3600, 7200]}],
        "caps": [a.cap],
        "time": True,
        "policies": policies,
        "models": {"0": str(ROOT / "models" / f"fast_{a.trace}.json")},
        "wa_fits": {a.trace: str(ROOT / "models" / f"wa_{a.trace}.json")},
    }
    with tempfile.TemporaryDirectory() as d:
        sp, out = Path(d) / "spec.json", Path(d) / "out.jsonl"
        sp.write_text(json.dumps(spec))
        subprocess.run([str(BIN), "sweep", "--spec", str(sp), "--out", str(out)], check=True)
        rows = {r["label"]: r for r in map(json.loads, out.read_text().splitlines())}
    lru = rows["LRU (vLLM default)"]["hit_ratio"]
    bel = rows["Belady (offline reference)"]["hit_ratio"]
    print(f"\n{a.trace}, cache = {100 * a.cap:g}% of unique blocks, test hour (t = 1-2 h)\n")
    print(f"{'policy':26s} {'hit ratio':>9s} {'vs LRU':>8s} {'gap closed':>11s} {'us/evict':>9s}")
    for p in policies:
        r = rows[p["label"]]
        h = r["hit_ratio"]
        closed = (h - lru) / (bel - lru) if bel > lru else float("nan")
        us = r["victim_ns"] / max(r["evictions"], 1) / 1000
        print(
            f"{p['label']:26s} {100 * h:8.2f}% {100 * (h - lru):+7.2f} "
            f"{100 * closed:10.1f}% {us:9.2f}"
        )


if __name__ == "__main__":
    main()

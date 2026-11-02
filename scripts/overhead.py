"""Decision cost per eviction on an idle machine: each policy replays the test hour of every
trace at 2% capacity, one run at a time on one thread, with victim selection timed."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "pacer"
TRACES = ["traceA", "traceB", "coder", "thinking"]
REPEATS = 3


def raw(t: str) -> Path:
    return ROOT / "data" / "raw" / f"qwen_{t}_blksz_16.jsonl"


def main() -> None:
    policies = [
        {"name": "lru"},
        {"name": "s3fifo"},
        {"name": "arc"},
        {"name": "wa", "params": {"fit_end": 3600}},
        {"name": "sampled-lru", "params": {"k": 32}},
    ]
    for i, t in enumerate(TRACES):
        policies.append(
            {
                "name": "pacer",
                "label": "pacer-fixed",
                "params": {"model": i, "mult": 1.0, "k": 32, "reuse": 4},
                "traces": [t],
            }
        )
        policies.append(
            {
                "name": "pacer",
                "label": "pacer-full-k64",
                "params": {"model": i + 10, "mult": 1.0, "k": 64},
                "traces": [t],
            }
        )
    models = {str(i): str(ROOT / f"models/v2/fast_{t}.json") for i, t in enumerate(TRACES)}
    models |= {str(i + 10): str(ROOT / f"models/v2/hazard_{t}.json") for i, t in enumerate(TRACES)}
    out: list[dict] = []
    with tempfile.TemporaryDirectory() as d:
        for rep in range(REPEATS):
            spec = {
                "traces": [{"path": str(raw(t)), "window": [3600, 7200]} for t in TRACES],
                "caps": [0.02],
                "time": True,
                "policies": policies,
                "models": models,
            }
            sp, res = Path(d) / "spec.json", Path(d) / f"out{rep}.jsonl"
            sp.write_text(json.dumps(spec))
            subprocess.run(
                [str(BIN), "sweep", "--spec", str(sp), "--out", str(res), "--threads", "1"],
                check=True,
                capture_output=True,
            )
            for line in res.read_text().splitlines():
                r = json.loads(line)
                out.append(
                    {
                        "repeat": rep,
                        "trace": r["trace"],
                        "label": r["label"],
                        "evictions": r["evictions"],
                        "decisions": r["decisions"],
                        "victim_ns": r["victim_ns"],
                        "ns_per_eviction": r["victim_ns"] / max(r["evictions"], 1),
                        "wall_s": r["wall_s"],
                        "hit_ratio": r["hit_ratio"],
                    }
                )
    (ROOT / "results" / "overhead.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out:
        if r["repeat"] == 0:
            print(f"{r['trace']:9s} {r['label']:15s} {r['ns_per_eviction']:9.0f} ns/eviction")


if __name__ == "__main__":
    main()

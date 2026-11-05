"""Two follow-up experiments, both on settings fixed in advance (nothing is tuned on the test hour).

cost: the fixed configuration with candidates scored by expected saved prefill (reuse
probability times 1 + alpha * depth), alpha taken from the M5 Pro prefill calibration. Run on the
validation window and the test hour.

tpot: the reported pipeline at other decode speeds. For each speed the hazard rows and compact
models are rebuilt (the rows depend on completion times), and every baseline plus the fixed
configuration is replayed on the test hour with the settings chosen at 20 ms/token.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "pacer"
TRACES = ["traceA", "traceB", "coder", "thinking"]


def sweep(spec: dict, out: Path, threads: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    sp = out.with_suffix(".spec.json")
    sp.write_text(json.dumps(spec, indent=1) + "\n")
    subprocess.run(
        [str(BIN), "sweep", "--spec", str(sp), "--out", str(out), "--threads", str(threads)],
        check=True,
    )


def fixed(
    model_dir: str, alpha: float = 0.0, label: str = "pacer-fixed"
) -> tuple[list[dict], dict[str, str]]:
    params = {"mult": 1.0, "k": 32, "reuse": 4}
    if alpha:
        params["cost_alpha"] = alpha
    return [
        {"name": "pacer", "label": label, "params": dict(params, model=i), "traces": [t]}
        for i, t in enumerate(TRACES)
    ], {str(i): f"{model_dir}/fast_{t}.json" for i, t in enumerate(TRACES)}


def cost(threads: int) -> None:
    alpha = json.loads((ROOT / "results/live/prefill_cost.json").read_text())["block_cost"]["alpha"]
    pols, models = fixed("models", alpha, "pacer-cost")
    for name, window in [("validation", [2700, 3600]), ("test", [3600, 7200])]:
        spec = {
            "traces": [
                {"path": f"data/raw/qwen_{t}_blksz_16.jsonl", "window": window} for t in TRACES
            ],
            "caps": [0.01, 0.02, 0.05, 0.1],
            "time": False,
            "policies": pols,
            "models": models,
        }
        sweep(spec, ROOT / f"results/research/cost_{name}.jsonl", threads)


def tpot(speeds: list[float], threads: int) -> None:
    base = json.loads((ROOT / "configs/test.json").read_text())
    for s in speeds:
        tag = f"tpot{round(s * 1000):03d}"
        rows = ROOT / "data/labels" / tag
        mdir = ROOT / "models/history" / tag
        rows.mkdir(parents=True, exist_ok=True)
        mdir.mkdir(parents=True, exist_ok=True)
        for t in TRACES:
            raw = f"data/raw/qwen_{t}_blksz_16.jsonl"
            subprocess.run(
                [str(BIN), "hazard", "--trace", raw, "--cutoff", "3600", "--rate", "0.15",
                 "--per-touch", "8", "--tpot", str(s), "--out", str(rows / f"{t}.npy")],
                check=True,
            )  # fmt: skip
            subprocess.run(
                [sys.executable, "-m", "pacer.hazard", str(rows / f"{t}.npy"), "--num-leaves", "31",
                 "--learning-rate", "0.25", "--wa-out", str(mdir / f"wa_{t}.json"),
                 "--out", str(mdir / f"fast_{t}.json"),
                 "--report", str(ROOT / f"results/research/{tag}_fast_{t}.json")],
                check=True,
                env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
            )  # fmt: skip
        pols, models = fixed(str(mdir.relative_to(ROOT)))
        spec = dict(base, tpot=s, time=False, models=models)
        spec["policies"] = base["policies"] + pols
        spec["wa_fits"] = {t: str((mdir / f"wa_{t}.json").relative_to(ROOT)) for t in TRACES}
        sweep(spec, ROOT / f"results/research/{tag}_test.jsonl", threads)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cost", "tpot"])
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.01, 0.04])
    ap.add_argument("--threads", type=int, default=14)
    a = ap.parse_args()
    if a.what == "cost":
        cost(a.threads)
    else:
        tpot(a.speeds, a.threads)


if __name__ == "__main__":
    main()

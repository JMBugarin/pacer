"""Train the time-to-next-use model on logged eviction candidates and export it for Rust.

Rows come from `pacer label`: [decision id, time, features..., ln(1 + seconds to next use)].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

FEATURES = [
    "age",
    "age_insert",
    "n_access",
    "depth",
    "is_output",
    "fanout",
    "turn",
    "rtype",
    "input_len",
    "output_len",
    "since_done",
    "prev_gap",
    "mean_gap",
    "session_age",
    "child_arrived",
    "arrivals_60s",
    "evict_age_ewma",
    "rel_depth",
]
N_COLS = len(FEATURES) + 3

DEFAULT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
}


def load_rows(paths: list[str | Path]) -> np.ndarray:
    parts = []
    offset = 0.0
    for p in paths:
        a = np.load(p)
        if a.ndim != 2 or a.shape[1] != N_COLS:
            raise ValueError(f"{p}: expected {N_COLS} columns, got {a.shape}")
        a = a.astype(np.float64)
        a[:, 0] += offset
        offset = a[:, 0].max() + 1 if len(a) else offset
        parts.append(a)
    return np.concatenate(parts)


def split(rows: np.ndarray, valid_from: float) -> tuple[np.ndarray, np.ndarray]:
    t = rows[:, 1]
    return rows[t < valid_from], rows[t >= valid_from]


def decision_metrics(rows: np.ndarray, score: np.ndarray) -> dict:
    """Per decision: does argmax(score) pick a candidate with the largest true next use?"""
    dec = rows[:, 0]
    y = rows[:, -1]
    order = np.lexsort((-score, dec))
    d_sorted = dec[order]
    first = np.r_[True, d_sorted[1:] != d_sorted[:-1]]
    chosen = order[first]
    starts = np.flatnonzero(first)
    ymax = np.maximum.reduceat(y[np.lexsort((dec,))], starts)
    ychosen = y[chosen]
    return {
        "decisions": len(chosen),
        "top1": float(np.mean(ychosen >= ymax - 1e-6)),
        "regret_log": float(np.mean(ymax - ychosen)),
        "chosen_ttnu_median_s": float(np.median(np.expm1(ychosen))),
    }


EWMA_COL = 2 + FEATURES.index("evict_age_ewma")


def targets(rows: np.ndarray, target: str, mult: float) -> np.ndarray:
    """`log`: ln(1 + seconds to next use). `beyond`: 1 if the next use comes later than `mult`
    times the cache's recent eviction age, i.e. the block would likely be gone by then anyway."""
    if target == "log":
        return rows[:, -1]
    horizon = np.maximum(mult * rows[:, EWMA_COL], 1.0)
    return (np.expm1(rows[:, -1]) > horizon).astype(np.float64)


def train(
    rows: np.ndarray,
    valid_from: float,
    rounds: int,
    params: dict | None = None,
    target: str = "log",
    mult: float = 2.0,
):
    p = dict(DEFAULT_PARAMS, **(params or {}))
    if target == "beyond":
        p["objective"] = "binary"
    tr, va = split(rows, valid_from)
    x_tr, y_tr = tr[:, 2:-1].astype(np.float32), targets(tr, target, mult)
    x_va, y_va = va[:, 2:-1].astype(np.float32), targets(va, target, mult)
    dtr = lgb.Dataset(x_tr, y_tr, feature_name=FEATURES, free_raw_data=False)
    dva = lgb.Dataset(x_va, y_va, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        p,
        dtr,
        num_boost_round=rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    pred = booster.predict(x_va, num_iteration=booster.best_iteration)
    age = va[:, 2]
    report = {
        "target": target,
        "mult": mult,
        "train_rows": len(tr),
        "valid_rows": len(va),
        "best_iteration": booster.best_iteration,
        "valid_rmse": float(np.sqrt(np.mean((pred - y_va) ** 2))),
        "valid_rmse_const": float(np.sqrt(np.mean((y_tr.mean() - y_va) ** 2))),
        "model": decision_metrics(va, pred),
        "age": decision_metrics(va, age),
        "importance_gain": dict(
            sorted(
                zip(FEATURES, booster.feature_importance("gain").tolist(), strict=True),
                key=lambda kv: -kv[1],
            )
        ),
    }
    return booster, report


def export(booster: lgb.Booster, path: str | Path) -> None:
    model = booster.dump_model(num_iteration=booster.best_iteration)
    model["feature_names"] = FEATURES
    Path(path).write_text(json.dumps(model))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--valid-from", type=float, default=2700.0)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--target", choices=["log", "beyond"], default="log")
    ap.add_argument("--mult", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    a = ap.parse_args()
    rows = load_rows(a.rows)
    booster, report = train(
        rows, a.valid_from, a.rounds, {"num_leaves": a.num_leaves}, a.target, a.mult
    )
    export(booster, a.out)
    text = json.dumps(report, indent=2)
    print(text)
    if a.report:
        Path(a.report).write_text(text + "\n")


if __name__ == "__main__":
    main()

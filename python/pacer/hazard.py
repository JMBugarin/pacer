"""Session-return hazard model: P(block is next used in (a, a + h] | not used for a seconds).

Rows come from `pacer hazard` and are independent of any cache policy: every sampled block
release in the training period, at several (elapsed a, horizon h) pairs, labeled only with uses
seen before the training cutoff. The baseline is the per-workload exponential model of
"KVCache Cache in the Wild" (ATC'25), fitted by maximum likelihood on the same rows; the same
fit drives the ATC'25 policy in the simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

FEATURES = [
    "a",
    "h",
    "n_access",
    "depth",
    "is_output",
    "fanout",
    "turn",
    "rtype",
    "input_len",
    "output_len",
    "prev_gap",
    "mean_gap",
    "session_age",
    "arrivals_60s",
    "rel_depth",
    "a_over_gap",
]
N_COLS = len(FEATURES) + 2
MAX_TURN = 5

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
    "deterministic": True,
    "force_row_wise": True,
}


def load(paths: list[str | Path]) -> np.ndarray:
    parts = [np.load(p).astype(np.float64) for p in paths]
    for p, a in zip(paths, parts, strict=True):
        if a.ndim != 2 or a.shape[1] != N_COLS:
            raise ValueError(f"{p}: expected {N_COLS} columns, got {a.shape}")
    return np.concatenate(parts)


def col(name: str) -> int:
    return 1 + FEATURES.index(name)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = y > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def workload(rows: np.ndarray) -> np.ndarray:
    turn = np.clip(rows[:, col("turn")], 1, MAX_TURN)
    return (rows[:, col("rtype")] * MAX_TURN + turn - 1).astype(int)


class WorkloadExp:
    """ATC'25-style model: per (type, turn), reuse probability and an exponential reuse time,
    fitted by maximum likelihood on the hazard rows (each row is an interval observation)."""

    def __init__(self, rows: np.ndarray):
        w = workload(rows)
        self.params: dict[int, tuple[float, float]] = {}
        grid_p = np.linspace(0.02, 0.98, 49)
        grid_tau = np.exp(np.linspace(np.log(5), np.log(5000), 60))
        for wk in np.unique(w):
            m = w == wk
            a, h, y = rows[m, col("a")], rows[m, col("h")], rows[m, -1]
            best = (-np.inf, 0.5, 300.0)
            for tau in grid_tau:
                for p in grid_p:
                    q = self._prob(p, tau, a, h)
                    ll = np.sum(y * np.log(q + 1e-9) + (1 - y) * np.log(1 - q + 1e-9))
                    if ll > best[0]:
                        best = (ll, p, tau)
            self.params[int(wk)] = (best[1], best[2])

    @staticmethod
    def _prob(p, tau, a, h):
        alive = p * np.exp(-a / tau)
        return alive * (1 - np.exp(-h / tau)) / (alive + 1 - p)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        w = workload(rows)
        out = np.empty(len(rows))
        for i, wk in enumerate(w):
            p, tau = self.params.get(int(wk), (0.3, 300.0))
            out[i] = self._prob(p, tau, rows[i, col("a")], rows[i, col("h")])
        return out


def train(
    rows: np.ndarray,
    valid_from: float,
    rounds: int = 2000,
    params: dict | None = None,
    stop_until: float | None = None,
):
    """Fits on rows whose whole window ends before `valid_from`, so no label looks into the
    validation period. Early stopping uses validation rows that start before `stop_until`;
    the reported metrics use the rest."""
    t = rows[:, 0]
    end = t + rows[:, col("a")] + rows[:, col("h")]
    if stop_until is None:
        stop_until = valid_from + (t.max() - valid_from) / 2
    tr = rows[end <= valid_from]
    es = rows[(t >= valid_from) & (t < stop_until)]
    va = rows[t >= stop_until]
    x_tr, y_tr = tr[:, 1:-1].astype(np.float32), tr[:, -1]
    x_es, y_es = es[:, 1:-1].astype(np.float32), es[:, -1]
    x_va, y_va = va[:, 1:-1].astype(np.float32), va[:, -1]
    p = dict(PARAMS, **(params or {}))
    dtr = lgb.Dataset(x_tr, y_tr, feature_name=FEATURES, free_raw_data=False)
    des = lgb.Dataset(x_es, y_es, reference=dtr, free_raw_data=False)
    booster = lgb.train(
        p, dtr, rounds, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    pred = booster.predict(x_va, num_iteration=booster.best_iteration)
    wexp = WorkloadExp(tr)
    pw = wexp.predict(va)
    h = va[:, col("h")]
    by_h = {}
    for lo, hi in [(0, 60), (60, 300), (300, 1e9)]:
        m = (h >= lo) & (h < hi)
        by_h[f"h in [{lo:g}, {hi:g})"] = {
            "rows": int(m.sum()),
            "gbdt_auc": auc(y_va[m], pred[m]),
            "workload_exp_auc": auc(y_va[m], pw[m]),
        }
    report = {
        "train_rows": len(tr),
        "early_stop_rows": len(es),
        "valid_rows": len(va),
        "valid_from": valid_from,
        "early_stop_until": stop_until,
        "valid_max_window_s": float((va[:, col("a")] + va[:, col("h")]).max()),
        "valid_positive_rate": float(y_va.mean()),
        "best_iteration": booster.best_iteration,
        "gbdt": {"auc": auc(y_va, pred), "logloss": logloss(y_va, pred)},
        "workload_exp": {"auc": auc(y_va, pw), "logloss": logloss(y_va, pw)},
        "constant": {"logloss": logloss(y_va, np.full(len(y_va), y_tr.mean()))},
        "auc_by_horizon": by_h,
        "importance_gain": dict(
            sorted(
                zip(FEATURES, booster.feature_importance("gain").tolist(), strict=True),
                key=lambda kv: -kv[1],
            )
        ),
    }
    return booster, report, wexp


def export_workload(wexp: WorkloadExp, path: str | Path) -> None:
    """Writes the ATC'25 workload parameters for the Rust policy (40 workloads)."""
    n = 8 * MAX_TURN
    p = [wexp.params.get(w, (0.3, 300.0))[0] for w in range(n)]
    tau = [wexp.params.get(w, (0.3, 300.0))[1] for w in range(n)]
    Path(path).write_text(json.dumps({"p_reuse": p, "tau": tau}) + "\n")


def export(booster: lgb.Booster, path: str | Path) -> None:
    model = booster.dump_model(num_iteration=booster.best_iteration)
    model["feature_names"] = FEATURES
    Path(path).write_text(json.dumps(model))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--valid-from", type=float, default=2700.0)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--wa-out", help="also write the fitted ATC'25 workload parameters")
    a = ap.parse_args()
    rows = load(a.rows)
    for name in a.drop:
        rows[:, col(name)] = 0.0
    booster, report, wexp = train(
        rows,
        a.valid_from,
        params={"num_leaves": a.num_leaves, "learning_rate": a.learning_rate},
    )
    report["dropped"] = a.drop
    export(booster, a.out)
    if a.wa_out:
        export_workload(wexp, a.wa_out)
    text = json.dumps(report, indent=2)
    print(text)
    if a.report:
        Path(a.report).write_text(text + "\n")


if __name__ == "__main__":
    main()

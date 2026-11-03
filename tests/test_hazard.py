from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest
from conftest import FIXTURES

hazard = pytest.importorskip("pacer.hazard")


def pairwise_auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", range(40))
def test_auc_matches_pairwise(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 120))
    y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(np.float64)
    s = rng.normal(size=n) + y * rng.uniform(0, 2)
    if seed % 2:
        s = np.round(s, 1)
    if seed % 5 == 0:
        s = rng.integers(0, 3, size=n).astype(np.float64)
    want = pairwise_auc(y, s)
    got = hazard.auc(y, s)
    if np.isnan(want):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(want, abs=1e-12)


def test_auc_edges():
    assert hazard.auc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 1.0
    assert hazard.auc(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert hazard.auc(np.array([1.0, 0.0, 1.0]), np.array([2.0, 2.0, 2.0])) == 0.5
    assert np.isnan(hazard.auc(np.ones(4), np.arange(4.0)))
    assert np.isnan(hazard.auc(np.zeros(4), np.arange(4.0)))


def hazard_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    rows = np.zeros((n, hazard.N_COLS))
    rows[:, 0] = np.sort(rng.uniform(0, 3600, n))
    for name in hazard.FEATURES:
        rows[:, hazard.col(name)] = rng.uniform(0, 10, n)
    rows[:, hazard.col("a")] = np.where(rng.random(n) < 0.2, 0.0, rng.uniform(5, 3600, n))
    rows[:, hazard.col("h")] = rng.uniform(10, 2000, n)
    rows[:, hazard.col("turn")] = rng.integers(1, 9, n)
    rows[:, hazard.col("rtype")] = rng.integers(0, 3, n)
    p = np.exp(-rows[:, hazard.col("a")] / 300) * 0.6
    rows[:, -1] = (rng.random(n) < p).astype(np.float64)
    return rows


def test_workload_matches_rust_definition():
    rows = np.zeros((4, hazard.N_COLS))
    rows[:, hazard.col("rtype")] = [0, 2, 7, 3]
    rows[:, hazard.col("turn")] = [1, 3, 9, 0]
    assert hazard.workload(rows).tolist() == [0, 12, 39, 15]


def test_workload_exp_probabilities_in_unit_interval():
    rng = np.random.default_rng(3)
    model = hazard.WorkloadExp(hazard_rows(rng, 600))
    for p, tau in model.params.values():
        assert 0.02 <= p <= 0.98 and 5 - 1e-9 <= tau <= 5000 + 1e-9
    probe = hazard_rows(rng, 400)
    extremes = list(product([0.0, 1e-3, 60.0, 1e6, 1e12], [0.0, 1.0, 600.0, 1e9]))
    for i, (a, h) in enumerate(extremes):
        probe[i, hazard.col("a")] = a
        probe[i, hazard.col("h")] = h
    probe[-5:, hazard.col("rtype")] = 7
    pred = model.predict(probe)
    assert pred.shape == (len(probe),)
    assert np.all(np.isfinite(pred)) and np.all(pred >= 0) and np.all(pred <= 1)


def test_workload_exp_prob_grid():
    a = np.array([0.0, 1.0, 100.0, 1e4, 1e8])
    for p, tau, h in product([0.02, 0.5, 0.98], [5.0, 300.0, 5000.0], [0.0, 10.0, 1e3, 1e9]):
        q = hazard.WorkloadExp._prob(p, tau, a, h)
        assert np.all(q >= 0) and np.all(q <= 1)
        assert np.all(np.diff(q) <= 1e-15), "survival-weighted reuse must not grow with age"
    assert hazard.WorkloadExp._prob(0.4, 300.0, 0.0, np.inf) == pytest.approx(0.4)


def tree_value(node: dict, x: list[float]) -> float:
    while "leaf_value" not in node:
        left = x[node["split_feature"]] <= node["threshold"]
        node = node["left_child"] if left else node["right_child"]
    return node["leaf_value"]


def test_gbdt_fixture_is_self_consistent():
    model = json.loads((FIXTURES / "gbdt_model.json").read_text())
    cases = json.loads((FIXTURES / "gbdt_cases.json").read_text())
    assert len(cases["rows"]) == len(cases["pred"]) == 200
    for x, want in zip(cases["rows"], cases["pred"], strict=True):
        assert np.float32(x).astype(np.float64).tolist() == x
        got = sum(tree_value(t["tree_structure"], x) for t in model["tree_info"])
        assert abs(got - want) < 1e-9

"""refsim (the executable spec) against the Rust CLI: identical hits, accesses, evictions, depth
sums and overflow for every policy both implement.

refsim's victim scans are O(cache), so the fast tier replays short prefixes of the fixtures plus
many small random traces (where policies actually disagree); the slow tier replays each whole
fixture."""

from __future__ import annotations

import json
import random
import subprocess
from functools import cache

import pytest
from conftest import TRACES, cap_blocks, fixture_path
from pacer import refsim
from pacer import trace as ptrace
from synth import random_rows, write_jsonl

POLICIES = ["lru", "fifo", "lfu", "belady", "sieve", "s3fifo", "arc", "lruqd"]
FAST_LIMIT = {"traceA": 30, "coder": 10, "thinking": 8, "traceB": 80}
FAST_CAPS = [0.2, 0.6]
SLOW_CAPS = [0.05, 0.2, 0.5]
RANDOM_CAPS = [0.05, 0.15, 0.4]
FIELDS = ["hits", "accesses", "evictions", "hit_depth_sum", "access_depth_sum", "max_overflow"]


@cache
def ref_trace(name: str, limit: int | None) -> ptrace.Trace:
    return ptrace.load(fixture_path(name), limit=limit, tpot=ptrace.DEFAULT_TPOT)


def rust_sim(binary, name: str, policy: str, frac: float, limit: int | None) -> dict:
    cmd = [binary, "sim", "--trace", fixture_path(name), "--policy", policy, "--cap", repr(frac)]
    cmd += ["--window", "0,inf", "--tpot", repr(ptrace.DEFAULT_TPOT)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def assert_same(got: dict, tr: ptrace.Trace, policy: str, cap: int) -> refsim.Stats:
    assert got["cap_blocks"] == cap
    st = refsim.simulate(tr, policy, cap)
    assert {f: got[f] for f in FIELDS} == {f: getattr(st, f) for f in FIELDS}
    return st


@pytest.mark.parametrize("frac", FAST_CAPS)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", TRACES)
def test_refsim_matches_rust_on_fixture_prefix(pacer_bin, name, policy, frac):
    tr = ref_trace(name, FAST_LIMIT[name])
    got = rust_sim(pacer_bin, name, policy, frac, FAST_LIMIT[name])
    assert_same(got, tr, policy, cap_blocks(tr.n_keys, frac))


@pytest.mark.slow
@pytest.mark.parametrize("frac", SLOW_CAPS)
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("name", TRACES)
def test_refsim_matches_rust_on_full_fixture(pacer_bin, name, policy, frac):
    tr = ref_trace(name, None)
    got = rust_sim(pacer_bin, name, policy, frac, None)
    assert_same(got, tr, policy, cap_blocks(tr.n_keys, frac))


def test_refsim_matches_rust_on_random_traces(pacer_bin, tmp_path):
    paths = {}
    for seed in range(160):
        rng = random.Random(seed)
        rows = random_rows(rng, rng.randint(10, 60), max_blocks=8, zero_duration=seed % 4 != 0)
        paths[f"r{seed}"] = write_jsonl(rows, tmp_path / f"r{seed}.jsonl")
    spec = {
        "traces": [{"path": str(p), "window": [0.0, 1e18]} for p in paths.values()],
        "caps": RANDOM_CAPS,
        "policies": [{"name": p} for p in POLICIES],
        "tpot": ptrace.DEFAULT_TPOT,
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    out = tmp_path / "out.jsonl"
    subprocess.run(
        [pacer_bin, "sweep", "--spec", tmp_path / "spec.json", "--out", out, "--threads", "2"],
        check=True,
        capture_output=True,
    )
    results = {}
    for line in out.read_text().splitlines():
        r = json.loads(line)
        results[r["trace"], r["policy"], r["cap_frac"]] = r
    assert len(results) == len(paths) * len(POLICIES) * len(RANDOM_CAPS)

    disagree = zero_duration = 0
    for name, path in paths.items():
        tr = ptrace.load(path)
        zero_duration += any(r.done == r.t for r in tr.requests)
        for frac in RANDOM_CAPS:
            hits = set()
            for policy in POLICIES:
                got = results[name, policy, frac]
                try:
                    hits.add(assert_same(got, tr, policy, cap_blocks(tr.n_keys, frac)).hits)
                except AssertionError as e:
                    raise AssertionError(f"{name} {policy} cap={frac}: {e}") from None
            disagree += len(hits) > 1
    assert disagree >= 40, "random traces too easy: policies rarely disagree"
    assert zero_duration >= 40


# Regression case: request 0 completes at its own arrival time. Belady once took request 0 itself
# as the next use of block A, kept A (never used again) and evicted D. Both simulators now
# measure next use from the first arrival not processed yet: 3 hits.
ZERO_DURATION_DIVERGENCE = [
    {"chat_id": 0, "parent_chat_id": -1, "timestamp": 0.0, "input_length": 25,
     "output_length": 0, "type": "coder", "turn": 1, "hash_ids": [900, 1]},
    {"chat_id": 1, "parent_chat_id": -1, "timestamp": 0.0, "input_length": 41,
     "output_length": 8, "type": "api", "turn": 1, "hash_ids": [902, 2, 3]},
    {"chat_id": 2, "parent_chat_id": 1, "timestamp": 0.01, "input_length": 49,
     "output_length": 8, "type": "coder", "turn": 2, "hash_ids": [902, 2, 4]},
]  # fmt: skip


def test_belady_zero_duration_divergence(pacer_bin, tmp_path):
    path = write_jsonl(ZERO_DURATION_DIVERGENCE, tmp_path / "zero.jsonl")
    tr = ptrace.load(path)
    assert tr.n_keys == 4 and tr.requests[0].done == tr.requests[0].t
    got = json.loads(
        subprocess.run(
            [pacer_bin, "sim", "--trace", path, "--policy", "belady", "--cap", "0.75"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert refsim.simulate(tr, "belady", 3).hits == 3
    assert got["hits"] == 3


@pytest.mark.parametrize(
    ("n_keys", "frac", "want"),
    [(10, 0.25, 3), (10, 0.15, 2), (1000, 0.0005, 1), (3, 0.01, 1), (36035, 0.02, 721)],
)
def test_cap_blocks_rounds_half_away_from_zero(n_keys, frac, want):
    assert cap_blocks(n_keys, frac) == want

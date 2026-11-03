from __future__ import annotations

import json
import random
import re
import subprocess

import pytest
from conftest import TRACES, fixture_path
from pacer import trace as ptrace
from synth import random_rows, write_jsonl


def read_rows(path, limit=None) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for i, line in enumerate(f) if limit is None or i < limit]


@pytest.fixture(scope="module", params=TRACES)
def loaded(request):
    rows = read_rows(fixture_path(request.param))
    return rows, ptrace.build(rows)


def chain(r: ptrace.Request) -> list[int]:
    return r.input_keys + r.output_keys


def parents(tr: ptrace.Trace) -> dict[int, int]:
    par: dict[int, int] = {}
    for r in tr.requests:
        c = chain(r)
        for j, k in enumerate(c):
            p = c[j - 1] if j else -1
            assert par.setdefault(k, p) == p, f"key {k} has two parents"
    return par


def check_invariants(rows: list[dict], tr: ptrace.Trace, tpot: float = ptrace.DEFAULT_TPOT):
    edge: dict[tuple[int, int], int] = {}
    depth: dict[int, int] = {}
    for row, r in zip(rows, tr.requests, strict=True):
        assert len(r.input_keys) == min(row["input_length"] // 16, len(row["hash_ids"]))
        prev = -1
        for j, (h, k) in enumerate(zip(row["hash_ids"], r.input_keys, strict=False)):
            assert edge.setdefault((prev, h), k) == k
            assert depth.setdefault(k, j) == j
            prev = k
    n_intern = len(edge)
    assert sorted(edge.values()) == list(range(n_intern)), "input keys are not dense"

    by_chat = {row["chat_id"]: i for i, row in enumerate(rows)}
    kids: dict[int, list[int]] = {}
    for i, row in enumerate(rows):
        p = by_chat.get(row["parent_chat_id"], -1) if row["parent_chat_id"] != -1 else -1
        if 0 <= p < i:
            kids.setdefault(p, []).append(i)
    synthetic_owner: dict[int, int] = {}
    for q in tr.requests:
        first = min(kids.get(q.idx, []), key=lambda i: (rows[i]["timestamp"], i), default=-1)
        assert q.first_child_idx == first

        f_in = len(q.input_keys)
        f_tot = (q.input_len + q.output_len) // 16
        assert len(q.output_keys) == f_tot - f_in
        m = 0
        if first != -1:
            ck = tr.requests[first].input_keys
            verbatim = tr.requests[first].input_len >= q.input_len + q.output_len
            prefix = f_in == 0 or (len(ck) >= f_in and ck[f_in - 1] == q.input_keys[-1])
            if verbatim and prefix:
                m = max(0, min(f_tot, len(ck)) - f_in)
            assert q.output_keys[:m] == ck[f_in : f_in + m]
        for k in q.output_keys[m:]:
            assert k >= n_intern, "a synthetic output key collides with an input key"
            assert synthetic_owner.setdefault(k, q.idx) == q.idx

        assert q.t <= q.done <= q.t + tpot * q.output_len
        if first != -1:
            tc = tr.requests[first].t
            assert q.done <= tc and (q.done < tc or tc - q.t <= 1e-3)
            assert q.done == q.t + min(tpot * q.output_len, max(0.0, tc - q.t - 1e-3))
        else:
            assert q.done == q.t + tpot * q.output_len

    all_keys = {k for r in tr.requests for k in chain(r)}
    assert all_keys == set(range(tr.n_keys)), "keys are not dense"
    parents(tr)


def test_fixture_invariants(loaded):
    rows, tr = loaded
    check_invariants(rows, tr)


def test_fixture_reuse(loaded):
    rows, tr = loaded
    n_intern = len({k for r in tr.requests for k in r.input_keys})
    reused = sum(len(r.input_keys) for r in tr.requests) - n_intern
    continued = sum(1 for r in tr.requests if r.output_keys and r.output_keys[0] < n_intern)
    synthetic = sum(1 for r in tr.requests if r.output_keys and r.output_keys[-1] >= n_intern)
    assert reused > 0 and synthetic > 0
    if any(r.first_child_idx != -1 for r in tr.requests):
        assert continued > 0


@pytest.mark.parametrize("seed", range(300))
def test_random_invariants(seed):
    rows = random_rows(random.Random(seed))
    check_invariants(rows, ptrace.build(rows))


def row(chat, parent, t, in_len, out_len, hashes, turn=1):
    return {
        "chat_id": chat,
        "parent_chat_id": parent,
        "timestamp": t,
        "input_length": in_len,
        "output_length": out_len,
        "type": "text",
        "turn": turn,
        "hash_ids": hashes,
    }


def test_worked_example():
    tr = ptrace.build(
        [
            row(0, -1, 0.0, 40, 24, [1, 2, 3]),
            row(1, 0, 0.5, 64, 0, [1, 2, 4, 5], turn=2),
            row(2, 0, 2.0, 48, 20, [1, 6, 7], turn=2),
            row(3, -1, 2.0, 10, 40, [8]),
        ]
    )
    r0, r1, r2, r3 = tr.requests
    assert (r0.input_keys, r0.output_keys) == ([0, 1], [2, 3])
    assert (r1.input_keys, r1.output_keys) == ([0, 1, 2, 3], [])
    assert (r2.input_keys, r2.output_keys) == ([0, 4, 5], [6])
    assert (r3.input_keys, r3.output_keys) == ([], [7, 8, 9])
    assert tr.n_keys == 10
    assert r0.first_child_idx == 1 and r0.done == 0.02 * 24
    assert (r1.done, r2.done, r3.done) == (0.5, 2.0 + 0.02 * 20, 2.0 + 0.02 * 40)


def test_shortened_child_gets_fresh_output_keys():
    """A thinking model's next turn can drop the reasoning: the child's prompt is then shorter
    than prompt + output, so its blocks are not the ones the server decoded."""
    tr = ptrace.build([row(0, -1, 0.0, 32, 64, [1, 2]), row(1, 0, 5.0, 64, 0, [1, 2, 3, 4])])
    r0, r1 = tr.requests
    assert r1.input_keys == [0, 1, 2, 3]
    assert len(r0.output_keys) == 4 and min(r0.output_keys) >= 4


def test_first_child_decides_output_keys():
    tr = ptrace.build(
        [
            row(7, -1, 0.0, 32, 64, [1, 2]),
            row(8, 7, 1.0, 48, 0, [1, 9, 3], turn=2),
            row(9, 7, 3.0, 64, 0, [1, 2, 3, 4], turn=2),
        ]
    )
    r0 = tr.requests[0]
    assert r0.first_child_idx == 1
    assert len(r0.output_keys) == 4
    assert set(r0.output_keys).isdisjoint(k for r in tr.requests[1:] for k in r.input_keys)
    assert r0.done == 1.0 - 1e-3


def test_empty_input_continues_into_child():
    tr = ptrace.build([row(0, -1, 0.0, 10, 40, [1]), row(1, 0, 0.0, 64, 0, [5, 6, 7, 8], turn=2)])
    r0, r1 = tr.requests
    assert r0.input_keys == [] and r1.input_keys == [0, 1, 2, 3]
    assert r0.output_keys == [0, 1, 2]
    assert parents(tr)[2] == 1
    assert r0.done == 0.0


def rust_core():
    return pytest.importorskip("pacer._core", reason="pacer._core is not built")


def assert_same_as_rust(core, path, rows):
    tr = ptrace.build(rows)
    rt = core.load_trace(str(path))
    assert (len(rt), rt.n_keys) == (len(tr.requests), tr.n_keys)
    for r in tr.requests:
        d = rt.request(r.idx)
        assert (d["input_keys"], d["output_keys"]) == (r.input_keys, r.output_keys), r.idx
        assert (d["t"], d["done"]) == (r.t, r.done), r.idx
        assert (d["input_len"], d["output_len"], d["turn"]) == (r.input_len, r.output_len, r.turn)
    for k, p in parents(tr).items():
        assert rt.key_parent(k) == p


@pytest.mark.parametrize("name", TRACES)
def test_python_matches_rust_on_fixture(name):
    core = rust_core()
    path = fixture_path(name)
    assert_same_as_rust(core, path, read_rows(path))


def test_python_matches_rust_on_random(tmp_path):
    core = rust_core()
    for seed in range(100):
        rows = random_rows(random.Random(10_000 + seed))
        assert_same_as_rust(core, write_jsonl(rows, tmp_path / f"r{seed}.jsonl"), rows)


@pytest.mark.parametrize("name", TRACES)
def test_cli_info_matches_python(pacer_bin, name):
    path = fixture_path(name)
    tr = ptrace.load(path)
    out = subprocess.run(
        [pacer_bin, "info", "--trace", path], check=True, capture_output=True, text=True
    ).stdout
    m = re.match(
        r"(\w+): (\d+) requests, (\d+) unique blocks \((\d+) decode-completed\), "
        r"(\d+) input block accesses",
        out,
    )
    assert m, out
    assert m.group(1) == name
    got = tuple(int(x) for x in m.groups()[1:])
    want = (
        len(tr.requests),
        tr.n_keys,
        sum(len(r.output_keys) for r in tr.requests),
        sum(len(r.input_keys) for r in tr.requests),
    )
    assert got == want

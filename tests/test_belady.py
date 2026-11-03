"""Belady's MIN against a brute-force optimum under the simulator's own semantics: the same event
order, prefix hits, pinning and leaf-only eviction, but every evictable victim is tried at every
eviction (memoized on the cached set, which is all the state that matters to an offline
schedule).

TIE_COUNTEREXAMPLE is a regression case: a zero-duration request (done == t) completes after the
arrivals at t that came before it, and Belady once counted those past arrivals as future uses.
Next use is now measured from the first arrival not processed yet."""

from __future__ import annotations

import heapq
import random
import sys
from functools import cache

import pytest
from pacer import refsim
from pacer import trace as ptrace
from synth import random_rows

ONLINE = ["lru", "fifo", "lfu", "sieve", "s3fifo", "arc", "lruqd"]
INF = float("inf")


def row(chat, t, in_len, out_len, hashes, parent=-1):
    return {
        "chat_id": chat,
        "parent_chat_id": parent,
        "timestamp": t,
        "input_length": in_len,
        "output_length": out_len,
        "type": "text",
        "turn": 1 if parent == -1 else 2,
        "hash_ids": hashes,
    }


# Request 1 has no full input block; its output block is request 2's first block (C), and its
# completion at t=0.3 comes after request 0's arrival at t=0.3. Belady sees that arrival as the
# next use of block B and evicts C, which request 2 needs immediately.
TIE_COUNTEREXAMPLE = [
    row(0, 0.3, 16, 0, [1]),
    row(1, 0.3, 9, 8, [7]),
    row(2, 0.3, 17, 0, [2], parent=1),
    row(3, 1.3, 16, 0, [1]),
]


def events(tr: ptrace.Trace) -> list[tuple[int, int]]:
    """(kind, request) in refsim.simulate order: completions (0) before arrivals (1) at equal
    times, completions by (done, idx), arrivals by (t, idx)."""
    reqs = sorted(tr.requests, key=lambda r: (r.t, r.idx))
    done: list[tuple[float, int]] = []
    out = []
    i = 0
    while i < len(reqs) or done:
        if done and (i >= len(reqs) or done[0][0] <= reqs[i].t):
            out.append((0, heapq.heappop(done)[1]))
            continue
        out.append((1, reqs[i].idx))
        heapq.heappush(done, (reqs[i].done, reqs[i].idx))
        i += 1
    return out


def optimum(tr: ptrace.Trace, cap: int) -> int:
    parent: dict[int, int] = {}
    for r in tr.requests:
        c = r.input_keys + r.output_keys
        for j, k in enumerate(c):
            parent[k] = c[j - 1] if j else -1
    evs = events(tr)
    pinned = []
    live: set[int] = set()
    for kind, idx in evs:
        (live.add if kind else live.discard)(idx)
        pinned.append(frozenset(k for j in live for k in tr.requests[j].input_keys))

    @cache
    def run(e: int, cached: frozenset[int]) -> int:
        if e == len(evs):
            return 0
        kind, idx = evs[e]
        r = tr.requests[idx]
        if kind == 0:
            return evict(e, cached | frozenset(r.output_keys))
        p = 0
        while p < len(r.input_keys) and r.input_keys[p] in cached:
            p += 1
        return p + evict(e, cached | frozenset(r.input_keys))

    @cache
    def evict(e: int, cached: frozenset[int]) -> int:
        if len(cached) > cap:
            inner = {parent[k] for k in cached}
            cand = [k for k in cached if k not in pinned[e] and k not in inner]
            if cand:
                return max(evict(e, cached - {k}) for k in cand)
        return run(e + 1, cached)

    return run(0, frozenset())


def case(seed: int, zero_duration: bool = True) -> tuple[ptrace.Trace, int]:
    rng = random.Random(seed)
    n = rng.randint(2, 9)
    tr = ptrace.build(random_rows(rng, n_req=n, max_blocks=5, zero_duration=zero_duration))
    return tr, rng.randint(1, 6)


@pytest.fixture(autouse=True)
def deep_recursion():
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20_000))
    yield
    sys.setrecursionlimit(old)


def test_events_match_simulator_order():
    for seed in range(100):
        tr, _ = case(seed)
        evs = events(tr)
        assert sorted(i for k, i in evs if k == 1) == list(range(len(tr.requests)))
        assert sorted(i for k, i in evs if k == 0) == list(range(len(tr.requests)))
        st = refsim.simulate(tr, "lru", 10**9)
        assert st.evictions == 0 and st.accesses == sum(len(r.input_keys) for r in tr.requests)


@pytest.mark.parametrize("chunk", range(8))
@pytest.mark.parametrize("zero_duration", [False, True])
def test_belady_is_optimal(chunk, zero_duration):
    evicting = beaten = 0
    for seed in range(chunk * 125, (chunk + 1) * 125):
        tr, cap = case(seed, zero_duration=zero_duration)
        best = optimum(tr, cap)
        bel = refsim.simulate(tr, "belady", cap)
        assert bel.hits == best, f"seed {seed}: belady {bel.hits} < optimum {best} at cap {cap}"
        online = [refsim.simulate(tr, name, cap).hits for name in ONLINE]
        assert max(online) <= best, seed
        evicting += bel.evictions > 0
        beaten += min(online) < best
    assert evicting >= 90 and beaten >= 8, (evicting, beaten)


def test_belady_tie_counterexample():
    tr = ptrace.build(TIE_COUNTEREXAMPLE)
    assert optimum(tr, 1) == 1
    assert refsim.simulate(tr, "lru", 1).hits == 1
    assert refsim.simulate(tr, "belady", 1).hits == 1

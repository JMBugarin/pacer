"""Pure-Python reference simulator. Slow and simple on purpose: it is the executable spec that
the Rust simulator is differentially tested against. Semantics are documented in
docs/simulator.md.
"""

from __future__ import annotations

import bisect
import heapq
from collections import deque
from dataclasses import dataclass

from .trace import Trace

INF = float("inf")


@dataclass
class Stats:
    hits: int = 0
    accesses: int = 0
    hit_depth_sum: int = 0
    access_depth_sum: int = 0
    evictions: int = 0
    max_overflow: int = 0

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


# `pos` handed to policies is the index of the first arrival not processed yet; Belady treats
# uses from that index on as the future.


class Core:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.parent: dict[int, int] = {}
        self.children: dict[int, int] = {}
        self.ref: dict[int, int] = {}
        self.depth: dict[int, int] = {}
        self.seq = 0

    def cached(self, k: int) -> bool:
        return k in self.parent

    def evictable(self, k: int) -> bool:
        return k in self.parent and self.ref[k] == 0 and self.children[k] == 0

    def size(self) -> int:
        return len(self.parent)

    def add(self, k: int, parent: int, depth: int) -> None:
        self.parent[k] = parent
        self.children[k] = 0
        self.ref[k] = 0
        self.depth[k] = depth
        if parent != -1:
            self.children[parent] += 1

    def remove(self, k: int) -> None:
        p = self.parent.pop(k)
        del self.children[k], self.ref[k], self.depth[k]
        if p != -1:
            self.children[p] -= 1

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq


class PriorityPolicy:
    """Evicts the evictable block with the smallest key(k). O(n) scan; tiny inputs only."""

    def __init__(self, core: Core):
        self.core = core

    def on_insert(self, k, now, pos):
        pass

    def on_access(self, k, now, pos):
        pass

    def on_release(self, k, now, pos):
        pass

    def on_evict(self, k):
        pass

    def key(self, k, pos):
        raise NotImplementedError

    def victim(self, pos):
        best = None
        best_key = None
        for k in self.core.parent:
            if self.core.evictable(k):
                kk = self.key(k, pos)
                if best is None or kk < best_key:
                    best, best_key = k, kk
        return best


class LRU(PriorityPolicy):
    def __init__(self, core):
        super().__init__(core)
        self.last = {}

    def on_insert(self, k, now, pos):
        self.last[k] = self.core.next_seq()

    on_access = on_insert
    on_release = on_insert

    def on_evict(self, k):
        del self.last[k]

    def key(self, k, pos):
        return self.last[k]


class FIFO(PriorityPolicy):
    def __init__(self, core):
        super().__init__(core)
        self.ins = {}

    def on_insert(self, k, now, pos):
        self.ins[k] = self.core.next_seq()

    def on_evict(self, k):
        del self.ins[k]

    def key(self, k, pos):
        return self.ins[k]


class LFU(PriorityPolicy):
    def __init__(self, core):
        super().__init__(core)
        self.count = {}
        self.last = {}

    def on_insert(self, k, now, pos):
        self.count[k] = 1
        self.last[k] = self.core.next_seq()

    def on_access(self, k, now, pos):
        self.count[k] += 1
        self.last[k] = self.core.next_seq()

    def on_evict(self, k):
        del self.count[k], self.last[k]

    def key(self, k, pos):
        return (self.count[k], self.last[k])


class Belady(PriorityPolicy):
    """Evict the block whose next use is farthest; ties go to the deeper block, then larger key."""

    def __init__(self, core, uses: dict[int, list[int]]):
        super().__init__(core)
        self.uses = uses

    def next_use(self, k, pos):
        lst = self.uses.get(k)
        if not lst:
            return INF
        i = bisect.bisect_left(lst, pos)
        return lst[i] if i < len(lst) else INF

    def key(self, k, pos):
        return (-self.next_use(k, pos), -self.core.depth[k], -k)


class SIEVE:
    def __init__(self, core: Core):
        self.core = core
        self.prev: dict[int, int | None] = {}
        self.next: dict[int, int | None] = {}
        self.head = None
        self.tail = None
        self.visited: dict[int, bool] = {}
        self.hand = None

    def on_insert(self, k, now, pos):
        self.next[k] = self.head
        self.prev[k] = None
        if self.head is not None:
            self.prev[self.head] = k
        self.head = k
        if self.tail is None:
            self.tail = k
        self.visited[k] = False
        self.core.next_seq()

    def on_access(self, k, now, pos):
        self.visited[k] = True
        self.core.next_seq()

    def on_release(self, k, now, pos):
        self.core.next_seq()

    def on_evict(self, k):
        p, n = self.prev.pop(k), self.next.pop(k)
        if p is not None:
            self.next[p] = n
        else:
            self.head = n
        if n is not None:
            self.prev[n] = p
        else:
            self.tail = p
        del self.visited[k]

    def victim(self, pos):
        o = self.hand if self.hand is not None else self.tail
        steps = 0
        limit = 2 * len(self.visited) + 2
        while o is not None and steps <= limit:
            if self.visited[o]:
                self.visited[o] = False
            elif self.core.evictable(o):
                self.hand = self.prev[o]
                return o
            o = self.prev[o] if self.prev[o] is not None else self.tail
            steps += 1
        return None


class S3FIFO:
    def __init__(self, core: Core, small_ratio: float = 0.1, threshold: int = 2):
        self.core = core
        self.small_cap = max(1, int(core.capacity * small_ratio))
        self.main_cap = core.capacity - self.small_cap
        self.threshold = threshold
        self.S: deque[int] = deque()
        self.M: deque[int] = deque()
        self.where: dict[int, str] = {}
        self.freq: dict[int, int] = {}
        self.ghost: dict[int, int] = {}
        self.ghost_q: deque[tuple[int, int]] = deque()
        self.ghost_ins = 0

    def _ghost_add(self, k):
        self.ghost_ins += 1
        self.ghost[k] = self.ghost_ins
        self.ghost_q.append((k, self.ghost_ins))
        while self.ghost_q and self.ghost_q[0][1] <= self.ghost_ins - self.main_cap:
            gk, gt = self.ghost_q.popleft()
            if self.ghost.get(gk) == gt:
                del self.ghost[gk]

    def on_insert(self, k, now, pos):
        self.freq[k] = 0
        if k in self.ghost:
            del self.ghost[k]
            self.M.appendleft(k)
            self.where[k] = "M"
        else:
            self.S.appendleft(k)
            self.where[k] = "S"
        self.core.next_seq()

    def on_access(self, k, now, pos):
        self.freq[k] = min(self.freq[k] + 1, 3)
        self.core.next_seq()

    def on_release(self, k, now, pos):
        self.core.next_seq()

    def on_evict(self, k):
        del self.where[k], self.freq[k]

    def victim(self, pos):
        if len(self.S) >= self.small_cap or not self.M:
            v = self._evict_s()
            return v if v is not None else self._evict_m()
        v = self._evict_m()
        return v if v is not None else self._evict_s()

    def _evict_s(self):
        skips = 0
        while self.S and skips <= len(self.S):
            t = self.S.pop()
            if self.freq[t] >= self.threshold:
                self.M.appendleft(t)
                self.where[t] = "M"
                if len(self.M) > self.main_cap:
                    v = self._evict_m()
                    if v is not None:
                        return v
            elif not self.core.evictable(t):
                self.S.appendleft(t)
                skips += 1
            else:
                self._ghost_add(t)
                return t
        return None

    def _evict_m(self):
        steps = 0
        limit = 4 * len(self.M) + 4
        while self.M and steps <= limit:
            t = self.M.pop()
            if self.freq[t] > 0:
                self.freq[t] -= 1
                self.M.appendleft(t)
            elif not self.core.evictable(t):
                self.M.appendleft(t)
            else:
                return t
            steps += 1
        return None


def build_uses(trace: Trace) -> dict[int, list[int]]:
    """Request indices that use each key, in arrival order."""
    uses: dict[int, list[int]] = {}
    for r in sorted(trace.requests, key=lambda r: (r.t, r.idx)):
        for k in r.input_keys:
            uses.setdefault(k, []).append(r.idx)
    return uses


def make_policy(name: str, core: Core, trace: Trace):
    if name == "lru":
        return LRU(core)
    if name == "fifo":
        return FIFO(core)
    if name == "lfu":
        return LFU(core)
    if name == "belady":
        return Belady(core, build_uses(trace))
    if name == "sieve":
        return SIEVE(core)
    if name == "s3fifo":
        return S3FIFO(core)
    if name == "arc":
        return ARC(core)
    if name == "lruqd":
        return LRUQD(core)
    raise ValueError(name)


def simulate(
    trace: Trace, policy: str, capacity: int, window: tuple[float, float] = (0.0, INF)
) -> Stats:
    core = Core(capacity)
    pol = make_policy(policy, core, trace)
    st = Stats()
    reqs = sorted(trace.requests, key=lambda r: (r.t, r.idx))
    done: list[tuple[float, int]] = []
    i = 0

    def evict_to_capacity(pos):
        while core.size() > capacity:
            v = pol.victim(pos)
            if v is None or not core.evictable(v):
                st.max_overflow = max(st.max_overflow, core.size() - capacity)
                return
            pol.on_evict(v)
            core.remove(v)
            st.evictions += 1

    while i < len(reqs) or done:
        if done and (i >= len(reqs) or done[0][0] <= reqs[i].t):
            tdone, idx = heapq.heappop(done)
            r = trace.requests[idx]
            pos = i
            chain = r.input_keys + r.output_keys
            f_in = len(r.input_keys)
            for j, k in enumerate(r.output_keys):
                if not core.cached(k):
                    par = chain[f_in + j - 1] if f_in + j > 0 else -1
                    core.add(k, par, f_in + j)
                    pol.on_insert(k, tdone, pos)
            for k in r.input_keys:
                core.ref[k] -= 1
            for k in reversed(chain):
                pol.on_release(k, tdone, pos)
            evict_to_capacity(pos)
            continue

        r = reqs[i]
        i += 1
        pos = i
        keys = r.input_keys
        p = 0
        while p < len(keys) and core.cached(keys[p]):
            p += 1
        if window[0] <= r.t < window[1]:
            st.hits += p
            st.accesses += len(keys)
            st.hit_depth_sum += p * (p - 1) // 2
            st.access_depth_sum += len(keys) * (len(keys) - 1) // 2
        for j in range(p, len(keys)):
            core.add(keys[j], keys[j - 1] if j > 0 else -1, j)
        for j in range(len(keys) - 1, -1, -1):
            if j >= p:
                pol.on_insert(keys[j], r.t, pos)
            else:
                pol.on_access(keys[j], r.t, pos)
        for k in keys:
            core.ref[k] += 1
        evict_to_capacity(pos)
        heapq.heappush(done, (r.done, r.idx))

    return st


class ARC:
    """Mirrors crates/pacer/src/policy/arc.rs: REPLACE takes the least recent evictable block of
    the preferred list, else of the other; ghosts are trimmed to |T1| + |B1| <= c and a total of
    at most 2c. Each list is a dict in stamp order (attach always appends)."""

    def __init__(self, core: Core):
        self.core = core
        self.c = float(core.capacity)
        self.p = 0.0
        self.lists: dict[str, dict[int, None]] = {n: {} for n in ("T1", "T2", "B1", "B2")}
        self.loc: dict[int, str] = {}

    def _detach(self, k):
        lst = self.loc.pop(k, None)
        if lst is not None:
            del self.lists[lst][k]

    def _attach(self, k, lst):
        self.loc[k] = lst
        self.lists[lst][k] = None

    def _trim(self):
        t1, t2, b1, b2 = (self.lists[n] for n in ("T1", "T2", "B1", "B2"))
        c = int(self.c)
        while len(t1) + len(b1) > c and b1:
            self._detach(next(iter(b1)))
        while len(t1) + len(t2) + len(b1) + len(b2) > 2 * c and b2:
            self._detach(next(iter(b2)))

    def on_insert(self, k, now, pos):
        b1, b2 = self.lists["B1"], self.lists["B2"]
        where = self.loc.get(k)
        if where == "B1":
            self.p = min(self.p + max(len(b2) / len(b1), 1.0), self.c)
            self._detach(k)
            self._attach(k, "T2")
        elif where == "B2":
            self.p = max(self.p - max(len(b1) / len(b2), 1.0), 0.0)
            self._detach(k)
            self._attach(k, "T2")
        else:
            self._detach(k)
            self._attach(k, "T1")
        self._trim()

    def on_access(self, k, now, pos):
        if self.loc.get(k) in ("T1", "T2"):
            self._detach(k)
            self._attach(k, "T2")

    def on_release(self, k, now, pos):
        pass

    def on_evict(self, k):
        where = self.loc.get(k)
        self._detach(k)
        if where == "T1":
            self._attach(k, "B1")
        elif where == "T2":
            self._attach(k, "B2")
        self._trim()

    def _lru_evictable(self, lst):
        return next((k for k in self.lists[lst] if self.core.evictable(k)), None)

    def victim(self, pos):
        t1 = self.lists["T1"]
        first, second = ("T1", "T2") if t1 and len(t1) > self.p else ("T2", "T1")
        v = self._lru_evictable(first)
        return v if v is not None else self._lru_evictable(second)


class LRUQD:
    """Mirrors crates/pacer/src/policy/qd.rs: new blocks enter a probation FIFO, a hit promotes
    a block to the LRU main queue, and probation is evicted first while it holds more than
    `small` of the cache. Dicts are kept in sequence order, like the Rust lazy heaps."""

    def __init__(self, core: Core, small: float = 0.1):
        self.core = core
        self.small_cap = max(1, int(core.capacity * small))
        self.prob: dict[int, None] = {}
        self.main: dict[int, None] = {}

    def on_insert(self, k, now, pos):
        self.main.pop(k, None)
        self.prob[k] = None

    def on_access(self, k, now, pos):
        self.prob.pop(k, None)
        self.main.pop(k, None)
        self.main[k] = None

    def on_release(self, k, now, pos):
        if k in self.main:
            del self.main[k]
            self.main[k] = None

    def on_evict(self, k):
        self.prob.pop(k, None)
        self.main.pop(k, None)

    def _first_evictable(self, q):
        return next((k for k in q if self.core.evictable(k)), None)

    def victim(self, pos):
        first, second = (
            (self.prob, self.main) if len(self.prob) > self.small_cap else (self.main, self.prob)
        )
        v = self._first_evictable(first)
        return v if v is not None else self._first_evictable(second)

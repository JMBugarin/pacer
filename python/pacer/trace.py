"""Bailian trace loading: prefix re-keying, decode-completed output blocks, completion times.

This module and crates/pacer/src/trace.rs implement the same preprocessing; the differential
tests compare simulator hit counts, so any disagreement between the two shows up there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

BLOCK = 16
DEFAULT_TPOT = 0.02
TYPES = ["text", "image", "search", "file", "api", "coder", "thinking"]


@dataclass
class Request:
    idx: int
    chat_id: int
    parent: int
    t: float
    input_len: int
    output_len: int
    rtype: int
    turn: int
    input_keys: list[int]
    output_keys: list[int] = field(default_factory=list)
    done: float = 0.0
    parent_idx: int = -1
    first_child_idx: int = -1


@dataclass
class Trace:
    requests: list[Request]
    n_keys: int

    def unique_blocks(self) -> int:
        return self.n_keys


def load(path: str | Path, limit: int | None = None, tpot: float = DEFAULT_TPOT) -> Trace:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return build(rows, tpot)


def build(rows: list[dict], tpot: float = DEFAULT_TPOT) -> Trace:
    intern: dict[tuple[int, int], int] = {}
    chains: list[list[int]] = []
    for r in rows:
        prev = -1
        chain = []
        for h in r["hash_ids"][: r["input_length"] // BLOCK]:
            k = intern.setdefault((prev, h), len(intern))
            chain.append(k)
            prev = k
        chains.append(chain)

    reqs: list[Request] = []
    by_chat: dict[int, int] = {}
    for i, r in enumerate(rows):
        f_in = len(chains[i])
        rtype = TYPES.index(r["type"]) if r["type"] in TYPES else len(TYPES)
        reqs.append(
            Request(
                idx=i,
                chat_id=r["chat_id"],
                parent=r["parent_chat_id"],
                t=float(r["timestamp"]),
                input_len=r["input_length"],
                output_len=r["output_length"],
                rtype=rtype,
                turn=r["turn"],
                input_keys=chains[i][:f_in],
            )
        )
        by_chat[r["chat_id"]] = i

    for q in reqs:
        p = by_chat.get(q.parent, -1) if q.parent != -1 else -1
        if p != -1 and p < q.idx:
            q.parent_idx = p
            par = reqs[p]
            if par.first_child_idx == -1 or q.t < reqs[par.first_child_idx].t:
                par.first_child_idx = q.idx

    next_key = len(intern)
    for q in reqs:
        f_in = len(q.input_keys)
        f_tot = (q.input_len + q.output_len) // BLOCK
        out: list[int] = []
        c = q.first_child_idx
        verbatim = c != -1 and reqs[c].input_len >= q.input_len + q.output_len
        if verbatim and f_tot > f_in:
            child = reqs[c]
            cf = len(child.input_keys)
            matches = f_in == 0 or (cf >= f_in and child.input_keys[f_in - 1] == q.input_keys[-1])
            if matches:
                out = child.input_keys[f_in : min(f_tot, cf)]
        while f_in + len(out) < f_tot:
            out.append(next_key)
            next_key += 1
        q.output_keys = out

        dur = tpot * q.output_len
        if c != -1:
            dur = min(dur, max(0.0, reqs[c].t - q.t - 1e-3))
        q.done = q.t + dur

    return Trace(reqs, next_key)

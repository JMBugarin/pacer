"""Tiny random Bailian-style traces: shared system prefixes, multi-turn sessions whose children
extend (or occasionally diverge from) the parent's chain, partial last blocks, equal timestamps
and zero-length outputs, so every preprocessing and event-ordering branch is reachable.

With zero_duration=False every request completes strictly after it arrives (no empty outputs,
no child within 2 ms of its parent), which avoids the known Belady next-use gap described in
tests/test_belady.py."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

SYSTEM_PREFIXES = [[], [900], [900, 901], [902]]


def random_rows(
    rng: random.Random, n_req: int | None = None, max_blocks: int = 4, zero_duration: bool = True
) -> list[dict]:
    n = n_req if n_req is not None else rng.randint(2, 7)
    fresh = itertools.count(1)
    rows: list[dict] = []
    t = 0.0
    for i in range(n):
        t = round(t + rng.choice([0.0, 0.0, 0.01, 0.1, 0.3, 1.0]), 3)
        parents = rows if zero_duration else [r for r in rows if r["timestamp"] <= t - 0.002]
        if parents and rng.random() < 0.55:
            par = rng.choice(parents)
            base = par["hash_ids"][: par["input_length"] // 16]
            if base and rng.random() < 0.2:
                base = [*base[:-1], next(fresh)]
            hashes = base + [next(fresh) for _ in range(rng.randint(0, 3))]
            parent, turn = par["chat_id"], par["turn"] + 1
        else:
            hashes = rng.choice(SYSTEM_PREFIXES) + [next(fresh) for _ in range(rng.randint(0, 2))]
            parent, turn = -1, 1
        hashes = hashes[:max_blocks]
        in_len = 16 * len(hashes) - rng.choice([0, 0, 0, 7]) if hashes else rng.choice([0, 9])
        rows.append(
            {
                "chat_id": i,
                "parent_chat_id": parent,
                "timestamp": t,
                "input_length": in_len,
                "output_length": rng.choice([0, 0, 8, 16, 24, 40] if zero_duration else [1, 8, 40]),
                "type": rng.choice(["text", "image", "api", "coder", "thinking", "other"]),
                "turn": turn,
                "hash_ids": hashes,
            }
        )
    return rows


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path

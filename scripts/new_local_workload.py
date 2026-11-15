"""Frozen synthetic workload generator; no real user requests."""

import json
import random
from pathlib import Path

rng = random.Random(20261115)
rows = []
for i in range(120):
    session = rng.randrange(12)
    hashes = list(range(1000 + 100 * session, 1016 + 100 * session))
    hashes += list(range(100000 + 10 * i, 100000 + 10 * i + rng.randrange(1, 9)))
    rows.append(
        dict(
            chat_id=i,
            parent_chat_id=-1,
            timestamp=float(i),
            input_length=16 * len(hashes),
            output_length=16,
            type="text",
            turn=1,
            hash_ids=hashes,
        )
    )
p = Path("docs/evidence/serving-v3/new-local.jsonl")
if p.exists():
    raise SystemExit("Refusing to overwrite a frozen workload")
p.write_text("".join(json.dumps(row) + "\n" for row in rows))

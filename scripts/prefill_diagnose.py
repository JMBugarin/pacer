"""Development-only chunk-shape diagnostic; never an independent final evaluation."""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from pacer.live.engine import Engine


def main():
    engine = Engine()
    records = []
    started = datetime.now(UTC).isoformat()
    for case in range(64):
        keys = list(range(100000 + case * 40, 100000 + case * 40 + 32))
        split = 1 + case % 30
        pool = engine.new_pool(split)
        pre = engine.run(None, [], keys[:split], logits=False)
        pool.write(keys[:split], pre.k_new, pre.v_new)
        got = engine.run(pool, keys[:split], keys[split:])
        cache = make_prompt_cache(engine.model)
        engine._forward(mx.array(engine.tokens(keys[:split])[None]), cache, False)
        stock = engine._forward(mx.array(engine.tokens(keys[split:])[None]), cache, True)
        full = engine.full_logits(keys)
        mx.eval(stock, full)
        records.append(
            dict(
                case=case,
                blocks=32,
                split=split,
                pooled_stock_bitwise=mx.array_equal(got.logits, stock).item(),
                pooled_stock_max=float(mx.max(mx.abs(got.logits - stock)).item()),
                stock_full_max=float(mx.max(mx.abs(stock - full)).item()),
                stock_token=int(mx.argmax(stock).item()),
                full_token=int(mx.argmax(full).item()),
            )
        )
    out = dict(
        kind="synthetic development fixtures; 64 fixed cases",
        started=started,
        ended=datetime.now(UTC).isoformat(),
        platform=platform.platform(),
        mlx=mx.__version__,
        code_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=records,
    )
    Path("docs/evidence/serving-v3/chunk-diagnostic.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                n=len(records),
                pooled_stock_equal=sum(x["pooled_stock_bitwise"] for x in records),
                stock_full_token_equal=sum(x["stock_token"] == x["full_token"] for x in records),
            )
        )
    )


if __name__ == "__main__":
    main()

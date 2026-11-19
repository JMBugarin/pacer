"""Live MLX checks: the KV block pool mirrors the Rust controller, and cached-prefix prefill
matches a full prefill. Skips without mlx, the pacer._core bindings, the trace or the model."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
core = pytest.importorskip("pacer._core")

from pacer.live import replay as rp  # noqa: E402
from pacer.live.engine import DEFAULT_MODEL, Engine, block_tokens  # noqa: E402
from pacer.live.pool import BLOCK, BlockPool, PoolFull  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "data/raw/qwen_traceA_blksz_16.jsonl"


@pytest.fixture(scope="module")
def engine():
    if not mx.metal.is_available():
        pytest.skip("no Metal device")
    from mlx_lm.utils import hf_repo_to_path

    try:
        path = hf_repo_to_path(DEFAULT_MODEL)
    except Exception:
        pytest.skip(f"{DEFAULT_MODEL} is not downloaded")
    return Engine(str(path))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The first 40 short-prompt requests of trace A: a small pool suffices to never overflow."""
    if not TRACE.exists():
        pytest.skip("trace A is not downloaded")
    rows = []
    with open(TRACE) as f:
        for line in f:
            if json.loads(line)["input_length"] <= 512:
                rows.append(line)
                if len(rows) == 40:
                    break
    path = tmp_path_factory.mktemp("live") / "tiny.jsonl"
    path.write_text("".join(rows))
    tr = core.load_trace(str(path))
    reqs = rp.requests(tr)
    return tr, reqs, rp.min_cap_blocks(reqs) + 40


def test_block_tokens_deterministic():
    a = block_tokens([5, 9, 5], vocab=1000)
    assert a.shape == (3 * BLOCK,)
    assert (a[:BLOCK] == a[2 * BLOCK :]).all()
    assert (a[:BLOCK] != a[BLOCK : 2 * BLOCK]).any()
    assert a.min() >= 0 and a.max() < 1000
    assert (block_tokens([5], 1000, seed=1) != a[:BLOCK]).any()


def test_pool_write_gather_free():
    pool = BlockPool(2, 2, 8, 4)
    data = [mx.random.normal((3, 2, BLOCK, 8)).astype(mx.bfloat16) for _ in range(2)]
    pool.write([10, 11, 12], data, data)
    ks, _ = pool.gather([12, 10], size=3)
    assert ks[1].shape == (1, 2, 3 * BLOCK, 8)
    want = data[1][mx.array([2, 0])].transpose(1, 0, 2, 3).reshape(1, 2, 2 * BLOCK, 8)
    assert mx.array_equal(ks[1][:, :, : 2 * BLOCK], want).item()
    with pytest.raises(KeyError):
        pool.write([11], [d[:1] for d in data], [d[:1] for d in data])
    with pytest.raises(PoolFull):
        pool.write([1, 2], [d[:2] for d in data], [d[:2] for d in data])
    pool.free([10, 11])
    pool.write([1, 2], [d[:2] for d in data], [d[:2] for d in data])
    assert len(pool) == 3 and 10 not in pool and pool.peak == 3


@pytest.mark.parametrize("policy", ["lru", "s3fifo"])
def test_replay_mirrors_controller(engine, tiny, policy):
    tr, reqs, cap = tiny
    spec = rp.policy_spec(policy)
    dry = rp.dry_run(tr, spec, cap)
    r = rp.replay(tr, reqs, spec, cap, engine, engine.new_pool(cap))
    assert r["stats"]["hits"] == dry["hits"] > 0
    assert r["stats"]["evictions"] == dry["evictions"] > 0
    assert r["prefill_tokens"] == dry["prefill_tokens"]
    assert r["completion"]["tokens"] == dry["output_tokens"]
    assert r["pool_peak"] <= cap
    assert all(t is not None and t > 0 for t in r["ttft_s"])


def test_pool_refuses_to_exceed_capacity(engine, tiny):
    tr, reqs, cap = tiny
    small = cap - 60
    with pytest.raises((PoolFull, AssertionError)):
        rp.replay(tr, reqs, rp.policy_spec("lru"), small, engine, engine.new_pool(small))


def test_cached_prefill_matches_full(engine, tiny):
    tr, reqs, cap = tiny
    spec = rp.policy_spec("lru")
    picked = rp.dry_run(tr, spec, cap)["hit_arrivals"][:8]
    r = rp.replay(
        tr,
        reqs,
        spec,
        cap,
        engine,
        engine.new_pool(cap),
        check=set(picked),
        stop_after=picked[-1],
    )
    s = rp.check_summary(r["checks"])
    assert s["n"] == len(picked)
    assert all(c["argmax_equal"] or c["ref_margin"] <= 2 * c["max_abs"] for c in r["checks"])
    assert s["max_abs_diff"]["max"] < 1.0
    assert s["control_max_abs_p50"] > 2 * s["max_abs_diff"]["p50"]


def test_full_hit_recomputes_last_token(engine):
    keys = list(range(1 << 30, (1 << 30) + 6))
    pool = engine.new_pool(len(keys))
    full = engine.run(None, [], keys)
    pool.write(keys, full.k_new, full.v_new)
    redo = engine.run(pool, keys, [])
    part = engine.run(pool, keys[:4], keys[4:])
    assert (redo.new_tokens, redo.forward_tokens) == (0, 1)
    assert part.new_tokens == 2 * BLOCK
    for p in (redo, part):
        assert p.token == full.token
        assert mx.max(mx.abs(p.logits - full.logits)).item() < 1.0


def test_pool_path_is_bitwise_stock_mlx_lm(engine):
    """Same kernel shapes as stock mlx-lm (prefix forward, then suffix forward on its cache),
    so gather, offsets and the in-place cache writes must reproduce it exactly."""
    from mlx_lm.models.cache import make_prompt_cache

    keys = list(range(1 << 31, (1 << 31) + 7))
    split = 5
    pool = engine.new_pool(split)
    pre = engine.run(None, [], keys[:split], logits=False)
    pool.write(keys[:split], pre.k_new, pre.v_new)
    got = engine.run(pool, keys[:split], keys[split:])

    cache = make_prompt_cache(engine.model)
    engine.model.model(mx.array(engine.tokens(keys[:split])[None]), cache)
    h = engine.model.model(mx.array(engine.tokens(keys[split:])[None]), cache)
    want = engine._head(h[:, -1, :].astype(mx.float32))[0]
    assert mx.array_equal(got.logits, want).item()
    stock_k = cache[3].keys[0, :, BLOCK * split : BLOCK * len(keys), :]
    mine_k = got.k_new[3].transpose(1, 0, 2, 3).reshape(stock_k.shape)
    assert mx.array_equal(mine_k, stock_k).item()


@pytest.fixture
def output(engine):
    """Sets the engine's output mode for one test and restores it after."""
    prev = engine.output

    def set_mode(mode: str):
        engine.output = mode
        return engine

    yield set_mode
    engine.output = prev


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """40 synthetic requests shaped like the serving-protocol workload: 16 shared session
    blocks plus 1-8 unique blocks, so every prompt is 17-24 blocks long."""
    import random

    rng = random.Random(7)
    rows = []
    for i in range(40):
        s = rng.randrange(4)
        keys = list(range(5000 + 100 * s, 5016 + 100 * s))
        keys += list(range(800000 + 10 * i, 800000 + 10 * i + rng.randrange(1, 9)))
        rows.append(
            dict(
                chat_id=i,
                parent_chat_id=-1,
                timestamp=float(i),
                input_length=BLOCK * len(keys),
                output_length=16,
                type="text",
                turn=1,
                hash_ids=keys,
            )
        )
    path = tmp_path_factory.mktemp("live") / "synth.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tr = core.load_trace(str(path), tpot=0.002)
    reqs = rp.requests(tr)
    return tr, reqs, max(rp.cap_blocks(tr, 0.3), rp.min_cap_blocks(reqs))


def test_verified_output_uses_full_prefill(engine, output):
    keys = list(range(90000, 90032))
    pool = engine.new_pool(16)
    pre = engine.run(None, [], keys[:16], logits=False)
    pool.write(keys[:16], pre.k_new, pre.v_new)
    output("verified")
    got = engine.run(pool, keys[:16], keys[16:])
    ref = engine.full_logits(keys)
    assert mx.array_equal(got.logits, ref).item()
    assert got.token == mx.argmax(ref).item()
    assert got.cached_logits is None and got.gather_s == 0.0
    assert (got.new_tokens, got.forward_tokens) == (16 * BLOCK, 32 * BLOCK)
    full = engine.run(None, [], keys, logits=False)
    assert mx.array_equal(got.k_new[2], full.k_new[2][16:]).item()


def test_verified_check_compares_the_cached_path(engine, output, synth):
    """Verified mode serves a full prefill, so its check must run the cached forward itself;
    comparing the served logits would compare the reference with itself."""
    tr, reqs, cap = synth
    output("verified")
    engine.reserve(max(len(r["input_keys"]) + len(r["output_keys"]) for r in reqs))
    spec = rp.policy_spec("lru")
    picked = rp.dry_run(tr, spec, cap)["hit_arrivals"]
    r = rp.replay(tr, reqs, spec, cap, engine, engine.new_pool(cap), check=set(picked))
    assert len(r["checks"]) == len(picked) > 0
    assert not any(c["served"] for c in r["checks"])
    assert max(c["max_abs"] for c in r["checks"]) > 0

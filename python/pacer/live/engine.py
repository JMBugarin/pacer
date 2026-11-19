"""MLX prefill engine over the block pool.

Prompts are synthetic: every block key owns 16 token ids drawn from a hash of the key, so a key
always denotes the same tokens (and, since keys are prefix-chain positions, the same KV). A
forward starts from the gathered KV of a cached block prefix and runs only the new blocks; the
new blocks' KV is then sliced out of the mlx-lm caches for the pool.

Output modes:
- "cached": logits of the cached-prefix forward (experimental, not token-equivalent).
- "verified": logits of a full prefill on every hit; the cached forward is skipped.
- "exact": the cached-prefix forward with every quantized matmul run at a row count where MLX
  picks the same kernel as the full prefill, so its logits equal the full prefill bit for bit.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from .pool import BLOCK, BlockPool

DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_CHUNK = 2048
OUTPUTS = ("cached", "verified", "exact")
MIN_CALIBRATED_ROWS = 512

_U = np.uint64


def _mix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + _U(0x9E3779B97F4A7C15)
        x = (x ^ (x >> _U(30))) * _U(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> _U(27))) * _U(0x94D049BB133111EB)
        return x ^ (x >> _U(31))


def block_tokens(keys: Sequence[int], vocab: int, seed: int = 0) -> np.ndarray:
    """The BLOCK token ids of each key, concatenated; ids are in [0, vocab)."""
    k = np.asarray(keys, dtype=_U).reshape(-1, 1)
    x = k * _U(BLOCK) + np.arange(BLOCK, dtype=_U) + (_U(seed) << _U(40))
    return (_mix(x) % _U(vocab)).astype(np.int32).reshape(-1)


def _token_bound(tok, vocab_size: int) -> int:
    ids = set(getattr(tok, "all_special_ids", None) or [])
    ids |= set(getattr(tok, "added_tokens_decoder", None) or {})
    return min(min(ids), vocab_size) if ids else vocab_size


def _cache(k: mx.array, v: mx.array, offset: int) -> KVCache:
    c = KVCache()
    c.keys, c.values, c.offset = k, v, offset
    return c


class RowStableLinear(nn.QuantizedLinear):
    """QuantizedLinear that, while `active`, zero-pads its input up to `min_rows` rows.

    MLX chooses the quantized matmul kernel from the row count; below a shape-dependent
    threshold it splits the reduction differently, so a row's result depends on how many rows
    share the call. Above the threshold every row is computed the same way regardless of the
    row count or the row's position, which is what makes a short suffix forward reproduce the
    rows of a long full prefill exactly.
    """

    active = False
    min_rows = 1

    def __call__(self, x: mx.array) -> mx.array:
        m = x.shape[-2]
        if not RowStableLinear.active or m >= self.min_rows:
            return super().__call__(x)
        pad = mx.zeros((*x.shape[:-2], self.min_rows - m, x.shape[-1]), x.dtype)
        return super().__call__(mx.concatenate([x, pad], axis=-2))[..., :m, :]


def _shape(m: nn.QuantizedLinear) -> tuple[int, int, int, int, str]:
    k = m.weight.shape[1] * 32 // m.bits
    return k, m.weight.shape[0], m.bits, m.group_size, str(getattr(m, "mode", "affine"))


def stable_rows(model: nn.Module, max_rows: int, seed: int = 0) -> dict[tuple, int]:
    """Per quantized matmul shape, the smallest row count from which every row matches the
    same row of a `max_rows` call bit for bit, checked at every count below `max_rows`."""
    mods: dict[tuple, nn.QuantizedLinear] = {}
    for _, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            mods.setdefault(_shape(m), m)
    key = mx.random.key(seed)
    out = {}
    for shape, m in mods.items():
        x = (mx.random.normal((1, max_rows, shape[0]), key=key) * 0.5).astype(mx.bfloat16)
        ref = nn.QuantizedLinear.__call__(m, x)
        mx.eval(ref)
        last = 0
        for rows in range(1, max_rows):
            y = nn.QuantizedLinear.__call__(m, x[:, :rows])
            if not mx.array_equal(y, ref[:, :rows]).item():
                last = rows
        out[shape] = last + 1
    return out


@dataclass
class Prefill:
    token: int | None
    seconds: float
    gather_s: float
    new_tokens: int
    forward_tokens: int
    logits: mx.array | None
    k_new: list[mx.array]
    v_new: list[mx.array]
    cached_logits: mx.array | None = None


class Engine:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        chunk: int = DEFAULT_CHUNK,
        seed: int = 0,
        output: str = "cached",
    ):
        self.model, tok = load(model)
        args = self.model.args
        self.name = model
        self.n_layers = args.num_hidden_layers
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.vocab = _token_bound(tok, args.vocab_size)
        self.chunk = chunk
        self.seed = seed
        probe = [KVCache() for _ in range(self.n_layers)]
        self.model.model(mx.array([[0]]), probe)
        self.dtype = probe[0].keys.dtype
        self._scratch = self.new_pool(1)
        self.workspace = 0
        self._stable: list[RowStableLinear] = []
        for _, m in self.model.model.named_modules():
            if type(m) is nn.QuantizedLinear:
                m.__class__ = RowStableLinear
                self._stable.append(m)
        self.stable_rows: dict[tuple, int] = {}
        self.calibrated_rows = 0
        self._output = "cached"
        self.output = output

    @property
    def output(self) -> str:
        return self._output

    @output.setter
    def output(self, mode: str) -> None:
        if mode not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, not {mode!r}")
        self._output = mode
        if mode == "exact":
            self._calibrate()

    def _calibrate(self) -> None:
        rows = max(MIN_CALIBRATED_ROWS, BLOCK * self.workspace)
        if rows <= self.calibrated_rows:
            return
        self.stable_rows = stable_rows(self.model.model, rows)
        for m in self._stable:
            m.min_rows = self.stable_rows[_shape(m)]
        self.calibrated_rows = rows

    @property
    def exact_min_rows(self) -> int:
        return max(self.stable_rows.values(), default=1)

    def exact_supported(self, n_blocks: int) -> bool:
        """Whether a prompt's full prefill runs every matmul in the row-stable regime, in one
        chunk, inside the calibrated range; only then can a cached forward reproduce it."""
        rows = BLOCK * n_blocks
        return self.exact_min_rows <= rows <= min(self.chunk, self.calibrated_rows)

    def reserve(self, n_blocks: int) -> None:
        """Sizes the per-forward KV workspace (in blocks) to fit every prompt to come. Prompts
        that fit share one buffer size, which MLX's allocator then reuses; a larger prompt still
        runs but gets a fresh buffer."""
        self.workspace = max(self.workspace, n_blocks)
        if self.output == "exact":
            self._calibrate()

    @property
    def workspace_bytes(self) -> int:
        return self.workspace * self.block_bytes

    @property
    def block_bytes(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * BLOCK * self.head_dim * self.dtype.size

    def tokens(self, keys: Sequence[int]) -> np.ndarray:
        return block_tokens(keys, self.vocab, self.seed)

    def new_pool(self, n_blocks: int) -> BlockPool:
        return BlockPool(self.n_layers, self.n_kv_heads, self.head_dim, n_blocks, self.dtype)

    @contextmanager
    def _row_stable(self, on: bool) -> Iterator[None]:
        prev, RowStableLinear.active = RowStableLinear.active, on
        try:
            yield
        finally:
            RowStableLinear.active = prev

    def _head(self, h: mx.array) -> mx.array:
        m = self.model
        if m.args.tie_word_embeddings:
            return m.model.embed_tokens.as_linear(h)
        return m.lm_head(h)

    def _forward(
        self, x: mx.array, cache: list[KVCache], logits: bool, stable: bool = False
    ) -> mx.array | None:
        n = x.shape[1]
        with self._row_stable(stable):
            for s in range(0, n, self.chunk):
                h = self.model.model(x[:, s : s + self.chunk], cache)
                if s + self.chunk < n:
                    mx.eval([c.keys for c in cache], [c.values for c in cache])
        # float32 head: bf16 logits would tie often enough to blur the argmax comparison.
        return self._head(h[:, -1, :].astype(mx.float32))[0] if logits else None

    def run(
        self,
        pool: BlockPool | None,
        ctx: Sequence[int],
        new: Sequence[int],
        logits: bool = True,
    ) -> Prefill:
        """Prefill `new` blocks on top of the pooled KV of the cached prefix `ctx`.

        `seconds` covers what precedes the first token: gather, forward and argmax for a cached
        forward, the full prefill when the output mode recomputes. With no new blocks (a full
        hit) the cached forward recomputes the last token (the last block in exact mode) to get
        its logits; that KV is not written back.
        """
        n_ctx, n_new = len(ctx), len(new)
        if n_ctx + n_new == 0:
            raise ValueError("empty prompt")
        if logits and n_ctx and self.output == "verified":
            return self._recompute([*ctx, *new], n_ctx)
        if self.output == "exact":
            if logits and not self.exact_supported(n_ctx + n_new):
                return self._recompute([*ctx, *new], n_ctx, stable_kv=True)
            return self._cached(pool, ctx, new, logits, stable=True)
        return self._cached(pool, ctx, new, logits, stable=False)

    def cached_logits(self, pool: BlockPool, ctx: Sequence[int], new: Sequence[int]) -> mx.array:
        """Logits of this mode's cached-prefix forward, whatever the mode serves."""
        return self._cached(pool, ctx, new, True, stable=self.output == "exact").logits

    def full_logits(self, keys: Sequence[int]) -> mx.array:
        """Last-position logits of the whole prompt prefilled from scratch (float32)."""
        return self._cached(None, [], keys, True, stable=False).logits

    def _new_kv(
        self, cache: list[KVCache], base: int, n_new: int
    ) -> tuple[list[mx.array], list[mx.array]]:
        end, shape = base + BLOCK * n_new, (self.n_kv_heads, n_new, BLOCK, self.head_dim)
        k_new = [c.keys[0, :, base:end].reshape(shape).transpose(1, 0, 2, 3) for c in cache]
        v_new = [c.values[0, :, base:end].reshape(shape).transpose(1, 0, 2, 3) for c in cache]
        return k_new, v_new

    def _cached(
        self,
        pool: BlockPool | None,
        ctx: Sequence[int],
        new: Sequence[int],
        logits: bool,
        stable: bool,
    ) -> Prefill:
        n_ctx, n_new = len(ctx), len(new)
        base = BLOCK * n_ctx
        redo = 0 if n_new else BLOCK if stable else 1
        ids = self.tokens(new) if n_new else self.tokens(ctx[-1:])[BLOCK - redo :]
        x = mx.array(ids[None])
        t0 = perf_counter()
        size = max(self.workspace, n_ctx + n_new)
        ks, vs = (self._scratch if pool is None else pool).gather(ctx, size)
        mx.eval(ks, vs)
        t1 = perf_counter()
        # The caches must hold the only references so their in-place slice updates can donate.
        cache = [_cache(k, v, base - redo) for k, v in zip(ks, vs, strict=True)]
        del ks, vs
        out = self._forward(x, cache, logits, stable)
        y = mx.argmax(out) if logits else None
        mx.eval(y, [c.keys for c in cache], [c.values for c in cache])
        t2 = perf_counter()
        k_new, v_new = self._new_kv(cache, base, n_new)
        return Prefill(
            token=y.item() if logits else None,
            seconds=t2 - t0,
            gather_s=t1 - t0,
            new_tokens=cache[0].offset - base,
            forward_tokens=x.shape[1],
            logits=out,
            k_new=k_new,
            v_new=v_new,
            cached_logits=out,
        )

    def _recompute(self, keys: Sequence[int], n_ctx: int, stable_kv: bool = False) -> Prefill:
        """Full prefill of `keys` for the output; the new blocks' KV comes from the same
        forward, or, with `stable_kv` and a prompt outside the row-stable regime, from a
        row-stable forward after the first token, so the pool only ever holds KV that a later
        exact cached forward can build on."""
        full = self._cached(None, [], keys, True, stable=False)
        k_new = [a[n_ctx:] for a in full.k_new]
        v_new = [a[n_ctx:] for a in full.v_new]
        if stable_kv and not self.exact_supported(len(keys)) and len(keys) > n_ctx:
            kv = self._cached(None, [], keys, False, stable=True)
            k_new = [a[n_ctx:] for a in kv.k_new]
            v_new = [a[n_ctx:] for a in kv.v_new]
        return Prefill(
            token=full.token,
            seconds=full.seconds,
            gather_s=0.0,
            new_tokens=BLOCK * (len(keys) - n_ctx),
            forward_tokens=full.forward_tokens,
            logits=full.logits,
            k_new=k_new,
            v_new=v_new,
        )

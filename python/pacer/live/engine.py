"""MLX prefill engine over the block pool.

Prompts are synthetic: every block key owns 16 token ids drawn from a hash of the key, so a key
always denotes the same tokens (and, since keys are prefix-chain positions, the same KV). A
forward starts from the gathered KV of a cached block prefix and runs only the new blocks; the
new blocks' KV is then sliced out of the mlx-lm caches for the pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from .pool import BLOCK, BlockPool

DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_CHUNK = 2048

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


class Engine:
    def __init__(self, model: str = DEFAULT_MODEL, chunk: int = DEFAULT_CHUNK, seed: int = 0):
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

    def reserve(self, n_blocks: int) -> None:
        """Sizes the per-forward KV workspace (in blocks) to fit every prompt to come. Prompts
        that fit share one buffer size, which MLX's allocator then reuses; a larger prompt still
        runs but gets a fresh buffer."""
        self.workspace = max(self.workspace, n_blocks)

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

    def _head(self, h: mx.array) -> mx.array:
        m = self.model
        if m.args.tie_word_embeddings:
            return m.model.embed_tokens.as_linear(h)
        return m.lm_head(h)

    def _forward(self, x: mx.array, cache: list[KVCache], logits: bool) -> mx.array | None:
        n = x.shape[1]
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

        With no new blocks (a full hit) the last prompt token is recomputed to get its logits,
        as servers do; its KV is not written back. `seconds` covers gather, forward and argmax.
        """
        n_ctx, n_new = len(ctx), len(new)
        if n_ctx + n_new == 0:
            raise ValueError("empty prompt")
        base = BLOCK * n_ctx
        redo = int(n_new == 0)
        ids = self.tokens(ctx[-1:])[-1:] if redo else self.tokens(new)
        x = mx.array(ids[None])
        t0 = perf_counter()
        size = max(self.workspace, n_ctx + n_new)
        ks, vs = (self._scratch if pool is None else pool).gather(ctx, size)
        mx.eval(ks, vs)
        t1 = perf_counter()
        # The caches must hold the only references so their in-place slice updates can donate.
        cache = [_cache(k, v, base - redo) for k, v in zip(ks, vs, strict=True)]
        del ks, vs
        out = self._forward(x, cache, logits)
        y = mx.argmax(out) if logits else None
        mx.eval(y, [c.keys for c in cache], [c.values for c in cache])
        t2 = perf_counter()
        end, shape = base + BLOCK * n_new, (self.n_kv_heads, n_new, BLOCK, self.head_dim)
        k_new = [c.keys[0, :, base:end].reshape(shape).transpose(1, 0, 2, 3) for c in cache]
        v_new = [c.values[0, :, base:end].reshape(shape).transpose(1, 0, 2, 3) for c in cache]
        return Prefill(
            token=y.item() if logits else None,
            seconds=t2 - t0,
            gather_s=t1 - t0,
            new_tokens=cache[0].offset - base,
            forward_tokens=x.shape[1],
            logits=out,
            k_new=k_new,
            v_new=v_new,
        )

    def full_logits(self, keys: Sequence[int]) -> mx.array:
        """Last-position logits of the whole prompt prefilled from scratch (float32)."""
        return self.run(None, [], keys).logits

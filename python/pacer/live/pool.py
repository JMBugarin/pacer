"""Paged KV block pool: the data plane that mirrors the Rust controller's inserts and evictions.

Each layer has preallocated K and V arrays [n_blocks, n_kv_heads, BLOCK, head_dim]. A block key
maps to a slot; freeing a key returns its slot to the free list without touching the data. The
pool can never hold more than `n_blocks` keys. Gathers use a Metal kernel that copies each
(block, head) chunk with 16-byte loads, about 3x faster than mx.take on the transposed pool.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

BLOCK = 16


_GATHER = """
    uint n = meta[0];
    uint gid = thread_position_in_grid.x;
    if (gid >= n * HEADS * PER) return;
    uint c = gid / PER;
    uint w = gid - c * PER;
    uint h = c / n;
    uint i = c - h * n;
    const device uint4* src = (const device uint4*)pool;
    device uint4* dst = (device uint4*)out;
    dst[(h * meta[1] + i) * PER + w] = src[(slots[i] * HEADS + h) * PER + w];
"""
_kernel = None


def _gather_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="pacer_block_gather",
            input_names=["pool", "slots", "meta"],
            output_names=["out"],
            source=_GATHER,
        )
    return _kernel


class PoolFull(RuntimeError):
    pass


class BlockPool:
    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_blocks: int,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        shape = (n_blocks, n_kv_heads, BLOCK, head_dim)
        self.k = [mx.zeros(shape, dtype) for _ in range(n_layers)]
        self.v = [mx.zeros(shape, dtype) for _ in range(n_layers)]
        mx.eval(self.k, self.v)
        self.n_blocks = n_blocks
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.slot: dict[int, int] = {}
        self._free: list[int] = []
        self.peak = 0
        self.reset()

    def __len__(self) -> int:
        return len(self.slot)

    def __contains__(self, key: int) -> bool:
        return key in self.slot

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k) + sum(a.nbytes for a in self.v)

    def reset(self) -> None:
        self.slot.clear()
        self._free = list(range(self.n_blocks - 1, -1, -1))
        self.peak = 0

    def free(self, keys: Sequence[int]) -> None:
        for k in keys:
            self._free.append(self.slot.pop(k))

    def _alloc(self, keys: Sequence[int]) -> list[int]:
        if len(keys) > len(self._free):
            raise PoolFull(f"need {len(keys)} slots, {len(self._free)} free of {self.n_blocks}")
        dup = [k for k in keys if k in self.slot]
        if dup:
            raise KeyError(f"blocks already in pool: {dup[:8]}")
        slots = [self._free.pop() for _ in keys]
        self.slot.update(zip(keys, slots, strict=True))
        self.peak = max(self.peak, len(self.slot))
        return slots

    def write(
        self, keys: Sequence[int], k_blocks: Sequence[mx.array], v_blocks: Sequence[mx.array]
    ) -> None:
        """Allocates `keys` and fills them; k_blocks[l] is [len(keys), H, BLOCK, D]."""
        if not keys:
            return
        idx = mx.array(self._alloc(keys), dtype=mx.uint32)
        for layer in range(len(self.k)):
            self.k[layer][idx] = k_blocks[layer]
            self.v[layer][idx] = v_blocks[layer]
        mx.eval(self.k, self.v)

    def gather(self, keys: Sequence[int], size: int) -> tuple[list[mx.array], list[mx.array]]:
        """Per-layer K and V of `keys` as the start of a sequence [1, H, size * BLOCK, D].

        Blocks past len(keys) are left uninitialized: they are the working space the next
        forward writes its new tokens into, so they are overwritten before being read. A fixed
        `size` lets MLX reuse the same buffers across calls instead of faulting in new memory.
        """
        n = len(keys)
        if size < n:
            raise ValueError(f"size {size} < {n} blocks")
        slots = mx.array([self.slot[k] for k in keys] or [0], dtype=mx.uint32)
        meta = mx.array([n, size], dtype=mx.uint32)
        per = BLOCK * self.head_dim * self.dtype.size // 16
        threads = max(n * self.n_kv_heads * per, 1)
        shape = (1, self.n_kv_heads, size * BLOCK, self.head_dim)
        kernel = _gather_kernel()

        def take(a: mx.array) -> mx.array:
            return kernel(
                inputs=[a, slots, meta],
                template=[("HEADS", self.n_kv_heads), ("PER", per)],
                grid=(threads, 1, 1),
                threadgroup=(min(256, threads), 1, 1),
                output_shapes=[shape],
                output_dtypes=[self.dtype],
            )[0]

        return [take(a) for a in self.k], [take(a) for a in self.v]

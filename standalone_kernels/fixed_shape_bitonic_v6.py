"""Fixed-shape TPU-v6 bitonic workarounds.

These kernels avoid the original bitonic Pallas rank-3 reshape by keeping all
kernel-local arrays in `(16, 256)` tiles and expressing compare-exchange with
static column slices plus concatenate.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .common import (
    NEG_INF,
    V6_TILE_ROWS,
    V6_TILE_COLS,
    V6_TILE_SIZE,
    next_power_of_2,
    sanitize_scores as _sanitize_scores,
)

def _row_bitonic_sort_16x256(keys: jax.Array, inds: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sort each 256-column row with static slice/concat compare-exchange.

    This intentionally avoids rank-changing reshapes inside the Pallas kernel.
    Keys are sorted ascending; callers pass -scores so the output is score
    descending, with smaller original index as the deterministic tie-breaker.
    """
    size = 2
    while size <= V6_TILE_COLS:
        stride = size // 2
        while stride >= 1:
            key_chunks = []
            idx_chunks = []
            for block in range(0, V6_TILE_COLS, 2 * stride):
                a_key = keys[:, block:block + stride]
                b_key = keys[:, block + stride:block + 2 * stride]
                a_idx = inds[:, block:block + stride]
                b_idx = inds[:, block + stride:block + 2 * stride]
                a_less = (a_key < b_key) | ((a_key == b_key) & (a_idx < b_idx))
                min_key = jnp.where(a_less, a_key, b_key)
                min_idx = jnp.where(a_less, a_idx, b_idx)
                max_key = jnp.where(a_less, b_key, a_key)
                max_idx = jnp.where(a_less, b_idx, a_idx)
                asc = (block & size) == 0
                first_key = min_key if asc else max_key
                first_idx = min_idx if asc else max_idx
                second_key = max_key if asc else min_key
                second_idx = max_idx if asc else min_idx
                key_chunks.extend((first_key, second_key))
                idx_chunks.extend((first_idx, second_idx))
            keys = jnp.concatenate(key_chunks, axis=1)
            inds = jnp.concatenate(idx_chunks, axis=1)
            stride //= 2
        size *= 2
    return keys, inds


def _bitonic_row256_kernel(row_k: int):
    def kernel(x_ref, vals_ref, idx_ref):
        bid = pl.program_id(0)
        base = (bid * V6_TILE_SIZE).astype(jnp.int32)
        row = jnp.arange(V6_TILE_ROWS, dtype=jnp.int32)[:, None]
        col = jnp.arange(V6_TILE_COLS, dtype=jnp.int32)[None, :]
        global_idx = base + row * jnp.int32(V6_TILE_COLS) + col
        keys = -_sanitize_scores(x_ref[...])
        keys, inds = _row_bitonic_sort_16x256(keys, global_idx)
        vals = -keys
        keep = col < jnp.int32(row_k)
        vals_ref[...] = jnp.where(keep, vals, NEG_INF)
        idx_ref[...] = jnp.where(keep, inds, -1)
    return kernel


@partial(jax.jit, static_argnames=("k", "row_k"))
def bitonic_row256_pallas_v6(
    x: jax.Array,
    *,
    k: int,
    row_k: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """A: row-local v6 bitonic candidates without rank-changing reshape.

    Each [16, 256] tile row is independently bitonic-sorted. The kernel emits
    row_k candidates per row, then XLA top_k merges the candidate matrix. This
    is exact when no row contributes more than row_k global winners; row_k >= k
    is exact for a single row but still an approximation across rows.
    """
    n_orig = x.shape[0]
    num_tiles = (n_orig + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    n_pad = num_tiles * V6_TILE_SIZE
    if row_k is None:
        row_k = min(k, V6_TILE_COLS)
    if row_k < k:
        raise ValueError("row_k must be >= k for the default exactness guard")
    if row_k > V6_TILE_COLS:
        raise ValueError("row_k must be <= 256 for row-local v6 bitonic")
    x_pad = jnp.pad(_sanitize_scores(x), (0, n_pad - n_orig), constant_values=NEG_INF)
    x_tiles = x_pad.reshape((num_tiles * V6_TILE_ROWS, V6_TILE_COLS))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda b: (b, 0))],
        out_specs=(
            pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda b: (b, 0)),
            pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda b: (b, 0)),
        ),
        grid=(num_tiles,),
    )
    cand_vals, cand_idx = pl.pallas_call(
        _bitonic_row256_kernel(row_k),
        out_shape=(
            jax.ShapeDtypeStruct((num_tiles * V6_TILE_ROWS, V6_TILE_COLS), x.dtype),
            jax.ShapeDtypeStruct((num_tiles * V6_TILE_ROWS, V6_TILE_COLS), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x_tiles)
    flat_vals = cand_vals.reshape((num_tiles * V6_TILE_SIZE,))
    flat_idx = cand_idx.reshape((num_tiles * V6_TILE_SIZE,))
    vals, pos = lax.top_k(flat_vals, k)
    idx = flat_idx[pos].astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx


def _bitonic_tile_partial_kernel(local_k: int, local_k_pad: int, row_k: int):
    def kernel(x_ref, vals_ref, idx_ref):
        bid = pl.program_id(0)
        base = (bid * V6_TILE_SIZE).astype(jnp.int32)
        row = jnp.arange(V6_TILE_ROWS, dtype=jnp.int32)[:, None]
        col = jnp.arange(V6_TILE_COLS, dtype=jnp.int32)[None, :]
        global_idx = base + row * jnp.int32(V6_TILE_COLS) + col
        keys = -_sanitize_scores(x_ref[...])
        keys, inds = _row_bitonic_sort_16x256(keys, global_idx)
        neg_inf = jnp.array(NEG_INF, dtype=jnp.float32)
        work = jnp.where(col < jnp.int32(row_k), (-keys).astype(jnp.float32), neg_inf)
        vals = []
        idx_out = []
        for _ in range(local_k):
            m = jnp.max(work, axis=(0, 1))
            candidates = jnp.where(work == m, inds, jnp.iinfo(jnp.int32).max)
            arg = jnp.min(candidates, axis=(0, 1)).astype(jnp.int32)
            vals.append(m)
            idx_out.append(arg)
            work = jnp.where(inds == arg, neg_inf, work)
        vals_ref[...] = jnp.pad(
            jnp.stack(vals),
            (0, local_k_pad - local_k),
            constant_values=neg_inf,
        )
        idx_ref[...] = jnp.pad(
            jnp.stack(idx_out),
            (0, local_k_pad - local_k),
            constant_values=-1,
        )
    return kernel


@partial(jax.jit, static_argnames=("k", "local_k", "row_k"))
def bitonic_tile_partial_pallas_v6(
    x: jax.Array,
    *,
    k: int,
    local_k: int | None = None,
    row_k: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """B: row bitonic prefilter plus tile-local iterative top-k.

    Rows are sorted by the reshape-free 256-lane bitonic network, only row_k
    candidates per row remain active, then each tile emits local_k candidates.
    Exactness requires both row_k and local_k to cover the concentration of true
    winners inside each row/tile; with row_k >= k and local_k >= k it is exact
    for the usual per-vector test cases but still has adversarial row/tile skew
    limits when used as a bucket approximation.
    """
    n_orig = x.shape[0]
    num_tiles = (n_orig + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    n_pad = num_tiles * V6_TILE_SIZE
    if local_k is None:
        local_k = min(k, V6_TILE_SIZE)
    if row_k is None:
        row_k = min(max(k, local_k), V6_TILE_COLS)
    if local_k < k:
        raise ValueError("local_k must be >= k for the default exactness guard")
    if row_k > V6_TILE_COLS:
        raise ValueError("row_k must be <= 256 for row-local v6 bitonic")
    local_k_pad = max(
        V6_TILE_COLS,
        ((local_k + V6_TILE_COLS - 1) // V6_TILE_COLS) * V6_TILE_COLS,
    )
    x_pad = jnp.pad(_sanitize_scores(x), (0, n_pad - n_orig), constant_values=NEG_INF)
    x_tiles = x_pad.reshape((num_tiles * V6_TILE_ROWS, V6_TILE_COLS))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda b: (b, 0))],
        out_specs=(
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
        ),
        grid=(num_tiles,),
    )
    cand_vals, cand_idx = pl.pallas_call(
        _bitonic_tile_partial_kernel(local_k, local_k_pad, row_k),
        out_shape=(
            jax.ShapeDtypeStruct((num_tiles * local_k_pad,), jnp.float32),
            jax.ShapeDtypeStruct((num_tiles * local_k_pad,), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x_tiles)
    vals, pos = lax.top_k(cand_vals, k)
    idx = cand_idx[pos].astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx


def _candidate_merge_v6_kernel(local_k: int, local_k_pad: int):
    def kernel(vals_in_ref, idx_in_ref, vals_ref, idx_ref):
        work = vals_in_ref[...].astype(jnp.float32)
        neg_inf = jnp.array(NEG_INF, dtype=jnp.float32)
        inds = idx_in_ref[...]
        vals = []
        idx_out = []
        for _ in range(local_k):
            m = jnp.max(work, axis=(0, 1))
            candidates = jnp.where(work == m, inds, jnp.iinfo(jnp.int32).max)
            arg = jnp.min(candidates, axis=(0, 1)).astype(jnp.int32)
            vals.append(m)
            idx_out.append(arg)
            work = jnp.where(inds == arg, neg_inf, work)
        vals_ref[...] = jnp.pad(
            jnp.stack(vals),
            (0, local_k_pad - local_k),
            constant_values=neg_inf,
        )
        idx_ref[...] = jnp.pad(
            jnp.stack(idx_out),
            (0, local_k_pad - local_k),
            constant_values=-1,
        )
    return kernel


def _merge_candidate_rows_v6(
    cand_vals: jax.Array,
    cand_idx: jax.Array,
    *,
    local_k: int,
    local_k_pad: int,
) -> tuple[jax.Array, jax.Array]:
    num_rows = cand_vals.shape[0]
    row_pad = ((num_rows + V6_TILE_ROWS - 1) // V6_TILE_ROWS) * V6_TILE_ROWS
    if row_pad != num_rows:
        cand_vals = jnp.pad(
            cand_vals,
            ((0, row_pad - num_rows), (0, 0)),
            constant_values=NEG_INF,
        )
        cand_idx = jnp.pad(
            cand_idx,
            ((0, row_pad - num_rows), (0, 0)),
            constant_values=-1,
        )
    num_groups = row_pad // V6_TILE_ROWS
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda b: (b, 0)),
            pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda b: (b, 0)),
        ],
        out_specs=(
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
        ),
        grid=(num_groups,),
    )
    return pl.pallas_call(
        _candidate_merge_v6_kernel(local_k, local_k_pad),
        out_shape=(
            jax.ShapeDtypeStruct((num_groups * local_k_pad,), cand_vals.dtype),
            jax.ShapeDtypeStruct((num_groups * local_k_pad,), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(cand_vals, cand_idx)


@partial(jax.jit, static_argnames=("k", "local_k", "row_k"))
def bitonic_hierarchical_pallas_v6(
    x: jax.Array,
    *,
    k: int,
    local_k: int | None = None,
    row_k: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """C: multi-stage fixed-shape Pallas candidate merge.

    Stage 1 uses the reshape-free row bitonic prefilter plus tile-local top-k.
    Later stages repeatedly merge 16 candidate rows of width 256 with a fixed
    [16, 256] Pallas block until only one candidate row remains. This avoids
    the final XLA top_k merge at the cost of additional Pallas kernel launches.
    """
    n_orig = x.shape[0]
    num_tiles = (n_orig + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    n_pad = num_tiles * V6_TILE_SIZE
    if local_k is None:
        local_k = min(k, V6_TILE_COLS)
    if row_k is None:
        row_k = min(max(k, local_k), V6_TILE_COLS)
    if local_k < k:
        raise ValueError("local_k must be >= k for exact hierarchical merge")
    if local_k > V6_TILE_COLS:
        raise ValueError("C currently supports local_k <= 256")
    if row_k > V6_TILE_COLS:
        raise ValueError("row_k must be <= 256 for row-local v6 bitonic")
    local_k_pad = V6_TILE_COLS
    x_pad = jnp.pad(_sanitize_scores(x), (0, n_pad - n_orig), constant_values=NEG_INF)
    x_tiles = x_pad.reshape((num_tiles * V6_TILE_ROWS, V6_TILE_COLS))
    stage1_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((V6_TILE_ROWS, V6_TILE_COLS), lambda b: (b, 0))],
        out_specs=(
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
        ),
        grid=(num_tiles,),
    )
    cand_vals, cand_idx = pl.pallas_call(
        _bitonic_tile_partial_kernel(local_k, local_k_pad, row_k),
        out_shape=(
            jax.ShapeDtypeStruct((num_tiles * local_k_pad,), jnp.float32),
            jax.ShapeDtypeStruct((num_tiles * local_k_pad,), jnp.int32),
        ),
        grid_spec=stage1_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x_tiles)
    num_rows = num_tiles
    cand_vals = cand_vals.reshape((num_rows, local_k_pad))
    cand_idx = cand_idx.reshape((num_rows, local_k_pad))
    while num_rows > 1:
        cand_vals, cand_idx = _merge_candidate_rows_v6(
            cand_vals,
            cand_idx,
            local_k=local_k,
            local_k_pad=local_k_pad,
        )
        num_rows = (num_rows + V6_TILE_ROWS - 1) // V6_TILE_ROWS
        cand_vals = cand_vals.reshape((num_rows, local_k_pad))
        cand_idx = cand_idx.reshape((num_rows, local_k_pad))
    vals = cand_vals.reshape((local_k_pad,))[:k]
    idx = cand_idx.reshape((local_k_pad,))[:k].astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx

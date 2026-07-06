"""Bucket-select top-k implementations.

Each block/tile emits local candidates using iterative max-mask. XLA top_k
then merges the candidate buffer. The v6 variant uses fixed `(16, 256)` input
tiles matching TPU v6 layout constraints.
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

def _block_iterative_kernel(local_k: int, local_k_pad: int, block_size: int):
    def kernel(x_ref, vals_ref, idx_ref):
        bid = pl.program_id(0)
        base = (bid * block_size).astype(jnp.int32)
        local_pos = jnp.arange(block_size, dtype=jnp.int32)
        global_idx = base + local_pos
        work = _sanitize_scores(x_ref[...]).astype(jnp.float32)
        neg_inf = jnp.array(NEG_INF, dtype=jnp.float32)
        vals = []
        idx_out = []
        for i in range(local_k):
            m = jnp.max(work, axis=0)
            candidates = jnp.where(work == m, global_idx, jnp.iinfo(jnp.int32).max)
            arg = jnp.min(candidates).astype(jnp.int32)
            vals.append(m)
            idx_out.append(arg)
            work = jnp.where(global_idx == arg, neg_inf, work)
        vals_ref[...] = jnp.pad(jnp.stack(vals), (0, local_k_pad - local_k), constant_values=neg_inf)
        idx_ref[...] = jnp.pad(jnp.stack(idx_out), (0, local_k_pad - local_k), constant_values=-1)
    return kernel


@partial(jax.jit, static_argnames=("k", "block_size", "local_k"))
def bucket_select_pallas(
    x: jax.Array,
    *,
    k: int,
    block_size: int = V6_TILE_SIZE,
    local_k: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Two-level block candidate selection.

    This is exact when local_k >= min(k, block_size). Smaller local_k is a bucket
    approximation that can fail on long-tail/adversarial distributions when more
    than local_k global winners land in one block.
    """
    n_orig = x.shape[0]
    num_blocks = (n_orig + block_size - 1) // block_size
    n_pad = num_blocks * block_size
    if local_k is None:
        local_k = min(k, block_size)
    local_k_pad = max(V6_TILE_COLS, ((local_k + V6_TILE_COLS - 1) // V6_TILE_COLS) * V6_TILE_COLS)
    x_pad = jnp.pad(_sanitize_scores(x), (0, n_pad - n_orig), constant_values=NEG_INF)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((block_size,), lambda b: (b,))],
        out_specs=(
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
            pl.BlockSpec((local_k_pad,), lambda b: (b,)),
        ),
        grid=(num_blocks,),
    )
    cand_vals, cand_idx = pl.pallas_call(
        _block_iterative_kernel(local_k, local_k_pad, block_size),
        out_shape=(
            jax.ShapeDtypeStruct((num_blocks * local_k_pad,), jnp.float32),
            jax.ShapeDtypeStruct((num_blocks * local_k_pad,), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x_pad)
    flat_vals = cand_vals
    flat_idx = cand_idx
    vals, pos = lax.top_k(flat_vals, k)
    idx = flat_idx[pos].astype(jnp.int32)
    idx = jnp.where(idx < n_orig, idx, -1).astype(jnp.int32)
    return vals.astype(x.dtype), idx


def _v6_tile_iterative_kernel(local_k: int, local_k_pad: int):
    def kernel(x_ref, vals_ref, idx_ref):
        bid = pl.program_id(0)
        base = (bid * V6_TILE_SIZE).astype(jnp.int32)
        row = jnp.arange(V6_TILE_ROWS, dtype=jnp.int32)[:, None]
        col = jnp.arange(V6_TILE_COLS, dtype=jnp.int32)[None, :]
        local_pos = row * jnp.int32(V6_TILE_COLS) + col
        global_idx = base + local_pos
        work = _sanitize_scores(x_ref[...]).astype(jnp.float32)
        neg_inf = jnp.array(NEG_INF, dtype=jnp.float32)
        vals = []
        idx_out = []
        for i in range(local_k):
            m = jnp.max(work, axis=(0, 1))
            candidates = jnp.where(work == m, global_idx, jnp.iinfo(jnp.int32).max)
            arg = jnp.min(candidates, axis=(0, 1)).astype(jnp.int32)
            vals.append(m)
            idx_out.append(arg)
            work = jnp.where(global_idx == arg, neg_inf, work)
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


@partial(jax.jit, static_argnames=("k", "local_k"))
def bucket_select_pallas_v6_tile(
    x: jax.Array,
    *,
    k: int,
    local_k: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Bucket select using TPU v6-friendly [16, 256] input tiles.

    Each Pallas program consumes one 4096-element tile shaped as [16, 256],
    emits a 256-column-aligned local candidate row, and XLA top_k merges the
    candidate buffer. Exactness requires local_k >= min(k, V6_TILE_SIZE).
    """
    n_orig = x.shape[0]
    num_tiles = (n_orig + V6_TILE_SIZE - 1) // V6_TILE_SIZE
    n_pad = num_tiles * V6_TILE_SIZE
    if local_k is None:
        local_k = min(k, V6_TILE_SIZE)
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
        _v6_tile_iterative_kernel(local_k, local_k_pad),
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

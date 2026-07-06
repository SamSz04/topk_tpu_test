"""TopK custom-call inspired implementations for TPU.

This file follows the algorithmic structure documented in
`Docs/jax_topk_tpu_llo_analysis.md`: reshape to a 2D `[rows, n]` layout,
then perform `k` rounds of stable argmax with winner deletion.

The exact TPU LLO custom-call uses ordered integer keys, `vxpose`, and
`vrot.slane`. Pallas does not expose those instructions directly here, so the
Pallas implementation below mimics the semantic structure rather than the exact
vector-register schedule.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .common import NEG_INF, sanitize_scores


@partial(jax.jit, static_argnames=("k",))
def xla_topk_2d_custom_call_hint(
    x: jax.Array,
    *,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    """Call `lax.top_k` after flattening leading dims into a 2D matrix.

    TPU XLA more often lowers small-k, large-last-dimension top_k to the
    `TopK` custom-call when the operand is shaped as `[rows, n]`. This wrapper
    preserves semantics while nudging the lowering toward that path. It is the
    first optimization to try before replacing XLA with a custom Pallas kernel.
    """
    n = x.shape[-1]
    leading = x.shape[:-1]
    x2 = sanitize_scores(x).reshape((-1, n))
    vals, idx = lax.top_k(x2, k)
    out_shape = leading + (k,)
    return vals.reshape(out_shape), idx.astype(jnp.int32).reshape(out_shape)


def _stable_argmax_delete_kernel(k: int, n: int, block_rows: int):
    def kernel(x_ref, vals_ref, idx_ref):
        idxs = jnp.arange(n, dtype=jnp.int32)
        work = sanitize_scores(x_ref[...]).astype(jnp.float32)
        vals = []
        idx_out = []
        for _ in range(k):
            best_val = jnp.max(work, axis=1)
            # Stable tie-break: lower index wins for equal values within each row.
            candidates = jnp.where(work == best_val[:, None], idxs[None, :], jnp.int32(n))
            best_idx = jnp.min(candidates, axis=1).astype(jnp.int32)
            vals.append(best_val.astype(x_ref.dtype))
            idx_out.append(best_idx)
            work = jnp.where(idxs[None, :] == best_idx[:, None], NEG_INF, work)
        vals_ref[...] = jnp.stack(vals, axis=1)
        idx_ref[...] = jnp.stack(idx_out, axis=1)
    return kernel


@partial(jax.jit, static_argnames=("k",))
def repeated_argmax_topk_pallas_2d(
    x: jax.Array,
    *,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    """Pallas mimic of TPU TopK custom-call repeated stable argmax.

    The input may be 1D or have arbitrary leading dimensions. The wrapper
    flattens leading dimensions to `[rows, n]`, launches one Pallas program per
    1-row or 8-row group, and reshapes outputs back to `x.shape[:-1] + (k,)`.

    This is exact for hard top-k with NaNs masked to `-inf`. It mimics the
    custom-call control flow, but it does not reproduce LLO-specific ordered
    integer key packing or lane-rotation scheduling.
    """
    n = x.shape[-1]
    leading = x.shape[:-1]
    x2 = sanitize_scores(x).reshape((-1, n))
    rows = x2.shape[0]
    block_rows = 1 if rows == 1 else 8
    row_groups = (rows + block_rows - 1) // block_rows
    row_pad = row_groups * block_rows
    if row_pad != rows:
        x2 = jnp.pad(x2, ((0, row_pad - rows), (0, 0)), constant_values=NEG_INF)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((block_rows, n), lambda g: (g, 0))],
        out_specs=(
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
        ),
        grid=(row_groups,),
    )
    vals, idx = pl.pallas_call(
        _stable_argmax_delete_kernel(k, n, block_rows),
        out_shape=(
            jax.ShapeDtypeStruct((row_pad, k), x.dtype),
            jax.ShapeDtypeStruct((row_pad, k), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x2)
    out_shape = leading + (k,)
    vals = vals[:rows, :]
    idx = idx[:rows, :]
    return vals.reshape(out_shape), idx.reshape(out_shape).astype(jnp.int32)



def _select_better_pair(cand_val, cand_idx, best_val, best_idx):
    take = (cand_val > best_val) | ((cand_val == best_val) & (cand_idx < best_idx))
    out_val = jnp.where(take, cand_val, best_val)
    out_idx = jnp.where(take, cand_idx, best_idx)
    return out_val, out_idx


def _reduce_128_pair(vals, idxs):
    """Static pair reduction over 128 columns using compare-select stages."""
    best_val = vals
    best_idx = idxs
    width = 128
    while width > 1:
        half = width // 2
        left_val = best_val[:, :half]
        right_val = best_val[:, half:width]
        left_idx = best_idx[:, :half]
        right_idx = best_idx[:, half:width]
        best_val, best_idx = _select_better_pair(right_val, right_idx, left_val, left_idx)
        width = half
    return best_val[:, 0], best_idx[:, 0]


def _llo_style_argmax_delete_kernel(k: int, n: int, block_rows: int):
    def kernel(x_ref, vals_ref, idx_ref):
        work = sanitize_scores(x_ref[...]).astype(jnp.float32)
        full_idx = jnp.arange(n, dtype=jnp.int32)
        for rank in range(k):
            best_val = jnp.full((block_rows,), NEG_INF, dtype=jnp.float32)
            best_idx = jnp.full((block_rows,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
            for start in range(0, n, 128):
                chunk_val = work[:, start:start + 128]
                chunk_idx = jnp.broadcast_to(
                    full_idx[start:start + 128][None, :],
                    (block_rows, 128),
                )
                chunk_best_val, chunk_best_idx = _reduce_128_pair(chunk_val, chunk_idx)
                best_val, best_idx = _select_better_pair(
                    chunk_best_val,
                    chunk_best_idx,
                    best_val,
                    best_idx,
                )
            vals_ref[:, rank] = best_val.astype(x_ref.dtype)
            idx_ref[:, rank] = best_idx
            work = jnp.where(full_idx[None, :] == best_idx[:, None], NEG_INF, work)
    return kernel


@partial(jax.jit, static_argnames=("k",))
def llo_style_repeated_argmax_pallas_2d(
    x: jax.Array,
    *,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    """LLO-style Pallas TopK: explicit 128-lane pair reductions.

    This implementation follows the custom-call analysis more closely than
    `repeated_argmax_topk_pallas_2d`: each rank performs explicit stable
    pair reduction over 128-column chunks with `(value desc, index asc)`
    compare-select, then deletes the winner before the next rank.

    Constraints: the last dimension must be divisible by 128, matching TPU
    vector-tile granularity used by the analyzed LLO path.
    """
    n = x.shape[-1]
    if n % 128 != 0:
        raise ValueError("llo_style_repeated_argmax_pallas_2d requires n % 128 == 0")
    leading = x.shape[:-1]
    x2 = sanitize_scores(x).reshape((-1, n))
    rows = x2.shape[0]
    block_rows = 1 if rows == 1 else 8
    row_groups = (rows + block_rows - 1) // block_rows
    row_pad = row_groups * block_rows
    if row_pad != rows:
        x2 = jnp.pad(x2, ((0, row_pad - rows), (0, 0)), constant_values=NEG_INF)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec((block_rows, n), lambda g: (g, 0))],
        out_specs=(
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
            pl.BlockSpec((block_rows, k), lambda g: (g, 0)),
        ),
        grid=(row_groups,),
    )
    vals, idx = pl.pallas_call(
        _llo_style_argmax_delete_kernel(k, n, block_rows),
        out_shape=(
            jax.ShapeDtypeStruct((row_pad, k), x.dtype),
            jax.ShapeDtypeStruct((row_pad, k), jnp.int32),
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x2)
    out_shape = leading + (k,)
    vals = vals[:rows, :]
    idx = idx[:rows, :]
    return vals.reshape(out_shape), idx.reshape(out_shape).astype(jnp.int32)
